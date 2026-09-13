#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>

namespace {

__device__ inline float half_to_float(const half value) { return __half2float(value); }

__global__ void small_q_attention_v0_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ output,
    int batch,
    int q_len,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages) {
  const int logical = blockIdx.x;
  const int qh = logical % num_query_heads;
  const int q_index = logical / num_query_heads;
  const int qi = q_index % q_len;
  const int b = q_index / q_len;
  if (b >= batch) return;

  const int d = threadIdx.x;
  if (d >= head_dim) return;

  const int group_size = num_query_heads / num_kv_heads;
  const int kvh = qh / group_size;
  const int total_len = seq_lens[b];
  const int valid_len = total_len - q_len + qi + 1;
  if (valid_len <= 0) return;

  const int q_base = ((b * q_len + qi) * num_query_heads + qh) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  // Read the query vector once per output lane. This v0 deliberately favors
  // readability; later variants will share Q/KV fragments across lanes.
  float max_score = -INFINITY;
  float denom = 0.0f;
  for (int key_index = 0; key_index < valid_len; ++key_index) {
    const int page_slot = key_index / page_size;
    const int page_offset = key_index % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;
    float score = 0.0f;
    for (int k = 0; k < head_dim; ++k) {
      score += half_to_float(query[q_base + k]) * half_to_float(key_cache[kv_base + k]);
    }
    score *= scale;
    const float new_max = fmaxf(max_score, score);
    denom = denom * expf(max_score - new_max) + expf(score - new_max);
    max_score = new_max;
  }

  float value = 0.0f;
  for (int key_index = 0; key_index < valid_len; ++key_index) {
    const int page_slot = key_index / page_size;
    const int page_offset = key_index % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;
    float score = 0.0f;
    for (int k = 0; k < head_dim; ++k) {
      score += half_to_float(query[q_base + k]) * half_to_float(key_cache[kv_base + k]);
    }
    score *= scale;
    value += expf(score - max_score) * half_to_float(value_cache[kv_base + d]);
  }

  const int out_base = ((b * q_len + qi) * num_query_heads + qh) * head_dim;
  output[out_base + d] = __float2half(value / denom);
}

// V1 changes only the score computation: one warp cooperatively evaluates
// each Q.K dot product, instead of repeating it once per output dimension.
__global__ void small_q_attention_v1_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ output,
    int batch,
    int q_len,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages) {
  const int logical = blockIdx.x;
  const int qh = logical % num_query_heads;
  const int q_index = logical / num_query_heads;
  const int qi = q_index % q_len;
  const int b = q_index / q_len;
  if (b >= batch) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int d = tid;
  if (d >= head_dim) return;

  const int group_size = num_query_heads / num_kv_heads;
  const int kvh = qh / group_size;
  const int total_len = seq_lens[b];
  const int valid_len = total_len - q_len + qi + 1;
  if (valid_len <= 0) return;

  const int q_base = ((b * q_len + qi) * num_query_heads + qh) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  __shared__ float score_shared;
  __shared__ float weight_shared;
  __shared__ float max_shared;
  __shared__ float denom_shared;

  if (tid == 0) {
    max_shared = -INFINITY;
    denom_shared = 0.0f;
  }
  __syncthreads();

  // First pass: warp 0 computes each score and thread 0 updates online softmax.
  for (int key_index = 0; key_index < valid_len; ++key_index) {
    const int page_slot = key_index / page_size;
    const int page_offset = key_index % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;
    float partial = 0.0f;
    if (warp == 0) {
      for (int k = lane; k < head_dim; k += 32) {
        partial += half_to_float(query[q_base + k]) * half_to_float(key_cache[kv_base + k]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
      }
      if (lane == 0) score_shared = partial * scale;
    }
    __syncthreads();
    if (tid == 0) {
      const float score = score_shared;
      const float new_max = fmaxf(max_shared, score);
      denom_shared = denom_shared * expf(max_shared - new_max) + expf(score - new_max);
      max_shared = new_max;
    }
    __syncthreads();
  }

  float value = 0.0f;
  // Second pass: recompute one shared score per key, then each lane loads V[d].
  for (int key_index = 0; key_index < valid_len; ++key_index) {
    const int page_slot = key_index / page_size;
    const int page_offset = key_index % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;
    float partial = 0.0f;
    if (warp == 0) {
      for (int k = lane; k < head_dim; k += 32) {
        partial += half_to_float(query[q_base + k]) * half_to_float(key_cache[kv_base + k]);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xffffffff, partial, offset);
      }
      if (lane == 0) score_shared = partial * scale;
    }
    __syncthreads();
    if (tid == 0) weight_shared = expf(score_shared - max_shared);
    __syncthreads();
    value += weight_shared * half_to_float(value_cache[kv_base + d]);
    __syncthreads();
  }

  const int out_base = ((b * q_len + qi) * num_query_heads + qh) * head_dim;
  output[out_base + d] = __float2half(value / denom_shared);
}

// V2 changes the mapping instead of the arithmetic. Two structural fixes over
// V0/V1:
//
//   1. The KV range of one (batch, query_row, query_head) output is split into
//      independent chunks, so the grid is no longer proportional to
//      batch * q_len * query_heads alone. At batch=1/q_len=2 that is 64 blocks
//      for V1 versus 64 * num_chunks for V2.
//   2. Inside a block, four warps stride over the chunk and each warp keeps its
//      own online-softmax state, so the KV loop needs no block-wide barrier at
//      all. V1 pays two __syncthreads() per key.
//
// The cost of split-KV is a second kernel that merges the per-chunk partials
// with a log-sum-exp combine. Partials are fp32; the merge writes fp16.

constexpr int V2_WARPS = 4;
constexpr int V2_THREADS = 32 * V2_WARPS;

__global__ void small_q_attention_v2_partial_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    int batch,
    int q_len,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages,
    int chunk_keys,
    int num_chunks) {
  const int logical = blockIdx.x;
  const int chunk = blockIdx.y;
  const int qh = logical % num_query_heads;
  const int q_index = logical / num_query_heads;
  const int qi = q_index % q_len;
  const int b = q_index / q_len;
  if (b >= batch) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group_size = num_query_heads / num_kv_heads;
  const int kvh = qh / group_size;
  const int total_len = seq_lens[b];
  const int valid_len = total_len - q_len + qi + 1;

  const int chunk_start = chunk * chunk_keys;
  const int out_base = (logical * num_chunks + chunk) * head_dim;

  // Chunks past the end of a short sequence still have to publish a neutral
  // partial so the merge kernel never reads uninitialised memory.
  if (valid_len <= 0 || chunk_start >= valid_len) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      partial_acc[out_base + lane + 32 * j] = 0.0f;
    }
    if (tid == 0) {
      partial_stats[(logical * num_chunks + chunk) * 2 + 0] = -INFINITY;
      partial_stats[(logical * num_chunks + chunk) * 2 + 1] = 0.0f;
    }
    return;
  }

  const int chunk_end = min(chunk_start + chunk_keys, valid_len);
  const int q_base = ((b * q_len + qi) * num_query_heads + qh) * head_dim;
  const float scale = rsqrtf(static_cast<float>(head_dim));

  // Each lane owns four head_dim slots, so K and V loads stay coalesced.
  float q_reg[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    q_reg[j] = half_to_float(query[q_base + lane + 32 * j]);
  }

  float running_max = -INFINITY;
  float running_denom = 0.0f;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int key = chunk_start + warp; key < chunk_end; key += V2_WARPS) {
    const int page_slot = key / page_size;
    const int page_offset = key % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;

    float partial = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      partial += q_reg[j] * half_to_float(key_cache[kv_base + lane + 32 * j]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    const float score = __shfl_sync(0xffffffffu, partial, 0) * scale;

    const float new_max = fmaxf(running_max, score);
    // running_max == -INFINITY on the first key, and exp(-inf) == 0 handles it.
    const float rescale = __expf(running_max - new_max);
    const float weight = __expf(score - new_max);
    running_denom = running_denom * rescale + weight;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[j] = acc[j] * rescale + weight * half_to_float(value_cache[kv_base + lane + 32 * j]);
    }
    running_max = new_max;
  }

  // One barrier per chunk, not per key: combine the four warps' partials.
  __shared__ float warp_acc[V2_WARPS][128];
  __shared__ float warp_max[V2_WARPS];
  __shared__ float warp_denom[V2_WARPS];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    warp_acc[warp][lane + 32 * j] = acc[j];
  }
  if (lane == 0) {
    warp_max[warp] = running_max;
    warp_denom[warp] = running_denom;
  }
  __syncthreads();

  float chunk_max = -INFINITY;
#pragma unroll
  for (int w = 0; w < V2_WARPS; ++w) {
    chunk_max = fmaxf(chunk_max, warp_max[w]);
  }
  float chunk_denom = 0.0f;
  float warp_scale[V2_WARPS];
#pragma unroll
  for (int w = 0; w < V2_WARPS; ++w) {
    warp_scale[w] = warp_denom[w] > 0.0f ? __expf(warp_max[w] - chunk_max) : 0.0f;
    chunk_denom += warp_denom[w] * warp_scale[w];
  }

#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float merged = 0.0f;
#pragma unroll
    for (int w = 0; w < V2_WARPS; ++w) {
      merged += warp_acc[w][lane + 32 * j] * warp_scale[w];
    }
    partial_acc[out_base + lane + 32 * j] = merged;
  }
  if (tid == 0) {
    partial_stats[(logical * num_chunks + chunk) * 2 + 0] = chunk_max;
    partial_stats[(logical * num_chunks + chunk) * 2 + 1] = chunk_denom;
  }
}

__global__ void small_q_attention_v2_merge_kernel(
    const float* __restrict__ partial_acc,
    const float* __restrict__ partial_stats,
    half* __restrict__ output,
    int num_chunks,
    int head_dim) {
  const int row = blockIdx.x;
  const int d = threadIdx.x;
  if (d >= head_dim) return;

  float global_max = -INFINITY;
  for (int c = 0; c < num_chunks; ++c) {
    global_max = fmaxf(global_max, partial_stats[(row * num_chunks + c) * 2 + 0]);
  }

  float denom = 0.0f;
  float merged = 0.0f;
  for (int c = 0; c < num_chunks; ++c) {
    const float chunk_max = partial_stats[(row * num_chunks + c) * 2 + 0];
    const float chunk_denom = partial_stats[(row * num_chunks + c) * 2 + 1];
    if (chunk_denom <= 0.0f) continue;
    const float scale = __expf(chunk_max - global_max);
    denom += chunk_denom * scale;
    merged += partial_acc[(row * num_chunks + c) * head_dim + d] * scale;
  }

  output[row * head_dim + d] = denom > 0.0f ? __float2half(merged / denom) : __float2half(0.0f);
}

// V3 keeps V2's split-KV grid and barrier-free KV loop, and removes the GQA
// redundancy: the query heads that share one kv head are computed by the same
// block, so each K/V element is loaded once and reused by all of them.
//
// The contract fixes Hq/Hkv = 32/8, so a block covers exactly four query heads.
// Per key and lane this costs four K loads and four V loads for all four heads,
// versus sixteen of each in V2. The partial layout is unchanged, so V3 reuses
// the V2 merge kernel.

constexpr int V3_GROUP = 4;
constexpr int V3_WARPS = 4;
constexpr int V3_THREADS = 32 * V3_WARPS;

__global__ void small_q_attention_v3_partial_kernel(
    const half* __restrict__ query,
    const half* __restrict__ key_cache,
    const half* __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_stats,
    int batch,
    int q_len,
    int num_query_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages,
    int chunk_keys,
    int num_chunks) {
  const int logical = blockIdx.x;
  const int chunk = blockIdx.y;
  const int kvh = logical % num_kv_heads;
  const int q_index = logical / num_kv_heads;
  const int qi = q_index % q_len;
  const int b = q_index / q_len;
  if (b >= batch) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  const int group_size = num_query_heads / num_kv_heads;
  if (group_size != V3_GROUP) return;

  const int total_len = seq_lens[b];
  const int valid_len = total_len - q_len + qi + 1;
  const int chunk_start = chunk * chunk_keys;
  const int row_base = (b * q_len + qi) * num_query_heads + kvh * group_size;

  // Publish neutral partials when this chunk is past the end of the sequence.
  if (valid_len <= 0 || chunk_start >= valid_len) {
    for (int h = 0; h < V3_GROUP; ++h) {
      const int acc_base = ((row_base + h) * num_chunks + chunk) * head_dim;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        partial_acc[acc_base + lane + 32 * j] = 0.0f;
      }
      if (tid == 0) {
        const int stat = ((row_base + h) * num_chunks + chunk) * 2;
        partial_stats[stat + 0] = -INFINITY;
        partial_stats[stat + 1] = 0.0f;
      }
    }
    return;
  }

  const int chunk_end = min(chunk_start + chunk_keys, valid_len);
  const float scale = rsqrtf(static_cast<float>(head_dim));

  // Q is small, so keep all four heads' slices in registers for the whole loop.
  float q_reg[V3_GROUP][4];
#pragma unroll
  for (int h = 0; h < V3_GROUP; ++h) {
    const int q_base = (row_base + h) * head_dim;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      q_reg[h][j] = half_to_float(query[q_base + lane + 32 * j]);
    }
  }

  float running_max[V3_GROUP];
  float running_denom[V3_GROUP];
  float acc[V3_GROUP][4];
#pragma unroll
  for (int h = 0; h < V3_GROUP; ++h) {
    running_max[h] = -INFINITY;
    running_denom[h] = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[h][j] = 0.0f;
    }
  }

  for (int key = chunk_start + warp; key < chunk_end; key += V3_WARPS) {
    const int page_slot = key / page_size;
    const int page_offset = key % page_size;
    const int physical_page = block_tables[b * max_pages + page_slot];
    const int kv_base = ((physical_page * page_size + page_offset) * num_kv_heads + kvh) * head_dim;

    // One K and one V slice per lane, reused by all four query heads.
    float k_reg[4];
    float v_reg[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      k_reg[j] = half_to_float(key_cache[kv_base + lane + 32 * j]);
      v_reg[j] = half_to_float(value_cache[kv_base + lane + 32 * j]);
    }

    float partial[V3_GROUP];
#pragma unroll
    for (int h = 0; h < V3_GROUP; ++h) {
      float dot = 0.0f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        dot += q_reg[h][j] * k_reg[j];
      }
      partial[h] = dot;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
#pragma unroll
      for (int h = 0; h < V3_GROUP; ++h) {
        partial[h] += __shfl_down_sync(0xffffffffu, partial[h], offset);
      }
    }

#pragma unroll
    for (int h = 0; h < V3_GROUP; ++h) {
      const float score = __shfl_sync(0xffffffffu, partial[h], 0) * scale;
      const float new_max = fmaxf(running_max[h], score);
      const float rescale = __expf(running_max[h] - new_max);
      const float weight = __expf(score - new_max);
      running_denom[h] = running_denom[h] * rescale + weight;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        acc[h][j] = acc[h][j] * rescale + weight * v_reg[j];
      }
      running_max[h] = new_max;
    }
  }

  // Same one-barrier-per-chunk warp merge as V2, but for four heads at once.
  __shared__ float warp_acc[V3_WARPS][V3_GROUP][128];
  __shared__ float warp_max[V3_WARPS][V3_GROUP];
  __shared__ float warp_denom[V3_WARPS][V3_GROUP];
#pragma unroll
  for (int h = 0; h < V3_GROUP; ++h) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      warp_acc[warp][h][lane + 32 * j] = acc[h][j];
    }
    if (lane == 0) {
      warp_max[warp][h] = running_max[h];
      warp_denom[warp][h] = running_denom[h];
    }
  }
  __syncthreads();

#pragma unroll
  for (int h = 0; h < V3_GROUP; ++h) {
    float chunk_max = -INFINITY;
#pragma unroll
    for (int w = 0; w < V3_WARPS; ++w) {
      chunk_max = fmaxf(chunk_max, warp_max[w][h]);
    }
    float warp_scale[V3_WARPS];
    float chunk_denom = 0.0f;
#pragma unroll
    for (int w = 0; w < V3_WARPS; ++w) {
      warp_scale[w] = warp_denom[w][h] > 0.0f ? __expf(warp_max[w][h] - chunk_max) : 0.0f;
      chunk_denom += warp_denom[w][h] * warp_scale[w];
    }
    const int acc_base = ((row_base + h) * num_chunks + chunk) * head_dim;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float merged = 0.0f;
#pragma unroll
      for (int w = 0; w < V3_WARPS; ++w) {
        merged += warp_acc[w][h][lane + 32 * j] * warp_scale[w];
      }
      partial_acc[acc_base + lane + 32 * j] = merged;
    }
    if (tid == 0) {
      const int stat = ((row_base + h) * num_chunks + chunk) * 2;
      partial_stats[stat + 0] = chunk_max;
      partial_stats[stat + 1] = chunk_denom;
    }
  }
}

}  // namespace

torch::Tensor small_q_attention_v0(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t page_size) {
  TORCH_CHECK(query.is_cuda(), "query must be CUDA");
  TORCH_CHECK(key_cache.is_cuda() && value_cache.is_cuda(), "KV cache must be CUDA");
  TORCH_CHECK(block_tables.is_cuda() && seq_lens.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16, "v0 supports FP16 query only");
  TORCH_CHECK(key_cache.scalar_type() == torch::kFloat16, "v0 supports FP16 key only");
  TORCH_CHECK(value_cache.scalar_type() == torch::kFloat16, "v0 supports FP16 value only");
  TORCH_CHECK(query.dim() == 4, "query must be [B, Q, Hq, D]");
  TORCH_CHECK(key_cache.dim() == 4, "key_cache must be [P, page, Hkv, D]");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape mismatch");
  TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous(), "inputs must be contiguous");
  TORCH_CHECK(block_tables.is_contiguous() && seq_lens.is_contiguous(), "metadata must be contiguous");

  const int batch = query.size(0);
  const int q_len = query.size(1);
  const int num_query_heads = query.size(2);
  const int head_dim = query.size(3);
  const int num_pages = key_cache.size(0);
  const int cache_page_size = key_cache.size(1);
  const int num_kv_heads = key_cache.size(2);
  const int max_pages = block_tables.size(1);
  TORCH_CHECK(q_len == 2 || q_len == 4 || q_len == 8, "v0 supports q_len 2, 4, or 8");
  TORCH_CHECK(head_dim == 128, "v0 supports head_dim 128");
  TORCH_CHECK(num_query_heads % num_kv_heads == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(page_size == cache_page_size, "page_size argument does not match cache");
  TORCH_CHECK(seq_lens.numel() == batch, "seq_lens batch mismatch");
  TORCH_CHECK(block_tables.size(0) == batch, "block_tables batch mismatch");
  TORCH_CHECK(num_pages > 0 && max_pages > 0, "empty KV cache is not supported");

  auto output = torch::empty_like(query);
  const dim3 grid(batch * q_len * num_query_heads);
  const dim3 block(128);
  small_q_attention_v0_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      batch, q_len, num_query_heads, num_kv_heads, head_dim, static_cast<int>(page_size), max_pages);
  const cudaError_t launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v0 launch failed: ", cudaGetErrorString(launch_error));
  return output;
}

torch::Tensor small_q_attention_v1(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t page_size) {
  TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(block_tables.is_cuda() && seq_lens.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 && key_cache.scalar_type() == torch::kFloat16 && value_cache.scalar_type() == torch::kFloat16, "v1 supports FP16 only");
  TORCH_CHECK(query.dim() == 4 && key_cache.dim() == 4, "invalid input rank");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape mismatch");
  TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() && block_tables.is_contiguous() && seq_lens.is_contiguous(), "inputs must be contiguous");
  const int batch = query.size(0);
  const int q_len = query.size(1);
  const int num_query_heads = query.size(2);
  const int head_dim = query.size(3);
  const int num_pages = key_cache.size(0);
  const int cache_page_size = key_cache.size(1);
  const int num_kv_heads = key_cache.size(2);
  const int max_pages = block_tables.size(1);
  TORCH_CHECK(q_len == 2 || q_len == 4 || q_len == 8, "v1 supports q_len 2, 4, or 8");
  TORCH_CHECK(head_dim == 128 && num_query_heads % num_kv_heads == 0, "invalid v1 head configuration");
  TORCH_CHECK(page_size == cache_page_size && seq_lens.numel() == batch && block_tables.size(0) == batch, "invalid v1 metadata");
  TORCH_CHECK(num_pages > 0 && max_pages > 0, "empty KV cache is not supported");

  auto output = torch::empty_like(query);
  small_q_attention_v1_kernel<<<dim3(batch * q_len * num_query_heads), dim3(128)>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(), seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()), batch, q_len,
      num_query_heads, num_kv_heads, head_dim, static_cast<int>(page_size), max_pages);
  const cudaError_t launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v1 launch failed: ", cudaGetErrorString(launch_error));
  return output;
}

torch::Tensor small_q_attention_v2(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t chunk_keys) {
  TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(block_tables.is_cuda() && seq_lens.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 && key_cache.scalar_type() == torch::kFloat16 &&
                  value_cache.scalar_type() == torch::kFloat16,
              "v2 supports FP16 only");
  TORCH_CHECK(query.dim() == 4 && key_cache.dim() == 4, "invalid input rank");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape mismatch");
  TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() &&
                  block_tables.is_contiguous() && seq_lens.is_contiguous(),
              "inputs must be contiguous");

  const int batch = query.size(0);
  const int q_len = query.size(1);
  const int num_query_heads = query.size(2);
  const int head_dim = query.size(3);
  const int num_pages = key_cache.size(0);
  const int cache_page_size = key_cache.size(1);
  const int num_kv_heads = key_cache.size(2);
  const int max_pages = block_tables.size(1);

  TORCH_CHECK(q_len == 2 || q_len == 4 || q_len == 8, "v2 supports q_len 2, 4, or 8");
  TORCH_CHECK(head_dim == 128, "v2 requires head_dim 128 so each warp lane owns four dims");
  TORCH_CHECK(num_query_heads % num_kv_heads == 0, "Hq must be divisible by Hkv");
  TORCH_CHECK(page_size == cache_page_size && seq_lens.numel() == batch &&
                  block_tables.size(0) == batch,
              "invalid v2 metadata");
  TORCH_CHECK(num_pages > 0 && max_pages > 0, "empty KV cache is not supported");
  TORCH_CHECK(chunk_keys > 0, "chunk_keys must be positive");

  // The grid is uniform, so the chunk count comes from the longest sequence.
  const int max_valid_len = static_cast<int>(seq_lens.max().item<int32_t>());
  TORCH_CHECK(max_valid_len > 0, "seq_lens must be positive");
  const int num_chunks = (max_valid_len + static_cast<int>(chunk_keys) - 1) / static_cast<int>(chunk_keys);
  const int rows = batch * q_len * num_query_heads;

  auto float_opts = torch::TensorOptions().device(query.device()).dtype(torch::kFloat32);
  auto partial_acc = torch::empty({rows, num_chunks, head_dim}, float_opts);
  auto partial_stats = torch::empty({rows, num_chunks, 2}, float_opts);
  auto output = torch::empty_like(query);

  small_q_attention_v2_partial_kernel<<<dim3(rows, num_chunks), dim3(V2_THREADS)>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      partial_acc.data_ptr<float>(),
      partial_stats.data_ptr<float>(),
      batch, q_len, num_query_heads, num_kv_heads, head_dim,
      static_cast<int>(page_size), max_pages, static_cast<int>(chunk_keys), num_chunks);
  cudaError_t launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v2 partial launch failed: ",
              cudaGetErrorString(launch_error));

  small_q_attention_v2_merge_kernel<<<dim3(rows), dim3(head_dim)>>>(
      partial_acc.data_ptr<float>(),
      partial_stats.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      num_chunks, head_dim);
  launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v2 merge launch failed: ",
              cudaGetErrorString(launch_error));
  return output;
}

torch::Tensor small_q_attention_v3(
    torch::Tensor query,
    torch::Tensor key_cache,
    torch::Tensor value_cache,
    torch::Tensor block_tables,
    torch::Tensor seq_lens,
    int64_t page_size,
    int64_t chunk_keys,
    int64_t max_seq_len) {
  TORCH_CHECK(query.is_cuda() && key_cache.is_cuda() && value_cache.is_cuda(), "inputs must be CUDA");
  TORCH_CHECK(block_tables.is_cuda() && seq_lens.is_cuda(), "metadata must be CUDA");
  TORCH_CHECK(query.scalar_type() == torch::kFloat16 && key_cache.scalar_type() == torch::kFloat16 &&
                  value_cache.scalar_type() == torch::kFloat16,
              "v3 supports FP16 only");
  TORCH_CHECK(query.dim() == 4 && key_cache.dim() == 4, "invalid input rank");
  TORCH_CHECK(value_cache.sizes() == key_cache.sizes(), "value_cache shape mismatch");
  TORCH_CHECK(query.is_contiguous() && key_cache.is_contiguous() && value_cache.is_contiguous() &&
                  block_tables.is_contiguous() && seq_lens.is_contiguous(),
              "inputs must be contiguous");

  const int batch = query.size(0);
  const int q_len = query.size(1);
  const int num_query_heads = query.size(2);
  const int head_dim = query.size(3);
  const int num_pages = key_cache.size(0);
  const int cache_page_size = key_cache.size(1);
  const int num_kv_heads = key_cache.size(2);
  const int max_pages = block_tables.size(1);

  TORCH_CHECK(q_len == 2 || q_len == 4 || q_len == 8, "v3 supports q_len 2, 4, or 8");
  TORCH_CHECK(head_dim == 128, "v3 requires head_dim 128");
  TORCH_CHECK(num_query_heads / num_kv_heads == 4 && num_query_heads % num_kv_heads == 0,
              "v3 requires a GQA group size of exactly 4 (Hq/Hkv = 32/8)");
  TORCH_CHECK(page_size == cache_page_size && seq_lens.numel() == batch &&
                  block_tables.size(0) == batch,
              "invalid v3 metadata");
  TORCH_CHECK(num_pages > 0 && max_pages > 0, "empty KV cache is not supported");
  TORCH_CHECK(chunk_keys > 0, "chunk_keys must be positive");

  // Passing max_seq_len from the caller avoids a device reduction plus a
  // device-to-host copy on every launch. 0 means "measure it".
  int longest = static_cast<int>(max_seq_len);
  if (longest <= 0) {
    longest = static_cast<int>(seq_lens.max().item<int32_t>());
  }
  TORCH_CHECK(longest > 0, "seq_lens must be positive");
  const int num_chunks = (longest + static_cast<int>(chunk_keys) - 1) / static_cast<int>(chunk_keys);
  const int rows = batch * q_len * num_query_heads;
  const int kv_rows = batch * q_len * num_kv_heads;

  auto float_opts = torch::TensorOptions().device(query.device()).dtype(torch::kFloat32);
  auto partial_acc = torch::empty({rows, num_chunks, head_dim}, float_opts);
  auto partial_stats = torch::empty({rows, num_chunks, 2}, float_opts);
  auto output = torch::empty_like(query);

  small_q_attention_v3_partial_kernel<<<dim3(kv_rows, num_chunks), dim3(V3_THREADS)>>>(
      reinterpret_cast<const half*>(query.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(key_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(value_cache.data_ptr<at::Half>()),
      block_tables.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      partial_acc.data_ptr<float>(),
      partial_stats.data_ptr<float>(),
      batch, q_len, num_query_heads, num_kv_heads, head_dim,
      static_cast<int>(page_size), max_pages, static_cast<int>(chunk_keys), num_chunks);
  cudaError_t launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v3 partial launch failed: ",
              cudaGetErrorString(launch_error));

  small_q_attention_v2_merge_kernel<<<dim3(rows), dim3(head_dim)>>>(
      partial_acc.data_ptr<float>(),
      partial_stats.data_ptr<float>(),
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      num_chunks, head_dim);
  launch_error = cudaGetLastError();
  TORCH_CHECK(launch_error == cudaSuccess, "small_q_attention_v3 merge launch failed: ",
              cudaGetErrorString(launch_error));
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &small_q_attention_v0, "Small-Q paged attention v0 (CUDA)");
  m.def("forward_v1", &small_q_attention_v1, "Small-Q paged attention v1 (CUDA)");
  m.def("forward_v2", &small_q_attention_v2, "Small-Q paged attention v2: split-KV, warp-tiled, CUDA");
  m.def("forward_v3", &small_q_attention_v3,
        "Small-Q paged attention v3: split-KV plus GQA-shared K/V, CUDA");
}
