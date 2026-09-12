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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &small_q_attention_v0, "Small-Q paged attention v0 (CUDA)");
  m.def("forward_v1", &small_q_attention_v1, "Small-Q paged attention v1 (CUDA)");
}
