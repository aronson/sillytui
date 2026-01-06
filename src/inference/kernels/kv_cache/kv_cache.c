/*
 * KV Cache - Dispatcher and Scalar Fallback
 */

#include "inference/kernels/kv_cache/kv_cache.h"
#include "inference/kernels/kv_cache/kv_cache_kernels.h"
#include <string.h>

static void kv_cache_append_f32_scalar(float *key_cache, float *value_cache,
                                       const float *key, const float *value,
                                       int cache_len, int num_tokens,
                                       int num_heads, int head_dim) {
  int head_size = num_heads * head_dim;

  for (int t = 0; t < num_tokens; t++) {
    int cache_offset = (cache_len + t) * head_size;
    int input_offset = t * head_size;

    memcpy(key_cache + cache_offset, key + input_offset,
           head_size * sizeof(float));
    memcpy(value_cache + cache_offset, value + input_offset,
           head_size * sizeof(float));
  }
}

static void kv_cache_append_bf16_scalar(uint16_t *key_cache,
                                        uint16_t *value_cache,
                                        const uint16_t *key,
                                        const uint16_t *value, int cache_len,
                                        int num_tokens, int num_heads,
                                        int head_dim) {
  int head_size = num_heads * head_dim;

  for (int t = 0; t < num_tokens; t++) {
    int cache_offset = (cache_len + t) * head_size;
    int input_offset = t * head_size;

    memcpy(key_cache + cache_offset, key + input_offset,
           head_size * sizeof(uint16_t));
    memcpy(value_cache + cache_offset, value + input_offset,
           head_size * sizeof(uint16_t));
  }
}

static void kv_cache_append_f16_scalar(uint16_t *key_cache,
                                       uint16_t *value_cache,
                                       const uint16_t *key,
                                       const uint16_t *value, int cache_len,
                                       int num_tokens, int num_heads,
                                       int head_dim) {
  int head_size = num_heads * head_dim;

  for (int t = 0; t < num_tokens; t++) {
    int cache_offset = (cache_len + t) * head_size;
    int input_offset = t * head_size;

    memcpy(key_cache + cache_offset, key + input_offset,
           head_size * sizeof(uint16_t));
    memcpy(value_cache + cache_offset, value + input_offset,
           head_size * sizeof(uint16_t));
  }
}

void kv_cache_append_f32(float *key_cache, float *value_cache, const float *key,
                         const float *value, int cache_len, int num_tokens,
                         int num_heads, int head_dim) {
  kv_cache_caps_t caps = kv_cache_get_capabilities();
  if (caps.has_neon) {
    kv_cache_append_f32_kernel(key_cache, value_cache, key, value, cache_len,
                               num_tokens, num_heads, head_dim);
  } else {
    kv_cache_append_f32_scalar(key_cache, value_cache, key, value, cache_len,
                               num_tokens, num_heads, head_dim);
  }
}

void kv_cache_append_bf16(uint16_t *key_cache, uint16_t *value_cache,
                          const uint16_t *key, const uint16_t *value,
                          int cache_len, int num_tokens, int num_heads,
                          int head_dim) {
  kv_cache_caps_t caps = kv_cache_get_capabilities();
  if (caps.has_neon) {
    kv_cache_append_bf16_kernel(key_cache, value_cache, key, value, cache_len,
                                num_tokens, num_heads, head_dim);
  } else {
    kv_cache_append_bf16_scalar(key_cache, value_cache, key, value, cache_len,
                                num_tokens, num_heads, head_dim);
  }
}

void kv_cache_append_f16(uint16_t *key_cache, uint16_t *value_cache,
                         const uint16_t *key, const uint16_t *value,
                         int cache_len, int num_tokens, int num_heads,
                         int head_dim) {
  kv_cache_caps_t caps = kv_cache_get_capabilities();
  if (caps.has_neon) {
    kv_cache_append_f16_kernel(key_cache, value_cache, key, value, cache_len,
                               num_tokens, num_heads, head_dim);
  } else {
    kv_cache_append_f16_scalar(key_cache, value_cache, key, value, cache_len,
                               num_tokens, num_heads, head_dim);
  }
}
