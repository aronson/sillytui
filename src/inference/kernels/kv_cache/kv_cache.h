/*
 * KV Cache - Public API
 *
 * Efficiently append key/value tensors to cache buffers during autoregressive
 * inference to avoid recomputing K/V for previous tokens.
 */

#ifndef KV_CACHE_H
#define KV_CACHE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Append key/value tensors to cache buffers (FP32)
 *
 * Parameters:
 *   key_cache:   [cache_len, num_heads, head_dim] - cache buffer (modified)
 *   value_cache: [cache_len, num_heads, head_dim] - cache buffer (modified)
 *   key:         [num_tokens, num_heads, head_dim] - new keys to append
 *   value:       [num_tokens, num_heads, head_dim] - new values to append
 *   cache_len:   current length of cache (before append)
 *   num_tokens:  number of new tokens to append
 *   num_heads:   number of attention heads
 *   head_dim:    dimension per head
 */
void kv_cache_append_f32(float *key_cache, float *value_cache, const float *key,
                         const float *value, int cache_len, int num_tokens,
                         int num_heads, int head_dim);

/*
 * Append key/value tensors to cache buffers (BF16)
 */
void kv_cache_append_bf16(uint16_t *key_cache, uint16_t *value_cache,
                          const uint16_t *key, const uint16_t *value,
                          int cache_len, int num_tokens, int num_heads,
                          int head_dim);

/*
 * Append key/value tensors to cache buffers (FP16)
 */
void kv_cache_append_f16(uint16_t *key_cache, uint16_t *value_cache,
                         const uint16_t *key, const uint16_t *value,
                         int cache_len, int num_tokens, int num_heads,
                         int head_dim);

#ifdef __cplusplus
}
#endif

#endif // KV_CACHE_H
