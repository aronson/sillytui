/*
 * KV Cache kernel interface for architecture-specific implementations
 */

#ifndef KV_CACHE_KERNELS_H
#define KV_CACHE_KERNELS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  bool has_neon;
} kv_cache_caps_t;

kv_cache_caps_t kv_cache_get_capabilities(void);

void kv_cache_append_f32_kernel(float *key_cache, float *value_cache,
                                const float *key, const float *value,
                                int cache_len, int num_tokens, int num_heads,
                                int head_dim);

void kv_cache_append_bf16_kernel(uint16_t *key_cache, uint16_t *value_cache,
                                 const uint16_t *key, const uint16_t *value,
                                 int cache_len, int num_tokens, int num_heads,
                                 int head_dim);

void kv_cache_append_f16_kernel(uint16_t *key_cache, uint16_t *value_cache,
                                const uint16_t *key, const uint16_t *value,
                                int cache_len, int num_tokens, int num_heads,
                                int head_dim);

#ifdef __cplusplus
}
#endif

#endif // KV_CACHE_KERNELS_H
