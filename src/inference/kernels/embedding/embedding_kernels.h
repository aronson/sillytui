/*
 * Embedding kernel interface for architecture-specific implementations
 */

#ifndef EMBEDDING_KERNELS_H
#define EMBEDDING_KERNELS_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  bool has_neon;
} embedding_caps_t;

embedding_caps_t embedding_get_capabilities(void);

/* FP32 kernels */
void embedding_lookup_f32_kernel(float *output, const int64_t *token_ids,
                                 const float *weight, int num_tokens,
                                 int vocab_size, int embedding_dim,
                                 int64_t padding_idx);

/* BF16 kernels */
void embedding_lookup_bf16_kernel(uint16_t *output, const int64_t *token_ids,
                                  const uint16_t *weight, int num_tokens,
                                  int vocab_size, int embedding_dim,
                                  int64_t padding_idx);

/* FP16 kernels */
void embedding_lookup_f16_kernel(uint16_t *output, const int64_t *token_ids,
                                 const uint16_t *weight, int num_tokens,
                                 int vocab_size, int embedding_dim,
                                 int64_t padding_idx);

#ifdef __cplusplus
}
#endif

#endif // EMBEDDING_KERNELS_H
