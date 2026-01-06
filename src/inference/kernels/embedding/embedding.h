/*
 * Embedding Lookup - Public API
 */

#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Embedding lookup for FP32
 *
 * Parameters:
 *   output:      [num_tokens, embedding_dim] output embeddings
 *   token_ids:   [num_tokens] token IDs (int64)
 *   weight:      [vocab_size, embedding_dim] embedding table
 *   num_tokens:  number of tokens to lookup
 *   vocab_size:  vocabulary size
 *   embedding_dim: dimension of each embedding vector
 *   padding_idx: token ID that maps to zero vector (-1 means no padding)
 */
void embedding_lookup_f32(float *output, const int64_t *token_ids,
                          const float *weight, int num_tokens, int vocab_size,
                          int embedding_dim, int64_t padding_idx);

/*
 * Embedding lookup for BF16
 */
void embedding_lookup_bf16(uint16_t *output, const int64_t *token_ids,
                           const uint16_t *weight, int num_tokens,
                           int vocab_size, int embedding_dim,
                           int64_t padding_idx);

/*
 * Embedding lookup for FP16
 */
void embedding_lookup_f16(uint16_t *output, const int64_t *token_ids,
                          const uint16_t *weight, int num_tokens,
                          int vocab_size, int embedding_dim,
                          int64_t padding_idx);

#ifdef __cplusplus
}
#endif

#endif // EMBEDDING_H
