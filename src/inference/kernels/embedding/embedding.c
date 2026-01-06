/*
 * Embedding Lookup - Dispatcher and Scalar Fallback
 */

#include "inference/kernels/embedding/embedding.h"
#include "inference/kernels/embedding/embedding_kernels.h"
#include <string.h>

// Scalar fallback implementations
static void embedding_lookup_f32_scalar(float *output, const int64_t *token_ids,
                                        const float *weight, int num_tokens,
                                        int vocab_size, int embedding_dim,
                                        int64_t padding_idx) {
  for (int i = 0; i < num_tokens; i++) {
    int64_t token_id = token_ids[i];
    float *out_vec = output + i * embedding_dim;

    if (token_id == padding_idx) {
      // Padding token: output zero vector
      memset(out_vec, 0, embedding_dim * sizeof(float));
    } else {
      // Clamp token_id to valid range
      if (token_id < 0) {
        token_id = 0;
      } else if (token_id >= vocab_size) {
        token_id = vocab_size - 1;
      }

      const float *weight_vec = weight + token_id * embedding_dim;
      memcpy(out_vec, weight_vec, embedding_dim * sizeof(float));
    }
  }
}

static void embedding_lookup_bf16_scalar(uint16_t *output,
                                         const int64_t *token_ids,
                                         const uint16_t *weight, int num_tokens,
                                         int vocab_size, int embedding_dim,
                                         int64_t padding_idx) {
  for (int i = 0; i < num_tokens; i++) {
    int64_t token_id = token_ids[i];
    uint16_t *out_vec = output + i * embedding_dim;

    if (token_id == padding_idx) {
      // Padding token: output zero vector
      memset(out_vec, 0, embedding_dim * sizeof(uint16_t));
    } else {
      // Clamp token_id to valid range
      if (token_id < 0) {
        token_id = 0;
      } else if (token_id >= vocab_size) {
        token_id = vocab_size - 1;
      }

      const uint16_t *weight_vec = weight + token_id * embedding_dim;
      memcpy(out_vec, weight_vec, embedding_dim * sizeof(uint16_t));
    }
  }
}

static void embedding_lookup_f16_scalar(uint16_t *output,
                                        const int64_t *token_ids,
                                        const uint16_t *weight, int num_tokens,
                                        int vocab_size, int embedding_dim,
                                        int64_t padding_idx) {
  for (int i = 0; i < num_tokens; i++) {
    int64_t token_id = token_ids[i];
    uint16_t *out_vec = output + i * embedding_dim;

    if (token_id == padding_idx) {
      // Padding token: output zero vector
      memset(out_vec, 0, embedding_dim * sizeof(uint16_t));
    } else {
      // Clamp token_id to valid range
      if (token_id < 0) {
        token_id = 0;
      } else if (token_id >= vocab_size) {
        token_id = vocab_size - 1;
      }

      const uint16_t *weight_vec = weight + token_id * embedding_dim;
      memcpy(out_vec, weight_vec, embedding_dim * sizeof(uint16_t));
    }
  }
}

// Dispatchers
void embedding_lookup_f32(float *output, const int64_t *token_ids,
                          const float *weight, int num_tokens, int vocab_size,
                          int embedding_dim, int64_t padding_idx) {
  embedding_caps_t caps = embedding_get_capabilities();
  if (caps.has_neon) {
    embedding_lookup_f32_kernel(output, token_ids, weight, num_tokens,
                                vocab_size, embedding_dim, padding_idx);
  } else {
    embedding_lookup_f32_scalar(output, token_ids, weight, num_tokens,
                                vocab_size, embedding_dim, padding_idx);
  }
}

void embedding_lookup_bf16(uint16_t *output, const int64_t *token_ids,
                           const uint16_t *weight, int num_tokens,
                           int vocab_size, int embedding_dim,
                           int64_t padding_idx) {
  embedding_caps_t caps = embedding_get_capabilities();
  if (caps.has_neon) {
    embedding_lookup_bf16_kernel(output, token_ids, weight, num_tokens,
                                 vocab_size, embedding_dim, padding_idx);
  } else {
    embedding_lookup_bf16_scalar(output, token_ids, weight, num_tokens,
                                 vocab_size, embedding_dim, padding_idx);
  }
}

void embedding_lookup_f16(uint16_t *output, const int64_t *token_ids,
                          const uint16_t *weight, int num_tokens,
                          int vocab_size, int embedding_dim,
                          int64_t padding_idx) {
  embedding_caps_t caps = embedding_get_capabilities();
  if (caps.has_neon) {
    embedding_lookup_f16_kernel(output, token_ids, weight, num_tokens,
                                vocab_size, embedding_dim, padding_idx);
  } else {
    embedding_lookup_f16_scalar(output, token_ids, weight, num_tokens,
                                vocab_size, embedding_dim, padding_idx);
  }
}
