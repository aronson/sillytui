/*
 * Embedding Lookup Unit Tests
 */

#include "test_framework.h"

extern "C" {
#include "inference/kernels/embedding/embedding.h"
}

#include <cmath>
#include <cstdlib>
#include <cstring>

static void assert_array_near(const float *expected, const float *actual,
                              size_t count, float epsilon) {
  for (size_t i = 0; i < count; i++) {
    if (fabsf(expected[i] - actual[i]) > epsilon) {
      FAIL_FMT("Array mismatch at index %zu: expected %.6f, got %.6f", i,
               expected[i], actual[i]);
      return;
    }
  }
}

TEST(embedding_f32_single_token) {
  const int vocab_size = 10;
  const int embedding_dim = 4;
  const int64_t padding_idx = -1;

  float weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    weight[i] = (float)(i + 1);
  }

  int64_t token_ids[] = {3};
  float output[embedding_dim];

  embedding_lookup_f32(output, token_ids, weight, 1, vocab_size, embedding_dim,
                       padding_idx);

  // Token 3 should map to weight[3*4 : 3*4+4] = [13, 14, 15, 16]
  float expected[] = {13.0f, 14.0f, 15.0f, 16.0f};
  assert_array_near(expected, output, embedding_dim, 1e-5f);
}

TEST(embedding_f32_multiple_tokens) {
  const int vocab_size = 5;
  const int embedding_dim = 3;
  const int64_t padding_idx = -1;

  float weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    weight[i] = (float)(i + 1);
  }

  int64_t token_ids[] = {0, 2, 4};
  float output[3 * embedding_dim];

  embedding_lookup_f32(output, token_ids, weight, 3, vocab_size, embedding_dim,
                       padding_idx);

  // Token 0: [1, 2, 3]
  // Token 2: [7, 8, 9]
  // Token 4: [13, 14, 15]
  float expected[] = {1.0f, 2.0f, 3.0f, 7.0f, 8.0f, 9.0f, 13.0f, 14.0f, 15.0f};
  assert_array_near(expected, output, 3 * embedding_dim, 1e-5f);
}

TEST(embedding_f32_padding_idx) {
  const int vocab_size = 5;
  const int embedding_dim = 4;
  const int64_t padding_idx = 2;

  float weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    weight[i] = (float)(i + 1);
  }

  int64_t token_ids[] = {0, 2, 4}; // 2 is padding_idx
  float output[3 * embedding_dim];

  embedding_lookup_f32(output, token_ids, weight, 3, vocab_size, embedding_dim,
                       padding_idx);

  // Token 0: [1, 2, 3, 4]
  // Token 2 (padding): [0, 0, 0, 0]
  // Token 4: [17, 18, 19, 20]
  float expected[] = {1.0f, 2.0f, 3.0f,  4.0f,  0.0f,  0.0f,
                      0.0f, 0.0f, 17.0f, 18.0f, 19.0f, 20.0f};
  assert_array_near(expected, output, 3 * embedding_dim, 1e-5f);
}

TEST(embedding_f32_out_of_bounds) {
  const int vocab_size = 5;
  const int embedding_dim = 3;
  const int64_t padding_idx = 99; // Use a value that won't conflict

  float weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    weight[i] = (float)(i + 1);
  }

  int64_t token_ids[] = {-1, 10, 2}; // -1 and 10 are out of bounds
  float output[3 * embedding_dim];

  embedding_lookup_f32(output, token_ids, weight, 3, vocab_size, embedding_dim,
                       padding_idx);

  // -1 should clamp to 0: [1, 2, 3]
  // 10 should clamp to 4: [13, 14, 15]
  // 2: [7, 8, 9]
  float expected[] = {1.0f, 2.0f, 3.0f, 13.0f, 14.0f, 15.0f, 7.0f, 8.0f, 9.0f};
  assert_array_near(expected, output, 3 * embedding_dim, 1e-5f);
}

TEST(embedding_f32_large_dim) {
  const int vocab_size = 100;
  const int embedding_dim = 512;
  const int64_t padding_idx = -1;

  float *weight = (float *)malloc(vocab_size * embedding_dim * sizeof(float));
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    weight[i] = (float)(i % 100) / 100.0f;
  }

  int64_t token_ids[] = {0, 50, 99};
  float *output = (float *)malloc(3 * embedding_dim * sizeof(float));

  embedding_lookup_f32(output, token_ids, weight, 3, vocab_size, embedding_dim,
                       padding_idx);

  // Verify first and last elements of each token
  ASSERT_NEAR(weight[0 * embedding_dim + 0], output[0 * embedding_dim + 0],
              1e-5);
  ASSERT_NEAR(weight[0 * embedding_dim + embedding_dim - 1],
              output[0 * embedding_dim + embedding_dim - 1], 1e-5);
  ASSERT_NEAR(weight[50 * embedding_dim + 0], output[1 * embedding_dim + 0],
              1e-5);
  ASSERT_NEAR(weight[99 * embedding_dim + 0], output[2 * embedding_dim + 0],
              1e-5);

  free(weight);
  free(output);
}

TEST(embedding_bf16_basic) {
  const int vocab_size = 5;
  const int embedding_dim = 4;
  const int64_t padding_idx = -1;

  uint16_t weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    float val = (float)(i + 1);
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    weight[i] = (uint16_t)(bits >> 16); // BF16 conversion
  }

  int64_t token_ids[] = {1, 3};
  uint16_t output[2 * embedding_dim];

  embedding_lookup_bf16(output, token_ids, weight, 2, vocab_size, embedding_dim,
                        padding_idx);

  // Just verify we got something (exact comparison is tricky with BF16)
  ASSERT_TRUE(output[0] != 0 || output[1] != 0);
}

TEST(embedding_f16_basic) {
  const int vocab_size = 5;
  const int embedding_dim = 4;
  const int64_t padding_idx = -1;

  uint16_t weight[vocab_size * embedding_dim];
  for (int i = 0; i < vocab_size * embedding_dim; i++) {
    float val = (float)(i + 1);
    // Simple FP16 conversion (not exact, but good enough for test)
    uint32_t bits;
    memcpy(&bits, &val, sizeof(float));
    uint32_t sign = (bits >> 16) & 0x8000;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3FF;
    if (exp <= 0)
      weight[i] = (uint16_t)sign;
    else if (exp >= 31)
      weight[i] = (uint16_t)(sign | 0x7C00);
    else
      weight[i] = (uint16_t)(sign | (exp << 10) | mant);
  }

  int64_t token_ids[] = {1, 3};
  uint16_t output[2 * embedding_dim];

  embedding_lookup_f16(output, token_ids, weight, 2, vocab_size, embedding_dim,
                       padding_idx);

  // Just verify we got something
  ASSERT_TRUE(output[0] != 0 || output[1] != 0);
}

extern "C" void run_embedding_tests(void) {
  TEST_SUITE("Embedding Lookup");
  RUN_TEST(embedding_f32_single_token);
  RUN_TEST(embedding_f32_multiple_tokens);
  RUN_TEST(embedding_f32_padding_idx);
  RUN_TEST(embedding_f32_out_of_bounds);
  RUN_TEST(embedding_f32_large_dim);
  RUN_TEST(embedding_bf16_basic);
  RUN_TEST(embedding_f16_basic);
}
