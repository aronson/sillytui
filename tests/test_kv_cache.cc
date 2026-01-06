/*
 * KV Cache Unit Tests
 */

#include "test_framework.h"

extern "C" {
#include "inference/kernels/kv_cache/kv_cache.h"
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

#define ASSERT_ARRAY_NEAR(expected, actual, count, epsilon)                    \
  assert_array_near(expected, actual, count, epsilon)

static inline float bf16_to_float_scalar(uint16_t bf16) {
  uint32_t bits = ((uint32_t)bf16) << 16;
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
}

static inline uint16_t float_to_bf16_scalar(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(float));
  return (uint16_t)(bits >> 16);
}

static inline float fp16_to_float_scalar(uint16_t fp16) {
  uint32_t sign = (fp16 & 0x8000) << 16;
  uint32_t exp = (fp16 >> 10) & 0x1F;
  uint32_t mant = fp16 & 0x3FF;
  uint32_t f32_bits;

  if (exp == 0) {
    f32_bits = (mant == 0) ? sign : (sign | (127 - 14) << 23 | mant << 13);
  } else if (exp == 31) {
    f32_bits = sign | 0x7F800000 | (mant << 13);
  } else {
    f32_bits = sign | (((uint32_t)(exp - 15 + 127)) << 23) | (mant << 13);
  }

  float result;
  memcpy(&result, &f32_bits, sizeof(float));
  return result;
}

static inline uint16_t float_to_fp16_scalar(float f) {
  uint32_t bits;
  memcpy(&bits, &f, sizeof(float));
  uint32_t sign = (bits >> 16) & 0x8000;
  int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (bits >> 13) & 0x3FF;

  if (exp <= 0) {
    return (uint16_t)sign;
  } else if (exp >= 31) {
    return (uint16_t)(sign | 0x7C00);
  }
  return (uint16_t)(sign | (exp << 10) | mant);
}

TEST(kv_cache_f32_single_token) {
  const int cache_len = 0;
  const int num_tokens = 1;
  const int num_heads = 2;
  const int head_dim = 4;

  float key_cache[10 * 2 * 4] = {0};
  float value_cache[10 * 2 * 4] = {0};

  float key[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float value[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  kv_cache_append_f32(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  ASSERT_ARRAY_NEAR(key, key_cache, num_heads * head_dim, 1e-5);
  ASSERT_ARRAY_NEAR(value, value_cache, num_heads * head_dim, 1e-5);
}

TEST(kv_cache_f32_multiple_tokens) {
  const int cache_len = 0;
  const int num_tokens = 3;
  const int num_heads = 2;
  const int head_dim = 4;

  float key_cache[10 * 2 * 4] = {0};
  float value_cache[10 * 2 * 4] = {0};

  float key[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f};
  float value[] = {10.0f,  20.0f,  30.0f,  40.0f,  50.0f,  60.0f,
                   70.0f,  80.0f,  110.0f, 120.0f, 130.0f, 140.0f,
                   150.0f, 160.0f, 170.0f, 180.0f, 210.0f, 220.0f,
                   230.0f, 240.0f, 250.0f, 260.0f, 270.0f, 280.0f};

  kv_cache_append_f32(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  ASSERT_ARRAY_NEAR(key, key_cache, num_tokens * num_heads * head_dim, 1e-5);
  ASSERT_ARRAY_NEAR(value, value_cache, num_tokens * num_heads * head_dim,
                    1e-5);
}

TEST(kv_cache_f32_append_to_existing) {
  const int cache_len = 2;
  const int num_tokens = 1;
  const int num_heads = 2;
  const int head_dim = 4;

  float key_cache[10 * 2 * 4] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                 7.0f,  8.0f,  11.0f, 12.0f, 13.0f, 14.0f,
                                 15.0f, 16.0f, 17.0f, 18.0f};
  float value_cache[10 * 2 * 4] = {
      10.0f,  20.0f,  30.0f,  40.0f,  50.0f,  60.0f,  70.0f,  80.0f,
      110.0f, 120.0f, 130.0f, 140.0f, 150.0f, 160.0f, 170.0f, 180.0f};

  float key[] = {21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f};
  float value[] = {210.0f, 220.0f, 230.0f, 240.0f,
                   250.0f, 260.0f, 270.0f, 280.0f};

  kv_cache_append_f32(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  float expected_key[3 * 2 * 4] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                   7.0f,  8.0f,  11.0f, 12.0f, 13.0f, 14.0f,
                                   15.0f, 16.0f, 17.0f, 18.0f, 21.0f, 22.0f,
                                   23.0f, 24.0f, 25.0f, 26.0f, 27.0f, 28.0f};
  float expected_value[3 * 2 * 4] = {
      10.0f,  20.0f,  30.0f,  40.0f,  50.0f,  60.0f,  70.0f,  80.0f,
      110.0f, 120.0f, 130.0f, 140.0f, 150.0f, 160.0f, 170.0f, 180.0f,
      210.0f, 220.0f, 230.0f, 240.0f, 250.0f, 260.0f, 270.0f, 280.0f};

  ASSERT_ARRAY_NEAR(expected_key, key_cache, 3 * num_heads * head_dim, 1e-5);
  ASSERT_ARRAY_NEAR(expected_value, value_cache, 3 * num_heads * head_dim,
                    1e-5);
}

TEST(kv_cache_bf16_basic) {
  const int cache_len = 0;
  const int num_tokens = 1;
  const int num_heads = 2;
  const int head_dim = 4;

  uint16_t key_cache[10 * 2 * 4] = {0};
  uint16_t value_cache[10 * 2 * 4] = {0};

  float key_f32[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float value_f32[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  uint16_t key[8], value[8];
  for (int i = 0; i < 8; i++) {
    key[i] = float_to_bf16_scalar(key_f32[i]);
    value[i] = float_to_bf16_scalar(value_f32[i]);
  }

  kv_cache_append_bf16(key_cache, value_cache, key, value, cache_len,
                       num_tokens, num_heads, head_dim);

  for (int i = 0; i < 8; i++) {
    ASSERT_NEAR(key_f32[i], bf16_to_float_scalar(key_cache[i]), 1e-2);
    ASSERT_NEAR(value_f32[i], bf16_to_float_scalar(value_cache[i]), 1e-2);
  }
}

TEST(kv_cache_f16_basic) {
  const int cache_len = 0;
  const int num_tokens = 1;
  const int num_heads = 2;
  const int head_dim = 4;

  uint16_t key_cache[10 * 2 * 4] = {0};
  uint16_t value_cache[10 * 2 * 4] = {0};

  float key_f32[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float value_f32[] = {10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f, 80.0f};

  uint16_t key[8], value[8];
  for (int i = 0; i < 8; i++) {
    key[i] = float_to_fp16_scalar(key_f32[i]);
    value[i] = float_to_fp16_scalar(value_f32[i]);
  }

  kv_cache_append_f16(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  for (int i = 0; i < 8; i++) {
    ASSERT_NEAR(key_f32[i], fp16_to_float_scalar(key_cache[i]), 1e-2);
    ASSERT_NEAR(value_f32[i], fp16_to_float_scalar(value_cache[i]), 1e-2);
  }
}

TEST(kv_cache_f32_large) {
  const int cache_len = 10;
  const int num_tokens = 32;
  const int num_heads = 8;
  const int head_dim = 64;

  int total_size = (cache_len + num_tokens) * num_heads * head_dim;
  float *key_cache = (float *)calloc(total_size, sizeof(float));
  float *value_cache = (float *)calloc(total_size, sizeof(float));
  float *key =
      (float *)malloc(num_tokens * num_heads * head_dim * sizeof(float));
  float *value =
      (float *)malloc(num_tokens * num_heads * head_dim * sizeof(float));

  for (int i = 0; i < num_tokens * num_heads * head_dim; i++) {
    key[i] = (float)i / 100.0f;
    value[i] = (float)i / 200.0f;
  }

  kv_cache_append_f32(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  int offset = cache_len * num_heads * head_dim;
  ASSERT_ARRAY_NEAR(key, key_cache + offset, num_tokens * num_heads * head_dim,
                    1e-5);
  ASSERT_ARRAY_NEAR(value, value_cache + offset,
                    num_tokens * num_heads * head_dim, 1e-5);

  free(key_cache);
  free(value_cache);
  free(key);
  free(value);
}

extern "C" void run_kv_cache_tests(void) {
  TEST_SUITE("KV Cache");
  RUN_TEST(kv_cache_f32_single_token);
  RUN_TEST(kv_cache_f32_multiple_tokens);
  RUN_TEST(kv_cache_f32_append_to_existing);
  RUN_TEST(kv_cache_bf16_basic);
  RUN_TEST(kv_cache_f16_basic);
  RUN_TEST(kv_cache_f32_large);
}
