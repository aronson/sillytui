/*
 * KV Cache PyTorch Accuracy Tests
 */

#include "test_framework.h"

extern "C" {
#include "inference/kernels/kv_cache/kv_cache.h"
}

#include "inference/model_loader/safetensors.hh"
#include <cmath>
#include <cstring>
#include <string>

static safetensors::safetensors_t g_st;
static bool g_loaded = false;

static bool load_reference_data() {
  if (g_loaded)
    return true;

  std::string warn, err;
  bool ok = safetensors::mmap_from_file("tests/kv_cache_reference.safetensors",
                                        &g_st, &warn, &err);
  if (!ok) {
    printf("Failed to load KV cache reference data: %s\n", err.c_str());
    return false;
  }

  g_loaded = true;
  return true;
}

static const float *get_f32_tensor(const std::string &name, size_t *count) {
  const auto &keys = g_st.tensors.keys();
  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i] == name) {
      safetensors::tensor_t tensor;
      g_st.tensors.at(i, &tensor);
      size_t data_len = tensor.data_offsets[1] - tensor.data_offsets[0];
      *count = data_len / sizeof(float);
      return reinterpret_cast<const float *>(g_st.databuffer_addr +
                                             tensor.data_offsets[0]);
    }
  }
  return nullptr;
}

static const uint16_t *get_bf16_tensor(const std::string &name, size_t *count) {
  const auto &keys = g_st.tensors.keys();
  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i] == name) {
      safetensors::tensor_t tensor;
      g_st.tensors.at(i, &tensor);
      size_t data_len = tensor.data_offsets[1] - tensor.data_offsets[0];
      *count = data_len / sizeof(uint16_t);
      return reinterpret_cast<const uint16_t *>(g_st.databuffer_addr +
                                                tensor.data_offsets[0]);
    }
  }
  return nullptr;
}

static const uint16_t *get_f16_tensor(const std::string &name, size_t *count) {
  const auto &keys = g_st.tensors.keys();
  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i] == name) {
      safetensors::tensor_t tensor;
      g_st.tensors.at(i, &tensor);
      size_t data_len = tensor.data_offsets[1] - tensor.data_offsets[0];
      *count = data_len / sizeof(uint16_t);
      return reinterpret_cast<const uint16_t *>(g_st.databuffer_addr +
                                                tensor.data_offsets[0]);
    }
  }
  return nullptr;
}

static int get_metadata_int(const std::string &name) {
  const auto &keys = g_st.metadata.keys();
  for (size_t i = 0; i < keys.size(); i++) {
    if (keys[i] == name) {
      std::string val;
      g_st.metadata.at(i, &val);
      return std::stoi(val);
    }
  }
  return -1;
}

static float compute_max_rel_error_f32(const float *expected,
                                       const float *actual, size_t count) {
  float max_err = 0.0f;
  for (size_t i = 0; i < count; i++) {
    float err = fabsf(expected[i] - actual[i]);
    float rel_err = (expected[i] != 0.0f) ? err / fabsf(expected[i]) : err;
    if (rel_err > max_err) {
      max_err = rel_err;
    }
  }
  return max_err;
}

static inline float bf16_to_float_scalar(uint16_t bf16) {
  uint32_t bits = ((uint32_t)bf16) << 16;
  float result;
  memcpy(&result, &bits, sizeof(float));
  return result;
}

static float compute_max_rel_error_bf16(const uint16_t *expected,
                                        const uint16_t *actual, size_t count) {
  float max_err = 0.0f;
  for (size_t i = 0; i < count; i++) {
    float exp_f = bf16_to_float_scalar(expected[i]);
    float act_f = bf16_to_float_scalar(actual[i]);
    float err = fabsf(exp_f - act_f);
    float mag = fabsf(exp_f);
    float rel_err = (mag > 1e-3f) ? err / mag : err;
    if (rel_err > max_err) {
      max_err = rel_err;
    }
  }
  return max_err;
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

static float compute_max_rel_error_f16(const uint16_t *expected,
                                       const uint16_t *actual, size_t count) {
  float max_err = 0.0f;
  for (size_t i = 0; i < count; i++) {
    float exp_f = fp16_to_float_scalar(expected[i]);
    float act_f = fp16_to_float_scalar(actual[i]);
    float err = fabsf(exp_f - act_f);
    float mag = fabsf(exp_f);
    float rel_err = (mag > 1e-3f) ? err / mag : err;
    if (rel_err > max_err) {
      max_err = rel_err;
    }
  }
  return max_err;
}

static void run_kv_cache_f32_test(const char *case_name) {
  if (!load_reference_data()) {
    ASSERT(false);
    return;
  }

  std::string prefix = std::string("f32_") + case_name;
  size_t key_count, value_count, cache_in_count, cache_out_count;

  int cache_len = get_metadata_int(prefix + "_cache_len");
  int num_tokens = get_metadata_int(prefix + "_num_tokens");
  int num_heads = get_metadata_int(prefix + "_num_heads");
  int head_dim = get_metadata_int(prefix + "_head_dim");

  const float *key = get_f32_tensor(prefix + "_key", &key_count);
  const float *value = get_f32_tensor(prefix + "_value", &value_count);
  const float *key_cache_in = nullptr;
  const float *value_cache_in = nullptr;
  if (cache_len > 0) {
    key_cache_in = get_f32_tensor(prefix + "_key_cache_in", &cache_in_count);
    value_cache_in =
        get_f32_tensor(prefix + "_value_cache_in", &cache_in_count);
  }
  const float *expected_key_cache =
      get_f32_tensor(prefix + "_key_cache_out", &cache_out_count);
  const float *expected_value_cache =
      get_f32_tensor(prefix + "_value_cache_out", &cache_out_count);

  if (!key || !value || !expected_key_cache || !expected_value_cache) {
    printf("Missing tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }
  if (cache_len > 0 && (!key_cache_in || !value_cache_in)) {
    printf("Missing cache input tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }

  int total_size = (cache_len + num_tokens) * num_heads * head_dim;
  float *key_cache = (float *)malloc(total_size * sizeof(float));
  float *value_cache = (float *)malloc(total_size * sizeof(float));

  if (cache_len > 0) {
    memcpy(key_cache, key_cache_in,
           cache_len * num_heads * head_dim * sizeof(float));
    memcpy(value_cache, value_cache_in,
           cache_len * num_heads * head_dim * sizeof(float));
  }

  kv_cache_append_f32(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  float max_err_key =
      compute_max_rel_error_f32(expected_key_cache, key_cache, total_size);
  float max_err_value =
      compute_max_rel_error_f32(expected_value_cache, value_cache, total_size);

  ASSERT_LT(max_err_key, 1e-5f);
  ASSERT_LT(max_err_value, 1e-5f);

  free(key_cache);
  free(value_cache);
}

static void run_kv_cache_bf16_test(const char *case_name) {
  if (!load_reference_data()) {
    ASSERT(false);
    return;
  }

  std::string prefix = std::string("bf16_") + case_name;
  std::string f32_prefix = std::string("f32_") + case_name;
  size_t key_count, value_count, cache_in_count, cache_out_count;

  int cache_len = get_metadata_int(f32_prefix + "_cache_len");
  int num_tokens = get_metadata_int(f32_prefix + "_num_tokens");
  int num_heads = get_metadata_int(f32_prefix + "_num_heads");
  int head_dim = get_metadata_int(f32_prefix + "_head_dim");

  const uint16_t *key = get_bf16_tensor(prefix + "_key", &key_count);
  const uint16_t *value = get_bf16_tensor(prefix + "_value", &value_count);
  const uint16_t *key_cache_in = nullptr;
  const uint16_t *value_cache_in = nullptr;
  if (cache_len > 0) {
    key_cache_in = get_bf16_tensor(prefix + "_key_cache_in", &cache_in_count);
    value_cache_in =
        get_bf16_tensor(prefix + "_value_cache_in", &cache_in_count);
  }
  const uint16_t *expected_key_cache =
      get_bf16_tensor(prefix + "_key_cache_out", &cache_out_count);
  const uint16_t *expected_value_cache =
      get_bf16_tensor(prefix + "_value_cache_out", &cache_out_count);

  if (!key || !value || !expected_key_cache || !expected_value_cache) {
    printf("Missing tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }
  if (cache_len > 0 && (!key_cache_in || !value_cache_in)) {
    printf("Missing cache input tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }
  int total_size = (cache_len + num_tokens) * num_heads * head_dim;
  uint16_t *key_cache = (uint16_t *)malloc(total_size * sizeof(uint16_t));
  uint16_t *value_cache = (uint16_t *)malloc(total_size * sizeof(uint16_t));

  if (cache_len > 0) {
    memcpy(key_cache, key_cache_in,
           cache_len * num_heads * head_dim * sizeof(uint16_t));
    memcpy(value_cache, value_cache_in,
           cache_len * num_heads * head_dim * sizeof(uint16_t));
  }

  kv_cache_append_bf16(key_cache, value_cache, key, value, cache_len,
                       num_tokens, num_heads, head_dim);

  float max_err_key =
      compute_max_rel_error_bf16(expected_key_cache, key_cache, total_size);
  float max_err_value =
      compute_max_rel_error_bf16(expected_value_cache, value_cache, total_size);

  ASSERT_LT(max_err_key, 1e-2f);
  ASSERT_LT(max_err_value, 1e-2f);

  free(key_cache);
  free(value_cache);
}

static void run_kv_cache_f16_test(const char *case_name) {
  if (!load_reference_data()) {
    ASSERT(false);
    return;
  }

  std::string prefix = std::string("f16_") + case_name;
  std::string f32_prefix = std::string("f32_") + case_name;
  size_t key_count, value_count, cache_in_count, cache_out_count;

  int cache_len = get_metadata_int(f32_prefix + "_cache_len");
  int num_tokens = get_metadata_int(f32_prefix + "_num_tokens");
  int num_heads = get_metadata_int(f32_prefix + "_num_heads");
  int head_dim = get_metadata_int(f32_prefix + "_head_dim");

  const uint16_t *key = get_f16_tensor(prefix + "_key", &key_count);
  const uint16_t *value = get_f16_tensor(prefix + "_value", &value_count);
  const uint16_t *key_cache_in = nullptr;
  const uint16_t *value_cache_in = nullptr;
  if (cache_len > 0) {
    key_cache_in = get_f16_tensor(prefix + "_key_cache_in", &cache_in_count);
    value_cache_in =
        get_f16_tensor(prefix + "_value_cache_in", &cache_in_count);
  }
  const uint16_t *expected_key_cache =
      get_f16_tensor(prefix + "_key_cache_out", &cache_out_count);
  const uint16_t *expected_value_cache =
      get_f16_tensor(prefix + "_value_cache_out", &cache_out_count);

  if (!key || !value || !expected_key_cache || !expected_value_cache) {
    printf("Missing tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }
  if (cache_len > 0 && (!key_cache_in || !value_cache_in)) {
    printf("Missing cache input tensors for %s\n", case_name);
    ASSERT(false);
    return;
  }
  int total_size = (cache_len + num_tokens) * num_heads * head_dim;
  uint16_t *key_cache = (uint16_t *)malloc(total_size * sizeof(uint16_t));
  uint16_t *value_cache = (uint16_t *)malloc(total_size * sizeof(uint16_t));

  if (cache_len > 0) {
    memcpy(key_cache, key_cache_in,
           cache_len * num_heads * head_dim * sizeof(uint16_t));
    memcpy(value_cache, value_cache_in,
           cache_len * num_heads * head_dim * sizeof(uint16_t));
  }

  kv_cache_append_f16(key_cache, value_cache, key, value, cache_len, num_tokens,
                      num_heads, head_dim);

  float max_err_key =
      compute_max_rel_error_f16(expected_key_cache, key_cache, total_size);
  float max_err_value =
      compute_max_rel_error_f16(expected_value_cache, value_cache, total_size);

  ASSERT_LT(max_err_key, 1e-2f);
  ASSERT_LT(max_err_value, 1e-2f);

  free(key_cache);
  free(value_cache);
}

TEST(kv_cache_pytorch_f32_small) { run_kv_cache_f32_test("small"); }
TEST(kv_cache_pytorch_f32_medium) { run_kv_cache_f32_test("medium"); }
TEST(kv_cache_pytorch_f32_large) { run_kv_cache_f32_test("large"); }
TEST(kv_cache_pytorch_f32_append_small) {
  run_kv_cache_f32_test("append_small");
}
TEST(kv_cache_pytorch_f32_append_medium) {
  run_kv_cache_f32_test("append_medium");
}

TEST(kv_cache_pytorch_bf16_small) { run_kv_cache_bf16_test("small"); }
TEST(kv_cache_pytorch_bf16_medium) { run_kv_cache_bf16_test("medium"); }
TEST(kv_cache_pytorch_bf16_large) { run_kv_cache_bf16_test("large"); }

TEST(kv_cache_pytorch_f16_small) { run_kv_cache_f16_test("small"); }
TEST(kv_cache_pytorch_f16_medium) { run_kv_cache_f16_test("medium"); }
TEST(kv_cache_pytorch_f16_large) { run_kv_cache_f16_test("large"); }

extern "C" void run_kv_cache_pytorch_tests(void) {
  TEST_SUITE("KV Cache PyTorch Accuracy");
  RUN_TEST(kv_cache_pytorch_f32_small);
  RUN_TEST(kv_cache_pytorch_f32_medium);
  RUN_TEST(kv_cache_pytorch_f32_large);
  RUN_TEST(kv_cache_pytorch_f32_append_small);
  RUN_TEST(kv_cache_pytorch_f32_append_medium);
  RUN_TEST(kv_cache_pytorch_bf16_small);
  RUN_TEST(kv_cache_pytorch_bf16_medium);
  RUN_TEST(kv_cache_pytorch_bf16_large);
  RUN_TEST(kv_cache_pytorch_f16_small);
  RUN_TEST(kv_cache_pytorch_f16_medium);
  RUN_TEST(kv_cache_pytorch_f16_large);
}
