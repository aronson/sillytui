/*
 * Activation PyTorch Accuracy Tests
 */

#include "inference/model_loader/safetensors.hh"
#include "test_framework.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" {
#include "inference/linalg/activation.h"
}

static safetensors::safetensors_t g_reference_data;
static bool g_reference_loaded = false;

static bool load_reference_data() {
  if (g_reference_loaded)
    return true;

  std::string warn, err;
  bool ret = safetensors::mmap_from_file(
      "tests/activation_reference.safetensors", &g_reference_data, &warn, &err);
  if (!ret) {
    fprintf(stderr, "Failed to load activation reference: %s\n", err.c_str());
    return false;
  }

  g_reference_loaded = true;
  return true;
}

static float *load_tensor(const char *name, size_t *out_size) {
  if (!load_reference_data())
    return nullptr;

  safetensors::tensor_t tensor;
  bool found = false;

  for (size_t i = 0; i < g_reference_data.tensors.size(); i++) {
    std::string key = g_reference_data.tensors.keys()[i];
    if (key == name) {
      g_reference_data.tensors.at(i, &tensor);
      found = true;
      break;
    }
  }

  if (!found)
    return nullptr;

  if (tensor.dtype != safetensors::dtype::kFLOAT32)
    return nullptr;

  size_t total = safetensors::get_shape_size(tensor);
  float *data = (float *)malloc(total * sizeof(float));
  if (!data)
    return nullptr;

  const uint8_t *src =
      g_reference_data.databuffer_addr + tensor.data_offsets[0];
  memcpy(data, src, total * sizeof(float));

  *out_size = total;
  return data;
}

static double compute_max_relative_error(const float *expected,
                                         const float *actual, size_t n,
                                         double atol = 1e-6) {
  double max_err = 0.0;
  for (size_t i = 0; i < n; i++) {
    double diff = fabs((double)expected[i] - (double)actual[i]);
    double mag = fmax(fabs((double)expected[i]), atol);
    double err = diff / mag;
    if (err > max_err)
      max_err = err;
  }
  return max_err;
}

/* ============ SiLU Tests ============ */

TEST(pytorch_silu_small_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("silu_small_input", &in_size);
  float *expected = load_tensor("silu_small_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_silu_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("silu_decode_input", &in_size);
  float *expected = load_tensor("silu_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_silu_prefill_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("silu_prefill_input", &in_size);
  float *expected = load_tensor("silu_prefill_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_f32(output, input, 512, 4096);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

/* ============ SwiGLU Tests ============ */

TEST(pytorch_swiglu_small_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("swiglu_small_input", &in_size);
  float *expected = load_tensor("swiglu_small_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_and_mul_f32(output, input, 1, (int)out_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_swiglu_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("swiglu_decode_input", &in_size);
  float *expected = load_tensor("swiglu_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_and_mul_f32(output, input, 1, (int)out_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_swiglu_prefill_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("swiglu_prefill_input", &in_size);
  float *expected = load_tensor("swiglu_prefill_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  silu_and_mul_f32(output, input, 512, 4096);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

/* ============ GELU Tests ============ */

TEST(pytorch_gelu_small_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_small_input", &in_size);
  float *expected = load_tensor("gelu_small_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_gelu_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_decode_input", &in_size);
  float *expected = load_tensor("gelu_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_geglu_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("geglu_decode_input", &in_size);
  float *expected = load_tensor("geglu_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_and_mul_f32(output, input, 1, (int)out_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

/* ============ GELU Tanh Tests ============ */

TEST(pytorch_gelu_tanh_small_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_tanh_small_input", &in_size);
  float *expected = load_tensor("gelu_tanh_small_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_tanh_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_gelu_tanh_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_tanh_decode_input", &in_size);
  float *expected = load_tensor("gelu_tanh_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_tanh_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_gelu_tanh_mul_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_tanh_mul_decode_input", &in_size);
  float *expected = load_tensor("gelu_tanh_mul_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_tanh_and_mul_f32(output, input, 1, (int)out_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 0.01) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 0.01);

  free(input);
  free(expected);
  free(output);
}

/* ============ GELU Quick Tests ============ */

TEST(pytorch_gelu_quick_small_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_quick_small_input", &in_size);
  float *expected = load_tensor("gelu_quick_small_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_quick_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_gelu_quick_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_quick_decode_input", &in_size);
  float *expected = load_tensor("gelu_quick_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_quick_f32(output, input, 1, (int)in_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

TEST(pytorch_gelu_quick_mul_decode_f32) {
  size_t in_size, out_size;
  float *input = load_tensor("gelu_quick_mul_decode_input", &in_size);
  float *expected = load_tensor("gelu_quick_mul_decode_output", &out_size);
  ASSERT(input && expected);

  float *output = (float *)malloc(out_size * sizeof(float));
  gelu_quick_and_mul_f32(output, input, 1, (int)out_size);

  double max_err = compute_max_relative_error(expected, output, out_size);
  if (max_err >= 1e-4) {
    printf("max_err = %.6e ", max_err);
  }
  ASSERT_LT(max_err, 1e-4);

  free(input);
  free(expected);
  free(output);
}

/* ============ Test Registration ============ */

extern "C" void run_activation_pytorch_tests(void) {
  RUN_TEST(pytorch_silu_small_f32);
  RUN_TEST(pytorch_silu_decode_f32);
  RUN_TEST(pytorch_silu_prefill_f32);
  RUN_TEST(pytorch_swiglu_small_f32);
  RUN_TEST(pytorch_swiglu_decode_f32);
  RUN_TEST(pytorch_swiglu_prefill_f32);
  RUN_TEST(pytorch_gelu_small_f32);
  RUN_TEST(pytorch_gelu_decode_f32);
  RUN_TEST(pytorch_geglu_decode_f32);
  RUN_TEST(pytorch_gelu_tanh_small_f32);
  RUN_TEST(pytorch_gelu_tanh_decode_f32);
  RUN_TEST(pytorch_gelu_tanh_mul_decode_f32);
  RUN_TEST(pytorch_gelu_quick_small_f32);
  RUN_TEST(pytorch_gelu_quick_decode_f32);
  RUN_TEST(pytorch_gelu_quick_mul_decode_f32);
}
