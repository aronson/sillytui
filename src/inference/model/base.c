#include "inference/model/base.h"
#include "inference/model/qwen3/qwen3.h"
#include <stdlib.h>
#include <string.h>

static bool qwen3_load_wrapper(inference_model_t *model,
                               const char *model_dir) {
  qwen3_model_t *qwen3 = (qwen3_model_t *)model->impl;
  return qwen3_model_load(qwen3, model_dir);
}

static void qwen3_free_wrapper(inference_model_t *model) {
  qwen3_model_t *qwen3 = (qwen3_model_t *)model->impl;
  qwen3_model_free(qwen3);
  free(qwen3);
  model->impl = NULL;
}

static void qwen3_reset_cache_wrapper(inference_model_t *model) {
  qwen3_model_t *qwen3 = (qwen3_model_t *)model->impl;
  qwen3_model_reset_cache(qwen3);
}

static bool qwen3_forward_wrapper(inference_model_t *model, float *logits,
                                  const int *token_ids, int num_tokens) {
  qwen3_model_t *qwen3 = (qwen3_model_t *)model->impl;
  return qwen3_forward(qwen3, logits, token_ids, num_tokens);
}

static int qwen3_generate_wrapper(inference_model_t *model, int *output_tokens,
                                  int max_tokens, const int *input_tokens,
                                  int num_input_tokens, float temperature,
                                  int top_k, float top_p) {
  qwen3_model_t *qwen3 = (qwen3_model_t *)model->impl;
  return qwen3_generate(qwen3, output_tokens, max_tokens, input_tokens,
                        num_input_tokens, temperature, top_k, top_p);
}

static const inference_model_ops_t qwen3_ops = {
    .load = qwen3_load_wrapper,
    .free = qwen3_free_wrapper,
    .reset_cache = qwen3_reset_cache_wrapper,
    .forward = qwen3_forward_wrapper,
    .generate = qwen3_generate_wrapper,
};

bool inference_model_load(inference_model_t *model, const char *model_type,
                          const char *model_dir) {
  if (!model || !model_type || !model_dir)
    return false;

  memset(model, 0, sizeof(*model));

  if (strcmp(model_type, "qwen3") == 0) {
    qwen3_model_t *qwen3 = (qwen3_model_t *)calloc(1, sizeof(qwen3_model_t));
    if (!qwen3)
      return false;

    model->ops = &qwen3_ops;
    model->impl = qwen3;

    return qwen3_model_load(qwen3, model_dir);
  }

  return false;
}

void inference_model_free(inference_model_t *model) {
  if (!model || !model->ops)
    return;

  if (model->ops->free) {
    model->ops->free(model);
  }
  memset(model, 0, sizeof(*model));
}

void inference_model_reset_cache(inference_model_t *model) {
  if (!model || !model->ops || !model->ops->reset_cache)
    return;
  model->ops->reset_cache(model);
}

bool inference_model_forward(inference_model_t *model, float *logits,
                             const int *token_ids, int num_tokens) {
  if (!model || !model->ops || !model->ops->forward)
    return false;
  return model->ops->forward(model, logits, token_ids, num_tokens);
}

int inference_model_generate(inference_model_t *model, int *output_tokens,
                             int max_tokens, const int *input_tokens,
                             int num_input_tokens, float temperature, int top_k,
                             float top_p) {
  if (!model || !model->ops || !model->ops->generate)
    return 0;
  return model->ops->generate(model, output_tokens, max_tokens, input_tokens,
                              num_input_tokens, temperature, top_k, top_p);
}
