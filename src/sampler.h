#ifndef SAMPLER_H
#define SAMPLER_H

#include "config.h"
#include <stdbool.h>

#define MAX_CUSTOM_SAMPLERS 32
#define CUSTOM_SAMPLER_NAME_LEN 64

typedef struct {
  char name[CUSTOM_SAMPLER_NAME_LEN];
  double value;
  bool is_int;
} CustomSampler;

typedef struct {
  double temperature;
  double top_p;
  int top_k;
  double min_p;
  double frequency_penalty;
  double presence_penalty;
  double repetition_penalty;
  double typical_p;
  double tfs;
  double top_a;
  double smoothing_factor;
  int min_tokens;
  int max_tokens;

  double dynatemp_min;
  double dynatemp_max;
  double dynatemp_exponent;

  int mirostat_mode;
  double mirostat_tau;
  double mirostat_eta;

  double dry_multiplier;
  double dry_base;
  int dry_allowed_length;
  int dry_range;

  double xtc_threshold;
  double xtc_probability;

  double nsigma;
  double skew;

  CustomSampler custom[MAX_CUSTOM_SAMPLERS];
  int custom_count;
} SamplerSettings;

void sampler_init_defaults(SamplerSettings *s);
bool sampler_load(SamplerSettings *s, ApiType api_type);
bool sampler_save(const SamplerSettings *s, ApiType api_type);
bool sampler_add_custom(SamplerSettings *s, const char *name, double value,
                        bool is_int);
bool sampler_remove_custom(SamplerSettings *s, int index);

#endif
