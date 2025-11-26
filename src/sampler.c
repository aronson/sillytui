#include "sampler.h"
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

void sampler_init_defaults(SamplerSettings *s) {
  memset(s, 0, sizeof(*s));
  s->temperature = 1.0;
  s->top_p = 1.0;
  s->top_k = -1;
  s->min_p = 0.0;
  s->frequency_penalty = 0.0;
  s->presence_penalty = 0.0;
  s->repetition_penalty = 1.0;
  s->typical_p = 1.0;
  s->tfs = 1.0;
  s->top_a = 0.0;
  s->smoothing_factor = 0.0;
  s->min_tokens = 0;
  s->max_tokens = 512;
  s->dynatemp_min = 0.0;
  s->dynatemp_max = 0.0;
  s->dynatemp_exponent = 1.0;
  s->mirostat_mode = 0;
  s->mirostat_tau = 0.0;
  s->mirostat_eta = 0.0;
  s->dry_multiplier = 0.0;
  s->dry_base = 1.75;
  s->dry_allowed_length = 2;
  s->dry_range = 0;
  s->xtc_threshold = 0.1;
  s->xtc_probability = 0.0;
  s->nsigma = 0.0;
  s->skew = 0.0;
}

static bool get_config_path(char *buf, size_t bufsize) {
  const char *home = getenv("HOME");
  if (!home) {
    struct passwd *pw = getpwuid(getuid());
    if (pw)
      home = pw->pw_dir;
  }
  if (!home)
    return false;
  snprintf(buf, bufsize, "%s/.config/sillytui", home);
  return true;
}

static bool get_samplers_path(char *buf, size_t bufsize, ApiType api_type) {
  char dir[500];
  if (!get_config_path(dir, sizeof(dir)))
    return false;
  int written = snprintf(buf, bufsize, "%s/samplers_%s.json", dir,
                         api_type_name(api_type));
  return written > 0 && (size_t)written < bufsize;
}

static char *skip_ws(char *p) {
  while (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
    p++;
  return p;
}

static char *parse_string(char *p, char *out, size_t outsize) {
  p = skip_ws(p);
  if (*p != '"')
    return NULL;
  p++;
  size_t i = 0;
  while (*p && *p != '"' && i < outsize - 1) {
    if (*p == '\\' && *(p + 1)) {
      p++;
      if (*p == 'n')
        out[i++] = '\n';
      else if (*p == 't')
        out[i++] = '\t';
      else
        out[i++] = *p;
      p++;
    } else {
      out[i++] = *p++;
    }
  }
  out[i] = '\0';
  if (*p == '"')
    p++;
  return p;
}

static double parse_double(char **pp) {
  char *p = *pp;
  p = skip_ws(p);
  char *end;
  double val = strtod(p, &end);
  *pp = end;
  return val;
}

static int parse_int(char **pp) {
  char *p = *pp;
  p = skip_ws(p);
  char *end;
  int val = (int)strtol(p, &end, 10);
  *pp = end;
  return val;
}

bool sampler_load(SamplerSettings *s, ApiType api_type) {
  sampler_init_defaults(s);

  char path[512];
  if (!get_samplers_path(path, sizeof(path), api_type))
    return false;

  FILE *f = fopen(path, "r");
  if (!f)
    return true;

  fseek(f, 0, SEEK_END);
  long len = ftell(f);
  fseek(f, 0, SEEK_SET);

  char *data = malloc(len + 1);
  if (!data) {
    fclose(f);
    return false;
  }
  size_t read_len = fread(data, 1, len, f);
  data[read_len] = '\0';
  fclose(f);

  char *p = data;
  p = skip_ws(p);
  if (*p != '{') {
    free(data);
    return false;
  }
  p++;

  while (*p) {
    p = skip_ws(p);
    if (*p == '}')
      break;
    if (*p == ',') {
      p++;
      continue;
    }

    char key[64];
    p = parse_string(p, key, sizeof(key));
    if (!p)
      break;
    p = skip_ws(p);
    if (*p == ':')
      p++;

    if (strcmp(key, "temperature") == 0)
      s->temperature = parse_double(&p);
    else if (strcmp(key, "top_p") == 0)
      s->top_p = parse_double(&p);
    else if (strcmp(key, "top_k") == 0)
      s->top_k = parse_int(&p);
    else if (strcmp(key, "min_p") == 0)
      s->min_p = parse_double(&p);
    else if (strcmp(key, "frequency_penalty") == 0)
      s->frequency_penalty = parse_double(&p);
    else if (strcmp(key, "presence_penalty") == 0)
      s->presence_penalty = parse_double(&p);
    else if (strcmp(key, "repetition_penalty") == 0)
      s->repetition_penalty = parse_double(&p);
    else if (strcmp(key, "typical_p") == 0)
      s->typical_p = parse_double(&p);
    else if (strcmp(key, "tfs") == 0)
      s->tfs = parse_double(&p);
    else if (strcmp(key, "top_a") == 0)
      s->top_a = parse_double(&p);
    else if (strcmp(key, "smoothing_factor") == 0)
      s->smoothing_factor = parse_double(&p);
    else if (strcmp(key, "min_tokens") == 0)
      s->min_tokens = parse_int(&p);
    else if (strcmp(key, "max_tokens") == 0)
      s->max_tokens = parse_int(&p);
    else if (strcmp(key, "dynatemp_min") == 0)
      s->dynatemp_min = parse_double(&p);
    else if (strcmp(key, "dynatemp_max") == 0)
      s->dynatemp_max = parse_double(&p);
    else if (strcmp(key, "dynatemp_exponent") == 0)
      s->dynatemp_exponent = parse_double(&p);
    else if (strcmp(key, "mirostat_mode") == 0)
      s->mirostat_mode = parse_int(&p);
    else if (strcmp(key, "mirostat_tau") == 0)
      s->mirostat_tau = parse_double(&p);
    else if (strcmp(key, "mirostat_eta") == 0)
      s->mirostat_eta = parse_double(&p);
    else if (strcmp(key, "dry_multiplier") == 0)
      s->dry_multiplier = parse_double(&p);
    else if (strcmp(key, "dry_base") == 0)
      s->dry_base = parse_double(&p);
    else if (strcmp(key, "dry_allowed_length") == 0)
      s->dry_allowed_length = parse_int(&p);
    else if (strcmp(key, "dry_range") == 0)
      s->dry_range = parse_int(&p);
    else if (strcmp(key, "xtc_threshold") == 0)
      s->xtc_threshold = parse_double(&p);
    else if (strcmp(key, "xtc_probability") == 0)
      s->xtc_probability = parse_double(&p);
    else if (strcmp(key, "nsigma") == 0)
      s->nsigma = parse_double(&p);
    else if (strcmp(key, "skew") == 0)
      s->skew = parse_double(&p);
    else if (strcmp(key, "custom") == 0) {
      p = skip_ws(p);
      if (*p == '[') {
        p++;
        while (*p && *p != ']') {
          p = skip_ws(p);
          if (*p == ',') {
            p++;
            continue;
          }
          if (*p == '{') {
            p++;
            char cname[CUSTOM_SAMPLER_NAME_LEN] = {0};
            double cval = 0, cmin = 0, cmax = 100, cstep = 0.1;
            bool cint = false;
            while (*p && *p != '}') {
              p = skip_ws(p);
              if (*p == ',') {
                p++;
                continue;
              }
              char ckey[32];
              p = parse_string(p, ckey, sizeof(ckey));
              if (!p)
                break;
              p = skip_ws(p);
              if (*p == ':')
                p++;
              p = skip_ws(p);
              if (strcmp(ckey, "name") == 0) {
                p = parse_string(p, cname, sizeof(cname));
              } else if (strcmp(ckey, "value") == 0) {
                cval = parse_double(&p);
              } else if (strcmp(ckey, "is_int") == 0) {
                if (strncmp(p, "true", 4) == 0) {
                  cint = true;
                  p += 4;
                } else if (strncmp(p, "false", 5) == 0) {
                  cint = false;
                  p += 5;
                }
              } else if (strcmp(ckey, "min") == 0) {
                cmin = parse_double(&p);
              } else if (strcmp(ckey, "max") == 0) {
                cmax = parse_double(&p);
              } else if (strcmp(ckey, "step") == 0) {
                cstep = parse_double(&p);
              }
            }
            if (*p == '}')
              p++;
            if (cname[0] && s->custom_count < MAX_CUSTOM_SAMPLERS) {
              CustomSampler *cs = &s->custom[s->custom_count];
              strncpy(cs->name, cname, CUSTOM_SAMPLER_NAME_LEN - 1);
              cs->value = cval;
              cs->is_int = cint;
              cs->min_val = cmin;
              cs->max_val = cmax;
              cs->step = cstep > 0 ? cstep : (cint ? 1.0 : 0.1);
              s->custom_count++;
            }
          } else {
            break;
          }
        }
        if (*p == ']')
          p++;
      }
    } else {
      while (*p && *p != ',' && *p != '}')
        p++;
    }
  }

  free(data);
  return true;
}

bool sampler_save(const SamplerSettings *s, ApiType api_type) {
  config_ensure_dir();

  char path[512];
  if (!get_samplers_path(path, sizeof(path), api_type))
    return false;

  FILE *f = fopen(path, "w");
  if (!f)
    return false;

  fprintf(f, "{\n");
  fprintf(f, "  \"temperature\": %.4g,\n", s->temperature);
  fprintf(f, "  \"top_p\": %.4g,\n", s->top_p);
  fprintf(f, "  \"top_k\": %d,\n", s->top_k);
  fprintf(f, "  \"min_p\": %.4g,\n", s->min_p);
  fprintf(f, "  \"frequency_penalty\": %.4g,\n", s->frequency_penalty);
  fprintf(f, "  \"presence_penalty\": %.4g,\n", s->presence_penalty);
  fprintf(f, "  \"repetition_penalty\": %.4g,\n", s->repetition_penalty);
  fprintf(f, "  \"typical_p\": %.4g,\n", s->typical_p);
  fprintf(f, "  \"tfs\": %.4g,\n", s->tfs);
  fprintf(f, "  \"top_a\": %.4g,\n", s->top_a);
  fprintf(f, "  \"smoothing_factor\": %.4g,\n", s->smoothing_factor);
  fprintf(f, "  \"min_tokens\": %d,\n", s->min_tokens);
  fprintf(f, "  \"max_tokens\": %d,\n", s->max_tokens);
  fprintf(f, "  \"dynatemp_min\": %.4g,\n", s->dynatemp_min);
  fprintf(f, "  \"dynatemp_max\": %.4g,\n", s->dynatemp_max);
  fprintf(f, "  \"dynatemp_exponent\": %.4g,\n", s->dynatemp_exponent);
  fprintf(f, "  \"mirostat_mode\": %d,\n", s->mirostat_mode);
  fprintf(f, "  \"mirostat_tau\": %.4g,\n", s->mirostat_tau);
  fprintf(f, "  \"mirostat_eta\": %.4g,\n", s->mirostat_eta);
  fprintf(f, "  \"dry_multiplier\": %.4g,\n", s->dry_multiplier);
  fprintf(f, "  \"dry_base\": %.4g,\n", s->dry_base);
  fprintf(f, "  \"dry_allowed_length\": %d,\n", s->dry_allowed_length);
  fprintf(f, "  \"dry_range\": %d,\n", s->dry_range);
  fprintf(f, "  \"xtc_threshold\": %.4g,\n", s->xtc_threshold);
  fprintf(f, "  \"xtc_probability\": %.4g,\n", s->xtc_probability);
  fprintf(f, "  \"nsigma\": %.4g,\n", s->nsigma);
  fprintf(f, "  \"skew\": %.4g", s->skew);

  if (s->custom_count > 0) {
    fprintf(f, ",\n  \"custom\": [\n");
    for (int i = 0; i < s->custom_count; i++) {
      const CustomSampler *cs = &s->custom[i];
      fprintf(f,
              "    {\"name\": \"%s\", \"value\": %.4g, \"is_int\": %s, "
              "\"min\": %.4g, \"max\": %.4g, \"step\": %.4g}%s\n",
              cs->name, cs->value, cs->is_int ? "true" : "false", cs->min_val,
              cs->max_val, cs->step, i < s->custom_count - 1 ? "," : "");
    }
    fprintf(f, "  ]\n");
  } else {
    fprintf(f, "\n");
  }
  fprintf(f, "}\n");

  fclose(f);
  return true;
}

bool sampler_add_custom(SamplerSettings *s, const char *name, double value,
                        bool is_int, double min_val, double max_val,
                        double step) {
  if (s->custom_count >= MAX_CUSTOM_SAMPLERS)
    return false;
  CustomSampler *cs = &s->custom[s->custom_count];
  strncpy(cs->name, name, CUSTOM_SAMPLER_NAME_LEN - 1);
  cs->name[CUSTOM_SAMPLER_NAME_LEN - 1] = '\0';
  cs->value = value;
  cs->is_int = is_int;
  cs->min_val = min_val;
  cs->max_val = max_val;
  cs->step = step > 0 ? step : (is_int ? 1.0 : 0.1);
  s->custom_count++;
  return true;
}

bool sampler_remove_custom(SamplerSettings *s, int index) {
  if (index < 0 || index >= s->custom_count)
    return false;
  for (int i = index; i < s->custom_count - 1; i++) {
    s->custom[i] = s->custom[i + 1];
  }
  s->custom_count--;
  return true;
}
