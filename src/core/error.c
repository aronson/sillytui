#include "core/error.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void (*g_user_error_callback)(const char *msg) = NULL;
static void (*g_internal_error_callback)(const char *msg) = NULL;
static void (*g_warning_callback)(const char *msg) = NULL;
static void (*g_info_callback)(const char *msg) = NULL;

void log_set_user_error_callback(void (*callback)(const char *msg)) {
  g_user_error_callback = callback;
}

void log_set_internal_error_callback(void (*callback)(const char *msg)) {
  g_internal_error_callback = callback;
}

void log_set_warning_callback(void (*callback)(const char *msg)) {
  g_warning_callback = callback;
}

void log_set_info_callback(void (*callback)(const char *msg)) {
  g_info_callback = callback;
}

static void format_message(char *buf, size_t buf_size, const char *fmt,
                           va_list args) {
  if (!buf || buf_size == 0)
    return;
  vsnprintf(buf, buf_size, fmt, args);
  if (buf_size > 0)
    buf[buf_size - 1] = '\0';
}

void log_error_user(const char *fmt, ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  format_message(msg, sizeof(msg), fmt, args);
  va_end(args);

  if (g_user_error_callback) {
    g_user_error_callback(msg);
  } else {
    fprintf(stderr, "Error: %s\n", msg);
  }
}

void log_error_internal(const char *fmt, ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  format_message(msg, sizeof(msg), fmt, args);
  va_end(args);

  if (g_internal_error_callback) {
    g_internal_error_callback(msg);
  } else {
    fprintf(stderr, "[ERROR] %s\n", msg);
  }
}

void log_warning_internal(const char *fmt, ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  format_message(msg, sizeof(msg), fmt, args);
  va_end(args);

  if (g_warning_callback) {
    g_warning_callback(msg);
  } else {
    fprintf(stderr, "[WARNING] %s\n", msg);
  }
}

void log_info_internal(const char *fmt, ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  format_message(msg, sizeof(msg), fmt, args);
  va_end(args);

  if (g_info_callback) {
    g_info_callback(msg);
  } else {
    fprintf(stderr, "[INFO] %s\n", msg);
  }
}
