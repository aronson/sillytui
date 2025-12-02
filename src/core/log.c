#include "core/log.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void (*g_log_callback)(LogLevel level, const char *file, int line,
                              const char *msg) = NULL;

void log_set_callback(void (*callback)(LogLevel level, const char *file,
                                       int line, const char *msg)) {
  g_log_callback = callback;
}

void log_message(LogLevel level, const char *file, int line, const char *fmt,
                 ...) {
  char msg[512];
  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, sizeof(msg), fmt, args);
  va_end(args);
  if (sizeof(msg) > 0)
    msg[sizeof(msg) - 1] = '\0';

  if (g_log_callback) {
    g_log_callback(level, file, line, msg);
  } else {
    const char *level_str = "DEBUG";
    switch (level) {
    case LOG_DEBUG:
      level_str = "DEBUG";
      break;
    case LOG_INFO:
      level_str = "INFO";
      break;
    case LOG_WARNING:
      level_str = "WARNING";
      break;
    case LOG_ERROR:
      level_str = "ERROR";
      break;
    }
    const char *filename = file ? strrchr(file, '/') : NULL;
    if (filename)
      filename++;
    else
      filename = file;
    fprintf(stderr, "[%s] %s:%d: %s\n", level_str, filename ? filename : "?",
            line, msg);
  }
}
