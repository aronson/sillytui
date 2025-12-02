#ifndef LOG_H
#define LOG_H

#include <stddef.h>

typedef enum { LOG_DEBUG = 0, LOG_INFO, LOG_WARNING, LOG_ERROR } LogLevel;

void log_message(LogLevel level, const char *file, int line, const char *fmt,
                 ...);
void log_set_callback(void (*callback)(LogLevel level, const char *file,
                                       int line, const char *msg));

#endif
