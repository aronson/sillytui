#ifndef ERROR_H
#define ERROR_H

#include <stdbool.h>

void log_error_user(const char *fmt, ...);
void log_error_internal(const char *fmt, ...);
void log_warning_internal(const char *fmt, ...);
void log_info_internal(const char *fmt, ...);

void log_set_user_error_callback(void (*callback)(const char *msg));
void log_set_internal_error_callback(void (*callback)(const char *msg));
void log_set_warning_callback(void (*callback)(const char *msg));
void log_set_info_callback(void (*callback)(const char *msg));

#endif
