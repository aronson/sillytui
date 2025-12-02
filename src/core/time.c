#include "core/time.h"
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

void get_timestamp(char *buf, size_t buf_size) {
  if (!buf || buf_size == 0)
    return;

  struct timeval tv;
  gettimeofday(&tv, NULL);

  struct tm *tm_info = localtime(&tv.tv_sec);
  if (!tm_info) {
    buf[0] = '\0';
    return;
  }

  size_t written = strftime(buf, buf_size, "%H:%M:%S", tm_info);
  if (written > 0 && written < buf_size - 4) {
    snprintf(buf + written, buf_size - written, ".%03ld",
             (long)(tv.tv_usec / 1000));
  }
  if (buf_size > 0) {
    buf[buf_size - 1] = '\0';
  }
}
