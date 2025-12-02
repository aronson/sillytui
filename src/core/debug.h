#ifndef DEBUG_H
#define DEBUG_H

#include "core/log.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef SILLYTUI_DEBUG
#define SILLYTUI_ENABLE_ASSERT 1
#else
#define SILLYTUI_ENABLE_ASSERT 0
#endif

#if defined(SILLYTUI_COVERAGE_TEST) || defined(SILLYTUI_DEBUG)
extern unsigned int g_coverage_counter;
#define testcase(X)                                                            \
  do {                                                                         \
    if (X) {                                                                   \
      g_coverage_counter += (unsigned)__LINE__;                                \
    }                                                                          \
  } while (0)
#else
#define testcase(X)
#endif

#if SILLYTUI_ENABLE_ASSERT
#define SILLYTUI_ASSERT(X)                                                     \
  do {                                                                         \
    if (!(X)) {                                                                \
      log_message(LOG_ERROR, __FILE__, __LINE__, "ASSERTION FAILED: %s", #X);  \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define ALWAYS(X)                                                              \
  ((X) ? 1                                                                     \
       : (log_message(LOG_ERROR, __FILE__, __LINE__, "ALWAYS failed: %s", #X), \
          abort(), 0))

#define NEVER(X)                                                               \
  ((X) ? (log_message(LOG_ERROR, __FILE__, __LINE__, "NEVER failed: %s", #X),  \
          abort(), 1)                                                          \
       : 0)

#else
#define SILLYTUI_ASSERT(X)
#define ALWAYS(X) (X)
#define NEVER(X) (X)
#endif

#define TESTONLY(X) X

#ifdef SILLYTUI_DEBUG
#define VVA_ONLY(X) X
#else
#define VVA_ONLY(X)
#endif

#define UNUSED_PARAM(x) (void)(x)

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

#define STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)

#define COMPILE_TIME_ASSERT(expr, name)                                        \
  typedef char assert_##name[(expr) ? 1 : -1]

#ifdef SILLYTUI_DEBUG
#define DEBUG_ONLY(X) X
#define DEBUG_PRINTF(fmt, ...)                                                 \
  log_message(LOG_DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#else
#define DEBUG_ONLY(X)
#define DEBUG_PRINTF(fmt, ...)
#endif

#define UNREACHABLE()                                                          \
  do {                                                                         \
    log_message(LOG_ERROR, __FILE__, __LINE__, "UNREACHABLE");                 \
    abort();                                                                   \
  } while (0)

#define CHECK_NOT_NULL(ptr)                                                    \
  do {                                                                         \
    if ((ptr) == NULL) {                                                       \
      log_message(LOG_ERROR, __FILE__, __LINE__, "NULL pointer: %s", #ptr);    \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define CHECK_BOUNDS(index, size)                                              \
  do {                                                                         \
    if ((size_t)(index) >= (size_t)(size)) {                                   \
      log_message(LOG_ERROR, __FILE__, __LINE__,                               \
                  "Bounds check failed: %s >= %s", #index, #size);             \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define IMPLIES(a, b) (!(a) || (b))

#define RANGE_CHECK(val, min, max)                                             \
  SILLYTUI_ASSERT((val) >= (min) && (val) <= (max))

#endif
