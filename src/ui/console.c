#include "ui/console.h"
#include "core/time.h"
#include <stdlib.h>
#include <string.h>

void console_init(ConsoleState *console) {
  if (!console)
    return;
  memset(console, 0, sizeof(ConsoleState));
  console->auto_scroll = true;

  // Read log level from environment variable
  const char *log_level_str = getenv("SILLYTUI_LOG_LEVEL");
  if (log_level_str) {
    if (strcmp(log_level_str, "DEBUG") == 0 ||
        strcmp(log_level_str, "debug") == 0) {
      console->min_level = LOG_DEBUG;
    } else if (strcmp(log_level_str, "INFO") == 0 ||
               strcmp(log_level_str, "info") == 0) {
      console->min_level = LOG_INFO;
    } else if (strcmp(log_level_str, "WARNING") == 0 ||
               strcmp(log_level_str, "warning") == 0 ||
               strcmp(log_level_str, "WARN") == 0 ||
               strcmp(log_level_str, "warn") == 0) {
      console->min_level = LOG_WARNING;
    } else if (strcmp(log_level_str, "ERROR") == 0 ||
               strcmp(log_level_str, "error") == 0) {
      console->min_level = LOG_ERROR;
    } else {
      // Invalid value, default to INFO
      console->min_level = LOG_INFO;
    }
  } else {
    // No env var set, default to INFO
    console->min_level = LOG_INFO;
  }
}

void console_free(ConsoleState *console) {
  // Nothing to free, all data is in the struct
  (void)console;
}

void console_add_log(ConsoleState *console, LogLevel level, const char *file,
                     int line, const char *message) {
  if (!console || !message)
    return;

  // Filter by minimum level
  if (level < console->min_level)
    return;

  // Get timestamp
  char timestamp[32];
  get_timestamp(timestamp, sizeof(timestamp));

  // Extract filename from path
  const char *filename = file ? strrchr(file, '/') : NULL;
  if (filename)
    filename++;
  else
    filename = file ? file : "?";

  // Add entry to ring buffer
  ConsoleLogEntry *entry;
  if (console->count < CONSOLE_MAX_ENTRIES) {
    // Buffer not full yet, add to tail
    entry = &console->entries[console->count];
    console->count++;
  } else {
    // Buffer full, overwrite oldest entry (at head)
    entry = &console->entries[console->head];
    console->head = (console->head + 1) % CONSOLE_MAX_ENTRIES;
  }

  // Copy data to entry
  strncpy(entry->timestamp, timestamp, sizeof(entry->timestamp) - 1);
  entry->timestamp[sizeof(entry->timestamp) - 1] = '\0';
  entry->level = level;
  strncpy(entry->file, filename, sizeof(entry->file) - 1);
  entry->file[sizeof(entry->file) - 1] = '\0';
  entry->line = line;
  strncpy(entry->message, message, sizeof(entry->message) - 1);
  entry->message[sizeof(entry->message) - 1] = '\0';

  // Auto-scroll to bottom if enabled and already at bottom
  if (console->auto_scroll && console->scroll_offset == 0) {
    console_scroll_to_bottom(console);
  }

  // Mark console as needing redraw if visible
  if (console->visible) {
    console->needs_redraw = true;
  }
}

void console_toggle(ConsoleState *console) {
  if (!console)
    return;
  if (console->fullscreen) {
    console->fullscreen = false;
    console->visible = false;
  } else {
    console->visible = !console->visible;
  }
  if (console->visible) {
    console_scroll_to_bottom(console);
  }
}

void console_set_visible(ConsoleState *console, bool visible) {
  if (!console)
    return;
  console->visible = visible;
  if (visible) {
    console_scroll_to_bottom(console);
  }
}

bool console_is_visible(const ConsoleState *console) {
  return console && console->visible;
}

void console_scroll(ConsoleState *console, int direction) {
  if (!console)
    return;

  int max_scroll = (int)console->count - 1;
  if (max_scroll < 0)
    max_scroll = 0;

  console->scroll_offset += direction;
  if (console->scroll_offset < 0)
    console->scroll_offset = 0;
  if (console->scroll_offset > max_scroll)
    console->scroll_offset = max_scroll;

  // Disable auto-scroll if user manually scrolls
  if (direction != 0)
    console->auto_scroll = (console->scroll_offset == 0);
}

void console_scroll_to_top(ConsoleState *console) {
  if (!console)
    return;
  int max_scroll = (int)console->count - 1;
  if (max_scroll < 0)
    max_scroll = 0;
  console->scroll_offset = max_scroll;
  console->auto_scroll = false;
}

void console_scroll_to_bottom(ConsoleState *console) {
  if (!console)
    return;
  console->scroll_offset = 0;
  console->auto_scroll = true;
}

void console_clear(ConsoleState *console) {
  if (!console)
    return;
  console->count = 0;
  console->head = 0;
  console->tail = 0;
  console->scroll_offset = 0;
}

void console_set_min_level(ConsoleState *console, LogLevel min_level) {
  if (!console)
    return;
  console->min_level = min_level;
}

void console_toggle_fullscreen(ConsoleState *console) {
  if (!console)
    return;
  console->fullscreen = !console->fullscreen;
  if (console->fullscreen) {
    console->visible = true;
    console->auto_scroll = true;
    console_scroll_to_bottom(console);
  }
}

bool console_is_fullscreen(const ConsoleState *console) {
  if (!console)
    return false;
  return console->fullscreen;
}
