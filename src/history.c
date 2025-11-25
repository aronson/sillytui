#include "history.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static char *dup_string(const char *source) {
  size_t len = strlen(source) + 1;
  char *copy = malloc(len);
  if (copy) {
    memcpy(copy, source, len);
  }
  return copy;
}

static void message_init(ChatMessage *msg) {
  msg->swipes = NULL;
  msg->swipe_count = 0;
  msg->active_swipe = 0;
}

static void message_free(ChatMessage *msg) {
  if (!msg)
    return;
  for (size_t i = 0; i < msg->swipe_count; i++) {
    free(msg->swipes[i]);
  }
  free(msg->swipes);
  msg->swipes = NULL;
  msg->swipe_count = 0;
  msg->active_swipe = 0;
}

void history_init(ChatHistory *history) {
  history->messages = NULL;
  history->count = 0;
  history->capacity = 0;
}

void history_free(ChatHistory *history) {
  if (!history)
    return;
  for (size_t i = 0; i < history->count; i++) {
    message_free(&history->messages[i]);
  }
  free(history->messages);
  history->messages = NULL;
  history->count = 0;
  history->capacity = 0;
}

size_t history_add(ChatHistory *history, const char *message) {
  if (history->count == history->capacity) {
    size_t new_capacity = history->capacity == 0 ? 8 : history->capacity * 2;
    ChatMessage *tmp =
        realloc(history->messages, new_capacity * sizeof(ChatMessage));
    if (!tmp)
      return SIZE_MAX;
    history->messages = tmp;
    history->capacity = new_capacity;
  }

  ChatMessage *msg = &history->messages[history->count];
  message_init(msg);

  msg->swipes = malloc(sizeof(char *));
  if (!msg->swipes)
    return SIZE_MAX;

  msg->swipes[0] = dup_string(message);
  if (!msg->swipes[0]) {
    free(msg->swipes);
    return SIZE_MAX;
  }
  msg->swipe_count = 1;
  msg->active_swipe = 0;

  history->count++;
  return history->count - 1;
}

void history_update(ChatHistory *history, size_t index, const char *message) {
  if (index >= history->count)
    return;

  ChatMessage *msg = &history->messages[index];
  if (msg->swipe_count == 0)
    return;

  char *copy = dup_string(message);
  if (!copy)
    return;

  free(msg->swipes[msg->active_swipe]);
  msg->swipes[msg->active_swipe] = copy;
}

const char *history_get(const ChatHistory *history, size_t index) {
  if (index >= history->count)
    return NULL;

  ChatMessage *msg = &history->messages[index];
  if (msg->swipe_count == 0 || msg->active_swipe >= msg->swipe_count)
    return NULL;

  return msg->swipes[msg->active_swipe];
}

const char *history_get_swipe(const ChatHistory *history, size_t msg_index,
                              size_t swipe_index) {
  if (msg_index >= history->count)
    return NULL;

  ChatMessage *msg = &history->messages[msg_index];
  if (swipe_index >= msg->swipe_count)
    return NULL;

  return msg->swipes[swipe_index];
}

void history_update_swipe(ChatHistory *history, size_t msg_index,
                          size_t swipe_index, const char *message) {
  if (msg_index >= history->count)
    return;

  ChatMessage *msg = &history->messages[msg_index];
  if (swipe_index >= msg->swipe_count)
    return;

  char *copy = dup_string(message);
  if (!copy)
    return;

  free(msg->swipes[swipe_index]);
  msg->swipes[swipe_index] = copy;
}

size_t history_add_swipe(ChatHistory *history, size_t msg_index,
                         const char *content) {
  if (msg_index >= history->count)
    return SIZE_MAX;

  ChatMessage *msg = &history->messages[msg_index];
  if (msg->swipe_count >= MAX_SWIPES)
    return SIZE_MAX;

  char **new_swipes =
      realloc(msg->swipes, (msg->swipe_count + 1) * sizeof(char *));
  if (!new_swipes)
    return SIZE_MAX;
  msg->swipes = new_swipes;

  msg->swipes[msg->swipe_count] = dup_string(content);
  if (!msg->swipes[msg->swipe_count])
    return SIZE_MAX;

  msg->swipe_count++;
  msg->active_swipe = msg->swipe_count - 1;
  return msg->active_swipe;
}

bool history_set_active_swipe(ChatHistory *history, size_t msg_index,
                              size_t swipe_index) {
  if (msg_index >= history->count)
    return false;

  ChatMessage *msg = &history->messages[msg_index];
  if (swipe_index >= msg->swipe_count)
    return false;

  msg->active_swipe = swipe_index;
  return true;
}

size_t history_get_swipe_count(const ChatHistory *history, size_t msg_index) {
  if (msg_index >= history->count)
    return 0;
  return history->messages[msg_index].swipe_count;
}

size_t history_get_active_swipe(const ChatHistory *history, size_t msg_index) {
  if (msg_index >= history->count)
    return 0;
  return history->messages[msg_index].active_swipe;
}

bool history_delete(ChatHistory *history, size_t index) {
  if (index >= history->count)
    return false;

  ChatMessage *msg = &history->messages[index];
  for (size_t i = 0; i < msg->swipe_count; i++) {
    free(msg->swipes[i]);
  }
  free(msg->swipes);

  for (size_t i = index; i < history->count - 1; i++) {
    history->messages[i] = history->messages[i + 1];
  }
  history->count--;

  return true;
}
