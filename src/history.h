#ifndef HISTORY_H
#define HISTORY_H

#include <stdbool.h>
#include <stddef.h>

#define MAX_SWIPES 16

typedef struct {
  char **swipes;
  size_t swipe_count;
  size_t active_swipe;
} ChatMessage;

typedef struct {
  ChatMessage *messages;
  size_t count;
  size_t capacity;
} ChatHistory;

void history_init(ChatHistory *history);
void history_free(ChatHistory *history);
size_t history_add(ChatHistory *history, const char *message);
void history_update(ChatHistory *history, size_t index, const char *message);

const char *history_get(const ChatHistory *history, size_t index);
const char *history_get_swipe(const ChatHistory *history, size_t msg_index,
                              size_t swipe_index);
void history_update_swipe(ChatHistory *history, size_t msg_index,
                          size_t swipe_index, const char *message);
size_t history_add_swipe(ChatHistory *history, size_t msg_index,
                         const char *content);
bool history_set_active_swipe(ChatHistory *history, size_t msg_index,
                              size_t swipe_index);
size_t history_get_swipe_count(const ChatHistory *history, size_t msg_index);
size_t history_get_active_swipe(const ChatHistory *history, size_t msg_index);
bool history_delete(ChatHistory *history, size_t index);

#endif
