#include "author_note.h"
#include <string.h>

void author_note_init(AuthorNote *note) {
  if (!note)
    return;
  note->text[0] = '\0';
  note->position = AN_POS_IN_CHAT;
  note->role = AN_ROLE_SYSTEM;
  note->depth = 4;
  note->interval = 1;
}

void author_note_free(AuthorNote *note) {
  if (!note)
    return;
  note->text[0] = '\0';
}

void author_note_set_text(AuthorNote *note, const char *text) {
  if (!note)
    return;
  if (!text) {
    note->text[0] = '\0';
    return;
  }
  strncpy(note->text, text, AN_TEXT_MAX - 1);
  note->text[AN_TEXT_MAX - 1] = '\0';
}

void author_note_set_depth(AuthorNote *note, int depth) {
  if (!note)
    return;
  note->depth = depth < 0 ? 0 : depth;
}

void author_note_set_interval(AuthorNote *note, int interval) {
  if (!note)
    return;
  note->interval = interval < 1 ? 1 : interval;
}

void author_note_set_position(AuthorNote *note, AuthorNotePosition pos) {
  if (!note)
    return;
  note->position = pos;
}

void author_note_set_role(AuthorNote *note, AuthorNoteRole role) {
  if (!note)
    return;
  note->role = role;
}

const char *author_note_position_to_string(AuthorNotePosition pos) {
  switch (pos) {
  case AN_POS_AFTER_SCENARIO:
    return "after_scenario";
  case AN_POS_IN_CHAT:
    return "in_chat";
  case AN_POS_BEFORE_SCENARIO:
    return "before_scenario";
  default:
    return "in_chat";
  }
}

AuthorNotePosition author_note_position_from_string(const char *str) {
  if (!str)
    return AN_POS_IN_CHAT;
  if (strcmp(str, "after_scenario") == 0 || strcmp(str, "after") == 0)
    return AN_POS_AFTER_SCENARIO;
  if (strcmp(str, "before_scenario") == 0 || strcmp(str, "before") == 0)
    return AN_POS_BEFORE_SCENARIO;
  return AN_POS_IN_CHAT;
}

const char *author_note_role_to_string(AuthorNoteRole role) {
  switch (role) {
  case AN_ROLE_SYSTEM:
    return "system";
  case AN_ROLE_USER:
    return "user";
  case AN_ROLE_ASSISTANT:
    return "assistant";
  default:
    return "system";
  }
}

AuthorNoteRole author_note_role_from_string(const char *str) {
  if (!str)
    return AN_ROLE_SYSTEM;
  if (strcmp(str, "user") == 0)
    return AN_ROLE_USER;
  if (strcmp(str, "assistant") == 0)
    return AN_ROLE_ASSISTANT;
  return AN_ROLE_SYSTEM;
}
