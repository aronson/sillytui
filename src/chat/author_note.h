#ifndef AUTHOR_NOTE_H
#define AUTHOR_NOTE_H

#include <stdbool.h>

#define AN_TEXT_MAX 4096

typedef enum {
  AN_POS_AFTER_SCENARIO = 0,
  AN_POS_IN_CHAT = 1,
  AN_POS_BEFORE_SCENARIO = 2
} AuthorNotePosition;

typedef enum {
  AN_ROLE_SYSTEM = 0,
  AN_ROLE_USER = 1,
  AN_ROLE_ASSISTANT = 2
} AuthorNoteRole;

typedef struct {
  char text[AN_TEXT_MAX];
  AuthorNotePosition position;
  AuthorNoteRole role;
  int depth;
  int interval;
} AuthorNote;

void author_note_init(AuthorNote *note);
void author_note_free(AuthorNote *note);

void author_note_set_text(AuthorNote *note, const char *text);
void author_note_set_depth(AuthorNote *note, int depth);
void author_note_set_interval(AuthorNote *note, int interval);
void author_note_set_position(AuthorNote *note, AuthorNotePosition pos);
void author_note_set_role(AuthorNote *note, AuthorNoteRole role);

const char *author_note_position_to_string(AuthorNotePosition pos);
AuthorNotePosition author_note_position_from_string(const char *str);
const char *author_note_role_to_string(AuthorNoteRole role);
AuthorNoteRole author_note_role_from_string(const char *str);

#endif
