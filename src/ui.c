#include "ui.h"
#include "chat.h"
#include "markdown.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

static bool g_ui_colors = false;

#define COLOR_PAIR_BORDER_DIM 11
#define COLOR_PAIR_ACCENT 12
#define COLOR_PAIR_STATUS 13

void ui_init_colors(void) {
  if (!has_colors()) {
    g_ui_colors = false;
    return;
  }
  init_pair(COLOR_PAIR_BORDER, COLOR_CYAN, -1);
  init_pair(COLOR_PAIR_TITLE, COLOR_CYAN, -1);
  init_pair(COLOR_PAIR_PROMPT, COLOR_CYAN, -1);
  init_pair(COLOR_PAIR_USER, COLOR_GREEN, -1);
  init_pair(COLOR_PAIR_BOT, COLOR_MAGENTA, -1);
  init_pair(COLOR_PAIR_HINT, 8, -1);
  init_pair(COLOR_PAIR_LOADING, COLOR_CYAN, -1);
  init_pair(COLOR_PAIR_SUGGEST_ACTIVE, COLOR_BLACK, COLOR_CYAN);
  init_pair(COLOR_PAIR_SUGGEST_DESC, COLOR_CYAN, -1);
  init_pair(COLOR_PAIR_BORDER_DIM, 8, -1);
  init_pair(COLOR_PAIR_ACCENT, COLOR_MAGENTA, -1);
  init_pair(COLOR_PAIR_STATUS, 8, -1);
  g_ui_colors = true;
}

static void draw_rounded_box(WINDOW *win, int color_pair, bool focused) {
  int h, w;
  getmaxyx(win, h, w);
  if (h < 2 || w < 2)
    return;

  int actual_color = focused ? color_pair : COLOR_PAIR_BORDER_DIM;
  if (g_ui_colors)
    wattron(win, COLOR_PAIR(actual_color));

  mvwaddstr(win, 0, 0, "╭");
  for (int x = 1; x < w - 1; x++)
    mvwaddstr(win, 0, x, "─");
  mvwaddstr(win, 0, w - 1, "╮");

  for (int y = 1; y < h - 1; y++) {
    mvwaddstr(win, y, 0, "│");
    mvwaddstr(win, y, w - 1, "│");
  }

  mvwaddstr(win, h - 1, 0, "╰");
  for (int x = 1; x < w - 1; x++)
    mvwaddstr(win, h - 1, x, "─");
  mvwaddstr(win, h - 1, w - 1, "╯");

  if (g_ui_colors)
    wattroff(win, COLOR_PAIR(actual_color));
}

static void draw_title(WINDOW *win, const char *title, int color_pair) {
  int w = getmaxx(win);
  int title_len = (int)strlen(title);
  int pos = (w - title_len - 2) / 2;
  if (pos < 2)
    pos = 2;

  if (g_ui_colors)
    wattron(win, COLOR_PAIR(color_pair) | A_BOLD);
  mvwprintw(win, 0, pos, " %s ", title);
  if (g_ui_colors)
    wattroff(win, COLOR_PAIR(color_pair) | A_BOLD);
}

void ui_layout_windows(WINDOW **chat_win, WINDOW **input_win) {
  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  int input_height = 3;
  if (rows < 6)
    input_height = 1;
  int chat_height = rows - input_height;
  if (chat_height < 3)
    chat_height = 3;

  if (*chat_win)
    delwin(*chat_win);
  if (*input_win)
    delwin(*input_win);

  *chat_win = newwin(chat_height, cols, 0, 0);
  *input_win = newwin(input_height, cols, chat_height, 0);
  keypad(*input_win, TRUE);
}

static bool starts_with(const char *str, const char *prefix) {
  return strncmp(str, prefix, strlen(prefix)) == 0;
}

typedef struct {
  const char *start;
  int len;
} WrappedLine;

static int wrap_text(const char *text, int width, WrappedLine *out,
                     int max_lines) {
  if (!text || width <= 0 || !out || max_lines <= 0)
    return 0;

  int line_count = 0;
  const char *p = text;

  while (*p && line_count < max_lines) {
    while (*p == ' ')
      p++;
    if (!*p)
      break;

    const char *line_start = p;
    const char *last_break = NULL;
    int col = 0;

    while (*p && *p != '\n' && col < width) {
      if (*p == ' ')
        last_break = p;
      p++;
      col++;
    }

    int line_len;
    if (!*p || *p == '\n') {
      line_len = (int)(p - line_start);
      if (*p == '\n')
        p++;
    } else if (last_break && last_break > line_start) {
      line_len = (int)(last_break - line_start);
      p = last_break + 1;
    } else {
      line_len = col;
    }

    out[line_count].start = line_start;
    out[line_count].len = line_len;
    line_count++;
  }

  if (line_count == 0 && text[0] != '\0') {
    out[0].start = text;
    out[0].len = 0;
    line_count = 1;
  }

  return line_count;
}

static int count_wrapped_lines(const char *text, int width) {
  if (!text || width <= 0)
    return 1;
  if (text[0] == '\0')
    return 1;

  int line_count = 0;
  const char *p = text;

  while (*p) {
    while (*p == ' ')
      p++;
    if (!*p)
      break;

    const char *line_start = p;
    const char *last_break = NULL;
    int col = 0;

    while (*p && *p != '\n' && col < width) {
      if (*p == ' ')
        last_break = p;
      p++;
      col++;
    }

    if (!*p || *p == '\n') {
      if (*p == '\n')
        p++;
    } else if (last_break && last_break > line_start) {
      p = last_break + 1;
    }

    line_count++;
  }

  return line_count > 0 ? line_count : 1;
}

typedef struct {
  int msg_index;
  int line_in_msg;
  const char *line_start;
  int line_len;
  bool is_user;
  bool is_bot;
  bool is_first_line;
  bool is_spacer;
  bool is_name_line;
  unsigned initial_style;
} DisplayLine;

static int build_display_lines(const ChatHistory *history, int content_width,
                               DisplayLine *lines, int max_lines) {
  int total = 0;
  int gutter_width = 3;
  int text_width = content_width - gutter_width;
  if (text_width < 10)
    text_width = 10;

  WrappedLine *wrapped = malloc(sizeof(WrappedLine) * 200);
  if (!wrapped)
    return 0;

  for (size_t i = 0; i < history->count && total < max_lines; i++) {
    const char *msg = history->items[i];
    const char *content = msg;
    bool is_user = starts_with(msg, "You: ");
    bool is_bot = starts_with(msg, "Bot:");

    if (is_user) {
      content = msg + 5;
    } else if (is_bot) {
      content = msg + 4;
      while (*content == ' ')
        content++;
    }

    if ((is_user || is_bot) && total < max_lines) {
      lines[total].msg_index = (int)i;
      lines[total].line_in_msg = -1;
      lines[total].line_start = NULL;
      lines[total].line_len = 0;
      lines[total].is_user = is_user;
      lines[total].is_bot = is_bot;
      lines[total].is_first_line = true;
      lines[total].is_spacer = false;
      lines[total].is_name_line = true;
      lines[total].initial_style = 0;
      total++;
    }

    int num_wrapped = wrap_text(content, text_width, wrapped, 200);

    unsigned running_style = 0;
    for (int line = 0; line < num_wrapped && total < max_lines; line++) {
      lines[total].msg_index = (int)i;
      lines[total].line_in_msg = line;
      lines[total].line_start = wrapped[line].start;
      lines[total].line_len = wrapped[line].len;
      lines[total].is_user = is_user;
      lines[total].is_bot = is_bot;
      lines[total].is_first_line = false;
      lines[total].is_spacer = false;
      lines[total].is_name_line = false;
      lines[total].initial_style = running_style;

      running_style = markdown_compute_style_after(
          wrapped[line].start, wrapped[line].len, running_style);

      total++;
    }

    if (i < history->count - 1 && total < max_lines) {
      lines[total].msg_index = (int)i;
      lines[total].line_in_msg = -1;
      lines[total].line_start = NULL;
      lines[total].line_len = 0;
      lines[total].is_user = false;
      lines[total].is_bot = false;
      lines[total].is_first_line = false;
      lines[total].is_spacer = true;
      lines[total].is_name_line = false;
      lines[total].initial_style = 0;
      total++;
    }
  }

  free(wrapped);
  return total;
}

int ui_get_total_lines(WINDOW *chat_win, const ChatHistory *history) {
  int height, width;
  getmaxyx(chat_win, height, width);
  (void)height;

  int content_width = width - 4;
  if (content_width <= 0)
    return 0;

  int gutter_width = 3;
  int text_width = content_width - gutter_width;
  if (text_width < 10)
    text_width = 10;

  int total = 0;
  for (size_t i = 0; i < history->count; i++) {
    const char *msg = history->items[i];
    const char *content = msg;
    bool is_user = starts_with(msg, "You: ");
    bool is_bot = starts_with(msg, "Bot:");

    if (is_user) {
      content = msg + 5;
    } else if (is_bot) {
      content = msg + 4;
      while (*content == ' ')
        content++;
    }

    if (is_user || is_bot)
      total++;
    total += count_wrapped_lines(content, text_width);
    if (i < history->count - 1)
      total++;
  }
  return total;
}

void ui_draw_chat(WINDOW *chat_win, const ChatHistory *history,
                  int scroll_offset, const char *model_name,
                  const char *user_name, const char *bot_name) {
  if (!user_name || !user_name[0])
    user_name = "You";
  if (!bot_name || !bot_name[0])
    bot_name = "Bot";

  werase(chat_win);
  draw_rounded_box(chat_win, COLOR_PAIR_BORDER, true);

  int height, width;
  getmaxyx(chat_win, height, width);

  char title[128];
  if (model_name && model_name[0]) {
    snprintf(title, sizeof(title), "✦ %s", model_name);
  } else {
    snprintf(title, sizeof(title), "✦ sillytui");
  }
  draw_title(chat_win, title, COLOR_PAIR_TITLE);

  int usable_lines = height - 2;
  int content_width = width - 4;
  if (usable_lines <= 0 || content_width <= 0) {
    wrefresh(chat_win);
    return;
  }

  int max_display = (int)history->count * 100 + 100;
  DisplayLine *all_lines = malloc(sizeof(DisplayLine) * max_display);
  if (!all_lines) {
    wrefresh(chat_win);
    return;
  }

  int total_display_lines =
      build_display_lines(history, content_width, all_lines, max_display);

  int max_scroll = total_display_lines - usable_lines;
  if (max_scroll < 0)
    max_scroll = 0;

  int start_line;
  if (scroll_offset < 0) {
    start_line = max_scroll;
  } else {
    start_line = scroll_offset;
    if (start_line > max_scroll)
      start_line = max_scroll;
  }

  int gutter_width = 3;
  int text_width = content_width - gutter_width;
  if (text_width < 10)
    text_width = 10;

  for (int row = 0;
       row < usable_lines && (start_line + row) < total_display_lines; row++) {
    DisplayLine *dl = &all_lines[start_line + row];
    int y = row + 1;
    int x = 2;

    if (dl->is_spacer)
      continue;

    if (dl->is_name_line) {
      if (dl->is_user) {
        if (g_ui_colors)
          wattron(chat_win, COLOR_PAIR(COLOR_PAIR_USER) | A_BOLD);
        mvwaddstr(chat_win, y, x, "▸ ");
        mvwaddstr(chat_win, y, x + 2, user_name);
        if (g_ui_colors)
          wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_USER) | A_BOLD);
      } else if (dl->is_bot) {
        if (g_ui_colors)
          wattron(chat_win, COLOR_PAIR(COLOR_PAIR_BOT) | A_BOLD);
        mvwaddstr(chat_win, y, x, "◆ ");
        mvwaddstr(chat_win, y, x + 2, bot_name);
        if (g_ui_colors)
          wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_BOT) | A_BOLD);
      }
      continue;
    }

    if (dl->is_user) {
      if (g_ui_colors)
        wattron(chat_win, COLOR_PAIR(COLOR_PAIR_USER) | A_DIM);
      mvwaddstr(chat_win, y, x, "│");
      if (g_ui_colors)
        wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_USER) | A_DIM);
    } else if (dl->is_bot) {
      if (g_ui_colors)
        wattron(chat_win, COLOR_PAIR(COLOR_PAIR_BOT) | A_DIM);
      mvwaddstr(chat_win, y, x, "│");
      if (g_ui_colors)
        wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_BOT) | A_DIM);
    }

    char line_buf[512];
    int to_copy = dl->line_len;
    if (to_copy > (int)sizeof(line_buf) - 1)
      to_copy = (int)sizeof(line_buf) - 1;
    strncpy(line_buf, dl->line_start, to_copy);
    line_buf[to_copy] = '\0';

    markdown_render_line_styled(chat_win, y, x + gutter_width, text_width,
                                line_buf, dl->initial_style);
  }

  if (total_display_lines > usable_lines) {
    int scrollbar_height = usable_lines;
    if (scrollbar_height > 2) {
      int thumb_size = (usable_lines * scrollbar_height) / total_display_lines;
      if (thumb_size < 1)
        thumb_size = 1;
      if (thumb_size > scrollbar_height - 2)
        thumb_size = scrollbar_height - 2;
      int thumb_pos =
          (max_scroll > 0)
              ? 1 + (start_line * (scrollbar_height - thumb_size - 2)) /
                        max_scroll
              : 1;

      for (int i = 1; i < scrollbar_height - 1; i++) {
        if (i >= thumb_pos && i < thumb_pos + thumb_size) {
          if (g_ui_colors)
            wattron(chat_win, COLOR_PAIR(COLOR_PAIR_BORDER));
          mvwaddstr(chat_win, i, width - 2, "█");
          if (g_ui_colors)
            wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_BORDER));
        } else {
          if (g_ui_colors)
            wattron(chat_win, COLOR_PAIR(COLOR_PAIR_BORDER_DIM));
          mvwaddstr(chat_win, i, width - 2, "░");
          if (g_ui_colors)
            wattroff(chat_win, COLOR_PAIR(COLOR_PAIR_BORDER_DIM));
        }
      }
    }
  }

  free(all_lines);
  wrefresh(chat_win);
}

void ui_draw_input(WINDOW *input_win, const char *buffer, int cursor_pos) {
  werase(input_win);
  draw_rounded_box(input_win, COLOR_PAIR_BORDER, true);

  int h, w;
  getmaxyx(input_win, h, w);
  (void)h;
  int buf_len = (int)strlen(buffer);

  if (g_ui_colors)
    wattron(input_win, COLOR_PAIR(COLOR_PAIR_PROMPT) | A_BOLD);
  mvwaddstr(input_win, 1, 2, "›");
  if (g_ui_colors)
    wattroff(input_win, COLOR_PAIR(COLOR_PAIR_PROMPT) | A_BOLD);

  mvwprintw(input_win, 1, 4, "%s", buffer);

  int screen_cursor = 4 + cursor_pos;
  wattron(input_win, A_REVERSE);
  char under_cursor = (cursor_pos < buf_len) ? buffer[cursor_pos] : ' ';
  mvwaddch(input_win, 1, screen_cursor, under_cursor);
  wattroff(input_win, A_REVERSE);

  if (buffer[0] == '\0' && g_ui_colors) {
    wattron(input_win, COLOR_PAIR(COLOR_PAIR_HINT) | A_DIM);
    mvwaddstr(input_win, 1, 5, "Type a message or / for commands...");
    wattroff(input_win, COLOR_PAIR(COLOR_PAIR_HINT) | A_DIM);
  }

  if (g_ui_colors)
    wattron(input_win, COLOR_PAIR(COLOR_PAIR_HINT) | A_DIM);
  mvwaddstr(input_win, 1, w - 14, "↑↓ scroll");
  if (g_ui_colors)
    wattroff(input_win, COLOR_PAIR(COLOR_PAIR_HINT) | A_DIM);

  wrefresh(input_win);
}

#define MAX_SUGGESTIONS 10
#define SUGGESTION_BOX_WIDTH 60

static bool fuzzy_match(const char *pattern, const char *str) {
  if (!pattern || !*pattern)
    return true;
  const char *p = pattern;
  const char *s = str;
  while (*p && *s) {
    if (tolower((unsigned char)*p) == tolower((unsigned char)*s))
      p++;
    s++;
  }
  return *p == '\0';
}

void suggestion_box_init(SuggestionBox *sb, const SlashCommand *commands,
                         int count) {
  memset(sb, 0, sizeof(*sb));
  sb->commands = commands;
  sb->command_count = count;
  sb->matched_indices = malloc(sizeof(int) * count);
  sb->matched_count = 0;
  sb->active_index = 0;
  sb->dynamic_items = malloc(sizeof(DynamicItem) * MAX_DYNAMIC_ITEMS);
  sb->dynamic_count = 0;
  sb->showing_dynamic = false;
}

void suggestion_box_free(SuggestionBox *sb) {
  if (sb->matched_indices) {
    free(sb->matched_indices);
    sb->matched_indices = NULL;
  }
  if (sb->dynamic_items) {
    free(sb->dynamic_items);
    sb->dynamic_items = NULL;
  }
  suggestion_box_close(sb);
}

static bool command_matches_filter(const char *cmd_name, const char *query) {
  size_t query_len = strlen(query);
  size_t cmd_len = strlen(cmd_name);

  if (query_len == 0)
    return true;

  if (strncmp(cmd_name, query, query_len) == 0)
    return true;

  if (query_len <= cmd_len && fuzzy_match(query, cmd_name))
    return true;

  return false;
}

static void load_chat_suggestions(SuggestionBox *sb, const char *filter_text) {
  sb->dynamic_count = 0;

  ChatList list;
  chat_list_init(&list);
  if (!chat_list_load(&list)) {
    chat_list_free(&list);
    return;
  }

  for (size_t i = 0; i < list.count && sb->dynamic_count < MAX_DYNAMIC_ITEMS;
       i++) {
    bool matches = true;
    if (filter_text && filter_text[0]) {
      matches = (strstr(list.chats[i].title, filter_text) != NULL ||
                 strstr(list.chats[i].id, filter_text) != NULL);
    }
    if (matches) {
      snprintf(sb->dynamic_items[sb->dynamic_count].name, 128, "%s",
               list.chats[i].title);
      snprintf(sb->dynamic_items[sb->dynamic_count].description, 128,
               "%zu msgs", list.chats[i].message_count);
      sb->dynamic_count++;
    }
  }

  chat_list_free(&list);
}

void suggestion_box_update(SuggestionBox *sb, const char *filter,
                           WINDOW *parent, int parent_y, int parent_x) {
  if (!filter || filter[0] != '/') {
    suggestion_box_close(sb);
    return;
  }

  const char *query = filter + 1;
  size_t query_len = strlen(query);

  const char *chat_load_prefix = "chat load ";
  size_t prefix_len = strlen(chat_load_prefix);

  if (strncmp(query, chat_load_prefix, prefix_len) == 0) {
    sb->showing_dynamic = true;
    const char *chat_filter = query + prefix_len;
    load_chat_suggestions(sb, chat_filter);

    if (sb->dynamic_count == 0) {
      suggestion_box_close(sb);
      return;
    }

    sb->matched_count = sb->dynamic_count;
    if (sb->active_index >= sb->matched_count)
      sb->active_index = sb->matched_count - 1;
  } else {
    sb->showing_dynamic = false;

    bool has_trailing_space = query_len > 0 && query[query_len - 1] == ' ';
    bool exact_match_exists = false;

    for (int i = 0; i < sb->command_count; i++) {
      if (strcmp(sb->commands[i].name, query) == 0) {
        exact_match_exists = true;
        break;
      }
    }

    if (exact_match_exists && has_trailing_space &&
        strcmp(query, "chat load ") != 0) {
      suggestion_box_close(sb);
      return;
    }

    sb->matched_count = 0;
    for (int i = 0; i < sb->command_count; i++) {
      if (command_matches_filter(sb->commands[i].name, query))
        sb->matched_indices[sb->matched_count++] = i;
    }

    if (sb->matched_count == 0) {
      suggestion_box_close(sb);
      return;
    }

    if (sb->active_index >= sb->matched_count)
      sb->active_index = sb->matched_count - 1;
  }

  int box_height = sb->matched_count + 2;
  if (box_height > MAX_SUGGESTIONS + 2)
    box_height = MAX_SUGGESTIONS + 2;
  sb->visible_count = box_height - 2;

  int max_y, max_x;
  getmaxyx(stdscr, max_y, max_x);
  (void)max_y;

  int box_width = SUGGESTION_BOX_WIDTH;
  if (box_width > max_x - 4)
    box_width = max_x - 4;

  int box_y = parent_y - box_height;
  int box_x = parent_x + 2;
  if (box_y < 0)
    box_y = 0;
  if (box_x + box_width > max_x)
    box_x = max_x - box_width;

  if (sb->win) {
    int old_h, old_w, old_y, old_x;
    getmaxyx(sb->win, old_h, old_w);
    getbegyx(sb->win, old_y, old_x);
    if (old_h != box_height || old_w != box_width || old_y != box_y ||
        old_x != box_x) {
      wclear(sb->win);
      wrefresh(sb->win);
      delwin(sb->win);
      sb->win = newwin(box_height, box_width, box_y, box_x);
    }
  } else {
    sb->win = newwin(box_height, box_width, box_y, box_x);
  }

  sb->filter = filter;
}

void suggestion_box_draw(SuggestionBox *sb) {
  if (!sb->win || sb->matched_count == 0)
    return;

  werase(sb->win);

  int h, w;
  getmaxyx(sb->win, h, w);

  if (g_ui_colors)
    wattron(sb->win, COLOR_PAIR(COLOR_PAIR_BORDER));
  mvwaddstr(sb->win, 0, 0, "╭");
  for (int x = 1; x < w - 1; x++)
    mvwaddstr(sb->win, 0, x, "─");
  mvwaddstr(sb->win, 0, w - 1, "╮");
  for (int y = 1; y < h - 1; y++) {
    mvwaddstr(sb->win, y, 0, "│");
    mvwaddstr(sb->win, y, w - 1, "│");
  }
  mvwaddstr(sb->win, h - 1, 0, "╰");
  for (int x = 1; x < w - 1; x++)
    mvwaddstr(sb->win, h - 1, x, "─");
  mvwaddstr(sb->win, h - 1, w - 1, "╯");
  if (g_ui_colors)
    wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_BORDER));

  if (sb->scroll_offset > 0) {
    if (g_ui_colors)
      wattron(sb->win, COLOR_PAIR(COLOR_PAIR_HINT));
    mvwaddstr(sb->win, 0, w / 2, "▲");
    if (g_ui_colors)
      wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_HINT));
  }
  if (sb->scroll_offset + sb->visible_count < sb->matched_count) {
    if (g_ui_colors)
      wattron(sb->win, COLOR_PAIR(COLOR_PAIR_HINT));
    mvwaddstr(sb->win, h - 1, w / 2, "▼");
    if (g_ui_colors)
      wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_HINT));
  }

  int name_col_width = 18;

  for (int i = 0;
       i < sb->visible_count && (sb->scroll_offset + i) < sb->matched_count;
       i++) {
    int y = i + 1;
    bool is_active = (sb->scroll_offset + i) == sb->active_index;
    const char *name = NULL;
    const char *desc = NULL;

    if (sb->showing_dynamic) {
      int idx = sb->scroll_offset + i;
      if (idx < sb->dynamic_count) {
        name = sb->dynamic_items[idx].name;
        desc = sb->dynamic_items[idx].description;
      }
    } else {
      int cmd_idx = sb->matched_indices[sb->scroll_offset + i];
      name = sb->commands[cmd_idx].name;
      desc = sb->commands[cmd_idx].description;
    }

    if (!name)
      continue;

    if (is_active) {
      wattron(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));
      mvwhline(sb->win, y, 1, ' ', w - 2);
    }

    if (sb->showing_dynamic) {
      int name_max = w - 4;
      if (name_max > 40)
        name_max = 40;
      mvwprintw(sb->win, y, 2, "%-*.*s", name_max, name_max, name);

      if (is_active)
        wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));

      if (desc) {
        int desc_x = 2 + name_max + 2;
        int desc_max = w - desc_x - 2;
        if (desc_max > 0) {
          if (is_active) {
            wattron(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));
          } else if (g_ui_colors) {
            wattron(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_DESC));
          }
          mvwprintw(sb->win, y, desc_x, "%.*s", desc_max, desc);
          if (is_active) {
            wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));
          } else if (g_ui_colors) {
            wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_DESC));
          }
        }
      }
    } else {
      mvwprintw(sb->win, y, 2, "/%-*s", name_col_width - 1, name);

      if (is_active)
        wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));

      if (desc) {
        int desc_x = 2 + name_col_width + 1;
        int desc_max = w - desc_x - 2;
        if (desc_max > 0) {
          if (is_active) {
            wattron(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));
          } else if (g_ui_colors) {
            wattron(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_DESC));
          }
          mvwprintw(sb->win, y, desc_x, "%.*s", desc_max, desc);
          if (is_active) {
            wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_ACTIVE));
          } else if (g_ui_colors) {
            wattroff(sb->win, COLOR_PAIR(COLOR_PAIR_SUGGEST_DESC));
          }
        }
      }
    }
  }

  wrefresh(sb->win);
}

void suggestion_box_navigate(SuggestionBox *sb, int direction) {
  if (sb->matched_count == 0)
    return;

  sb->active_index += direction;
  if (sb->active_index < 0) {
    sb->active_index = sb->matched_count - 1;
  } else if (sb->active_index >= sb->matched_count) {
    sb->active_index = 0;
  }

  if (sb->active_index < sb->scroll_offset) {
    sb->scroll_offset = sb->active_index;
  } else if (sb->active_index >= sb->scroll_offset + sb->visible_count) {
    sb->scroll_offset = sb->active_index - sb->visible_count + 1;
  }
}

const char *suggestion_box_get_selected(SuggestionBox *sb) {
  if (sb->matched_count == 0 || sb->active_index < 0)
    return NULL;
  if (sb->showing_dynamic) {
    if (sb->active_index < sb->dynamic_count)
      return sb->dynamic_items[sb->active_index].name;
    return NULL;
  }
  int cmd_idx = sb->matched_indices[sb->active_index];
  return sb->commands[cmd_idx].name;
}

void suggestion_box_close(SuggestionBox *sb) {
  if (sb->win) {
    werase(sb->win);
    wrefresh(sb->win);
    delwin(sb->win);
    sb->win = NULL;
  }
  sb->matched_count = 0;
  sb->active_index = 0;
  sb->scroll_offset = 0;
}

bool suggestion_box_is_open(const SuggestionBox *sb) {
  return sb->win != NULL && sb->matched_count > 0;
}
