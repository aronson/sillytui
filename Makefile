BUILD_DIR ?= build
SRC_FILES := $(shell find src -name '*.c' -o -name '*.h')
TOKENIZE_SRCS := examples/tokenize.c src/tokenizer/tiktoken.c src/tokenizer/gpt2bpe.c \
	src/tokenizer/sentencepiece.c src/tokenizer/simd.c src/tokenizer/simd_arm64.S \
	src/tokenizer/unicode_tables.c

.PHONY: all configure build run clean distclean format tokenize example

all: build

configure:
	cmake -S $(CURDIR) -B $(BUILD_DIR)

build: configure
	cmake --build $(BUILD_DIR)

run: build
	./$(BUILD_DIR)/sillytui

clean:
	[ ! -d $(BUILD_DIR) ] || cmake --build $(BUILD_DIR) --target clean

distclean:
	rm -rf $(BUILD_DIR)

format:
	clang-format -i $(SRC_FILES)

tokenize: $(BUILD_DIR)/tokenize

$(BUILD_DIR)/tokenize: $(TOKENIZE_SRCS)
	@mkdir -p $(BUILD_DIR)
	$(CC) -O3 -o $@ $(TOKENIZE_SRCS) -Isrc

example: tokenize
	@./$(BUILD_DIR)/tokenize $(ARGS)

