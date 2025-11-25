BUILD_DIR ?= build
SRC_FILES := $(wildcard src/*.c src/*.h)

.PHONY: all configure build run clean distclean format

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

