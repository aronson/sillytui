project('sillytui', ['c', 'cpp'],
  version : '0.1',
  default_options : [
    'c_std=c11',
    'cpp_std=c++23',
    'warning_level=2',
  ]
)

cc = meson.get_compiler('c')
cpp = meson.get_compiler('cpp')

# Compiler flags
c_args = []
cpp_args = []

if cc.get_id() == 'gcc'
  c_args += '-Wno-stringop-truncation'
endif

# Check for GCC version for C++23 support
if cpp.get_id() == 'gcc'
  if cpp.version().version_compare('<13.0')
    error('GCC version 13.0 or later is required for C++23 support.')
  endif
endif

# System-specific flags
if host_machine.system() == 'linux'
  add_project_arguments('-D_XOPEN_SOURCE=500', '-D_GNU_SOURCE', language : 'c')
endif

if host_machine.cpu_family() == 'x86_64'
  # Note: meson doesn't have a direct equivalent to -march=x86-64-v3 detection in the same way,
  # but we can add the flag. Ideally, this should be a user option.
  # For this port, we'll verify if the compiler supports it or add it unconditionally if strictly following CMake.
  # However, adding architecture flags unconditionally can be dangerous for portability.
  # We will stick to the CMake logic: if x86_64, add it.
  add_project_arguments('-march=x86-64-v3', language : ['c', 'cpp'])
endif

# Dependencies
use_static = get_option('static_deps')

curses_dep = dependency('ncursesw', required : false, static : use_static)
if not curses_dep.found()
  curses_dep = dependency('ncurses', required : true, static : use_static)
endif
curl_dep = dependency('libcurl', required : true, static : use_static)
thread_dep = dependency('threads')
m_dep = cc.find_library('m', required : false, static : use_static)

deps = [curses_dep, curl_dep, thread_dep, m_dep]

if host_machine.system() == 'darwin'
  accelerate_dep = dependency('appleframeworks', modules : 'Accelerate')
  deps += accelerate_dep
endif

# Common definitions
add_project_arguments('-D_XOPEN_SOURCE_EXTENDED', '-DNCURSES_WIDECHAR=1', language : 'c')

# Include directories
inc_dirs = include_directories('src', 'src/ui/syntax/ulight/include', 'src/inference/model/qwen3')

# ULight Sources (C++)
ulight_sources = files(
  'src/ui/syntax/ulight/chars.cpp',
  'src/ui/syntax/ulight/io.cpp',
  'src/ui/syntax/ulight/parse_utils.cpp',
  'src/ui/syntax/ulight/ulight.cpp',
  'src/ui/syntax/ulight/lang/bash.cpp',
  'src/ui/syntax/ulight/lang/cowel.cpp',
  'src/ui/syntax/ulight/lang/cpp.cpp',
  'src/ui/syntax/ulight/lang/css.cpp',
  'src/ui/syntax/ulight/lang/diff.cpp',
  'src/ui/syntax/ulight/lang/ebnf.cpp',
  'src/ui/syntax/ulight/lang/html.cpp',
  'src/ui/syntax/ulight/lang/js.cpp',
  'src/ui/syntax/ulight/lang/json.cpp',
  'src/ui/syntax/ulight/lang/kotlin.cpp',
  'src/ui/syntax/ulight/lang/llvm.cpp',
  'src/ui/syntax/ulight/lang/lua.cpp',
  'src/ui/syntax/ulight/lang/nasm.cpp',
  'src/ui/syntax/ulight/lang/python.cpp',
  'src/ui/syntax/ulight/lang/rust.cpp',
  'src/ui/syntax/ulight/lang/tex.cpp',
  'src/ui/syntax/ulight/lang/xml.cpp',
)

# Specific flags for ulight sources
# Meson doesn't support per-file flags easily without a separate static_library or declaring them one by one.
# We will create a static library for ulight to apply flags.
ulight_lib = static_library('ulight',
  ulight_sources,
  include_directories : inc_dirs,
  cpp_args : ['-Wno-trigraphs', '-Wno-error'],
)

# Handle weights.c being C++
# We copy it to .cpp so Meson treats it as C++
fs = import('fs')
weights_cpp = configure_file(
  input : 'src/inference/model/qwen3/weights.c',
  output : 'weights.cpp', # We keep it flat but we need to fix include paths
  copy : true
)

# Main Sources
app_sources = files(
  'src/main.c',
  'src/core/config.c',
  'src/core/macros.c',
  'src/core/time.c',
  'src/core/error.c',
  'src/core/log.c',
  'src/ui/ui.c',
  'src/ui/modal.c',
  'src/ui/markdown.c',
  'src/ui/console.c',
  'src/ui/syntax/syntax_ncurses.c',
  'src/ui/syntax/c/highlight.c',
  'src/ui/syntax/c/hashtable.c',
  'src/chat/chat.c',
  'src/chat/history.c',
  'src/chat/author_note.c',
  'src/llm/llm.c',
  'src/llm/common.c',
  'src/llm/sampler.c',
  'src/llm/backends/openai.c',
  'src/llm/backends/anthropic.c',
  'src/llm/backends/kobold.c',
  'src/character/character.c',
  'src/character/persona.c',
  'src/lore/lorebook.c',
  'src/inference/tokenizer/tiktoken.c',
  'src/inference/tokenizer/sentencepiece.c',
  'src/inference/tokenizer/gpt2bpe.c',
  'src/inference/tokenizer/selector.c',
  'src/inference/tokenizer/simd.c',
  'src/inference/tokenizer/unicode_tables.c',
  'src/inference/kernels/gemm/gemm.c',
  'src/inference/kernels/gemm/gemm_neon.c',
  'src/inference/kernels/gemm/gemm_amx.c',
  'src/inference/kernels/norm/layernorm.c',
  'src/inference/kernels/norm/layernorm_neon.c',
  'src/inference/kernels/activation/activation.c',
  'src/inference/kernels/activation/activation_neon.c',
  'src/inference/kernels/rope/rope.c',
  'src/inference/kernels/rope/rope_neon.c',
  'src/inference/kernels/softmax/softmax.c',
  'src/inference/kernels/softmax/softmax_neon.c',
  'src/inference/kernels/attention/attention.c',
  'src/inference/kernels/attention/attention_neon.c',
  'src/inference/kernels/embedding/embedding.c',
  'src/inference/kernels/embedding/embedding_neon.c',
  'src/inference/kernels/sampling/sampling.c',
  'src/inference/kernels/sampling/sampling_neon.c',
  'src/inference/kernels/kv_cache/kv_cache.c',
  'src/inference/kernels/kv_cache/kv_cache_neon.c',
  'src/inference/model/base.c',
  'src/inference/model/qwen3/config.c',
  # weights.c is replaced by weights_cpp
  'src/inference/model/qwen3/ffn.c',
  'src/inference/model/qwen3/attention_layer.c',
  'src/inference/model/qwen3/transformer_layer.c',
  'src/inference/model/qwen3/qwen3.c',
)

if host_machine.cpu_family() == 'aarch64'
  app_sources += files('src/inference/tokenizer/simd_arm64.S')
elif host_machine.cpu_family() == 'x86_64'
  app_sources += files('src/inference/tokenizer/simd_x86_64.S')
endif

# Executable
executable('sillytui',
  app_sources + [weights_cpp],
  include_directories : inc_dirs,
  dependencies : deps,
  link_with : ulight_lib,
  c_args : c_args,
  cpp_args : cpp_args,
  install : true,
)

# Tests
test_sources = files(
    'tests/run_tests.c',
    'tests/test_history.c',
    'tests/test_macros.c',
    'tests/test_config.c',
    'tests/test_sampler.c',
    'tests/test_simd.c',
    'tests/test_tokenizer.c',
    'tests/test_modal.c',
    'tests/test_attachments.c',
    'tests/safetensors_impl.cc',
    'tests/test_safetensors.cc',
    'tests/kernels/test_gemm.cc',
    'tests/kernels/test_gemm_pytorch_accuracy.cc',
    'tests/kernels/test_layernorm.cc',
    'tests/kernels/test_activation.cc',
    'tests/kernels/test_activation_pytorch_accuracy.cc',
    'tests/kernels/test_rope.cc',
    'tests/kernels/test_rope_pytorch_accuracy.cc',
    'tests/kernels/test_softmax.cc',
    'tests/kernels/test_softmax_pytorch_accuracy.cc',
    'tests/kernels/test_attention.cc',
    'tests/kernels/test_embedding.cc',
    'tests/kernels/test_embedding_pytorch_accuracy.cc',
    'tests/kernels/test_sampling.cc',
    'tests/kernels/test_sampling_pytorch_accuracy.cc',
    'tests/kernels/test_kv_cache.cc',
    'tests/kernels/test_kv_cache_pytorch_accuracy.cc',
    'src/core/config.c',
    'src/core/macros.c',
    'src/core/time.c',
    'src/core/error.c',
    'src/core/log.c',
    'src/chat/history.c',
    'src/chat/author_note.c',
    'src/llm/sampler.c',
    'src/llm/common.c',
    'src/llm/llm.c',
    'src/llm/backends/openai.c',
    'src/llm/backends/anthropic.c',
    'src/llm/backends/kobold.c',
    'src/inference/tokenizer/tiktoken.c',
    'src/inference/tokenizer/simd.c',
    'src/inference/tokenizer/unicode_tables.c',
    'src/inference/tokenizer/selector.c',
    'src/inference/tokenizer/gpt2bpe.c',
    'src/inference/tokenizer/sentencepiece.c',
    'src/inference/kernels/gemm/gemm.c',
    'src/inference/kernels/gemm/gemm_neon.c',
    'src/inference/kernels/gemm/gemm_amx.c',
    'src/inference/kernels/norm/layernorm.c',
    'src/inference/kernels/norm/layernorm_neon.c',
    'src/inference/kernels/activation/activation.c',
    'src/inference/kernels/activation/activation_neon.c',
    'src/inference/kernels/rope/rope.c',
    'src/inference/kernels/rope/rope_neon.c',
    'src/inference/kernels/softmax/softmax.c',
    'src/inference/kernels/softmax/softmax_neon.c',
    'src/inference/kernels/attention/attention.c',
    'src/inference/kernels/attention/attention_neon.c',
    'src/inference/kernels/embedding/embedding.c',
    'src/inference/kernels/embedding/embedding_neon.c',
    'src/inference/kernels/sampling/sampling.c',
    'src/inference/kernels/sampling/sampling_neon.c',
    'src/inference/kernels/kv_cache/kv_cache.c',
    'src/inference/kernels/kv_cache/kv_cache_neon.c',
    'src/ui/modal.c',
    'src/ui/ui.c',
    'src/ui/markdown.c',
    'src/ui/console.c',
    'src/ui/syntax/syntax_ncurses.c',
    'src/ui/syntax/c/highlight.c',
    'src/ui/syntax/c/hashtable.c',
    'src/chat/chat.c',
    'src/character/character.c',
    'src/character/persona.c',
    'src/lore/lorebook.c',
)

if host_machine.cpu_family() == 'aarch64'
  test_sources += files('src/inference/tokenizer/simd_arm64.S')
elif host_machine.cpu_family() == 'x86_64'
  test_sources += files('src/inference/tokenizer/simd_x86_64.S')
endif

# Test executable
test_exe = executable('run_tests',
  test_sources,
  include_directories : [inc_dirs, include_directories('tests')],
  dependencies : deps,
  link_with : ulight_lib,
  cpp_args : ['-Wno-error', '-Wno-ignored-qualifiers', '-Wno-unused-parameter', '-Wno-unused-variable']
)
test('unit_tests', test_exe, workdir : meson.current_source_dir())

# Integration Tests
integration_test_sources = files(
    'tests/integration/run_integration_tests.c',
    'tests/integration/test_chat_integration.c',
    'tests/integration/test_config_integration.c',
    'tests/integration/test_persona_integration.c',
    'tests/integration/test_tokenizer_integration.c',
    'src/core/config.c',
    'src/core/macros.c',
    'src/core/time.c',
    'src/core/error.c',
    'src/core/log.c',
    'src/chat/chat.c',
    'src/chat/history.c',
    'src/chat/author_note.c',
    'src/llm/sampler.c',
    'src/character/character.c',
    'src/character/persona.c',
    'src/inference/tokenizer/tiktoken.c',
    'src/inference/tokenizer/gpt2bpe.c',
    'src/inference/tokenizer/sentencepiece.c',
    'src/inference/tokenizer/simd.c',
    'src/inference/tokenizer/unicode_tables.c',
    'src/inference/tokenizer/selector.c',
    'src/ui/modal.c',
    'src/ui/ui.c',
    'src/ui/markdown.c',
    'src/ui/console.c',
    'src/ui/syntax/syntax_ncurses.c',
    'src/ui/syntax/c/highlight.c',
    'src/ui/syntax/c/hashtable.c',
)

if host_machine.cpu_family() == 'aarch64'
  integration_test_sources += files('src/inference/tokenizer/simd_arm64.S')
elif host_machine.cpu_family() == 'x86_64'
  integration_test_sources += files('src/inference/tokenizer/simd_x86_64.S')
endif

integration_test_exe = executable('run_integration_tests',
  integration_test_sources,
  include_directories : [inc_dirs, include_directories('tests', 'tests/integration')],
  dependencies : deps,
  link_with : ulight_lib,
)
test('integration_tests', integration_test_exe, workdir : meson.current_source_dir())

# Generators
gen_unicode = executable('gen_unicode_tests', 'tests/generators/gen_unicode_tests.c')
gen_utf8 = executable('gen_utf8_tests', 'tests/generators/gen_utf8_tests.c')
gen_pretokenize = executable('gen_pretokenize_tests', 'tests/generators/gen_pretokenize_tests.c')

# We can use custom_target to run these and generate files, but for now we assume they are run manually or we can add a run target.
# Meson prefers generating files at build time if they are inputs. 
# CMake had a custom target 'generate_tests' that runs them.

custom_target('generate_unicode_tests',
  output : 'test_unicode_gen.c',
  command : [gen_unicode],
  capture : true,
  build_by_default : true,
)
custom_target('generate_utf8_tests',
  output : 'test_utf8_gen.c',
  command : [gen_utf8],
  capture : true,
  build_by_default : true,
)
custom_target('generate_pretokenize_tests',
  output : 'test_pretokenize_gen.c',
  command : [gen_pretokenize],
  capture : true,
  build_by_default : true,
)

# Note: The 'run_all_tests' target in CMake depends on these generated files existing in 'tests/generated'.
# In Meson, we should point to the generated files (output of custom_target) instead of source tree files if we want to be clean.
# However, the CMake script writes them back to the source tree.
# To keep it simple and non-intrusive, I won't automatically overwrite source files in Meson unless requested.
# But 'run_all_tests' needs them. I will assume they exist or use the generated ones if I were to link them.
# Given the CMakeLists.txt runs them to generated/test_*.c, I will skip defining 'run_all_tests' fully if it relies on side-effects, 
# OR I can link the generated files from the build dir.

# Let's add load_model example
executable('load_model',
  'examples/load_model.cc',
  include_directories : inc_dirs,
  cpp_args : ['-fno-rtti', '-fno-exceptions', '-Wno-ignored-qualifiers'],
)

# qwen3_inference example
qwen3_sources = files(
    'examples/qwen3_inference.c',
    'src/inference/model/base.c',
    'src/inference/model/qwen3/config.c',
    # weights.c -> weights_cpp
    'src/inference/model/qwen3/ffn.c',
    'src/inference/model/qwen3/attention_layer.c',
    'src/inference/model/qwen3/transformer_layer.c',
    'src/inference/model/qwen3/qwen3.c',
    'src/inference/tokenizer/gpt2bpe.c',
    'src/inference/tokenizer/simd.c',
    'src/inference/tokenizer/unicode_tables.c',
    'src/inference/kernels/gemm/gemm.c',
    'src/inference/kernels/gemm/gemm_neon.c',
    'src/inference/kernels/gemm/gemm_amx.c',
    'src/inference/kernels/norm/layernorm.c',
    'src/inference/kernels/norm/layernorm_neon.c',
    'src/inference/kernels/activation/activation.c',
    'src/inference/kernels/activation/activation_neon.c',
    'src/inference/kernels/rope/rope.c',
    'src/inference/kernels/rope/rope_neon.c',
    'src/inference/kernels/embedding/embedding.c',
    'src/inference/kernels/embedding/embedding_neon.c',
    'src/inference/kernels/attention/attention.c',
    'src/inference/kernels/attention/attention_neon.c',
    'src/inference/kernels/kv_cache/kv_cache.c',
    'src/inference/kernels/kv_cache/kv_cache_neon.c',
    'src/inference/kernels/sampling/sampling.c',
    'src/inference/kernels/sampling/sampling_neon.c',
)

if host_machine.cpu_family() == 'aarch64'
  qwen3_sources += files('src/inference/tokenizer/simd_arm64.S')
elif host_machine.cpu_family() == 'x86_64'
  qwen3_sources += files('src/inference/tokenizer/simd_x86_64.S')
endif

executable('qwen3_inference',
  qwen3_sources + [weights_cpp],
  include_directories : inc_dirs,
  dependencies : deps,
  c_args : ['-Wno-ignored-qualifiers', '-Wno-unused-parameter', '-Wno-unused-variable'],
)
