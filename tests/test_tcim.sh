#!/usr/bin/env bash
# DS4 TCIM CLI selection, memory-budget discovery and link-isolation regressions.
set -euo pipefail

cd "$(dirname "$0")/.."

BIN=${DS4_TCIM_BIN:-./ds4}
OBJECTS_ARG=${DS4_TCIM_OBJECTS:-}
LINK_ARGS=${DS4_TCIM_LINK_ARGS:-}
PROVIDER_REGEX=${TCIM_PROVIDER_REGEX:-'(^|/)ds4_tcim\.o$'}

fail() {
    printf 'FAIL test_tcim: %s\n' "$*" >&2
    exit 1
}

pass() {
    printf 'ok test_tcim: %s\n' "$*"
}

show_file() {
    local path=$1
    printf '%s\n' "--- $path ---" >&2
    sed -n '1,80p' "$path" >&2 || true
}

require_tool() {
    command -v "$1" >/dev/null 2>&1 || fail "required tool not found: $1"
}

assert_has() {
    local label=$1
    local pattern=$2
    local path=$3
    if ! grep -Eq -- "$pattern" "$path"; then
        show_file "$path"
        fail "$label (missing pattern: $pattern)"
    fi
}

assert_lacks() {
    local label=$1
    local pattern=$2
    local path=$3
    if grep -Eiq -- "$pattern" "$path"; then
        show_file "$path"
        fail "$label (forbidden pattern: $pattern)"
    fi
}

for tool in awk cmp grep mktemp nm readelf sed sort uniq; do
    require_tool "$tool"
done

[[ -x "$BIN" ]] || fail "TCIM CLI is not executable: $BIN"
[[ -n "$OBJECTS_ARG" ]] || fail \
    "DS4_TCIM_OBJECTS must list the linked TCIM host objects"
[[ -n "$LINK_ARGS" ]] || fail \
    "DS4_TCIM_LINK_ARGS must list the TCIM link arguments"

TMP_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ds4-tcim-host.XXXXXX")
trap 'rm -rf -- "$TMP_DIR"' EXIT

# Help is the public build identity.  Do not grep for generic CUDA/Metal words:
# shared SSD help still describes the other production builds.
"$BIN" --help >"$TMP_DIR/help" 2>&1
assert_has "help advertises the TCIM selector" \
    '--tcim[[:space:]]*\|[[:space:]]*--cpu' "$TMP_DIR/help"
assert_has "help exposes only the TCIM/CPU backend names" \
    'Backend name: tcim or cpu\.' "$TMP_DIR/help"
assert_has "help documents TCIM auto memory probing" \
    '--gpu-vram[[:space:]]+N\|auto' "$TMP_DIR/help"
assert_has "help documents logical device 0" \
    '--gpu-devices[[:space:]]+0' "$TMP_DIR/help"
assert_lacks "help omits the CUDA selector" \
    '--metal[[:space:]]*\|[[:space:]]*--cuda' "$TMP_DIR/help"
assert_lacks "help omits the ROCm selector" \
    '--metal[[:space:]]*\|[[:space:]]*--rocm' "$TMP_DIR/help"
pass "TCIM help identity"

run_identity_gate() {
    local label=$1
    local output=$2
    shift 2
    local rc

    set +e
    "$BIN" "$@" -m /dev/null --inspect >"$output" 2>&1
    rc=$?
    set -e

    [[ $rc -ne 0 ]] || fail "$label unexpectedly accepted /dev/null as a GGUF"
    assert_lacks "$label parses before model loading" \
        'unknown option|invalid backend' "$output"
    assert_has "$label selects the TCIM backend identity" \
        'Linux tcim backend' "$output"
    assert_has "$label reaches the deterministic model gate" \
        'model file is too small to be GGUF' "$output"
}

run_identity_gate "default selector" "$TMP_DIR/default"
run_identity_gate "--tcim" "$TMP_DIR/tcim-flag" --tcim
run_identity_gate "--backend tcim" "$TMP_DIR/tcim-name" --backend tcim

if ! cmp -s "$TMP_DIR/default" "$TMP_DIR/tcim-flag" ||
   ! cmp -s "$TMP_DIR/default" "$TMP_DIR/tcim-name"; then
    show_file "$TMP_DIR/default"
    show_file "$TMP_DIR/tcim-flag"
    show_file "$TMP_DIR/tcim-name"
    fail "default, --tcim, and --backend tcim did not reach the same path"
fi
pass "default, --tcim, and --backend tcim identity equivalence"

run_auto_probe() {
    local label=$1
    local output=$2
    shift 2
    local budget layout_count rc

    set +e
    "$BIN" "$@" -m /dev/null --inspect >"$output" 2>&1
    rc=$?
    set -e

    [[ $rc -ne 0 ]] || fail "$label unexpectedly accepted /dev/null as a GGUF"
    assert_lacks "$label parses before model loading" \
        'unknown option|invalid backend' "$output"
    assert_has "$label reports one logical device" \
        '^ds4: GPU config: 1 device \[0\] requested, budgets [0-9]+ GB; auto=true$' \
        "$output"
    layout_count=$(grep -Ec '^ds4: GPU config:' "$output" || true)
    [[ $layout_count -eq 1 ]] || {
        show_file "$output"
        fail "$label emitted $layout_count GPU layout lines (expected 1)"
    }
    budget=$(sed -nE \
        's/^ds4: GPU config: 1 device \[0\] requested, budgets ([0-9]+) GB; auto=true$/\1/p' \
        "$output")
    [[ "$budget" =~ ^[0-9]+$ ]] || {
        show_file "$output"
        fail "$label emitted a non-numeric device budget"
    }
    (( 10#$budget > 0 )) || {
        show_file "$output"
        fail "$label emitted a zero device budget"
    }
    assert_has "$label stays on the TCIM backend" 'Linux tcim backend' "$output"
    assert_has "$label reaches the deterministic model gate" \
        'model file is too small to be GGUF' "$output"
    pass "$label"
}

# These calls query allocator free memory only.  They do not allocate model
# buffers or execute a TCIM payload.
run_auto_probe "--gpu-vram auto" "$TMP_DIR/auto" --gpu-vram auto
run_auto_probe "--tcim with auto logical device 0" "$TMP_DIR/auto-device-0" \
    --tcim --gpu-vram auto --gpu-devices 0
run_auto_probe "--backend tcim with logical device 0" "$TMP_DIR/device-0" \
    --backend tcim --gpu-devices 0

run_bad_filter() {
    local label=$1
    local output=$2
    shift 2
    local rc

    set +e
    "$BIN" "$@" -m /dev/null --inspect >"$output" 2>&1
    rc=$?
    set -e

    [[ $rc -ne 0 ]] || fail "$label unexpectedly succeeded"
    assert_lacks "$label emits no accepted layout" '^ds4: GPU config:' "$output"
    assert_has "$label explains the single-device contract" \
        'logical device 0' "$output"
    pass "$label"
}

run_bad_filter "logical device 1 is rejected" "$TMP_DIR/device-1" \
    --gpu-vram auto --gpu-devices 1

# Audit the exact arguments used by the canonical make target. Shared ds4.c
# contains backend-related diagnostics, so scanning binary strings would not
# identify which backend objects or runtimes were actually linked.
read -r -a link_args <<<"$LINK_ARGS"
printf '%s\n' "${link_args[@]}" >"$TMP_DIR/link-inputs"

assert_lacks "link arguments exclude CUDA objects and libraries" \
    'ds4_cuda\.(o|a|so)|/cuda/|cuda/mmq|lib(cuda|cudart|cublas|nvrtc)([.[:space:]()_-]|$)' \
    "$TMP_DIR/link-inputs"
assert_lacks "link arguments exclude ROCm objects and libraries" \
    'ds4_rocm[^/[:space:]()]*\.(o|a|so)|/rocm/|lib(amdhip64|hiprtc|hipblas|rocblas|hsa-runtime64)([.[:space:]()_-]|$)' \
    "$TMP_DIR/link-inputs"
assert_lacks "link arguments exclude Metal objects and libraries" \
    'ds4_metal\.(o|a|so)|/metal/|lib(Metal|objc)([.[:space:]()_-]|$)' \
    "$TMP_DIR/link-inputs"
assert_lacks "link arguments exclude an HMM loader" \
    '(^|[/_.() -])[^/[:space:]()]*hmm[^/[:space:]()]*\.(o|a|so)|hmonnx' \
    "$TMP_DIR/link-inputs"
assert_lacks "link arguments exclude a C++ host runtime" \
    'libstdc\+\+|libc\+\+' "$TMP_DIR/link-inputs"
pass "link-argument backend isolation"

readelf -d "$BIN" >"$TMP_DIR/dynamic"
assert_lacks "ELF dependencies exclude CUDA" \
    '\(NEEDED\).*(libcuda|libcudart|libcublas|libnvrtc)' "$TMP_DIR/dynamic"
assert_lacks "ELF dependencies exclude ROCm" \
    '\(NEEDED\).*(libamdhip64|libhiprtc|libhipblas|librocblas|libhsa-runtime64)' \
    "$TMP_DIR/dynamic"
assert_lacks "ELF dependencies exclude Metal/Objective-C" \
    '\(NEEDED\).*(libMetal|libobjc)' "$TMP_DIR/dynamic"
assert_lacks "ELF dependencies exclude HMM" \
    '\(NEEDED\).*hmm' "$TMP_DIR/dynamic"
assert_lacks "ELF dependencies exclude libstdc++/libc++" \
    '\(NEEDED\).*(libstdc\+\+|libc\+\+)' "$TMP_DIR/dynamic"
pass "ELF dependency isolation"

# Inspect every TCIM host object, rather than trusting the final executable's
# flattened symbol table to tell us which object supplied a definition.
: >"$TMP_DIR/object-nm"
object_count=0
inspect_object() {
    local object=$1
    [[ -f "$object" ]] || fail "linked TCIM host object is missing: $object"
    object_count=$((object_count + 1))
    if ! nm -A -g --defined-only "$object" >>"$TMP_DIR/object-nm"; then
        fail "nm could not inspect TCIM host object: $object"
    fi
}

read -r -a objects <<<"$OBJECTS_ARG"
for object in "${objects[@]}"; do
    inspect_object "$object"
done
[[ $object_count -gt 0 ]] || fail "no linked TCIM host objects were selected"

awk '
    $NF ~ /^ds4_gpu_[[:alnum:]_]+$/ {
        object = $1
        sub(/:[^:]*$/, "", object)
        print object "\t" $(NF - 1) "\t" $NF
    }
' "$TMP_DIR/object-nm" | sort -k3,3 >"$TMP_DIR/gpu-defs"

[[ -s "$TMP_DIR/gpu-defs" ]] || fail "TCIM objects define no ds4_gpu_* symbols"

awk '{ print $3 }' "$TMP_DIR/gpu-defs" | sort | uniq -d \
    >"$TMP_DIR/duplicate-gpu-defs"
if [[ -s "$TMP_DIR/duplicate-gpu-defs" ]]; then
    show_file "$TMP_DIR/duplicate-gpu-defs"
    fail "more than one TCIM object defines the same ds4_gpu_* symbol"
fi

awk '$2 != "T" { print }' "$TMP_DIR/gpu-defs" \
    >"$TMP_DIR/non-function-gpu-defs"
if [[ -s "$TMP_DIR/non-function-gpu-defs" ]]; then
    show_file "$TMP_DIR/non-function-gpu-defs"
    fail "TCIM ds4_gpu_* providers must be strong function definitions"
fi

while IFS=$'\t' read -r object type symbol; do
    if ! grep -Eq -- "$PROVIDER_REGEX" <<<"$object"; then
        printf '%s\t%s\t%s\n' "$object" "$type" "$symbol" >&2
        fail "ds4_gpu_* definition is outside the TCIM provider (regex: $PROVIDER_REGEX)"
    fi
done <"$TMP_DIR/gpu-defs"

nm -u "$BIN" >"$TMP_DIR/binary-undefined"
if awk '{ name = $NF; sub(/@.*/, "", name); \
          if (name ~ /^ds4_gpu_[[:alnum:]_]+$/) print }' \
    "$TMP_DIR/binary-undefined" >"$TMP_DIR/unresolved-gpu" &&
   [[ -s "$TMP_DIR/unresolved-gpu" ]]; then
    show_file "$TMP_DIR/unresolved-gpu"
    fail "final TCIM CLI retains unresolved ds4_gpu_* symbols"
fi
pass "single strong TCIM ds4_gpu_* provider"

printf 'test_tcim: DS4 CLI and link-isolation tests passed\n'
