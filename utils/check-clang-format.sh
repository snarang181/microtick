#!/usr/bin/env bash
set -euo pipefail

CLANG_FORMAT_BIN=${CLANG_FORMAT_BIN:-clang-format}

staged_files=$(
  git ls-files | grep -E '\.(c|cc|cpp|cxx|h|hh|hpp|hxx)$' || true
)

if [ -z "$staged_files" ]; then
  exit 0
fi

tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT

failed=0

for file in $staged_files; do
  "$CLANG_FORMAT_BIN" "$file" > "$tmpdir/$(basename "$file")"
  if ! diff -u "$file" "$tmpdir/$(basename "$file")" >/dev/null; then
    echo "clang-format mismatch: $file"
    # Run clang-format on the file with -i option to add suggested changes
    "$CLANG_FORMAT_BIN" -i "$file"
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "Some files are not clang-formatted."
  exit 1
fi

echo "All files match clang-format."
exit 0