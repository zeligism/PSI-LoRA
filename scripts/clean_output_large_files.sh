#!/usr/bin/env bash
set -euo pipefail

target_dir="${1:-output}"
min_size="${2:-50M}"

if [[ ! -d "$target_dir" ]]; then
  echo "Directory not found: $target_dir" >&2
  exit 1
fi

found_any=false

while IFS= read -r -d '' file; do
  found_any=true
  size_bytes=$(stat -c %s -- "$file")
  size_human=$(numfmt --to=iec --suffix=B "$size_bytes" 2>/dev/null || echo "${size_bytes}B")
  printf 'Delete %s (%s)? [y/N] ' "$file" "$size_human"
  read -r reply < /dev/tty
  if [[ "$reply" == "y" || "$reply" == "Y" ]]; then
    rm -f -- "$file"
    echo "Deleted: $file"
  else
    echo "Skipped: $file"
  fi
done < <(find "$target_dir" -type f -size +"$min_size" -print0)

if [[ "$found_any" == false ]]; then
  echo "No files larger than $min_size found in $target_dir."
fi
