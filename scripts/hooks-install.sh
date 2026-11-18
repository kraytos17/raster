#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
hook_path="$repo_root/.githooks"
current_path="$(git config --local --get core.hooksPath || true)"

if [[ -n "$current_path" && "$current_path" != "$hook_path" ]]; then
    echo "core.hooksPath is already set to '$current_path'; leaving it unchanged." >&2
    exit 1
fi

git config --local core.hooksPath "$hook_path"
echo "Installed raster pre-commit hook (core.hooksPath=.githooks)."
