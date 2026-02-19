#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
TARGET_DIR="${1:-assets/mujoco_menagerie}"
TARGET_BRANCH="${2:-main}"

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required but not found in PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$TARGET_DIR")"

if [ -d "$TARGET_DIR/.git" ]; then
  echo "Updating existing menagerie checkout at: $TARGET_DIR"
  git -C "$TARGET_DIR" fetch --depth 1 origin "$TARGET_BRANCH"
  git -C "$TARGET_DIR" checkout "$TARGET_BRANCH"
  git -C "$TARGET_DIR" pull --ff-only --depth 1 origin "$TARGET_BRANCH"
else
  if [ -e "$TARGET_DIR" ] && [ ! -d "$TARGET_DIR/.git" ]; then
    echo "Error: target exists but is not a git checkout: $TARGET_DIR" >&2
    exit 1
  fi
  echo "Cloning MuJoCo Menagerie into: $TARGET_DIR"
  git clone --depth 1 --branch "$TARGET_BRANCH" "$REPO_URL" "$TARGET_DIR"
fi

echo "Done. Menagerie path: $TARGET_DIR"
