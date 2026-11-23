#!/usr/bin/env bash
# Build a zip of the project to share (e.g. with the professor), WITHOUT the dataset.
#
# Includes every file git tracks plus new files that are not git-ignored (so a report or
# slides added later are picked up). Leaves out the dataset and everything generated from it
# (data/), the virtual environment, training run folders (results/runs/), caches and
# local settings. The dashboard runs from the small demo bundle in results/demo/.
#
# Usage:  bash scripts/make_share_zip.sh            -> dist/rees46-purchase-intention-<date>.zip
# Re-run it after adding the report and presentation.
set -euo pipefail
cd "$(dirname "$0")/.."
NAME=rees46-purchase-intention
OUT="dist/$NAME-$(date +%Y%m%d).zip"
mkdir -p dist
rm -f "$OUT"
LIST=$(mktemp)
git ls-files -co --exclude-standard \
  | grep -vE '^(\.claude/|dist/|data/|\.env$)' \
  | grep -vE '(^|/)(~\$[^/]*|\.DS_Store)$' > "$LIST" || true
git ls-files 'data/*.gitkeep' 'data/**/.gitkeep' >> "$LIST"   # keep the empty data folders
# a top-level folder inside the zip, so it unzips into one directory
STAGE=$(mktemp -d)
ln -s "$PWD" "$STAGE/$NAME"
( cd "$STAGE" && sed "s|^|$NAME/|" "$LIST" | zip -q -X "$OLDPWD/$OUT" -@ )
rm -rf "$STAGE" "$LIST"
echo "$OUT  $(du -h "$OUT" | cut -f1)  $(unzip -Z1 "$OUT" | wc -l | tr -d ' ') files"
