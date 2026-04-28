#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$ROOT_DIR/dist"
mkdir -p "$OUT_DIR"

PANDOC_OPTS=(
  --from "markdown+tex_math_dollars"
  --pdf-engine "xelatex"
  -V "mainfont=DejaVu Sans"
  -V "monofont=DejaVu Sans Mono"
  -V "geometry:margin=25mm"
)

build_one () {
  local md_path="$1"
  local base_name
  base_name="$(basename "$md_path" .md)"

  local md_dir
  md_dir="$(cd "$(dirname "$md_path")" && pwd)"

  echo "Building PDF for: $md_path"
  (
    cd "$md_dir"
    pandoc "${PANDOC_OPTS[@]}" "$(basename "$md_path")" -o "$OUT_DIR/${base_name}.pdf"
  )
}

build_one "$ROOT_DIR/1dz_1part/Отчет_ИУ6_Н1_вариант7.md"
build_one "$ROOT_DIR/1dz_2part/Отчет_Н2_вариант7.md"
build_one "$ROOT_DIR/1dz_3part/Отчет_Н3_вариант7.md"

echo "PDFs written to: $OUT_DIR"

