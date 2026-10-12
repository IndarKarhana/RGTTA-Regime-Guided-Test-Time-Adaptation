#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error ieee_revision.tex
pdflatex -interaction=nonstopmode -halt-on-error ieee_revision.tex
