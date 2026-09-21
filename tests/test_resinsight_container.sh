#!/usr/bin/env bash
# Proves the image recipe end to end, the way the platform builds it: the
# default base (python:3.12-slim), this repository's setup.sh run as root, the
# Python installs, then — as the platform's non-root user (uid 10001) —
# tests/container_probe.py simulates a deck with OPM Flow and reads the results
# back through ResInsight under xvfb.
#
#   DECK=/path/to/SPE-2-cartesian-equi.DATA \
#   RESINSIGHT_LOCAL_ZIP=/path/to/ResInsight-Ubuntu-24.04-gcc.zip \
#   tests/test_resinsight_container.sh
#
# Needs docker and a deck. The zip is optional: without it setup.sh downloads
# the pinned release asset. The full JAX CUDA wheels are not needed for the
# proof, so the image installs only rips and opm-simulators.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DECK:?set DECK to an Eclipse .DATA file}"
TAG=${TAG:-gNN-recipe:test}
ctx=$(mktemp -d); trap 'rm -rf "$ctx"' EXIT
cp setup.sh tests/container_probe.py "$ctx/"
local_zip=""
if [ -n "${RESINSIGHT_LOCAL_ZIP:-}" ]; then cp "$RESINSIGHT_LOCAL_ZIP" "$ctx/ResInsight.zip"; local_zip=/app/ResInsight.zip; fi
cat > "$ctx/Dockerfile" <<DF
# Mirrors perd_api.build.render_dockerfile (the setup.sh line included).
FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY . /app
RUN if [ -f /app/setup.sh ]; then RESINSIGHT_LOCAL_ZIP=$local_zip bash /app/setup.sh; fi
RUN pip install rips==2026.6.1.1 opm-simulators==2026.4
RUN useradd --uid 10001 --no-create-home perd
USER 10001
ENV PYTHONPATH=/app HOME=/home/perd
DF
docker build -t "$TAG" "$ctx"
docker run --rm -v "$(realpath "$DECK"):/data/$(basename "$DECK"):ro" -e DECK="/data/$(basename "$DECK")" \
  "$TAG" python /app/container_probe.py
