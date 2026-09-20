#!/usr/bin/env bash
# Image setup for this workflow. The PERD builder runs a repository's root
# `setup.sh` ONCE, as root, right after copying the repository to /app and
# before `pip install -r requirements.txt` — on the platform's default base
# (python:3.12-slim, Debian). Everything ResInsight needs lives here; the
# platform knows nothing about it.
#
# It leaves behind:
#   /opt/ResInsight                       the ResInsight nightly (Qt bundled)
#   /opt/ResInsight/bin/resinsight-xvfb   what rips launches: xvfb-run wrapper
#   /home/perd                            a writable HOME for uid 10001
#
# RESINSIGHT_URL / RESINSIGHT_SHA256 pin the exact nightly proven against
# rips==2026.6.1.1; the nightly link is a moving target, so the default is
# a release asset and the checksum is what makes the build reproducible.
set -euo pipefail

RESINSIGHT_URL="${RESINSIGHT_URL:-https://github.com/Mechtanium/Delta-PINN/releases/download/resinsight-2026.07/ResInsight-Ubuntu-24.04-gcc.zip}"
RESINSIGHT_FALLBACK_URL="https://nightly.link/OPM/ResInsight/workflows/ResInsightWithCache/dev/ResInsight-Ubuntu%2024.04%20gcc.zip"
RESINSIGHT_SHA256="${RESINSIGHT_SHA256:-c49e2331b2eec0b5616528e0ed060ac715e790edceb50aa7bf71411b18d47be1}"
RESINSIGHT_LOCAL_ZIP="${RESINSIGHT_LOCAL_ZIP:-}"   # tests: skip the download

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl unzip \
  xvfb xauth \
  libgl1 libegl1 libglx-mesa0 libopengl0 libgomp1 \
  libx11-6 libxcb1 libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 \
  libxkbcommon0 libxkbcommon-x11-0 \
  libfontconfig1 libfreetype6 libglib2.0-0 libdbus-1-3 libkrb5-3 \
  fonts-dejavu-core

tmp=$(mktemp -d)
if [ -n "$RESINSIGHT_LOCAL_ZIP" ] && [ -f "$RESINSIGHT_LOCAL_ZIP" ]; then
  cp "$RESINSIGHT_LOCAL_ZIP" "$tmp/ri.zip"
else
  curl -fsSL -o "$tmp/ri.zip" "$RESINSIGHT_URL" || curl -fsSL -o "$tmp/ri.zip" "$RESINSIGHT_FALLBACK_URL"
fi
echo "$RESINSIGHT_SHA256  $tmp/ri.zip" | sha256sum -c -
mkdir -p /opt/ResInsight
unzip -q "$tmp/ri.zip" -d /opt/ResInsight
chmod +x /opt/ResInsight/bin/ResInsight
rm -rf "$tmp"

# rips launches RESINSIGHT_EXECUTABLE directly; the nightly ships only the
# xcb platform plugin, so wrap it in a virtual X server.
cat > /opt/ResInsight/bin/resinsight-xvfb <<'EOF'
#!/usr/bin/env bash
exec xvfb-run -a --server-args="-screen 0 1280x1024x24" /opt/ResInsight/bin/ResInsight "$@"
EOF
chmod +x /opt/ResInsight/bin/resinsight-xvfb

# The platform's non-root user (uid 10001) needs a writable HOME for
# ResInsight's settings; the workflow sets HOME to it before launching.
install -d -o 10001 -g 10001 /home/perd

apt-get clean
rm -rf /var/lib/apt/lists/*
