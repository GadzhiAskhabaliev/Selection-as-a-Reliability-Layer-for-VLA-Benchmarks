#!/usr/bin/env bash
# Unpacks Mesa OSMesa (llvmpipe) into $RM/sw-render without root, for containers that expose
# CUDA but not the NVIDIA graphics stack. Then run with RM_SOFTWARE_RENDERING=1.
set -euo pipefail
source "$(dirname "$0")/env.sh"
mkdir -p "$RM/sw-render/debs" && cd "$RM/sw-render/debs"
apt-get download libosmesa6 libglapi-mesa libllvm20 libdrm2 libelf1t64 libz3-4 libedit2 libxml2 libicu74 libsensors5
for deb in *.deb; do dpkg -x "$deb" ..; done
echo "software rendering libraries in $RM/sw-render"
