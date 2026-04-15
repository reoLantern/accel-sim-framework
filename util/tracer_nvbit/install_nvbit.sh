#!/bin/bash
# NVBit v1.8 (2026-04-06) is required for:
#   - getSassBinary() : raw instruction encoding for control bits extraction
#   - native TMA APIs : nvbit_parse_tma_transfer_info() etc. for Hopper
# Requires CUDA 13.2+ headers.
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

NVBIT_VER=1.8
NVBIT_TARBALL=nvbit-Linux-x86_64-${NVBIT_VER}.tar.bz2
NVBIT_URL=https://github.com/NVlabs/NVBit/releases/download/v${NVBIT_VER}/${NVBIT_TARBALL}

rm -rf "$BASH_ROOT/nvbit_release"
mkdir -p "$BASH_ROOT/nvbit_release"
wget "$NVBIT_URL"
tar -xf "$NVBIT_TARBALL" -C "$BASH_ROOT/nvbit_release" --strip-components=1
rm "$NVBIT_TARBALL"
