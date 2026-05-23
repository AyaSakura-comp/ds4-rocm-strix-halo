#!/bin/bash
# Unified DS4 Start Script (ROCm/HIP optimized)
# Fuses cleanup, memory flushing, and server execution.

set -e

echo "--- Step 1: Killing Stale Server Instances ---"
pkill -9 -x ds4-server || true
rm -f /tmp/ds4.lock

echo "--- Step 2: Cleaning System Cache Memory ---"
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches

echo "--- Step 3: Setting ROCm Environment ---"
# Never use DS4_HIP_COPY_MODEL on this UMA APU: it hipMalloc's a 2nd 84GiB and
# memcpy's RAM->RAM (pointless, no separate VRAM) and thrashes/wedges for >10min.
unset DS4_HIP_COPY_MODEL DS4_HIP_COPY_MODEL_CHUNKED
# DS4_HIP_LOAD_TO_MEMORY: read the whole model into a pinned host buffer via
# sequential read() (~3.9 GB/s, ~22s for 84GiB), then expose it to the GPU with
# no copy (UMA). Fully resident + pinned => not evictable mid-inference. Inference
# speed ~= zero-copy (same DRAM). Makes --warm-weights redundant (omit it).
export DS4_HIP_LOAD_TO_MEMORY=1
# DS4_METAL_PREFILL_CHUNK is the correct variable for both Metal and ROCm backends
# (DS4_HIP_PREFILL_CHUNK is a no-op due to incomplete rename in the port)
# WARNING: chunk=8192 hangs the GPU prefill kernel on Strix Halo (ROCm 7.2.2):
# hipDeviceSynchronize() busy-spins forever waiting on a kernel that never
# completes for large batches. The engine default for prompts >2048 is 2048,
# which prefills fine (~60 t/s). Keep this at 2048 (or unset to use the default).
export DS4_METAL_PREFILL_CHUNK=2048
export DS4_HIP_PREFILL_CHUNK=2048

# Enable verbose JIT compilation logs for ROCm Code Object Manager (COMGR)
export AMD_COMGR_EMIT_VERBOSE_LOGS=1
export AMD_COMGR_REDIRECT_LOGS="/tmp/ds4-comgr.log"

echo "--- Step 4: Starting DS4 Server ---"
# We run this in the background if it's called with --bg, otherwise we tail the log.
# For simplicity, we'll always use a log file to track initialization.
LOG_FILE="/tmp/ds4-server.log"

cd "$(dirname "$0")"
# ulimit -l unlimited allows mlockall to prevent model weight pages from being evicted
ulimit -l unlimited
# No --warm-weights: DS4_HIP_LOAD_TO_MEMORY already reads the whole model resident.
nohup ./ds4-server --rocm --ctx 65536 \
    --port 8001 \
    --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 16384 \
    > "$LOG_FILE" 2>&1 &

echo "--- Step 5: Waiting for Initialization ---"
until grep -q "listening on" "$LOG_FILE" || ! kill -0 $! 2>/dev/null; do
    sleep 1
done

echo "--- DONE: Server is running at http://127.0.0.1:8001 ---"
tail -f "$LOG_FILE"
