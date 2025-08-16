# DeepEP with NIXL - Build and Setup Guide

## Overview

This guide covers building and running DeepEP with NIXL integration, which enables **elastic scaling capabilities** for dynamic addition and removal of processes (ranks) during runtime.

### Build Dependencies

Follow the build instructions in the [NIXL repository](https://github.com/ai-dynamo/nixl) to install:
- **NIXL** (NVIDIA Inference Xfer Library)
- **UCX** (Unified Communication X)
- **ETCD** and ETCD C++ client library
- **DOCA** (with GPUNetIO)

## Building DeepEP with NIXL

### Step 1: Configure Environment Variables

Edit `scripts/set_env.sh` to match your installation paths and source the environment:
```bash
source scripts/set_env.sh
```

### Step 2: Build DeepEP with NIXL

Edit the paths in `scripts/build.sh` to match your installation paths and build DeepEP using the provided build script:

```bash
./scripts/build.sh
```

**Build output**:
- Compiled library: `build/lib.linux-x86_64-3.10/deep_ep_cpp.cpython-310-x86_64-linux-gnu.so`

## Running Elastic Tests

### Adjust UCX Network Devices

Edit `tests/elastic/elastic.py` / `tests/test_internode.py` to adjust the UCX network devices to match your system:
```python
pxb_nics = ["mlx5_0", "mlx5_3", "mlx5_4", "mlx5_5", "mlx5_6", "mlx5_9", "mlx5_10", "mlx5_11"]
tcp_nics = ',ibp154s0,ibp192s0,ibp206s0,ibp220s0,ibp94s0'
os.environ['UCX_NET_DEVICES'] = f'cuda{local_rank}-{pxb_nics[local_rank]}:1' + tcp_nics
```

**Note**: This is a workaround to force UCX to chose correct network devices on some systems.

### Start ETCD Server

If not already running:
```bash
# Local test (single node)
etcd --listen-client-urls http://127.0.0.1:2379 --advertise-client-urls http://127.0.0.1:2379

# Multi-node setup (on master node)
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://<MASTER_IP>:2379
```

### Set Runtime Environment

```bash
export UCX_LOG_LEVEL=error
export LD_PRELOAD=$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_common.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_gpunetio.so:$DOCA_HOME/lib/x86_64-linux-gnu/libdoca_verbs.so
export LD_LIBRARY_PATH=$UCX_HOME/lib:$LD_LIBRARY_PATH
```

### Run Elastic Scaling Test

#### Single Node (8 ranks, 4→8 expansion):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 8 \
    --etcd-server http://127.0.0.1:2379
```

#### Multi-Node Setup:

**Node 1** (will launch the first phase with 4 ranks):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4 \
```

**Node 2** (will join the second phase with additional 4 ranks):
```bash
python3 tests/elastic/elastic.py \
    --plan tests/elastic/single_expansion.json \
    --num-processes 4 \
    --rank-server $MASTER_IP \
    --etcd-server http://$MASTER_IP:2379
```

### Available Test Plans

- `no_expansion.json`: Static 4 ranks (baseline)
- `single_expansion.json`: 4 → 8 ranks (single expansion)
- `double_expansion.json`: 4 → 6 → 8 ranks (two expansions)
- `expansion_contraction.json`: 4 → 8 → 6 ranks (scale up then down)
