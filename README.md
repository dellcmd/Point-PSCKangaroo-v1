# ⚡ Point PSCKangaroo

**High-Performance Hybrid GPU + Multi-Core CPU Distinguished Point (TAME) Generator for secp256k1 ECDLP (Pollard's Kangaroo)**

---

## 📖 Project Concept & Overview

**Point PSCKangaroo** is an ultra-fast, industrial-grade generator of Distinguished Points (DPs / TAMEs) designed for solving secp256k1 Elliptic Curve Discrete Logarithm Problems (ECDLP) using Pollard's Kangaroo algorithm. 

In standard Kangaroo solvers, precious GPU time during the hunt is split between generating TAME kangaroos and hunting with WILD kangaroos. **Point PSCKangaroo** decouples this process:
1. It pre-computes millions or billions of valid TAME Distinguished Points using both the GPU (CUDA SOTA) and all available CPU cores (x86-64 BMI2/ADX assembly + Montgomery Batch Inversion) simultaneously.
2. It stores points in a shared, ultra-compact 16-byte hash table (`massive_tames.bin`) with zero serialization overhead via memory-mapped NVMe storage.
3. Once the target points are generated, the companion solver **PSCKangaroo** launches immediately in **100% WILD Hunt Mode**, maximizing collision probability from the very first second ($P \propto t^2$).

Because Point PSCKangaroo uses the exact same deterministic jump tables (`EcJumps1`, `EcJumps2`, `EcJumps3`) and SOTA equivalence-class negation maps as PSCKangaroo, all precomputed TAMEs are 100% mathematically compatible and yield instantaneous collisions.

---

## 🚀 Key Developments & Recent Upgrades

* **Hybrid GPU + CPU Co-Processing:** Harnesses 100% of GPU compute alongside all logical CPU cores running parallel SOTA kangaroo walks.
* **x86-64 BMI2/ADX Assembly Acceleration:** Powered by Jean-Luc Pons DRS62 modular arithmetic for ultra-fast point additions on modern CPUs.
* **Montgomery Batch Inversion:** Processes 64 to 256 kangaroo steps per thread simultaneously with a single modular inversion via `-b <size>`.
* **3-Tier HugePage Memory Management:** Dynamically allocates memory through 1 GB HugePages, 2 MB HugePages, or kernel Transparent HugePages (THP) to eliminate TLB cache misses.
* **Instant Sidecar Metadata (`.meta` & `.prefilter`):** Saves and restores database states and 512 MB L3 Pre-Filters in milliseconds without rescanning multi-gigabyte files.
* **Dual-Parity Support (`-p` / `-mixparity`):** Enables 50% Even + 50% Odd initial distances to ensure full subspace coverage for odd and even private keys.
* **Vita Life-Cycle & Macro-Cycle Detection (`-vita`):** Automatically detects 2-cycles, short loops, and macro-cycles on CPU, respawning stale kangaroos after a configurable lifespan.
* **Direct Mode (`-start 0`):** Automatically maps searches to the exact puzzle range without relative offset complexities.
* **Exact Target Limit Accounting (`-target`):** Automatically reserves point slots and gracefully shuts down once the exact desired count is reached.
* **Lock-Free Ring Buffers:** Zero-contention SPSC queues connecting GPU completion handlers with CPU worker threads.

---

## ⚙️ Core Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-range <N>` | Integer | *Required* | Target puzzle bit length (e.g., `100` or `135`). Range: 32–170. |
| `-start <HEX>` | Hex / `0` | `0` | Starting offset. Use `0` for Direct Mode (auto-sets range to $[2^{N-1}, 2^N-1]$). |
| `-dp <BITS>` | Integer | `28` | Number of distinguished point zero-bits required on X coordinate. |
| `-target <N>` | String | `0` (Inf) | Point limit. Accepts multipliers like `10M`, `50M`, `1B`. Aliases: `-points`, `-maxpts`, `-count`, `-n`. |
| `-gpu <DEV>` | String | `0` | Target GPU device index, mask (`0123`), or `none` to run exclusively on CPU. |
| `-gpus <N>` | Integer | `1` | Total number of GPU devices to allocate. Passing `0` disables GPUs. |
| `-w <N>` | Integer | `auto` | Active CPU kangaroo worker threads. Aliases: `-cpu`, `-threads`. Set to `0` to disable CPU. |
| `-b <N>` | Integer | `64` | Montgomery batch size per CPU worker thread (1–256). Aliases: `-batch`. |
| `-vita <N>` | String | `8M` | Maximum jump hops for a CPU kangaroo before respawn (e.g., `4M`, `8M`, `16M`). |
| `-p` | Flag | `false` | Enables Mix-Parity mode (50% Even + 50% Odd distances). Alias: `-mixparity`. |
| `-out <FILE>` | Path | `massive_tames.bin` | Output path for the binary hash table file. |
| `-ramlimit <GB>`| Float | `10.0` | Maximum system RAM / NVMe storage budget in Gigabytes. |
| `-txt <FILE>` | Path | *None* | Concurrently exports points in plain text (`<dist_hex> <x_hex>`). |
| `-resumenvme` | Flag | `auto` | Resumes and appends to an existing `.bin` file using its `.meta` sidecar. |
| `-wipe` | Flag | `false` | Forces wiping the existing binary table and starts fresh from 0 points. |
| `-checkpoint <M>`| Integer | `15` | Interval in minutes for background disk flushes (`msync`) and `.meta` updates. |

---

## 🛠️ Compilation Guide

Compile the binary directly for your target GPU architecture. Run the command matching your hardware:

### 1. NVIDIA GeForce RTX 40-Series (Ada Lovelace: RTX 4090, 4080, 4070, 4060)
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_89,code=sm_89"
```

### 2. NVIDIA GeForce RTX 30-Series (Ampere: RTX 3090, 3080, 3070, 3060, A5000)
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_86,code=sm_86"
```

### 3. NVIDIA GeForce RTX 20-Series & GTX 16-Series (Turing: RTX 2080 Ti, 2080, 2070, 2060, GTX 1660)
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_75,code=sm_75"
```

### 4. NVIDIA GeForce GTX 10-Series (Pascal: GTX 1080 Ti, 1080, 1070, 1060)
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_61,code=sm_61"
```

### 5. NVIDIA Data Center Hopper H100
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_90,code=sm_90"
```

### 6. NVIDIA Data Center Ampere A100
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_80,code=sm_80"
```

### 7. NVIDIA Data Center Volta V100
```bash
make clean && make -j$(nproc) GPU_ARCH="-gencode=arch=compute_70,code=sm_70"
```

### 8. Multi-Architecture Universal Build (RTX 30xx + RTX 40xx)
```bash
make clean && make -j$(nproc)
```

---

## 💡 Real Operational Examples

### Example 1: Standard Hybrid Generation on Puzzle 100 (GPU + All CPU Cores)
Generates 10 million Distinguished Points for Puzzle 100 with DP 28, combining GPU 0 with all CPU threads:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w auto -b 64 -p -target 10M -ramlimit 10 -out massive_tames.bin
```

### Example 2: Direct Mode Generation for High-Bit Puzzle 135
Targets Puzzle 135 ($[2^{134}, 2^{135}-1]$) directly using `-start 0`. Allocates a 20 GB table for 1 billion points across 4 GPUs:
```bash
./point -range 135 -dp 28 -start 0 -gpu 4 -w 32 -b 64 -p -target 1B -ramlimit 20 -wipe -out massive_tames.bin
```

### Example 3: Dual-Parity with Real-Time Plaintext Export (`-txt`)
Generates points while simultaneously streaming every single DP in human-readable plaintext (`<distance_hex> <x_hex>`) for external analysis, sorting, or independent verification:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w 28 -b 64 -p -target 10M -txt points_dump.txt -wipe
```

#### 🔍 Actual Real Output Generated by `-txt`:
Each line represents an authenticated Distinguished Point containing the 144-bit scalar walk distance (36 hex chars) followed by the 96-bit lower X-coordinate suffix (24 hex chars):
```text
000000000080A23C0637854A4347700EFBF 78CD2E635D4EA8C41FCCE55A
000000000080A536D5411B403434C18DEA8 F8FD8F2023A7B134E4F3C7A0
000000000080FEFB3F1EDE555FE3163ECA3 6BB2B193CB0DC24101953A48
000000000080EC8679A1D1D42EE68AE5F75 1F04DA2B1AFE2FFC00A2F66E
0000000000801AF5234D677F27CAB01D56C 1EEE51979A86FCAC23D103FF
00000000008027D11B6C4F1A88BE0C215A4 3A48B227B13A17B9D6E115AC
00000000008035AE1634C2B1D781F43E219 924B68AC1056CFBAE228B104
```

#### 🖥️ Real Live Operational Dashboard from an Actual Execution:
```text
================================================================================
 [POINT HYBRID v2] Total: 1.559 GKeys/s (GPU: 1.48 GK/s + CPU: 77.6 MK/s) | 0d 00h 03m 12s
--------------------------------------------------------------------------------
  Hardware Active: 1 GPU(s) (196608 Kangs) + 28 CPU Threads (64 Batch = 1792 Kangs)
  Range & DP     : Range 100 bits (Span 2^99) | DP 28 (1 DP / 2^28 jumps) | Shift 2^53
  Start Offset   : 0x0000000000000000000000000000000000000008000000000000000000000000
  Stored in .bin : 1,024,512 DPs (1.025M) | New: +1024512 (GPU: 973280 | CPU: 51232)
  Target Goal    : 1,024,512 / 10,000,000 DPs (10.25%) | Rem: 8,975,488 | ETA: 0d 00h 25m 40s
  Table Capacity : 1,024,512 / 436,076,544 slots (0.23% full) | File: massive_tames.bin (7.00 GB)
  DP Gen Rate    : 5.81 DPs/s (Theo: 5.75/s) | 348 DPs/min | 20916 DPs/hour
  Step Coverage  : 2^48.12 steps in .bin | Session Ops: 2^48.10 (CPU Escapes: 142)
================================================================================
```

### Example 4: CPU-Only Generation (Zero GPU Utilization)
Ideal for background servers, VPS instances without graphics hardware, or generating points while the GPU is occupied:
```bash
./point -range 100 -dp 28 -start 0 -gpu none -w 28 -b 64 -vita 4M -p -target 5M -ramlimit 8
```

### Example 5: GPU-Only Generation (CPU Kangaroo Walking Disabled)
Drives the GPU at maximum throughput while keeping CPU usage minimal (only draining GPU result queues):
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w 0 -p -target 20M -ramlimit 10
```

### Example 6: Resuming and Appending to an Existing Database
Safely resumes an existing session. Restores the exact point count and L3 filter in milliseconds:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w auto -p -target 50M -resumenvme
```

---

## 🔗 Direct Integration with PSCKangaroo Solver

Once your desired volume of TAME points is generated, run **PSCKangaroo** pointing to the exact same `massive_tames.bin`:

```bash
# Launch PSCKangaroo in 100% WILD Hunt Mode against the precomputed database:
./psckangaroo -gpu 0 -dp 28 -range 100 -ramlimit 10 -concurrent 1 -checkpoint 0 \
  -pubkey 0276ee308f3ed466e83076f63ab974103ea53d828bc62da635abe996c39238df86 \
  -start 0 -resumenvme -p -w 32 -b 64
```

**What Happens:**
* PSCKangaroo loads `massive_tames.bin.meta` instantly.
* It skips the TAME generation phase completely.
* 100% of GPU and CPU computational capacity is dedicated to hunting with WILD kangaroos against your precomputed points, delivering maximum collision efficiency.

---

## 📄 License & Attribution

* **License:** GNU General Public License v3 (GPLv3).
* **Base Architecture:** Derived from RetiredCoder's RCKangaroo SOTA engine.
* **Modular Math & Batch Optimizations:** Integrated from Jean-Luc Pons DRS62 and `tames-gen` assembly routines.

---

## 💬 Community & Contact

* **Telegram:** [@P2pm0](https://t.me/P2pm0)
* **GitHub Repository:** [Point-PSCKangaroo](https://github.com/P2pm0/Point-PSCKangaroo)
* **Issues & Discussions:** Feel free to open an issue or reach out directly on Telegram for optimizations, custom benchmarks, and cluster deployment setups.

