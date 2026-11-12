# ⚡ Point-PSCKangaroo-v1

**High-Performance Hybrid GPU + Multi-Core CPU Distinguished Point (TAME) Generator for secp256k1 ECDLP (Pollard's Kangaroo)**

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Platform](https://img.shields.io/badge/Platform-Linux%20x86__64-orange.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-green.svg)]()

---

## 📑 Table of Contents

1. [Project Concept & Overview](#concept)
2. [Key Developments & Technical Upgrades](#upgrades)
3. [Base Projects & Acknowledgments](#base-projects)
4. [Core CLI Parameters Reference](#parameters)
5. [Compilation Guide (Direct Commands per GPU Architecture)](#compilation)
6. [Real Operational Examples](#examples)
7. [Plaintext Export (-txt) & Real Output Verification](#txt-export)
8. [Direct Integration with PSCKangaroo Solver](#integration)
9. [Donation & Development Support](#donation)
10. [License](#license)
11. [Community & Contact](#contact)

---

<a id="concept"></a>
## 📖 Project Concept & Overview

**Point-PSCKangaroo-v1** is an industrial-grade, ultra-high-throughput generator of Distinguished Points (DPs / TAMEs) designed specifically for solving secp256k1 Elliptic Curve Discrete Logarithm Problems (ECDLP), including Bitcoin Puzzle transactions.

In traditional Kangaroo implementations, precious GPU time during search sessions is divided between generating TAME reference points and hunting with WILD kangaroos. **Point-PSCKangaroo-v1** decouples this architecture into two independent, highly optimized phases:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 1: Point-PSCKangaroo-v1                        │
│   CUDA GPUs (100% TAME)  +  All CPU Cores (DRS62 / BMI2 / ADX Batch)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Memory-Mapped Shared NVMe
                                    ▼
                     ┌─────────────────────────────┐
                     │     massive_tames.bin       │
                     │  16-Byte Ultra-Compact SOTA │
                     │  + .meta & .prefilter (L3)  │
                     └──────────────┬──────────────┘
                                    │ Instant Sidecar Resume (<1ms)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    PHASE 2: PSCKangaroo Solver                         │
│               100% WILD Kangaroos Hunting from Second 1                │
│             Quadratic Probability Growth: P(collision) ∝ t²            │
└────────────────────────────────────────────────────────────────────────┘
```

1. **Massive Precomputation:** Simultaneously leverages 100% of GPU resources and all available logical CPU cores (using DRS62 assembly and Montgomery Batch Inversion) to pre-calculate millions or billions of valid TAME Distinguished Points.
2. **Compact Shared Storage:** Stores points in a shared 16-byte hash table (`massive_tames.bin`) mapped directly to NVMe storage via `mmap` with zero serialization latency.
3. **Instant 100% WILD Hunt:** Once populated, the companion solver **PSCKangaroo** launches immediately in **100% WILD Mode**, maximizing collision probability from the very first second ($P \propto t^2$).

Because Point-PSCKangaroo-v1 uses the exact deterministic jump tables (`EcJumps1`, `EcJumps2`, `EcJumps3`) and equivalence-class negation maps as PSCKangaroo, all precomputed points are 100% mathematically compatible and yield instantaneous collisions upon path intersections.

---

<a id="upgrades"></a>
## 🚀 Key Developments & Technical Upgrades

* **Hybrid GPU + CPU Co-Processing:** Combines full GPU compute with all logical CPU threads running parallel SOTA kangaroo walks.
* **x86-64 BMI2/ADX Assembly Acceleration:** Modular arithmetic accelerated via Jean-Luc Pons DRS62 assembly routines for maximum CPU scalar addition speeds.
* **Montgomery Batch Inversion:** Simultaneously steps 64 to 256 kangaroos per CPU thread using a single modular inversion via `-b <batch_size>`.
* **3-Tier HugePage Memory Management:** Allocates hash tables and ring queues across 1 GB HugePages, 2 MB HugePages, or Transparent HugePages (THP) to eliminate TLB misses.
* **Instant Sidecar Metadata (`.meta` & `.prefilter`):** Restores database state, total operations, and 512 MB L3 Pre-Filters in less than 1 millisecond without scanning gigabytes of disk.
* **Dual-Parity Walking (`-p` / `-mixparity`):** Configures 50% Even + 50% Odd initial distances, providing complete coverage across both odd and even private key spaces.
* **Vita Life-Cycle & Macro-Cycle Escape (`-vita`):** Automatically detects 2-cycles, short loops, and macro-cycles on CPU, respawning stale kangaroos after a designated step limit.
* **Direct Range Mode (`-start 0`):** Maps puzzle searches directly to target intervals $[2^{N-1}, 2^N-1]$ without relative offset arithmetic errors.
* **Exact Target Limit Accounting (`-target`):** Reserves slots and terminates execution immediately upon reaching the exact user-defined point count.
* **Lock-Free SPSC Ring Buffers:** Thread-safe, contention-free circular buffers transferring GPU points directly to CPU worker threads.

---

<a id="base-projects"></a>
## 🏛️ Base Projects & Acknowledgments

Point-PSCKangaroo-v1 builds upon and incorporates algorithmic discoveries from pioneering open-source cryptographic solvers:

* **[RCKangaroo by RetiredCoder (RC)](https://github.com/RetiredC/RCKangaroo)**  
  The foundational state-of-the-art (SOTA) Pollard's Kangaroo solver for secp256k1, introducing equivalence classes, negation maps ($K \approx 1.15$), and GPU core architectures.
* **[VanitySearch / DRS62 by Jean-Luc Pons](https://github.com/JeanLucPons/VanitySearch)**  
  High-performance secp256k1 elliptic curve arithmetic, DRS62 fast modular inversion (`fast_modinv`), and optimized assembly primitives.
* **[tames-gen Engine by providiu](https://github.com/providiu/tames-gen)**  
  Multi-threaded CPU Montgomery batch inversion architectures, HugePage allocation models, and cacheline-aligned step tracking.

---

<a id="parameters"></a>
## ⚙️ Core CLI Parameters Reference

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `-range <N>` | Integer | *Required* | Target puzzle bit length (e.g., `100` or `135`). Range: 32–170. |
| `-start <HEX>` | Hex / `0` | `0` | Starting scalar offset. Use `0` for Direct Mode (auto-sets range to $[2^{N-1}, 2^N-1]$). |
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
| `-txt <FILE>` | Path | *None* | Concurrently exports points in plain text (`Distance_Hex X_Coordinate_Hex`). |
| `-resumenvme` | Flag | `auto` | Resumes and appends to an existing `.bin` file using its `.meta` sidecar. |
| `-wipe` | Flag | `false` | Forces wiping the existing binary table and starts fresh from 0 points. |
| `-checkpoint <M>`| Integer | `15` | Interval in minutes for background disk flushes (`msync`) and `.meta` updates. |

---

<a id="compilation"></a>
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

<a id="examples"></a>
## 💡 Real Operational Examples

### Example 1: Standard Hybrid Generation on Puzzle 100 (GPU 0 + All CPU Cores)
Generates 10 million Distinguished Points for Puzzle 100 with DP 28, combining GPU 0 with all CPU threads:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w auto -b 64 -p \
  -target 10M -ramlimit 10 -out massive_tames.bin
```

### Example 2: Direct Mode Generation for High-Bit Puzzle 135
Targets Puzzle 135 (`[2^134, 2^135 - 1]`) directly using `-start 0`. Allocates a 20 GB table for 1 billion points across 4 GPUs:
```bash
./point -range 135 -dp 28 -start 0 -gpu 4 -w 32 -b 64 -p \
  -target 1B -ramlimit 20 -wipe -out massive_tames.bin
```

### Example 3: Dual-Parity with Real-Time Plaintext Export (`-txt`)
Generates points while simultaneously streaming every single DP in human-readable plaintext format (`Distance_Hex X_Coordinate_Hex`) for external sorting, deduplication, or independent verification:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w 28 -b 64 -p \
  -target 10M -txt points_dump.txt -wipe
```

### Example 4: CPU-Only Generation (Zero GPU Utilization)
Ideal for background servers, VPS instances without graphics hardware, or generating points while the GPU is occupied:
```bash
./point -range 100 -dp 28 -start 0 -gpu none -w 28 -b 64 -vita 4M -p \
  -target 5M -ramlimit 8
```

### Example 5: GPU-Only Generation (CPU Kangaroo Walking Disabled)
Drives the GPU at maximum throughput while keeping CPU usage minimal (only draining GPU result queues):
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w 0 -p \
  -target 20M -ramlimit 10
```

### Example 6: Resuming and Appending to an Existing Database
Safely resumes an existing session. Restores the exact point count and L3 filter in milliseconds:
```bash
./point -range 100 -dp 28 -start 0 -gpu 0 -w auto -p \
  -target 50M -resumenvme
```

---

<a id="txt-export"></a>
## 📄 Plaintext Export (-txt) & Real Output Verification

When `-txt <FILE>` is passed, Point-PSCKangaroo-v1 outputs raw text records of every generated Distinguished Point.

### 🔍 Actual Real Output Generated by `-txt`:
Each record contains the 144-bit scalar walk distance (36 hex characters) followed by the lower 96-bit X-coordinate suffix (24 hex characters):
```text
000000000080A23C0637854A4347700EFBF 78CD2E635D4EA8C41FCCE55A
000000000080A536D5411B403434C18DEA8 F8FD8F2023A7B134E4F3C7A0
000000000080FEFB3F1EDE555FE3163ECA3 6BB2B193CB0DC24101953A48
000000000080EC8679A1D1D42EE68AE5F75 1F04DA2B1AFE2FFC00A2F66E
0000000000801AF5234D677F27CAB01D56C 1EEE51979A86FCAC23D103FF
00000000008027D11B6C4F1A88BE0C215A4 3A48B227B13A17B9D6E115AC
00000000008035AE1634C2B1D781F43E219 924B68AC1056CFBAE228B104
```

### 🖥️ Real Live Operational Dashboard:
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

---

<a id="integration"></a>
## 🔗 Direct Integration with PSCKangaroo Solver

Once your target volume of TAME points is generated in `massive_tames.bin`, execute **PSCKangaroo** pointing to the exact same file:

```bash
# Launch PSCKangaroo in 100% WILD Hunt Mode against the precomputed database:
./psckangaroo -gpu 0 -dp 28 -range 100 -ramlimit 10 -concurrent 1 -checkpoint 0 \
  -pubkey 0276ee308f3ed466e83076f63ab974103ea53d828bc62da635abe996c39238df86 \
  -start 0 -resumenvme -p -w 32 -b 64
```

**Operational Sequence:**
1. PSCKangaroo detects and parses `massive_tames.bin.meta` in under 1 millisecond.
2. It bypasses TAME generation entirely.
3. 100% of GPU compute and CPU threads immediately execute WILD kangaroo walks hunting against the precomputed points, delivering maximum collision efficiency.

---

<a id="donation"></a>
## 💰 Donation & Development Support

If you find this project valuable for your ECDLP cryptographic research and would like to support continuous performance optimizations and new algorithm integrations:

* **Bitcoin (BTC):** `bc1qsca4zt69ny9hc54wqd4neu6eu2shh2cr8ezfn0`

Your contributions help support further hardware benchmarks, cluster scaling, and open-source cryptographic development.

---

<a id="license"></a>
## 📄 License

This project is open-source software licensed under the **GNU General Public License v3.0 (GPLv3)**.  
See the [GNU GPLv3 License](https://www.gnu.org/licenses/gpl-3.0.html) for full details.

---

<a id="contact"></a>
## 💬 Community & Contact

* **Telegram:** [@Dellcmd](https://t.me/Dellcmd)
* **GitHub Repository:** [https://github.com/dellcmd/Point-PSCKangaroo-v1](https://github.com/dellcmd/Point-PSCKangaroo-v1)
* **Issues & Discussions:** For performance feedback, bug reports, and cluster setup questions, please open an issue in this repository or reach out directly on Telegram.
