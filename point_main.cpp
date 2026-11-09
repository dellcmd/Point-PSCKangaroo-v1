// ============================================================================
// POINT v2.0 — Ultra-Fast Hybrid GPU + CPU Distinguished Point (TAME) Generator
// Compatible 100% with PSCKangaroo (TameStore v56C 16-byte NVMe mmap format)
// Integrates tames-gen-main:
//   - x86-64 BMI2/ADX Inline Assembly Field Math (G_stub.c / G.h)
//   - Jean-Luc Pons DRS62 Modular Inversion (fast_inv.h) + Montgomery Batch Inv
//   - Full Hybrid GPU + Multi-Core CPU SOTA Kangaroo Engine (L1S2 + Jumps3)
//   - 3-Tier HugePage Allocator (1GB -> 2MB -> THP MADV_HUGEPAGE)
//   - Instant Disk Metadata Sidecar (.meta) + Non-Blocking MS_ASYNC Checkpoints
//   - Exact -target <COUNT> Auto-Stop Reservation System
// ============================================================================

#include <iostream>
#include <vector>
#include <signal.h>
#include <cmath>
#include <ctime>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <thread>
#include <mutex>

#ifndef _WIN32
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#endif

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "TameStore.h"
#include "G.h"
#include "fast_inv.h"

// ============================================================================
// Globals required by GpuKang.cpp and TameStore.h
// ============================================================================
Ec ec;
TameStore gTameStore;

uint8_t* gTameStoreData = nullptr;
size_t gTameStoreDataSize = 0;
bool gUsingExternalMemory = false;
bool gResumeNVMe = true;

bool gGenMode = true;           // Always 100% TAME generation mode
EcInt Int_HalfRange;            // Kept at 0 in point generator
EcInt gGenStartOffset;          // Starting scalar offset for TAME kangaroos
int gGenSpanBits = 0;           // Bit-width of random span added to gGenStartOffset

u32 gDP = 28;                   // Default DP = 28
u32 gRange = 0;                 // Target range in bits
EcInt gStart;
bool gStartSet = false;
double gRAMLimitGB = 10.0;      // Default 10 GB -> 7.00 GB massive_tames.bin
bool gUserSetRamLimit = false;
u64 gTargetDPs = 0;             // 0 = unlimited (stop on Ctrl+C or table full)
std::atomic<bool> gTargetReached{false};
bool gForceWipe = false;        // Only wipe existing .bin if -wipe is specified
bool gMixParityMode = false;    // -p / -mixparity: 50% Even + 50% Odd initial TAME distances
char gBinFileName[256] = "massive_tames.bin";
char gExportTxtFile[256] = {0};
FILE* gTxtFp = nullptr;
std::mutex gTxtMutex;

int gGroupCnt = V45_PNT_GROUP_CNT;
RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt = 0;
volatile long ThrCnt = 0;
u32 gGPUs_Mask[MAX_GPU_CNT];
int gRequestedGpuCount = 0;
bool gDisableGpu = false;

volatile bool gStopRequested = false;
u32 gTotalErrors = 0;
u64 PntTotalOps = 0;
volatile int PntIndex = 0;
u8* pPntList = nullptr;
u8* pPntList2 = nullptr;
CriticalSection csAddPoints;

// ============================================================================
// Hybrid CPU Kangaroo Engine Parameters & Jump Tables
// ============================================================================
#define MAX_WORKER_THREADS 256
#define MAX_CPU_BATCH_SIZE 256
#define DEFAULT_CPU_BATCH  64
#define VISITED_DP_RING    16

int gNumWorkers = 0;            // 0 = auto-detect all logical CPU cores
int gCpuThreadsActive = -1;     // -1 = auto (all worker threads also run CPU kangaroos)
int gCpuBatchSize = DEFAULT_CPU_BATCH;
u64 gVitaLimit = 8388608ULL;    // Default 2^23 steps life limit per CPU kangaroo
int gCheckpointMins = 15;       // Auto-sync .bin + .meta every 15 minutes (0 = disable)

EcJMP gEcJumps1[JMP_CNT];
EcJMP gEcJumps2[JMP_CNT];
EcJMP gEcJumps3[JMP_CNT];

// Precomputed (-J.p.y mod p) for branchless Y-parity jump inversion on CPU
alignas(64) uint64_t gJumps1_negY[JMP_CNT][4];
alignas(64) uint64_t gJumps2_negY[JMP_CNT][4];
alignas(64) uint64_t gJumps3_negY[JMP_CNT][4];

// Cacheline-padded per-worker CPU step counters (zero false sharing, from tames-gen-main)
alignas(64) volatile uint64_t gCpuWorkerSteps[MAX_WORKER_THREADS * 8] = {0};
std::atomic<u64> gGpuSessionDPs{0};
std::atomic<u64> gCpuSessionDPs{0};
std::atomic<u64> gSessionDPs{0};
std::atomic<u64> gInitialRecoveredDPs{0};
std::atomic<u64> gInitialRecoveredOps{0};
std::atomic<u64> gCpuCycleEscapes{0};
std::atomic<u64> gCpuRespawns{0};

// ============================================================================
// 3-Tier HugePage Memory Allocator (from tames-gen-main)
// Tier 1: 1GB HugePages -> Tier 2: 2MB HugePages -> Tier 3: mmap + MADV_HUGEPAGE
// ============================================================================
#ifndef MAP_HUGETLB
#define MAP_HUGETLB 0x40000
#endif
#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif
#ifndef MAP_HUGE_1GB
#define MAP_HUGE_1GB (30 << MAP_HUGE_SHIFT)
#endif

static void* hugepage_alloc(size_t size, const char* tag) {
    void* ptr = MAP_FAILED;
    if (size >= (1ULL << 30)) {
        size_t aligned_1g = (size + (1ULL << 30) - 1) & ~((1ULL << 30) - 1);
        ptr = mmap(NULL, aligned_1g, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_1GB, -1, 0);
        if (ptr != MAP_FAILED) {
            if (tag) printf("[MEM] %s: %.2f MB via 1GB HugePages\n", tag, size / 1048576.0);
            return ptr;
        }
    }
    if (size >= (2ULL << 20)) {
        size_t aligned_2m = (size + (2ULL << 20) - 1) & ~((2ULL << 20) - 1);
        ptr = mmap(NULL, aligned_2m, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (ptr != MAP_FAILED) {
            if (tag) printf("[MEM] %s: %.2f MB via 2MB HugePages\n", tag, size / 1048576.0);
            return ptr;
        }
    }
    size_t aligned_4k = (size + 4095ULL) & ~4095ULL;
    ptr = mmap(NULL, aligned_4k, PROT_READ | PROT_WRITE,
               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return nullptr;
#ifdef MADV_HUGEPAGE
    madvise(ptr, aligned_4k, MADV_HUGEPAGE);
#endif
    if (tag) printf("[MEM] %s: %.2f MB via Transparent HugePages (MADV_HUGEPAGE)\n", tag, size / 1048576.0);
    return ptr;
}

static void hugepage_free(void* ptr, size_t size) {
    if (!ptr || ptr == MAP_FAILED) return;
    if (size >= (1ULL << 30)) {
        size_t a1g = (size + (1ULL << 30) - 1) & ~((1ULL << 30) - 1);
        if (munmap(ptr, a1g) == 0) return;
    }
    if (size >= (2ULL << 20)) {
        size_t a2m = (size + (2ULL << 20) - 1) & ~((2ULL << 20) - 1);
        if (munmap(ptr, a2m) == 0) return;
    }
    size_t a4k = (size + 4095ULL) & ~4095ULL;
    munmap(ptr, a4k);
}

// ============================================================================
// Instant Disk Metadata Sidecar (<binfile>.meta) from tames-gen-main
// Restores DP count & step count in < 1 ms on resume without scanning GBs
// ============================================================================
#define META_MAGIC   0x54414D454D455432ULL // "TAMEMET2"
#define META_VERSION 2

struct DiskMeta {
    uint64_t magic;
    uint32_t version;
    uint32_t range_bits;
    uint32_t dp_bits;
    uint32_t reserved;
    uint64_t dp_count;
    uint64_t total_ops;
    uint64_t file_size;
    uint64_t start_offset[4];
};

static u64 GetTotalCpuSteps() {
    u64 sum = 0;
    for (int i = 0; i < gNumWorkers; i++) {
        sum += gCpuWorkerSteps[i * 8];
    }
    return sum;
}

static void SaveDiskMeta() {
    char meta_path[300];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", gBinFileName);
    DiskMeta m;
    memset(&m, 0, sizeof(m));
    m.magic      = META_MAGIC;
    m.version    = META_VERSION;
    m.range_bits = gRange;
    m.dp_bits    = gDP;
    m.dp_count   = gTameStore.GetWild1Count();
    m.total_ops  = gInitialRecoveredOps.load() + PntTotalOps + GetTotalCpuSteps();
    m.file_size  = gTameStoreDataSize;
    memcpy(m.start_offset, gGenStartOffset.data, 32);

    FILE* fp = fopen(meta_path, "wb");
    if (fp) {
        fwrite(&m, sizeof(m), 1, fp);
        fflush(fp);
        fsync(fileno(fp));
        fclose(fp);
    }
}

static bool LoadDiskMeta(u64& out_dp_count, u64& out_total_ops) {
    char meta_path[300];
    snprintf(meta_path, sizeof(meta_path), "%s.meta", gBinFileName);
    FILE* fp = fopen(meta_path, "rb");
    if (!fp) return false;
    DiskMeta m;
    size_t rd = fread(&m, sizeof(m), 1, fp);
    fclose(fp);
    if (rd != 1 || m.magic != META_MAGIC || m.version != META_VERSION) return false;
    if (m.file_size != gTameStoreDataSize) return false;
    if (m.range_bits != gRange || m.dp_bits != gDP) return false;
    if (memcmp(m.start_offset, gGenStartOffset.data, 32) != 0) return false;
    out_dp_count  = m.dp_count;
    out_total_ops = m.total_ops;
    return true;
}

// ============================================================================
// Lock-Free SPSC Ring Buffers + Hybrid Worker Pool with CPU Core Affinity
// ============================================================================
#pragma pack(push, 1)
struct DBRec {
    u8 x[12];
    u8 d[22];
    u8 type;
};
#pragma pack(pop)

#define RING_QUEUE_SIZE (256 * 1024)
#define RING_QUEUE_MASK (RING_QUEUE_SIZE - 1)

struct alignas(64) LockFreeRingQueue {
    DBRec* data = nullptr;
    alignas(64) std::atomic<uint64_t> head{0};
    alignas(64) std::atomic<uint64_t> tail{0};

    bool push(const DBRec& item) {
        uint64_t h = head.load(std::memory_order_relaxed);
        uint64_t t = tail.load(std::memory_order_acquire);
        if (h - t >= RING_QUEUE_SIZE) return false;
        data[h & RING_QUEUE_MASK] = item;
        head.store(h + 1, std::memory_order_release);
        return true;
    }

    bool pop(DBRec& item) {
        uint64_t t = tail.load(std::memory_order_relaxed);
        uint64_t h = head.load(std::memory_order_acquire);
        if (t == h) return false;
        item = data[t & RING_QUEUE_MASK];
        tail.store(t + 1, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return head.load(std::memory_order_acquire) == tail.load(std::memory_order_acquire);
    }
};

LockFreeRingQueue gWorkerQueues[MAX_WORKER_THREADS];
DBRec* gRingBufferPool = nullptr;
size_t gRingBufferPoolBytes = 0;
std::vector<std::thread> gWorkers;
std::atomic<bool> gWorkersRunning{false};

static void ExportPointToTxt(const u8* x_bytes, const u8* dist_bytes) {
    if (!gTxtFp) return;
    char d_hex[64];
    int pos = 0;
    for (int i = 17; i >= 0; i--) {
        sprintf(d_hex + pos, "%02X", dist_bytes[i]);
        pos += 2;
    }
    d_hex[pos] = 0;

    char x_hex[32];
    pos = 0;
    for (int i = 11; i >= 0; i--) {
        sprintf(x_hex + pos, "%02X", x_bytes[i]);
        pos += 2;
    }
    x_hex[pos] = 0;

    std::lock_guard<std::mutex> lk(gTxtMutex);
    fprintf(gTxtFp, "%s %s\n", d_hex, x_hex);
}

// ----------------------------------------------------------------------------
// -target support: exact auto-stop once the requested number of DPs is stored.
// ----------------------------------------------------------------------------
std::atomic<u64> gReservedDPs{0};

static inline void TriggerTargetStop() {
    if (!gTargetReached.exchange(true)) {
        gStopRequested = true;
        for (int i = 0; i < GpuCnt; i++) {
            if (GpuKangs[i]) GpuKangs[i]->Stop();
        }
    }
}

static inline bool CheckAndTriggerTargetStop() {
    if (gTargetDPs > 0 && gReservedDPs.load(std::memory_order_relaxed) >= gTargetDPs) {
        TriggerTargetStop();
        return true;
    }
    return false;
}

static inline bool ReserveTargetSlot() {
    if (gTargetDPs == 0) return true;
    u64 idx = gReservedDPs.fetch_add(1, std::memory_order_relaxed);
    if (idx >= gTargetDPs) {
        TriggerTargetStop();
        return false;
    }
    if (idx + 1 >= gTargetDPs) TriggerTargetStop();
    return true;
}

// ============================================================================
// Thread-Local xorshift64* PRNG + Montgomery Batch Inversion (tames-gen-main)
// ============================================================================
static __thread uint64_t tl_rng_state = 0;

static void tl_rng_seed(int worker_id) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    uint64_t s = (uint64_t)ts.tv_nsec ^ ((uint64_t)ts.tv_sec * 1000000007ULL)
               ^ ((uint64_t)(worker_id + 1) * 0x9E3779B97F4A7C15ULL)
               ^ ((uint64_t)getpid() << 32);
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        uint64_t ur = 0;
        if (read(fd, &ur, sizeof(ur)) == sizeof(ur)) s ^= ur;
        close(fd);
    }
    tl_rng_state = s ? s : 0xCAFEBABE12345678ULL;
}

static inline uint64_t tl_rng_next() {
    uint64_t x = tl_rng_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    tl_rng_state = x;
    return x * 0x2545F4914F6CDD1DULL;
}

// Generate random 192-bit TAME distance in [gGenStartOffset, gGenStartOffset + 2^gGenSpanBits)
// and compute its affine curve point (px, py) = d * G via G_stub.c's mul() + fast_modinv()
static void SpawnCpuKangaroo(uint64_t px[4], uint64_t py[4], uint64_t d[3],
                             uint8_t& l1s2, uint8_t& use_jmp3,
                             uint64_t d_hist[MD_LEN], uint8_t& hist_idx,
                             uint64_t visited_dp[VISITED_DP_RING], uint8_t& vdp_idx,
                             uint64_t& vita) {
    uint64_t rnd[4] = {0, 0, 0, 0};
    int span = (gGenSpanBits > 0) ? gGenSpanBits : ((int)gRange - 1);
    int full_limbs = span / 64;
    int rem_bits   = span % 64;

    for (int i = 0; i < full_limbs && i < 3; i++) {
        rnd[i] = tl_rng_next();
    }
    if (rem_bits > 0 && full_limbs < 3) {
        rnd[full_limbs] = tl_rng_next() & ((1ULL << rem_bits) - 1ULL);
    }
    // Keep even distance step parity unless -p / -mixparity is enabled
    if (!gMixParityMode) {
        rnd[0] &= ~1ULL;
    }
    if (rnd[0] == 0 && rnd[1] == 0 && rnd[2] == 0) rnd[0] = 2;

    // Add gGenStartOffset (192-bit addition)
    __uint128_t c = (__uint128_t)rnd[0] + gGenStartOffset.data[0];
    d[0] = (uint64_t)c; c >>= 64;
    c += (__uint128_t)rnd[1] + gGenStartOffset.data[1];
    d[1] = (uint64_t)c; c >>= 64;
    c += (__uint128_t)rnd[2] + gGenStartOffset.data[2];
    d[2] = (uint64_t)c;

    uint64_t k256[4] = {d[0], d[1], d[2], 0};
    uint64_t jx[4], jy[4], jz[4], inv_z[4];
    mul(jx, jy, jz, k256);
    fast_modinv(inv_z, jz);
    jac_to_aff(px, py, jx, jy, inv_z);

    l1s2 = 0;
    use_jmp3 = 0;
    hist_idx = 0;
    vdp_idx = 0;
    vita = 0;
    for (int k = 0; k < MD_LEN; k++) d_hist[k] = 0;
    for (int k = 0; k < VISITED_DP_RING; k++) visited_dp[k] = 0;
}

// Montgomery Batch Inversion using Jean-Luc Pons DRS62 fast_modinv (from tames-gen-main)
static inline void batch_inv_with_scratch(uint64_t inv[][4], const uint64_t val[][4],
                                          int n, uint64_t prefix[][4]) {
    memcpy(prefix[0], val[0], 32);
    for (int i = 1; i < n; i++) {
        aecc_mul(prefix[i], prefix[i - 1], val[i]);
    }
    uint64_t total_inv[4];
    fast_modinv(total_inv, prefix[n - 1]);

    for (int i = n - 1; i > 0; i--) {
        aecc_mul(inv[i], total_inv, prefix[i - 1]);
        aecc_mul(total_inv, total_inv, val[i]);
    }
    memcpy(inv[0], total_inv, 32);
}

// Helper to drain all currently queued GPU DPs for a worker
static inline bool DrainWorkerQueue(int worker_id) {
    DBRec rec;
    bool did_work = false;
    while (gWorkerQueues[worker_id].pop(rec)) {
        did_work = true;
        if (!ReserveTargetSlot()) continue;

        u8 x_bytes[16];
        u8 dist_bytes[24];
        memcpy(x_bytes, rec.x, 12);
        memset(x_bytes + 12, 0, 4);
        memcpy(dist_bytes, rec.d, 22);
        dist_bytes[22] = TAME;
        dist_bytes[23] = 0;

        gTameStore.AddTame(x_bytes, dist_bytes);
        gGpuSessionDPs.fetch_add(1, std::memory_order_relaxed);
        gSessionDPs.fetch_add(1, std::memory_order_relaxed);
        if (gTxtFp) ExportPointToTxt(x_bytes, dist_bytes);
    }
    return did_work;
}

// ============================================================================
// Hybrid Worker Thread:
//   1. Drains GPU DPs from lock-free ring buffer (zero contention)
//   2. Steps a batch of B SOTA TAME Kangaroos on CPU using BMI2/ADX + DRS62
// ============================================================================
void WorkerThread(int worker_id) {
#ifndef _WIN32
    int num_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cpus > 0) {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(worker_id % num_cpus, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    }
    // When GPUs are active, give CPU kangaroo workers a polite nice value (+5)
    // so GPU kernel management threads (kang_thr_proc) preempt instantly with 0 latency.
    if (GpuCnt > 0) {
        setpriority(PRIO_PROCESS, 0, 5);
    }
#endif

    const bool run_cpu_kangs = (worker_id < gCpuThreadsActive);
    const int B = gCpuBatchSize;

    // Allocate aligned per-thread batch state on stack/heap if CPU kangaroos enabled
    uint64_t (*px)[4]         = nullptr;
    uint64_t (*py)[4]         = nullptr;
    uint64_t (*dist)[3]       = nullptr;
    uint64_t (*dx)[4]         = nullptr;
    uint64_t (*inv_dx)[4]     = nullptr;
    uint64_t (*scratch)[4]    = nullptr;
    const uint64_t (**jx_ptr) = nullptr;
    const uint64_t (**jy_ptr) = nullptr;
    uint32_t* jmp_cmd         = nullptr;
    uint8_t*  l1s2            = nullptr;
    uint8_t*  use_jmp3        = nullptr;
    uint64_t (*d_hist)[MD_LEN]= nullptr;
    uint8_t*  hist_idx        = nullptr;
    uint64_t (*visited_dp)[VISITED_DP_RING] = nullptr;
    uint8_t*  vdp_idx         = nullptr;
    uint64_t* vita            = nullptr;

    if (run_cpu_kangs && B > 0) {
        tl_rng_seed(worker_id);
        px         = (uint64_t(*)[4])aligned_alloc(64, B * 32);
        py         = (uint64_t(*)[4])aligned_alloc(64, B * 32);
        dist       = (uint64_t(*)[3])aligned_alloc(64, B * 24);
        dx         = (uint64_t(*)[4])aligned_alloc(64, B * 32);
        inv_dx     = (uint64_t(*)[4])aligned_alloc(64, B * 32);
        scratch    = (uint64_t(*)[4])aligned_alloc(64, B * 32);
        jx_ptr     = (const uint64_t**)malloc(B * sizeof(uint64_t*));
        jy_ptr     = (const uint64_t**)malloc(B * sizeof(uint64_t*));
        jmp_cmd    = (uint32_t*)malloc(B * sizeof(uint32_t));
        l1s2       = (uint8_t*)malloc(B);
        use_jmp3   = (uint8_t*)malloc(B);
        d_hist     = (uint64_t(*)[MD_LEN])calloc(B, sizeof(uint64_t[MD_LEN]));
        hist_idx   = (uint8_t*)calloc(B, 1);
        visited_dp = (uint64_t(*)[VISITED_DP_RING])calloc(B, sizeof(uint64_t[VISITED_DP_RING]));
        vdp_idx    = (uint8_t*)calloc(B, 1);
        vita       = (uint64_t*)calloc(B, sizeof(uint64_t));

        for (int i = 0; i < B; i++) {
            SpawnCpuKangaroo(px[i], py[i], dist[i], l1s2[i], use_jmp3[i],
                             d_hist[i], hist_idx[i], visited_dp[i], vdp_idx[i], vita[i]);
        }
    }

    int idle_spins = 0;
    while (gWorkersRunning.load(std::memory_order_relaxed) || !gWorkerQueues[worker_id].empty()) {
        // Priority 1: Always drain any pending GPU DPs from our lock-free ring buffer
        bool drained = DrainWorkerQueue(worker_id);

        if (!run_cpu_kangs || gStopRequested || gTargetReached.load(std::memory_order_relaxed)) {
            if (!drained) {
                if (!gWorkersRunning.load(std::memory_order_relaxed)) break;
                if (++idle_spins < 64) {
                    asm volatile("pause" ::: "memory");
                } else {
                    usleep(200);
                }
            } else {
                idle_spins = 0;
            }
            continue;
        }

        // Priority 2: Step B SOTA TAME Kangaroos on this CPU core (32 steps per burst)
        const int STEPS_PER_BURST = 32;
        for (int step = 0; step < STEPS_PER_BURST; step++) {
            // Phase 1: Select jump point (Jumps1, Jumps2 for L1S2, or Jumps3 for loop escape),
            // apply Y-parity SOTA sign inversion, update 192-bit distance, and compute dx[i]
            for (int i = 0; i < B; i++) {
                u32 j_ind = (u32)(px[i][0] % JMP_CNT);
                u32 inv_flag = (u32)(py[i][0] & 1ULL);
                const uint64_t* d_jmp;

                if (__builtin_expect(use_jmp3[i], 0)) {
                    jx_ptr[i] = (const uint64_t*)gEcJumps3[j_ind].p.x.data;
                    jy_ptr[i] = inv_flag ? (const uint64_t*)gJumps3_negY[j_ind] : (const uint64_t*)gEcJumps3[j_ind].p.y.data;
                    d_jmp     = (const uint64_t*)gEcJumps3[j_ind].dist.data;
                    jmp_cmd[i]= j_ind | (inv_flag ? INV_FLAG : 0) | JMP2_FLAG;
                    use_jmp3[i] = 0;
                    l1s2[i] = 0;
                } else if (l1s2[i]) {
                    jx_ptr[i] = (const uint64_t*)gEcJumps2[j_ind].p.x.data;
                    jy_ptr[i] = inv_flag ? (const uint64_t*)gJumps2_negY[j_ind] : (const uint64_t*)gEcJumps2[j_ind].p.y.data;
                    d_jmp     = (const uint64_t*)gEcJumps2[j_ind].dist.data;
                    jmp_cmd[i]= j_ind | (inv_flag ? INV_FLAG : 0) | JMP2_FLAG;
                } else {
                    jx_ptr[i] = (const uint64_t*)gEcJumps1[j_ind].p.x.data;
                    jy_ptr[i] = inv_flag ? (const uint64_t*)gJumps1_negY[j_ind] : (const uint64_t*)gEcJumps1[j_ind].p.y.data;
                    d_jmp     = (const uint64_t*)gEcJumps1[j_ind].dist.data;
                    jmp_cmd[i]= j_ind | (inv_flag ? INV_FLAG : 0);
                }

                // Update 192-bit scalar distance: d += d_jmp (if y even) or d -= d_jmp (if y odd)
                if (inv_flag) {
                    __uint128_t b = 0, diff;
                    diff = (__uint128_t)dist[i][0] - d_jmp[0]; dist[i][0] = (uint64_t)diff; b = diff >> 127;
                    diff = (__uint128_t)dist[i][1] - d_jmp[1] - b; dist[i][1] = (uint64_t)diff; b = diff >> 127;
                    diff = (__uint128_t)dist[i][2] - d_jmp[2] - b; dist[i][2] = (uint64_t)diff;
                } else {
                    __uint128_t c = (__uint128_t)dist[i][0] + d_jmp[0];
                    dist[i][0] = (uint64_t)c; c >>= 64;
                    c += (__uint128_t)dist[i][1] + d_jmp[1];
                    dist[i][1] = (uint64_t)c; c >>= 64;
                    c += (__uint128_t)dist[i][2] + d_jmp[2];
                    dist[i][2] = (uint64_t)c;
                }

                aecc_sub(dx[i], px[i], jx_ptr[i]);
            }

            // Phase 2: Single DRS62 Modular Inversion for all B kangaroos via Montgomery trick
            batch_inv_with_scratch(inv_dx, dx, B, scratch);

            // Phase 3: Complete affine point addition, update L1S2 & cycle history, and check DPs
            for (int i = 0; i < B; i++) {
                uint64_t dy[4], lambda[4], l2[4], nx[4], ny[4], dx2[4];
                aecc_sub(dy, py[i], jy_ptr[i]);
                aecc_mul(lambda, dy, inv_dx[i]);
                aecc_sqr(l2, lambda);
                aecc_sub(nx, l2, jx_ptr[i]);
                aecc_sub(nx, nx, px[i]);
                aecc_sub(dx2, px[i], nx);
                aecc_mul(ny, lambda, dx2);
                aecc_sub(py[i], ny, py[i]);
                memcpy(px[i], nx, 32);

                // Update L1S2 2-cycle escape state (exact match to RCGpuCore.cu lines 388-398)
                if (!(jmp_cmd[i] & JMP2_FLAG)) {
                    u32 jmp_next = (u32)(px[i][0] % JMP_CNT);
                    jmp_next |= ((u32)py[i][0] & 1ULL) ? 0 : INV_FLAG;
                    l1s2[i] = (jmp_cmd[i] == jmp_next) ? 1 : 0;
                } else {
                    l1s2[i] = 0;
                }

                // Short-cycle detection on d[0] (MD_LEN = 10, exact match to KernelB)
                uint64_t cur_d0 = dist[i][0];
                bool short_loop = false;
                for (int k = 0; k < MD_LEN; k++) {
                    if (d_hist[i][k] == cur_d0 && cur_d0 != 0) {
                        short_loop = true;
                        break;
                    }
                }
                d_hist[i][hist_idx[i]] = cur_d0;
                hist_idx[i] = (hist_idx[i] + 1) % MD_LEN;

                if (__builtin_expect(short_loop, 0)) {
                    // Escape short cycle on next step using EcJumps3
                    use_jmp3[i] = 1;
                    gCpuCycleEscapes.fetch_add(1, std::memory_order_relaxed);
                    continue;
                }

                // Check Distinguished Point condition: top gDP bits of X == 0
                if (__builtin_expect(IS_DP(px[i][3], gDP), 0)) {
                    // Check visited_dp ring for macro-cycle (from tames-gen-main)
                    uint64_t dp_sig = px[i][0] ^ px[i][1];
                    bool dp_loop = false;
                    for (int k = 0; k < VISITED_DP_RING; k++) {
                        if (visited_dp[i][k] == dp_sig && dp_sig != 0) {
                            dp_loop = true;
                            break;
                        }
                    }
                    if (dp_loop) {
                        gCpuCycleEscapes.fetch_add(1, std::memory_order_relaxed);
                        gCpuRespawns.fetch_add(1, std::memory_order_relaxed);
                        SpawnCpuKangaroo(px[i], py[i], dist[i], l1s2[i], use_jmp3[i],
                                         d_hist[i], hist_idx[i], visited_dp[i], vdp_idx[i], vita[i]);
                        continue;
                    }
                    visited_dp[i][vdp_idx[i]] = dp_sig;
                    vdp_idx[i] = (vdp_idx[i] + 1) % VISITED_DP_RING;

                    if (ReserveTargetSlot()) {
                        u8 x_bytes[16];
                        u8 dist_bytes[24];
                        memcpy(x_bytes, px[i], 12);
                        memset(x_bytes + 12, 0, 4);
                        memcpy(dist_bytes, dist[i], 22);
                        dist_bytes[22] = TAME;
                        dist_bytes[23] = 0;

                        gTameStore.AddTame(x_bytes, dist_bytes);
                        gCpuSessionDPs.fetch_add(1, std::memory_order_relaxed);
                        gSessionDPs.fetch_add(1, std::memory_order_relaxed);
                        if (gTxtFp) ExportPointToTxt(x_bytes, dist_bytes);
                    }
                }

                // Check vita life limit (from tames-gen-main)
                if (++vita[i] >= gVitaLimit) {
                    gCpuRespawns.fetch_add(1, std::memory_order_relaxed);
                    SpawnCpuKangaroo(px[i], py[i], dist[i], l1s2[i], use_jmp3[i],
                                     d_hist[i], hist_idx[i], visited_dp[i], vdp_idx[i], vita[i]);
                }
            }
        }

        // Update cacheline-padded step counter once per burst (zero false sharing)
        gCpuWorkerSteps[worker_id * 8] += (uint64_t)B * STEPS_PER_BURST;
    }

    if (run_cpu_kangs && B > 0) {
        free(px);
        free(py);
        free(dist);
        free(dx);
        free(inv_dx);
        free(scratch);
        free(jx_ptr);
        free(jy_ptr);
        free(jmp_cmd);
        free(l1s2);
        free(use_jmp3);
        free(d_hist);
        free(hist_idx);
        free(visited_dp);
        free(vdp_idx);
        free(vita);
    }
}

void StartWorkerThreads() {
    if (gWorkersRunning.load()) return;
    gWorkersRunning.store(true);
    gWorkers.clear();
    for (int i = 0; i < gNumWorkers; i++) {
        gWorkers.emplace_back(WorkerThread, i);
    }
}

void StopWorkerThreads() {
    if (!gWorkersRunning.load()) return;
    gWorkersRunning.store(false);
    for (auto& t : gWorkers) {
        if (t.joinable()) t.join();
    }
    gWorkers.clear();
}

void SigIntHandler(int sig) {
    if (!gStopRequested) {
        gStopRequested = true;
        printf("\n\n[!] Ctrl+C detected! Stopping GPUs & CPU workers and syncing points safely to %s...\n", gBinFileName);
        for (int i = 0; i < GpuCnt; i++) {
            if (GpuKangs[i]) GpuKangs[i]->Stop();
        }
    } else {
        printf("\n[!] Force exit!\n");
        _exit(1);
    }
}

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt) {
    csAddPoints.Enter();
    if (pnt_cnt > 0 && data) {
        if (PntIndex + pnt_cnt >= MAX_CNT_LIST) {
            csAddPoints.Leave();
            return;
        }
        memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
        PntIndex += pnt_cnt;
    }
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

void ProcessNewPoints() {
    csAddPoints.Enter();
    if (!PntIndex) {
        csAddPoints.Leave();
        return;
    }
    int cnt = PntIndex;
    u8* localBuffer = pPntList2;
    memcpy(localBuffer, pPntList, GPU_DP_SIZE * cnt);
    PntIndex = 0;
    csAddPoints.Leave();

    if (CheckAndTriggerTargetStop()) return;

    if (gWorkersRunning.load() && cnt > 64) {
        static int current_worker = 0;
        for (int i = 0; i < cnt; i++) {
            if (gTargetReached.load(std::memory_order_relaxed)) break;
            DBRec nrec;
            u8* p = localBuffer + i * GPU_DP_SIZE;
            memcpy(nrec.x, p, 12);
            memcpy(nrec.d, p + 16, 22);
            nrec.type = TAME;

            int spins = 0;
            while (!gWorkerQueues[current_worker].push(nrec)) {
                current_worker = (current_worker + 1) % gNumWorkers;
                if (++spins >= gNumWorkers) {
                    std::this_thread::yield();
                    spins = 0;
                }
            }
            current_worker = (current_worker + 1) % gNumWorkers;
        }
    } else {
        for (int i = 0; i < cnt; i++) {
            if (!ReserveTargetSlot()) break;

            u8* p = localBuffer + i * GPU_DP_SIZE;
            u8 x_bytes[16];
            u8 dist_bytes[24];
            memcpy(x_bytes, p, 12);
            memset(x_bytes + 12, 0, 4);
            memcpy(dist_bytes, p + 16, 22);
            dist_bytes[22] = TAME;
            dist_bytes[23] = 0;

            gTameStore.AddTame(x_bytes, dist_bytes);
            gGpuSessionDPs.fetch_add(1, std::memory_order_relaxed);
            gSessionDPs.fetch_add(1, std::memory_order_relaxed);
            if (gTxtFp) ExportPointToTxt(x_bytes, dist_bytes);
        }
    }
}

void* kang_thr_proc(void* data) {
    RCGpuKang* Kang = (RCGpuKang*)data;
    Kang->Execute();
    __sync_fetch_and_sub(&ThrCnt, 1);
    return nullptr;
}

static int GetBitLength(const EcInt& val) {
    for (int limb = 4; limb >= 0; limb--) {
        if (val.data[limb] != 0) {
            return limb * 64 + (64 - __builtin_clzll(val.data[limb]));
        }
    }
    return 0;
}

void InitGpus() {
    GpuCnt = 0;
    if (gDisableGpu) {
        printf("GPU Mode: Disabled (-gpu none). Running in 100%% CPU Assembly Mode.\n");
        return;
    }
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT) gcnt = MAX_GPU_CNT;
    if (!gcnt) return;

    if (gRequestedGpuCount > 0) {
        memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
        for (int i = 0; i < gcnt && i < gRequestedGpuCount; i++) {
            gGPUs_Mask[i] = 1;
        }
    }

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    printf("CUDA Devices Detected: %d (Driver %d.%d / Runtime %d.%d)\n",
           gcnt, drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    for (int i = 0; i < gcnt; i++) {
        if (!gGPUs_Mask[i]) continue;
        if (cudaSetDevice(i) != cudaSuccess) continue;

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        if (prop.major < 6) continue;

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = prop.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = prop.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = prop.l2CacheSize < 16 * 1024 * 1024;
        GpuKangs[GpuCnt]->GroupCnt = gGroupCnt;

        printf("  [GPU %d] %s | %.2f GB VRAM | %d SMs | Cap %d.%d | %d Kangaroos (100%% TAME)\n",
               i, prop.name, (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.multiProcessorCount, prop.major, prop.minor,
               GpuKangs[GpuCnt]->CalcKangCnt());
        GpuCnt++;
    }
}

void ShowDashboard(u64 tm_start, u64 total_gpu_kangs, const char* start_hex_str, int jump_shift) {
    static u64 prev_cpu_steps = 0;
    static u64 prev_tm_ms = 0;

    int gpu_speed_mkeys = 0;
    for (int i = 0; i < GpuCnt; i++) {
        gpu_speed_mkeys += GpuKangs[i]->GetStatsSpeed();
    }

    u64 now_ms = GetTickCount64();
    u64 elapsed_ms = now_ms - tm_start;
    double elapsed_sec = (elapsed_ms > 0) ? (elapsed_ms / 1000.0) : 0.001;

    u64 cur_cpu_steps = GetTotalCpuSteps();
    double cpu_mkeys_s = 0.0;
    if (prev_tm_ms > 0 && now_ms > prev_tm_ms) {
        double dt = (now_ms - prev_tm_ms) / 1000.0;
        cpu_mkeys_s = (double)(cur_cpu_steps - prev_cpu_steps) / (dt * 1e6);
    } else if (elapsed_sec > 0.1) {
        cpu_mkeys_s = (double)cur_cpu_steps / (elapsed_sec * 1e6);
    }
    prev_cpu_steps = cur_cpu_steps;
    prev_tm_ms = now_ms;

    double total_mkeys_s = (double)gpu_speed_mkeys + cpu_mkeys_s;
    double total_gkeys_s = total_mkeys_s / 1000.0;

    u64 sec = (u64)elapsed_sec;
    u64 days = sec / 86400;
    int hours = (int)((sec % 86400) / 3600);
    int mins  = (int)((sec % 3600) / 60);
    int secs  = (int)(sec % 60);

    u64 session_dps = gSessionDPs.load();
    u64 gpu_dps     = gGpuSessionDPs.load();
    u64 cpu_dps     = gCpuSessionDPs.load();
    u64 total_stored = gTameStore.GetWild1Count();
    u64 table_cap = gTameStore.GetPerTableSize();
    u64 dups = gTameStore.GetDuplicatePoints();
    u64 overflows = gTameStore.GetSpatialOverwrites();
    u64 escapes = gCpuCycleEscapes.load();

    double fill_pct = (table_cap > 0) ? ((double)total_stored * 100.0 / (double)table_cap) : 0.0;
    double dps_per_sec = (elapsed_sec > 0.5) ? ((double)session_dps / elapsed_sec) : 0.0;
    double theo_dps_sec = (total_mkeys_s * 1000000.0) / pow(2.0, (double)gDP);

    u64 total_session_ops = PntTotalOps + cur_cpu_steps;
    double covered_steps_log2 = (total_stored > 0) ? (log2((double)total_stored) + (double)gDP) : 0.0;
    double session_ops_log2 = (total_session_ops > 0) ? log2((double)total_session_ops) : 0.0;

    printf("================================================================================\n");
    printf(" [POINT HYBRID v2] Total: %.3f GKeys/s (GPU: %.2f GK/s + CPU: %.1f MK/s) | %llud %02dh %02dm %02ds\n",
           total_gkeys_s, gpu_speed_mkeys / 1000.0, cpu_mkeys_s,
           (unsigned long long)days, hours, mins, secs);
    printf("--------------------------------------------------------------------------------\n");
    printf("  Hardware Active: %d GPU(s) (%llu Kangs) + %d CPU Threads (%d Batch = %d Kangs)\n",
           GpuCnt, (unsigned long long)total_gpu_kangs,
           gCpuThreadsActive, gCpuBatchSize, gCpuThreadsActive * gCpuBatchSize);
    printf("  Range & DP     : Range %u bits (Span 2^%d) | DP %u (1 DP / 2^%u jumps) | Shift 2^%d\n",
           gRange, gGenSpanBits, gDP, gDP, jump_shift);
    printf("  Start Offset   : 0x%s\n", start_hex_str);
    printf("  Stored in .bin : %llu DPs (%.3fM) | New: +%llu (GPU: %llu | CPU: %llu) | Dups: %llu\n",
           (unsigned long long)total_stored, total_stored / 1000000.0,
           (unsigned long long)session_dps, (unsigned long long)gpu_dps,
           (unsigned long long)cpu_dps, (unsigned long long)dups);
    if (gTargetDPs > 0) {
        double target_pct = (double)total_stored * 100.0 / (double)gTargetDPs;
        if (target_pct > 100.0) target_pct = 100.0;
        u64 remaining = (total_stored < gTargetDPs) ? (gTargetDPs - total_stored) : 0;
        double rate_for_eta = (dps_per_sec > 0.01) ? dps_per_sec : theo_dps_sec;
        u64 eta_sec = (rate_for_eta > 0.001) ? (u64)(remaining / rate_for_eta) : 0;
        u64 eta_d = eta_sec / 86400;
        int eta_h = (int)((eta_sec % 86400) / 3600);
        int eta_m = (int)((eta_sec % 3600) / 60);
        int eta_s = (int)(eta_sec % 60);
        printf("  Target Goal    : %llu / %llu DPs (%.2f%%) | Rem: %llu | ETA: %llud %02dh %02dm %02ds\n",
               (unsigned long long)total_stored, (unsigned long long)gTargetDPs, target_pct,
               (unsigned long long)remaining, (unsigned long long)eta_d, eta_h, eta_m, eta_s);
    }
    printf("  Table Capacity : %llu / %llu slots (%.4f%% full) | File: %s (%.2f GB)\n",
           (unsigned long long)total_stored, (unsigned long long)table_cap,
           fill_pct, gBinFileName, (double)gTameStoreDataSize / (1024.0 * 1024.0 * 1024.0));
    printf("  DP Gen Rate    : %.2f DPs/s (Theo: %.2f/s) | %.0f DPs/min | %.0f DPs/hour\n",
           dps_per_sec, theo_dps_sec, dps_per_sec * 60.0, dps_per_sec * 3600.0);
    printf("  Step Coverage  : 2^%.2f steps in .bin | Session Ops: 2^%.2f (CPU Escapes: %llu)\n",
           covered_steps_log2, session_ops_log2, (unsigned long long)escapes);
    if (overflows > 0) {
        printf("  [!] Probe Full : %llu overflows (table nearing capacity)\n", (unsigned long long)overflows);
    }
    printf("================================================================================\n");
    fflush(stdout);
}

static u64 ParsePointCount(const char* str) {
    if (!str || !*str) return 0;
    char* endptr = nullptr;
    double val = strtod(str, &endptr);
    if (val <= 0.0) return 0;
    if (endptr && *endptr) {
        char c = *endptr;
        if (c == 'k' || c == 'K') val *= 1e3;
        else if (c == 'm' || c == 'M') val *= 1e6;
        else if (c == 'b' || c == 'B' || c == 'g' || c == 'G') val *= 1e9;
        else if (c == 't' || c == 'T') val *= 1e12;
    }
    return (u64)(val + 0.5);
}

void PrintUsage(const char* prog) {
    printf("\nUsage:\n");
    printf("  %s -range <BITS> -start <HEX_OFFSET> -gpu <GPU_IDS_OR_COUNT> [options]\n\n", prog);
    printf("Required / Core Parameters:\n");
    printf("  -range <N>       Target puzzle range in bits (e.g. 100 for Puzzle 100)\n");
    printf("  -start <HEX>     Range start offset in hex (or 0 for default 2^(Range-1))\n");
    printf("  -gpu <IDs>       GPU indices (e.g. -gpu 0, -gpu 0123), count (-gpu 10), or -gpu none\n\n");
    printf("Hybrid CPU + Pro Parameters (integrated from tames-gen-main):\n");
    printf("  -w <N> / -cpu <N> Number of CPU threads generating DPs (default: ALL logical cores)\n");
    printf("                   Pass -cpu 0 to disable CPU kangaroo stepping (GPU-only mode)\n");
    printf("  -b <N>           CPU batch inversion size per thread (1..256, default: 64)\n");
    printf("  -vita <STEPS>    Max life steps per CPU kangaroo before respawn (default: 8388608)\n");
    printf("  -checkpoint <M>  Non-blocking disk msync + .meta save interval in minutes (default: 15)\n\n");
    printf("Target & Storage Parameters:\n");
    printf("  -target <COUNT>  Stop automatically once stored DPs reach <COUNT>\n");
    printf("                   Supports suffixes: K, M, B (e.g. -target 100M, -target 1B, -target 50000000)\n");
    printf("                   (Aliases: -points <COUNT>, -maxpts <COUNT>, -count <COUNT>, -n <COUNT>)\n");
    printf("  -gpus <N>        Use the first N GPUs (e.g. -gpus 10 for GPUs 0..9)\n");
    printf("  -dp <N>          Distinguished Point bits (6..60, default: 28)\n");
    printf("  -ramlimit <GB>   Size allocation for new .bin file in GB (default: 10 -> 7.00 GB .bin)\n");
    printf("  -out <FILE.bin>  Output binary filename (default: massive_tames.bin)\n");
    printf("  -txt <FILE.txt>  Also export generated DPs as text (<dist_hex> <x_hex>)\n");
    printf("  -p / -mixparity  Enable 50%% Even + 50%% Odd initial TAME distances\n");
    printf("  -wipe            Force wipe existing .bin file and start from 0 points\n\n");
}

bool ParseCmdLine(int argc, char* argv[]) {
    for (int i = 0; i < MAX_GPU_CNT; i++) gGPUs_Mask[i] = 1; // Default all GPUs

    int ci = 1;
    while (ci < argc) {
        char* arg = argv[ci++];
        if (strcmp(arg, "-gpu") == 0) {
            if (ci >= argc) return false;
            char* gstr = argv[ci++];
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            if (strcmp(gstr, "none") == 0 || strcmp(gstr, "off") == 0 || strcmp(gstr, "-1") == 0) {
                gDisableGpu = true;
            } else {
                int len = (int)strlen(gstr);
                if (gstr[0] == '0') {
                    for (int k = 0; k < len; k++) {
                        if (gstr[k] >= '0' && gstr[k] <= '9') {
                            int idx = gstr[k] - '0';
                            if (idx < MAX_GPU_CNT) gGPUs_Mask[idx] = 1;
                        }
                    }
                } else {
                    gRequestedGpuCount = atoi(gstr);
                    if (gRequestedGpuCount < 1) gRequestedGpuCount = 1;
                }
            }
        } else if (strcmp(arg, "-gpus") == 0) {
            if (ci >= argc) return false;
            gRequestedGpuCount = atoi(argv[ci++]);
            if (gRequestedGpuCount == 0) gDisableGpu = true;
        } else if (strcmp(arg, "-w") == 0 || strcmp(arg, "-cpu") == 0 || strcmp(arg, "-threads") == 0) {
            if (ci >= argc) return false;
            char* wstr = argv[ci++];
            if (strcmp(wstr, "all") == 0 || strcmp(wstr, "auto") == 0) {
                gCpuThreadsActive = -1;
            } else {
                gCpuThreadsActive = atoi(wstr);
                if (gCpuThreadsActive < 0) gCpuThreadsActive = 0;
                if (gCpuThreadsActive > MAX_WORKER_THREADS) gCpuThreadsActive = MAX_WORKER_THREADS;
            }
        } else if (strcmp(arg, "-b") == 0 || strcmp(arg, "-batch") == 0) {
            if (ci >= argc) return false;
            gCpuBatchSize = atoi(argv[ci++]);
            if (gCpuBatchSize < 1) gCpuBatchSize = 1;
            if (gCpuBatchSize > MAX_CPU_BATCH_SIZE) gCpuBatchSize = MAX_CPU_BATCH_SIZE;
        } else if (strcmp(arg, "-vita") == 0 || strcmp(arg, "--vita") == 0) {
            if (ci >= argc) return false;
            gVitaLimit = ParsePointCount(argv[ci++]);
            if (gVitaLimit < 1024) gVitaLimit = 1024;
        } else if (strcmp(arg, "-checkpoint") == 0) {
            if (ci >= argc) return false;
            gCheckpointMins = atoi(argv[ci++]);
        } else if (strcmp(arg, "-range") == 0) {
            if (ci >= argc) return false;
            gRange = (u32)atoi(argv[ci++]);
        } else if (strcmp(arg, "-start") == 0) {
            if (ci >= argc) return false;
            if (!gStart.SetHexStr(argv[ci++])) return false;
            gStartSet = true;
        } else if (strcmp(arg, "-dp") == 0) {
            if (ci >= argc) return false;
            gDP = (u32)atoi(argv[ci++]);
        } else if (strcmp(arg, "-target") == 0 || strcmp(arg, "-points") == 0 ||
                   strcmp(arg, "-maxpts") == 0 || strcmp(arg, "-count") == 0 ||
                   strcmp(arg, "-n") == 0) {
            if (ci >= argc) return false;
            gTargetDPs = ParsePointCount(argv[ci++]);
            if (gTargetDPs == 0) {
                printf("[!] Error: Invalid point count for %s\n", arg);
                return false;
            }
        } else if (strcmp(arg, "-ramlimit") == 0) {
            if (ci >= argc) return false;
            gRAMLimitGB = atof(argv[ci++]);
            gUserSetRamLimit = true;
        } else if (strcmp(arg, "-out") == 0) {
            if (ci >= argc) return false;
            strncpy(gBinFileName, argv[ci++], sizeof(gBinFileName) - 1);
        } else if (strcmp(arg, "-txt") == 0) {
            if (ci >= argc) return false;
            strncpy(gExportTxtFile, argv[ci++], sizeof(gExportTxtFile) - 1);
        } else if (strcmp(arg, "-wipe") == 0) {
            gForceWipe = true;
        } else if (strcmp(arg, "-resumenvme") == 0) {
            gForceWipe = false;
        } else if (strcmp(arg, "-p") == 0 || strcmp(arg, "-mixparity") == 0) {
            if (ci < argc && (strcmp(argv[ci], "0") == 0 || strcmp(argv[ci], "1") == 0)) {
                gMixParityMode = (atoi(argv[ci++]) != 0);
            } else {
                gMixParityMode = true;
            }
        } else {
            printf("Unknown argument: %s\n", arg);
            return false;
        }
    }

    if (gRange < 32 || gRange > 170) {
        printf("[!] Error: -range <32..170> is required.\n");
        return false;
    }
    if (!gStartSet) {
        gStart.SetZero();
        gStartSet = true;
    }
    return true;
}

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("================================================================================\n");
    printf(" ⚡ POINT v2.0 — Hybrid GPU + All-Core CPU Distinguished Point (TAME) Generator\n");
    printf("    CUDA SOTA + BMI2/ADX Assembly + DRS62 Batch Inv + HugePages + Instant .meta\n");
    printf("================================================================================\n");

    if (!ParseCmdLine(argc, argv)) {
        PrintUsage(argv[0]);
        return 1;
    }

    InitEc();
    aecc_init();

    // Detect CPU logical cores and configure hybrid worker pool
    int detected_cpus = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (detected_cpus < 1) detected_cpus = 8;
    if (detected_cpus > MAX_WORKER_THREADS) detected_cpus = MAX_WORKER_THREADS;

    if (gCpuThreadsActive < 0) {
        // Default: activate ALL logical CPU cores/threads!
        gCpuThreadsActive = detected_cpus;
    }
    gNumWorkers = (gCpuThreadsActive > detected_cpus) ? gCpuThreadsActive : detected_cpus;
    if (gNumWorkers < 1) gNumWorkers = 1;

    // Determine exact Start Offset, Span Bits, and Jump Table Range
    if (gStart.IsZero()) {
        gGenStartOffset.Set(1);
        gGenStartOffset.ShiftLeft(gRange - 1);
        gGenSpanBits = (int)gRange - 1;
    } else {
        gGenStartOffset = gStart;
        int sbits = GetBitLength(gStart);
        if (sbits == (int)gRange) {
            gGenSpanBits = (int)gRange - 1;
        } else if (sbits == (int)gRange + 1) {
            gGenSpanBits = (int)gRange;
            gRange = (u32)sbits;
        } else {
            gGenSpanBits = (int)gRange - 1;
        }
    }

    InitGpus();
    if (!GpuCnt && gCpuThreadsActive == 0) {
        printf("[!] Error: Neither GPUs nor CPU Kangaroo threads are active.\n");
        return 1;
    }

    printf("[CPU] Activated %d / %d logical CPU threads | Batch Size: %d (%d CPU Kangaroos) | BMI2/ADX + DRS62\n",
           gCpuThreadsActive, detected_cpus, gCpuBatchSize, gCpuThreadsActive * gCpuBatchSize);

    // Allocate lock-free ring buffers via 3-Tier HugePage Allocator
    gRingBufferPoolBytes = (size_t)gNumWorkers * RING_QUEUE_SIZE * sizeof(DBRec);
    gRingBufferPool = (DBRec*)hugepage_alloc(gRingBufferPoolBytes, "Lock-Free Ring Queues");
    if (!gRingBufferPool) {
        printf("[!] Error: Failed to allocate lock-free ring buffer pool\n");
        return 1;
    }
    for (int i = 0; i < gNumWorkers; i++) {
        gWorkerQueues[i].data = gRingBufferPool + (size_t)i * RING_QUEUE_SIZE;
    }

    if (GpuCnt > 0) {
        cudaHostAlloc((void**)&pPntList, MAX_CNT_LIST * GPU_DP_SIZE, cudaHostAllocDefault);
        cudaHostAlloc((void**)&pPntList2, MAX_CNT_LIST * GPU_DP_SIZE, cudaHostAllocDefault);
    } else {
        pPntList  = (u8*)aligned_alloc(64, MAX_CNT_LIST * GPU_DP_SIZE);
        pPntList2 = (u8*)aligned_alloc(64, MAX_CNT_LIST * GPU_DP_SIZE);
    }

    if (gExportTxtFile[0] != 0) {
        gTxtFp = fopen(gExportTxtFile, "a");
        if (gTxtFp) {
            printf("[*] Text export enabled: appending DPs to '%s'\n", gExportTxtFile);
        }
    }

    // ========================================================================
    // Setup or Resume massive_tames.bin (NVMe mmap + MADV_HUGEPAGE + .meta)
    // ========================================================================
    double store_ram = gRAMLimitGB - 3.0;
    if (store_ram < 4.0) store_ram = 4.0;
    if (gTargetDPs > 0 && !gUserSetRamLimit) {
        double needed_gb = ceil(((double)gTargetDPs * 16.0 / 0.80) / (1024.0 * 1024.0 * 1024.0) + 1.0);
        if (needed_gb > store_ram) store_ram = needed_gb;
    }

    struct stat st;
    bool file_exists = (stat(gBinFileName, &st) == 0 && st.st_size >= (off_t)(4ULL << 30));
    uint64_t massive_db_bytes = 0;

    if (file_exists && !gForceWipe) {
        massive_db_bytes = (uint64_t)st.st_size;
        store_ram = (double)massive_db_bytes / (1024.0 * 1024.0 * 1024.0);
        printf("[*] Found existing '%s' (%.2f GB). Resuming & adding points (use -wipe to reset).\n",
               gBinFileName, store_ram);

        int fd = open(gBinFileName, O_RDWR, 0666);
        if (fd == -1) {
            printf("[!] FATAL: Cannot open '%s'!\n", gBinFileName);
            return 1;
        }
        gTameStoreData = (uint8_t*)mmap(NULL, massive_db_bytes, PROT_READ | PROT_WRITE,
                                        MAP_SHARED | MAP_NORESERVE, fd, 0);
        close(fd);
    } else {
        struct statvfs statv;
        if (statvfs(".", &statv) == 0) {
            uint64_t free_bytes = (uint64_t)statv.f_bavail * statv.f_frsize;
            massive_db_bytes = (uint64_t)(store_ram * 1024ULL * 1024ULL * 1024ULL);
            if (massive_db_bytes > (uint64_t)(free_bytes * 0.95)) {
                massive_db_bytes = (uint64_t)(free_bytes * 0.95);
            }
            store_ram = (double)massive_db_bytes / (1024.0 * 1024.0 * 1024.0);
        } else {
            massive_db_bytes = (uint64_t)(store_ram * 1024ULL * 1024ULL * 1024ULL);
        }

        printf("[*] Creating '%s' (%.2f GB)...\n", gBinFileName, store_ram);
        int fd = open(gBinFileName, O_RDWR | O_CREAT, 0666);
        if (fd == -1) {
            printf("[!] FATAL: Cannot create '%s'!\n", gBinFileName);
            return 1;
        }
        if (fallocate(fd, 0, 0, massive_db_bytes) != 0) {
            if (ftruncate(fd, massive_db_bytes) == -1) {
                printf("[!] FATAL: ftruncate failed on '%s'!\n", gBinFileName);
                close(fd);
                return 1;
            }
        }
        gTameStoreData = (uint8_t*)mmap(NULL, massive_db_bytes, PROT_READ | PROT_WRITE,
                                        MAP_SHARED | MAP_NORESERVE, fd, 0);
        close(fd);
    }

    if (gTameStoreData == MAP_FAILED) {
        printf("[!] FATAL: mmap failed!\n");
        return 1;
    }
#ifdef MADV_HUGEPAGE
    madvise(gTameStoreData, massive_db_bytes, MADV_HUGEPAGE);
#endif
    gTameStoreDataSize = massive_db_bytes;
    gUsingExternalMemory = true;

    if (!gTameStore.InitAllTame((size_t)store_ram, 0)) {
        printf("[!] FATAL: Failed to initialize TameStore!\n");
        return 1;
    }
    gTameStore.SetNoRotation(true);

    // Resume from instant .meta sidecar if available, else 16-thread parallel scan
    if (file_exists && !gForceWipe) {
        u64 meta_dps = 0, meta_ops = 0;
        if (LoadDiskMeta(meta_dps, meta_ops)) {
            gTameStore.SetWildCounts(meta_dps, 0);
            gInitialRecoveredDPs.store(meta_dps);
            gInitialRecoveredOps.store(meta_ops);
            gReservedDPs.store(meta_dps);
            printf("[*] Instant .meta Resume: Recovered %llu DPs (2^%.2f steps) in < 1 ms!\n",
                   (unsigned long long)meta_dps, (meta_ops > 0) ? log2((double)meta_ops) : 0.0);
        } else {
            WildEntryCompact* ext_table = (WildEntryCompact*)gTameStoreData;
            uint64_t max_ext_entries = gTameStoreDataSize / sizeof(WildEntryCompact);
            int scan_threads = 16;
            std::vector<std::thread> scanners;
            std::vector<uint64_t> thr_rec(scan_threads, 0);
            uint64_t per_thr = max_ext_entries / scan_threads;

            for (int t = 0; t < scan_threads; t++) {
                scanners.emplace_back([&, t]() {
                    uint64_t s_idx = t * per_thr;
                    uint64_t e_idx = (t == scan_threads - 1) ? max_ext_entries : (s_idx + per_thr);
                    uint64_t cnt = 0;
                    for (uint64_t i = s_idx; i < e_idx; i++) {
                        if (ext_table[i].part1 != 0 || ext_table[i].part2 != 0) cnt++;
                    }
                    thr_rec[t] = cnt;
                });
            }
            uint64_t total_rec = 0;
            for (int t = 0; t < scan_threads; t++) {
                scanners[t].join();
                total_rec += thr_rec[t];
            }
            gTameStore.SetWildCounts(total_rec, 0);
            gInitialRecoveredDPs.store(total_rec);
            gReservedDPs.store(total_rec);
            SaveDiskMeta();
            printf("[*] Parallel Scan Resume: Recovered %llu existing DPs in '%s' (saved .meta).\n",
                   (unsigned long long)total_rec, gBinFileName);
        }

        u64 cur_recovered = gInitialRecoveredDPs.load();
        if (gTargetDPs > 0 && cur_recovered >= gTargetDPs) {
            printf("[✓] Target of %llu DPs is already reached in '%s' (%llu DPs stored)!\n",
                   (unsigned long long)gTargetDPs, gBinFileName, (unsigned long long)cur_recovered);
            printf("    Specify a higher -target or pass -wipe to start from 0.\n");
            munmap(gTameStoreData, gTameStoreDataSize);
            return 0;
        }
    } else {
        printf("[*] Wiping '%s' using %d threads...\n", gBinFileName, gNumWorkers);
        int num_threads = gNumWorkers;
        std::vector<std::thread> wipers;
        uint64_t chunk_size = gTameStoreDataSize / num_threads;
        for (int t = 0; t < num_threads; t++) {
            wipers.emplace_back([&, t]() {
                uint64_t start_ofs = t * chunk_size;
                uint64_t sz = (t == num_threads - 1) ? (gTameStoreDataSize - start_ofs) : chunk_size;
                memset((char*)gTameStoreData + start_ofs, 0, sz);
            });
        }
        for (auto& th : wipers) th.join();
        gTameStore.SetWildCounts(0, 0);
        gReservedDPs.store(0);
        SaveDiskMeta();
        printf("[*] Wipe complete. Ready to store new DPs.\n");
    }

    // ========================================================================
    // Build deterministic Jump Tables (identical to PSCKangaroo: SetRndSeed(0))
    // ========================================================================
    SetRndSeed(0);
    EcInt minjump, t;
    int jump_shift = gRange / 2 + 3;
    minjump.Set(1);
    minjump.ShiftLeft(jump_shift);
    for (int i = 0; i < JMP_CNT; i++) {
        gEcJumps1[i].dist = minjump;
        t.RndMax(minjump);
        gEcJumps1[i].dist.Add(t);
        gEcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        gEcJumps1[i].p = ec.MultiplyG(gEcJumps1[i].dist);
        aecc_sub(gJumps1_negY[i], pp, (const uint64_t*)gEcJumps1[i].p.y.data);
    }

    minjump.Set(1);
    minjump.ShiftLeft(gRange - 10);
    for (int i = 0; i < JMP_CNT; i++) {
        gEcJumps2[i].dist = minjump;
        t.RndMax(minjump);
        gEcJumps2[i].dist.Add(t);
        gEcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        gEcJumps2[i].p = ec.MultiplyG(gEcJumps2[i].dist);
        aecc_sub(gJumps2_negY[i], pp, (const uint64_t*)gEcJumps2[i].p.y.data);
    }

    minjump.Set(1);
    minjump.ShiftLeft(gRange - 12);
    for (int i = 0; i < JMP_CNT; i++) {
        gEcJumps3[i].dist = minjump;
        t.RndMax(minjump);
        gEcJumps3[i].dist.Add(t);
        gEcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFEULL;
        gEcJumps3[i].p = ec.MultiplyG(gEcJumps3[i].dist);
        aecc_sub(gJumps3_negY[i], pp, (const uint64_t*)gEcJumps3[i].p.y.data);
    }

    // High-entropy seed for GPU kangaroo starting positions
    SetRndSeed(GetTickCount64() ^ ((u64)getpid() << 16));

    Int_HalfRange.SetZero();
    gGenMode = true;

    EcPoint dummyPnt;
    dummyPnt.x.SetZero();
    dummyPnt.y.SetZero();

    signal(SIGINT, SigIntHandler);
    signal(SIGTERM, SigIntHandler);

    u64 total_gpu_kangs = 0;
    for (int i = 0; i < GpuCnt; i++) {
        total_gpu_kangs += GpuKangs[i]->CalcKangCnt();
        if (!GpuKangs[i]->Prepare(dummyPnt, gRange, gDP, gEcJumps1, gEcJumps2, gEcJumps3)) {
            GpuKangs[i]->Failed = true;
            printf("[!] GPU %d Prepare failed!\n", GpuKangs[i]->CudaIndex);
            return 1;
        }
    }

    char start_hex[100];
    gGenStartOffset.GetHexStr(start_hex);

    printf("\n================================================================================\n");
    printf(" 🚀 STARTING HYBRID GPU + CPU 100%% TAME DP GENERATION\n");
    printf("    Range       : %u bits (Span: 2^%d, Jump Shift: 2^%d)\n", gRange, gGenSpanBits, jump_shift);
    printf("    Start Point : 0x%s\n", start_hex);
    printf("    Target DP   : DP %u (1 DP every ~%.2fM jumps)\n", gDP, pow(2.0, gDP) / 1e6);
    if (gTargetDPs > 0) {
        printf("    Target Goal : %llu DPs (Auto-Stop immediately when reached)\n",
               (unsigned long long)gTargetDPs);
    }
    printf("    GPU Engine  : %llu TAME kangaroos across %d GPU(s)\n", (unsigned long long)total_gpu_kangs, GpuCnt);
    printf("    CPU Engine  : %d TAME kangaroos across %d CPU threads (Batch=%d, Vita=%llu)\n",
           gCpuThreadsActive * gCpuBatchSize, gCpuThreadsActive, gCpuBatchSize, (unsigned long long)gVitaLimit);
    printf("    Output .bin : %s (Capacity: %llu DPs | Auto-Checkpoint: %d min)\n",
           gBinFileName, (unsigned long long)gTameStore.GetPerTableSize(), gCheckpointMins);
    printf("    Press Ctrl+C at any time to sync & exit safely.\n");
    printf("================================================================================\n\n");

    StartWorkerThreads();

    pthread_t thr_handles[MAX_GPU_CNT];
    ThrCnt = GpuCnt;
    for (int i = 0; i < GpuCnt; i++) {
        pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
    }

    u64 tm0 = GetTickCount64();
    u64 tm_stats = tm0;
    u64 tm_ckpt  = tm0;

    while (!gStopRequested) {
        if (GpuCnt > 0) {
            ProcessNewPoints();
        }
        Sleep(2);

        if (gTargetReached.load() || CheckAndTriggerTargetStop()) {
            break;
        }

        u64 now_ms = GetTickCount64();
        if (now_ms - tm_stats >= 3000) {
            ShowDashboard(tm0, total_gpu_kangs, start_hex, jump_shift);
            tm_stats = now_ms;
        }

        // Periodic non-blocking background disk sync + .meta checkpoint (from tames-gen-main)
        if (gCheckpointMins > 0 && (now_ms - tm_ckpt) >= (u64)gCheckpointMins * 60000ULL) {
            msync(gTameStoreData, gTameStoreDataSize, MS_ASYNC);
            SaveDiskMeta();
            printf("[CKPT] Non-blocking MS_ASYNC + %s.meta updated (%llu DPs stored).\n",
                   gBinFileName, (unsigned long long)gTameStore.GetWild1Count());
            tm_ckpt = now_ms;
        }

        if (gTameStore.IsTableFrozen(0)) {
            printf("\n[*] Table capacity reached (FROZEN). Stopping generation...\n");
            gStopRequested = true;
            for (int i = 0; i < GpuCnt; i++) GpuKangs[i]->Stop();
            break;
        }
    }

    // Wait for GPU threads to finish current kernel batch
    for (int i = 0; i < GpuCnt; i++) {
        pthread_join(thr_handles[i], NULL);
    }
    if (!gTargetReached.load() && GpuCnt > 0) {
        ProcessNewPoints();
    }
    StopWorkerThreads();

    if (gTargetReached.load()) {
        printf("\n================================================================================\n");
        printf(" 🎯 TARGET GOAL REACHED! Stored %llu / %llu DPs in '%s'\n",
               (unsigned long long)gTameStore.GetWild1Count(),
               (unsigned long long)gTargetDPs, gBinFileName);
        printf("================================================================================\n");
    }

    printf("\n[*] Final Statistics:\n");
    ShowDashboard(tm0, total_gpu_kangs, start_hex, jump_shift);

    if (gTxtFp) {
        fclose(gTxtFp);
        gTxtFp = nullptr;
    }

    if (gUsingExternalMemory) {
        if (gSessionDPs.load() > 0 || gForceWipe) {
            printf("[*] Syncing %llu new DPs (GPU: %llu | CPU: %llu) to '%s' using %d threads...\n",
                   (unsigned long long)gSessionDPs.load(),
                   (unsigned long long)gGpuSessionDPs.load(),
                   (unsigned long long)gCpuSessionDPs.load(),
                   gBinFileName, gNumWorkers);
            int num_threads = gNumWorkers;
            uint64_t chunk = (gTameStoreDataSize / num_threads / 4096) * 4096;
            std::vector<std::thread> syncers;
            for (int t = 0; t < num_threads; t++) {
                syncers.emplace_back([&, t]() {
                    uint64_t start_ofs = t * chunk;
                    uint64_t sz = (t == num_threads - 1) ? (gTameStoreDataSize - start_ofs) : chunk;
                    msync((uint8_t*)gTameStoreData + start_ofs, sz, MS_SYNC);
                });
            }
            for (auto& th : syncers) th.join();
        }
        SaveDiskMeta();
        munmap(gTameStoreData, gTameStoreDataSize);
        printf("[*] '%s' and '%s.meta' safely synced and closed.\n", gBinFileName, gBinFileName);
    }

    for (int i = 0; i < GpuCnt; i++) delete GpuKangs[i];
    hugepage_free(gRingBufferPool, gRingBufferPoolBytes);
    if (GpuCnt > 0) {
        cudaFreeHost(pPntList2);
        cudaFreeHost(pPntList);
    } else {
        free(pPntList2);
        free(pPntList);
    }
    DeInitEc();

    return 0;
}
