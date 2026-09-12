// AMD XDNA (Ryzen AI NPU) backend
//
// Design: a narrow accelerator backend in the spirit of ggml-blas. It lives on host
// memory (CPU buffer type) so the scheduler can hand it individual ops out of a
// CPU graph with no tensor copies, and it only claims MUL_MATs that map onto the
// precompiled block GEMM kernels it finds at startup. Everything else stays on CPU.
//
// Kernel contract (see kernels/README.md; built with mlir-aie whole_array.py using
// --b-col-maj 1 --c-col-maj 1 --dtype_in bf16 --dtype_out f32):
//   A : [M x K] bf16 row-major   -> a block of weight rows (ggml src0, ne01 x ne00)
//   B : [N x K] bf16 row-major   -> a block of activation rows (ggml src1, ne11 x ne10)
//   C : [N x M] f32  row-major   -> C[t][r] = sum_k A[r][k] * B[t][k], i.e. exactly the
//                                   ggml dst layout (ne0 = ne01 features, ne1 = tokens)
// so no transposes are needed on either side; larger matmuls are tiled over (M, K, N)
// blocks and partial K products are accumulated on the host.
//
// CPU + GPU + NPU: a first-generation XDNA part is slower than eight Zen 4 cores at dense GEMM,
// and far slower than the integrated Radeon, so each admitted MUL_MAT is additionally split by
// output features between the NPU, a private Vulkan backend on the integrated GPU (zero-copy
// activations/dst on UMA, cached device weights) and a private CPU backend, all computing
// their rows concurrently; the split can adapt to the measured times of the workers
// (see ggml_backend_xdna_mul_mat).

#include "ggml-xdna.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#ifdef GGML_XDNA_HAVE_CPU
#include "ggml-cpu.h"
#endif
#ifdef GGML_XDNA_HAVE_VULKAN
#include "ggml-vulkan.h"
#endif

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#define GGML_XDNA_PAGE_ALLOC(size) _aligned_malloc((size), 4096)
#define GGML_XDNA_PAGE_FREE(ptr)   _aligned_free(ptr)
#else
#define GGML_XDNA_PAGE_ALLOC(size) std::aligned_alloc(4096, (size))
#define GGML_XDNA_PAGE_FREE(ptr)   std::free(ptr)
#endif

namespace fs = std::filesystem;

#define GGML_XDNA_KERNEL_PREFIX "MLIR_AIE"   // kernel name prefix inside IRON-built xclbins
#define GGML_XDNA_OPCODE_TXN    3u           // opcode for a transaction-binary instruction stream

// a precompiled block GEMM for one fixed (M, K, N)
struct ggml_xdna_kernel {
    int64_t M = 0;   // A rows  (weight features per block)
    int64_t K = 0;   // reduction length per block
    int64_t N = 0;   // B rows  (tokens per block)

    std::string xclbin_path;
    std::string insts_path;
    bool        w8 = false;   // int8 weight rows in A (mm_w8_i8bf16_f32_*); the host applies their scales

    bool loaded = false;
    bool broken = false;

    xrt::xclbin     xclbin;
    xrt::hw_context hwctx;
    xrt::kernel     kernel;
    xrt::bo         bo_instr;
    size_t          n_instr = 0;

    // two sets of operand buffers, each with its own reusable run, so the host can stage the
    // next block while the NPU computes the current one
    struct slot {
        xrt::bo       a, b, c;
        xrt::run      run;
        ggml_bf16_t * a_map  = nullptr;   // bf16 kernels
        int8_t *      a8_map = nullptr;   // int8-weight kernels
        ggml_bf16_t * b_map = nullptr;
        float       * c_map = nullptr;
    };
    slot slots[2];

    double launch_us = 0.0;   // measured NPU wall time per block while pipelined (EMA), 0 until known
    int    n_ops     = 0;
};

// a weight queued for the load-time warm-up
struct ggml_xdna_warm_item {
    ggml_tensor t;       // copy of the weight's metadata; its data outlives the XDNA backends
    int64_t     n_tok;   // batch size of the reserved graph, for the kernel choice
};

// process-wide device + kernel table, shared by the device (supports_op) and the backend
struct ggml_xdna_state {
    std::mutex mutex;

    bool probed    = false;
    bool available = false;

    std::string description = "AMD XDNA NPU";
    std::string kernel_dir;

    std::unique_ptr<xrt::device>  device;
    std::vector<ggml_xdna_kernel> kernels;

    int64_t min_batch = 32;   // smallest ne11 (tokens) worth sending to the NPU

    std::atomic<bool> probe_done{false};   // lets supports_op skip the lock once probed

    // bf16 copies of the weight rows the NPU handles, shared by all XDNA backends and keyed by
    // weight plane. The NPU always takes the first rows of an op, so rows [0, n_rows) are kept and
    // extended when a later split gives it more; a sampled signature of the plane catches reused
    // addresses. Up to GGML_XDNA_NPU_CACHE_MB, filled on first use or by the warm-up thread, which
    // only ever inserts new entries (so an op's pointer into an entry stays valid).
    struct npu_weight {
        std::vector<ggml_bf16_t> data;
        int64_t                  n_rows = 0;
        uint64_t                 sig    = 0;
    };
    std::mutex                         cache_mutex;   // taken after mutex, never before it
    std::map<const void *, npu_weight> npu_weights;

    // int8 counterpart for GGML_XDNA_NPU_W8: rows [0, n_rows) as int8, with one scale per row and
    // GGML_XDNA_W8_GROUP weights; same rules, same byte budget
    struct npu_weight_w8 {
        std::vector<int8_t> q;
        std::vector<float>  scales;
        int64_t             n_rows = 0;
        uint64_t            sig    = 0;
    };
    std::map<const void *, npu_weight_w8> npu_weights_w8;
    size_t                             npu_weight_bytes = 0;

    // load-time warm-up: while llama.cpp reserves its graphs, supports_op queues the weights the NPU
    // will handle, and a background thread loads their kernels and converts the NPU's expected rows,
    // so that the first prompt does not pay for it. The first NPU op stops it.
    std::mutex                       warm_mutex;
    std::condition_variable          warm_cv;
    std::thread                      warm_thread;
    bool                             warm_running = false;
    std::vector<ggml_xdna_warm_item> warm_queue;
    std::map<const void *, bool>     warm_seen;
    std::atomic<bool>                warm_stop{false};
    float                            warm_share = 0.0f;   // expected NPU share of an op's rows (0: no warm-up)
    int                              n_backends = 0;

    ~ggml_xdna_state() {
        warm_stop = true;
        warm_cv.notify_all();
        if (warm_thread.joinable()) {
            warm_thread.join();
        }
    }
};

static ggml_xdna_state & ggml_xdna_get_state() {
    static ggml_xdna_state state;
    return state;
}

// getenv that treats an empty value as unset (shells clear variables that way)
static const char * ggml_xdna_getenv(const char * name) {
    const char * v = std::getenv(name);
    return (v != nullptr && v[0] != '\0') ? v : nullptr;
}

static bool ggml_xdna_debug() {
    static const bool debug = ggml_xdna_getenv("GGML_XDNA_DEBUG") != nullptr;
    return debug;
}

// memory for the NPU's cached bf16 weights: GGML_XDNA_NPU_CACHE_MB (default 4096)
static size_t ggml_xdna_npu_cache_limit() {
    static const size_t limit = []() {
        const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_CACHE_MB");
        return env != nullptr ? (size_t) std::max(0LL, std::atoll(env)) << 20 : (size_t) 4 << 30;
    }();
    return limit;
}

// the CPU backend's repack buffer holds weights re-laid out for its SIMD kernels; its buffer type
// is internal to ggml-cpu, so it is recognized by name
static bool ggml_xdna_buft_is_cpu_repack(ggml_backend_buffer_type_t buft) {
    return std::strcmp(ggml_backend_buft_name(buft), "CPU_REPACK") == 0;
}

static bool ggml_xdna_is_cpu_repack(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_xdna_buft_is_cpu_repack(ggml_backend_buffer_get_type(t->buffer));
}

// whether matmuls on repacked weights may be taken (GGML_XDNA_REPACK=0 leaves them to the CPU)
static bool ggml_xdna_repack_enabled() {
#if !(defined(__x86_64__) || defined(_M_X64))
    return false;   // the layout converters below follow the x86 (AVX2) repack formats
#else
    static const bool enabled = [] {
        const char * v = ggml_xdna_getenv("GGML_XDNA_REPACK");
        return v == nullptr || std::atoi(v) != 0;
    }();
    return enabled;
#endif
}

static void ggml_xdna_scan_kernels(ggml_xdna_state & st, const std::string & dir) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return;
    }

    static const std::regex re("^mm_(bf16|w8_i8bf16)_f32_M([0-9]+)_K([0-9]+)_N([0-9]+)\\.xclbin$");

    for (const auto & entry : fs::directory_iterator(dir, ec)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch m;
        if (!std::regex_match(name, m, re)) {
            continue;
        }

        ggml_xdna_kernel k;
        k.w8 = m[1] == "w8_i8bf16";
        k.M  = std::stoll(m[2]);
        k.K  = std::stoll(m[3]);
        k.N  = std::stoll(m[4]);
        k.xclbin_path = entry.path().string();
        k.insts_path  = (entry.path().parent_path() / (name.substr(0, name.size() - 7) + ".insts.bin")).string();

        if (!fs::is_regular_file(k.insts_path, ec)) {
            GGML_LOG_WARN("%s: %s has no matching .insts.bin, skipping\n", __func__, name.c_str());
            continue;
        }
        st.kernels.push_back(std::move(k));
    }

    std::sort(st.kernels.begin(), st.kernels.end(), [](const ggml_xdna_kernel & a, const ggml_xdna_kernel & b) {
        return a.M*a.K*a.N < b.M*b.K*b.N;
    });
}

// open the NPU and enumerate kernels; cheap, done once
static void ggml_xdna_probe_locked(ggml_xdna_state & st);

static void ggml_xdna_probe(ggml_xdna_state & st) {
    if (st.probe_done.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.mutex);
    ggml_xdna_probe_locked(st);
    st.probe_done.store(true, std::memory_order_release);
}

static void ggml_xdna_probe_locked(ggml_xdna_state & st) {
    if (st.probed) {
        return;
    }
    st.probed = true;

    // GGML_XDNA_DISABLE=1: report no device and never touch XRT (e.g. for side processes such as
    // --version / --list-devices running next to a server that owns the NPU)
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_DISABLE")) {
        if (std::atoi(env) != 0) {
            return;
        }
    }

    try {
        st.device = std::make_unique<xrt::device>(0u);
        st.description = "AMD XDNA NPU (" + st.device->get_info<xrt::info::device::name>() + ")";
        st.available = true;
    } catch (const std::exception & e) {
        GGML_LOG_DEBUG("%s: no XRT device: %s\n", __func__, e.what());
        st.device.reset();
        return;
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_MIN_BATCH")) {
        st.min_batch = std::max<int64_t>(1, std::atoll(env));
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_KERNEL_DIR")) {
        st.kernel_dir = env;
    } else {
#ifdef GGML_XDNA_DEFAULT_KERNEL_DIR
        st.kernel_dir = GGML_XDNA_DEFAULT_KERNEL_DIR;
#else
        st.kernel_dir = "xdna-kernels";
#endif
    }
    ggml_xdna_scan_kernels(st, st.kernel_dir);

    if (st.kernels.empty()) {
        GGML_LOG_WARN("%s: no kernels found in %s - the XDNA backend will not claim any ops\n",
                      __func__, st.kernel_dir.c_str());
    } else {
        for (const auto & k : st.kernels) {
            GGML_LOG_INFO("%s: kernel M=%" PRId64 " K=%" PRId64 " N=%" PRId64 " (%s)\n",
                          __func__, k.M, k.K, k.N, fs::path(k.xclbin_path).filename().string().c_str());
        }
    }
}

static bool ggml_xdna_kernel_load(ggml_xdna_state & st, ggml_xdna_kernel & k) {
    if (k.loaded) {
        return true;
    }
    if (k.broken) {
        return false;
    }

    try {
        std::ifstream f(k.insts_path, std::ios::binary);
        if (!f) {
            throw std::runtime_error("cannot open " + k.insts_path);
        }
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bytes.empty() || bytes.size() % 4 != 0) {
            throw std::runtime_error("malformed instruction stream " + k.insts_path);
        }
        k.n_instr = bytes.size() / sizeof(uint32_t);

        k.xclbin = xrt::xclbin(k.xclbin_path);

        std::string kname;
        for (const auto & xk : k.xclbin.get_kernels()) {
            const std::string name = xk.get_name();
            if (name.rfind(GGML_XDNA_KERNEL_PREFIX, 0) == 0) {
                kname = name;
                break;
            }
        }
        if (kname.empty()) {
            throw std::runtime_error("no " GGML_XDNA_KERNEL_PREFIX "* kernel in " + k.xclbin_path);
        }

        st.device->register_xclbin(k.xclbin);
        k.hwctx  = xrt::hw_context(*st.device, k.xclbin.get_uuid());
        k.kernel = xrt::kernel(k.hwctx, kname);

        // argument layout of an IRON runtime sequence: (opcode, instr, n_instr, A, B, C)
        k.bo_instr = xrt::bo(*st.device, bytes.size(), xrt::bo::flags::cacheable, k.kernel.group_id(1));
        std::memcpy(k.bo_instr.map<void *>(), bytes.data(), bytes.size());
        k.bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        for (auto & s : k.slots) {
            s.a = xrt::bo(*st.device, k.M*k.K*(k.w8 ? sizeof(int8_t) : sizeof(ggml_bf16_t)), xrt::bo::flags::host_only, k.kernel.group_id(3));
            s.b = xrt::bo(*st.device, k.N*k.K*sizeof(ggml_bf16_t), xrt::bo::flags::host_only, k.kernel.group_id(4));
            s.c = xrt::bo(*st.device, k.N*k.M*sizeof(float),       xrt::bo::flags::host_only, k.kernel.group_id(5));
            if (k.w8) {
                s.a8_map = s.a.map<int8_t *>();
            } else {
                s.a_map = s.a.map<ggml_bf16_t *>();
            }
            s.b_map = s.b.map<ggml_bf16_t *>();
            s.c_map = s.c.map<float *>();

            s.run = xrt::run(k.kernel);
            s.run.set_arg(0, GGML_XDNA_OPCODE_TXN);
            s.run.set_arg(1, k.bo_instr);
            s.run.set_arg(2, k.n_instr);
            s.run.set_arg(3, s.a);
            s.run.set_arg(4, s.b);
            s.run.set_arg(5, s.c);
        }

        k.loaded = true;
        GGML_LOG_INFO("%s: loaded %s (kernel %s, %zu instruction words)\n",
                      __func__, fs::path(k.xclbin_path).filename().string().c_str(), kname.c_str(), k.n_instr);
        return true;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("%s: failed to load %s: %s\n", __func__, k.xclbin_path.c_str(), e.what());
        k.broken = true;
        return false;
    }
}

// submit the staged operands of slot s to the NPU; returns without waiting
static void ggml_xdna_slot_start(ggml_xdna_kernel::slot & s) {
    s.a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    s.b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    s.run.start();
}

// wait for slot s and make its result visible to the host; XRT errors propagate as exceptions
static bool ggml_xdna_slot_wait(ggml_xdna_kernel::slot & s) {
    const ert_cmd_state state = s.run.wait();
    if (state != ERT_CMD_STATE_COMPLETED) {
        GGML_LOG_ERROR("%s: kernel did not complete (ert state %d)\n", __func__, (int) state);
        return false;
    }
    s.c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    return true;
}

// estimated wall time of one launch of k: measured once it has run, otherwise a model fitted
// on a Ryzen 7 8700G (~0.2 ms fixed dispatch + sync cost, ~200 GMAC/s of block compute; the
// 512x512x128 block measures ~360 us)
static double ggml_xdna_kernel_launch_us(const ggml_xdna_kernel & k) {
    if (k.launch_us > 0.0) {
        return k.launch_us;
    }
    return 200.0 + (double) k.M*k.K*k.N/200.0e3;
}

// pick the kernel with the least estimated NPU time for this problem; padded work and the
// fixed per-launch cost both count, so a small block only wins when it saves real work
static ggml_xdna_kernel * ggml_xdna_select_kernel(ggml_xdna_state & st, int64_t n_feat, int64_t n_k, int64_t n_tok, bool w8) {
    std::lock_guard<std::mutex> lock(st.mutex);   // launch_us is updated by running kernels
    ggml_xdna_kernel * best = nullptr;
    double best_cost = 0.0;
    for (auto & k : st.kernels) {
        if (k.broken || k.w8 != w8) {
            continue;
        }
        const double launches = (double) ((n_feat + k.M - 1)/k.M) * ((n_k + k.K - 1)/k.K) * ((n_tok + k.N - 1)/k.N);
        const double cost = launches * ggml_xdna_kernel_launch_us(k);
        if (best == nullptr || cost < best_cost) {
            best = &k;
            best_cost = cost;
        }
    }
    return best;
}

// -------------------------------------------------------------------------------------------------
// host-side helpers

struct ggml_backend_xdna_context {
    int n_threads = GGML_DEFAULT_N_THREADS;

    std::vector<ggml_bf16_t> w_bf16;   // weight rows converted per call when not cached, [rows x K]
    std::vector<int8_t>      w_i8;     // same for int8 weights (GGML_XDNA_NPU_W8), [rows x K]
    std::vector<float>       w_scales; // and their scales, [rows x groups]
    std::vector<ggml_bf16_t> x_bf16;   // staged activations for the current plane,   [tokens x K]
    std::vector<std::future<void>> tasks;

    bool warned_fallback = false;

    // Work split: the output features of each MUL_MAT are divided between the NPU, a private
    // Vulkan backend (the integrated GPU) and a private CPU backend, all running concurrently.
    // share[] holds the fraction per worker: fixed (defaults below, or GGML_XDNA_NPU_SHARE /
    // GGML_XDNA_VK_SHARE, CPU takes the rest), or chosen per operation from the workers' measured
    // speeds (auto: the default without a Vulkan worker, or GGML_XDNA_NPU_SHARE=auto).
    ggml_backend_t cpu = nullptr;
    ggml_backend_t vk  = nullptr;
    bool  vk_tried = false;
    float share[3] = { 0.4f, 0.0f, 0.6f };   // NPU, VK, CPU
    bool  share_fixed = true;
    bool  shares_ready = false;
    double npu_min_gflop = 0.0;               // ops below this skip the NPU (set when a GPU exists)

    // Vulkan zero-copy: on a UMA GPU the worker imports the host ranges of the operands as
    // device buffers (VK_EXT_external_memory_host) instead of copying them. Imports are cached
    // by page-aligned range since weights and scheduler buffers keep their addresses.
    bool      vk_zero_copy = true;
    uintptr_t vk_import_align = 4096;
    std::map<std::pair<uintptr_t, uintptr_t>, ggml_backend_buffer_t> vk_imports;
    void * vk_out      = nullptr;   // page-aligned host scratch for the worker's result (imported once)
    size_t vk_out_size = 0;

    // Weight slices are copied once into device memory and kept (mmap'd model files cannot be
    // imported), keyed by (data pointer, row range) and guarded by a sampled content signature.
    struct vk_weight {
        ggml_context *        wctx = nullptr;
        ggml_backend_buffer_t buf  = nullptr;
        ggml_tensor *         a    = nullptr;
        uint64_t              sig  = 0;
        size_t                bytes = 0;
    };
    std::map<std::tuple<const void *, int64_t, int64_t>, vk_weight> vk_weights;
    size_t vk_weight_bytes = 0;
    size_t vk_weight_limit = (size_t) 8 << 30;

    // auto split: each worker's measured speed per matrix shape (n_feat, K, src0 type), in
    // rows*tokens per ms, and its average rate in MACs per ms as the prior for unseen shapes
    struct shape_perf {
        double thr[3] = { 0.0, 0.0, 0.0 };   // NPU, VK, CPU
        bool   npu_seen  = false;             // the NPU has run this shape (its first run is not learned)
        int    npu_skips = 0;                 // ops planned without the NPU since it was measured
    };
    std::map<std::tuple<int64_t, int64_t, int>, shape_perf> shape_perf_map;
    double mac_thr[3] = { 0.0, 0.0, 0.0 };   // NPU, VK, CPU
    bool   npu_cold   = false;                // the current op loaded an NPU kernel (its time is not learned)
};

enum ggml_xdna_worker {
    GGML_XDNA_WORKER_NPU = 0,
    GGML_XDNA_WORKER_VK  = 1,
    GGML_XDNA_WORKER_CPU = 2,
    GGML_XDNA_N_WORKERS  = 3,
};

static ggml_backend_t ggml_xdna_cpu_backend(ggml_backend_xdna_context * ctx);

static ggml_backend_t ggml_xdna_vk_backend(ggml_backend_xdna_context * ctx) {
#ifdef GGML_XDNA_HAVE_VULKAN
    if (ctx->vk == nullptr && !ctx->vk_tried) {
        ctx->vk_tried = true;
        const int n_dev = ggml_backend_vk_get_device_count();
        int dev = -1;
        if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_DEVICE")) {
            dev = std::atoi(env);
        } else {
            // prefer the integrated AMD GPU that shares the die with the NPU
            for (int i = 0; i < n_dev; i++) {
                char desc[256];
                ggml_backend_vk_get_device_description(i, desc, sizeof(desc));
                if (std::strstr(desc, "AMD") || std::strstr(desc, "Radeon")) {
                    dev = i;
                    break;
                }
            }
            if (dev < 0 && n_dev > 0) {
                dev = 0;
            }
        }
        if (dev >= 0 && dev < n_dev) {
            ctx->vk = ggml_backend_vk_init(dev);
            char desc[256];
            ggml_backend_vk_get_device_description(dev, desc, sizeof(desc));
            GGML_LOG_INFO("%s: Vulkan worker on device %d (%s)\n", __func__, dev, desc);
        }
    }
    return ctx->vk;
#else
    GGML_UNUSED(ctx);
    return nullptr;
#endif
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the Vulkan backend: the weight slice and the
// activations are copied into a device buffer, the product is read back into dst
static bool ggml_xdna_mul_mat_vk_copy(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t vk = ggml_xdna_vk_backend(ctx);
    if (vk == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n_rows = feat1 - feat0;

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_4d(gctx, src0->type, src0->ne[0], n_rows, src0->ne[2], src0->ne[3]);
    ggml_tensor * b = ggml_new_tensor_4d(gctx, src1->type, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    ggml_tensor * c = ggml_mul_mat(gctx, a, b);

    const auto t0 = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point s) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
    };

    bool ok = ggml_backend_supports_op(vk, c);
    ggml_backend_buffer_t buf = ok ? ggml_backend_alloc_ctx_tensors(gctx, vk) : nullptr;
    if (buf == nullptr) {
        ggml_free(gctx);
        return false;
    }
    const double t_alloc = ms_since(t0);

    // weight rows [feat0, feat1) are contiguous within each plane of the (contiguous) src0
    const size_t plane_bytes = n_rows*src0->nb[1];
    for (int64_t i3 = 0; i3 < src0->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < src0->ne[2]; i2++) {
            const char * src = (const char *) src0->data + feat0*src0->nb[1] + i2*src0->nb[2] + i3*src0->nb[3];
            ggml_backend_tensor_set(a, src, (i2 + i3*src0->ne[2])*plane_bytes, plane_bytes);
        }
    }
    ggml_backend_tensor_set(b, src1->data, 0, ggml_nbytes(src1));
    const double t_upload = ms_since(t0);

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, c);
    ok = ggml_backend_graph_compute(vk, graph) == GGML_STATUS_SUCCESS;
    const double t_compute = ms_since(t0);

    if (ok) {
        // c is [n_rows x tokens x ne2 x ne3] contiguous; scatter each token row into dst
        std::vector<float> tmp(ggml_nelements(c));
        ggml_backend_tensor_get(c, tmp.data(), 0, ggml_nbytes(c));
        for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
                for (int64_t t = 0; t < dst->ne[1]; t++) {
                    float * drow = (float *) ((char *) dst->data + t*dst->nb[1] + i2*dst->nb[2] + i3*dst->nb[3]) + feat0;
                    std::memcpy(drow, tmp.data() + ((i3*dst->ne[2] + i2)*dst->ne[1] + t)*n_rows, n_rows*sizeof(float));
                }
            }
        }
    }
    const double t_download = ms_since(t0);

    ggml_backend_buffer_free(buf);
    ggml_free(gctx);

    if (ggml_xdna_debug()) {
        static int n_logged = 0;
        if (n_logged++ < 12) {
            GGML_LOG_INFO("%s: %s[%" PRId64 " x %" PRId64 "] x [%" PRId64 " tok]: alloc %.2f ms, upload %.2f ms, compute %.2f ms, download %.2f ms, free %.2f ms\n",
                          __func__, ggml_type_name(src0->type), src0->ne[0], n_rows, src1->ne[1],
                          t_alloc, t_upload - t_alloc, t_compute - t_upload, t_download - t_compute, ms_since(t0) - t_download);
        }
    }
    return ok;
}

// import the host range [ptr, ptr + size) into the Vulkan device (page-rounded) and return the
// buffer plus the address the tensor must be placed at; nullptr if the device cannot import
static ggml_backend_buffer_t ggml_xdna_vk_import(ggml_backend_xdna_context * ctx, const void * ptr, size_t size, void ** tensor_addr) {
    const uintptr_t align = ctx->vk_import_align;
    const uintptr_t base  = (uintptr_t) ptr & ~(align - 1);
    const uintptr_t end   = ((uintptr_t) ptr + size + align - 1) & ~(align - 1);

    ggml_backend_buffer_t buf = nullptr;
    auto it = ctx->vk_imports.find({base, end});
    if (it != ctx->vk_imports.end()) {
        buf = it->second;
    } else {
        if (ctx->vk_imports.size() >= 4096) {
            for (auto & kv : ctx->vk_imports) {
                ggml_backend_buffer_free(kv.second);
            }
            ctx->vk_imports.clear();
        }
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vk);
        buf = ggml_backend_dev_buffer_from_host_ptr(dev, (void *) base, end - base, end - base);
        if (buf == nullptr) {
            return nullptr;
        }
        ctx->vk_imports[{base, end}] = buf;
    }
    *tensor_addr = (char *) ggml_backend_buffer_get_base(buf) + ((uintptr_t) ptr - base);
    return buf;
}

// whether t lives in a model-weights buffer: weights are immutable, so derived copies can be
// cached; anything else (activations, KV cache views) must be read fresh on every call
static bool ggml_xdna_is_weight(const ggml_tensor * t) {
    return t->buffer != nullptr && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

// cheap content signature of a weight slice (16 samples), to catch a reused address
static uint64_t ggml_xdna_weight_sig(const void * data, size_t bytes, int type, int64_t ne0) {
    uint64_t h = 1469598103934665603ull ^ (uint64_t) type ^ ((uint64_t) ne0 << 8) ^ ((uint64_t) bytes << 24);
    const size_t n = 16;
    for (size_t i = 0; i < n && bytes >= sizeof(uint64_t); i++) {
        uint64_t v;
        std::memcpy(&v, (const char *) data + (bytes - sizeof(uint64_t))*i/(n - 1), sizeof(v));
        h = (h ^ v)*1099511628211ull;
    }
    return h;
}

// device-resident copy of weight rows [feat0, feat1): cached across calls when within budget
static ggml_tensor * ggml_xdna_vk_weight_slice(ggml_backend_xdna_context * ctx, const ggml_tensor * src0, int64_t feat0, int64_t feat1,
                                               ggml_backend_xdna_context::vk_weight & temp, bool & cached) {
    const int64_t n_rows = feat1 - feat0;
    const char *  data   = (const char *) src0->data + feat0*src0->nb[1];
    const size_t  bytes  = n_rows*src0->nb[1];
    const uint64_t sig   = ggml_xdna_weight_sig(data, bytes, src0->type, src0->ne[0]);

    const auto key = std::make_tuple((const void *) src0->data, feat0, feat1);
    const bool cacheable = ggml_xdna_is_weight(src0);   // never cache activations (e.g. KV views)
    auto it = cacheable ? ctx->vk_weights.find(key) : ctx->vk_weights.end();
    if (it != ctx->vk_weights.end()) {
        if (it->second.sig == sig) {
            cached = true;
            return it->second.a;
        }
        // the address was reused by different weights: drop the stale copy
        ggml_backend_buffer_free(it->second.buf);
        ggml_free(it->second.wctx);
        ctx->vk_weight_bytes -= it->second.bytes;
        ctx->vk_weights.erase(it);
    }

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*2,
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_backend_xdna_context::vk_weight w;
    w.wctx  = ggml_init(ip);
    w.a     = ggml_new_tensor_2d(w.wctx, src0->type, src0->ne[0], n_rows);
    w.buf   = ggml_backend_alloc_ctx_tensors(w.wctx, ctx->vk);
    w.sig   = sig;
    w.bytes = bytes;
    if (w.buf == nullptr) {
        ggml_free(w.wctx);
        return nullptr;
    }
    ggml_backend_tensor_set(w.a, data, 0, bytes);

    if (cacheable && ctx->vk_weight_bytes + bytes <= ctx->vk_weight_limit) {
        ctx->vk_weights[key] = w;
        ctx->vk_weight_bytes += bytes;
        cached = true;
    } else {
        temp   = w;   // caller frees after the compute
        cached = false;
    }
    return w.a;
}

// page-aligned host scratch for the Vulkan worker's result, grown on demand; the old block's
// cached import is released before the block itself
static float * ggml_xdna_vk_out_scratch(ggml_backend_xdna_context * ctx, size_t bytes) {
    if (bytes <= ctx->vk_out_size) {
        return (float *) ctx->vk_out;
    }
    if (ctx->vk_out != nullptr) {
        const uintptr_t lo = (uintptr_t) ctx->vk_out;
        const uintptr_t hi = lo + ctx->vk_out_size;
        for (auto it = ctx->vk_imports.begin(); it != ctx->vk_imports.end(); ) {
            if (it->first.first >= lo && it->first.second <= hi) {
                ggml_backend_buffer_free(it->second);
                it = ctx->vk_imports.erase(it);
            } else {
                ++it;
            }
        }
        GGML_XDNA_PAGE_FREE(ctx->vk_out);
    }
    const size_t size = ((std::max(bytes, 2*ctx->vk_out_size) + (1u << 20) - 1) >> 20) << 20;   // whole MiB
    ctx->vk_out      = GGML_XDNA_PAGE_ALLOC(size);
    ctx->vk_out_size = ctx->vk_out != nullptr ? size : 0;
    return (float *) ctx->vk_out;
}

// zero-copy variant: the activations are the caller's host memory imported into the device, the
// result lands in an imported host scratch and is scattered into dst's rows on the CPU (Vulkan's
// split-k matmul path needs a contiguous destination); the weight slice is a cached device copy.
// 2-D operands only.
static bool ggml_xdna_mul_mat_vk_zero_copy(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t vk = ggml_xdna_vk_backend(ctx);
    if (vk == nullptr || !ctx->vk_zero_copy) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const int64_t n_rows = feat1 - feat0;

    if (src0->ne[2]*src0->ne[3] != 1 || src1->ne[2]*src1->ne[3] != 1) {
        return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    auto ms_since = [](std::chrono::steady_clock::time_point s) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - s).count();
    };

    // activations: imported in place; result: contiguous imported scratch (whole block, one import)
    void * addr_b = nullptr;
    void * addr_c = nullptr;
    const size_t c_bytes = (size_t) n_rows*src1->ne[1]*sizeof(float);
    ggml_backend_buffer_t buf_b = ggml_xdna_vk_import(ctx, src1->data, ggml_nbytes(src1), &addr_b);
    float * out = buf_b != nullptr ? ggml_xdna_vk_out_scratch(ctx, c_bytes) : nullptr;
    ggml_backend_buffer_t buf_c = out != nullptr ? ggml_xdna_vk_import(ctx, out, ctx->vk_out_size, &addr_c) : nullptr;
    if (buf_c == nullptr) {
        return false;
    }
    const double t_import = ms_since(t0);

    ggml_backend_xdna_context::vk_weight temp;
    bool cached = false;
    ggml_tensor * a = ggml_xdna_vk_weight_slice(ctx, src0, feat0, feat1, temp, cached);
    if (a == nullptr) {
        return false;
    }
    const double t_weights = ms_since(t0);

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * b = ggml_new_tensor_2d(gctx, src1->type, src1->ne[0], src1->ne[1]);
    ggml_tensor * c = ggml_mul_mat(gctx, a, b);

    bool ok = ggml_backend_supports_op(vk, c) &&
              ggml_backend_tensor_alloc(buf_b, b, addr_b) == GGML_STATUS_SUCCESS &&
              ggml_backend_tensor_alloc(buf_c, c, addr_c) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_cgraph * graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(graph, c);
        ok = ggml_backend_graph_compute(vk, graph) == GGML_STATUS_SUCCESS;
    }
    ggml_free(gctx);

    if (ok) {
        // c is [n_rows x tokens] contiguous: scatter each token row into dst
        for (int64_t t = 0; t < src1->ne[1]; t++) {
            std::memcpy((char *) dst->data + t*dst->nb[1] + feat0*sizeof(float), out + t*n_rows, n_rows*sizeof(float));
        }
    }

    if (!cached) {
        ggml_backend_buffer_free(temp.buf);
        ggml_free(temp.wctx);
    }

    if (ggml_xdna_debug()) {
        static int n_logged = 0;
        if (n_logged++ < 12) {
            GGML_LOG_INFO("%s: %s[%" PRId64 " x %" PRId64 "] x [%" PRId64 " tok]: import %.2f ms, weights %.2f ms (%s), compute %.2f ms%s (%zu imports, %.0f MB weights cached)\n",
                          __func__, ggml_type_name(src0->type), src0->ne[0], n_rows, src1->ne[1],
                          t_import, t_weights - t_import, cached ? "cached" : "copied", ms_since(t0) - t_weights, ok ? "" : " FAILED",
                          ctx->vk_imports.size(), ctx->vk_weight_bytes/1048576.0);
        }
    }
    return ok;
}

static bool ggml_xdna_mul_mat_vk(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    if (ggml_xdna_mul_mat_vk_zero_copy(ctx, dst, feat0, feat1)) {
        return true;
    }
    if (ctx->vk_zero_copy && ggml_xdna_debug()) {
        GGML_LOG_INFO("%s: zero-copy path unavailable for this op, using the copy path\n", __func__);
    }
    return ggml_xdna_mul_mat_vk_copy(ctx, dst, feat0, feat1);
}

// pick the default split once we know which workers exist; env overrides
static void ggml_xdna_init_shares(ggml_backend_xdna_context * ctx) {
    if (ctx->shares_ready) {
        return;
    }
    ctx->shares_ready = true;

    const bool have_vk  = ggml_xdna_vk_backend(ctx)  != nullptr;
    const bool have_cpu = ggml_xdna_cpu_backend(ctx) != nullptr;

    // measured on a Ryzen 7 8700G: the 780M (Vulkan) is ~10x the 4-column XDNA1 at batched
    // GEMM and the 8 Zen 4 cores sit in between, so next to the GPU worker the NPU only ever
    // lengthened the critical path (Qwen3-0.6B and gemma-4-E2B, 476 tokens). It therefore stays
    // out by default when a GPU worker exists; GGML_XDNA_NPU_SHARE opts it back in, and
    // GGML_XDNA_NPU_MIN_GFLOP keeps it to the large products where it has a chance.
    float npu = have_vk ? 0.0f : 0.4f;
    float vkf = have_vk ? 0.8f : 0.0f;
    // without a GPU the split is chosen per operation from measured speeds (auto, see
    // ggml_xdna_plan_auto); the 0.4 / 0.6 shares only serve the first measurement
    ctx->share_fixed = have_vk;
    ctx->npu_min_gflop = have_vk ? 4.0 : 0.0;
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_MIN_GFLOP")) {
        ctx->npu_min_gflop = std::max(0.0, std::atof(env));
    }

    if (const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_SHARE")) {
        if (std::strcmp(env, "auto") == 0) {
            ctx->share_fixed = false;
        } else {
            npu = (float) std::min(1.0, std::max(0.0, std::atof(env)));
            ctx->share_fixed = true;   // an explicit share pins the split
        }
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_SHARE")) {
        vkf = have_vk ? (float) std::min(1.0, std::max(0.0, std::atof(env))) : 0.0f;
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_ZERO_COPY")) {
        ctx->vk_zero_copy = std::atoi(env) != 0;
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_IMPORT_ALIGN")) {
        ctx->vk_import_align = (uintptr_t) std::max(4096LL, std::atoll(env));
    }
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_VK_CACHE_MB")) {
        ctx->vk_weight_limit = (size_t) std::max(0LL, std::atoll(env)) << 20;
    }
    if (npu + vkf > 1.0f) {
        vkf = 1.0f - npu;
    }
    float cpu = have_cpu ? 1.0f - npu - vkf : 0.0f;
    if (!have_cpu) {
        // no CPU worker: its share goes to the GPU if present, else to the NPU
        if (have_vk) { vkf = 1.0f - npu; } else { npu = 1.0f; }
    }

    ctx->share[GGML_XDNA_WORKER_NPU] = npu;
    ctx->share[GGML_XDNA_WORKER_VK]  = vkf;
    ctx->share[GGML_XDNA_WORKER_CPU] = cpu;
    GGML_LOG_INFO("%s: matmul split NPU %.2f / Vulkan %.2f / CPU %.2f%s\n", __func__, npu, vkf, cpu,
                  ctx->share_fixed ? "" : " (auto)");
}

static ggml_backend_t ggml_xdna_cpu_backend(ggml_backend_xdna_context * ctx) {
#ifdef GGML_XDNA_HAVE_CPU
    if (ctx->cpu == nullptr) {
        ctx->cpu = ggml_backend_cpu_init();
        if (ctx->cpu == nullptr) {
            return nullptr;
        }
    }
    ggml_backend_cpu_set_n_threads(ctx->cpu, ctx->n_threads);
    return ctx->cpu;
#else
    GGML_UNUSED(ctx);
    return nullptr;
#endif
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the CPU backend, as a tiny standalone graph
// built over views of the operands (no allocation, no copies)
static bool ggml_xdna_mul_mat_cpu(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    ggml_backend_t cpu = ggml_xdna_cpu_backend(ctx);
    if (cpu == nullptr) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    struct ggml_init_params ip = {
        /* .mem_size   = */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true,
    };
    ggml_context * gctx = ggml_init(ip);

    ggml_tensor * a = ggml_new_tensor_4d(gctx, src0->type, src0->ne[0], feat1 - feat0, src0->ne[2], src0->ne[3]);
    a->data = (char *) src0->data + feat0*src0->nb[1];
    std::memcpy(a->nb, src0->nb, sizeof(a->nb));
    if (ggml_xdna_is_cpu_repack(src0)) {
        // keep the CPU's repacked kernels: they pick the layout from the weight's buffer type and
        // extra traits, and a slice starting on an 8-row group is itself a valid repacked tensor
        a->buffer = src0->buffer;
        a->extra  = src0->extra;
    }

    ggml_tensor * b = ggml_new_tensor_4d(gctx, src1->type, src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    b->data = src1->data;
    std::memcpy(b->nb, src1->nb, sizeof(b->nb));

    ggml_tensor * c = ggml_mul_mat(gctx, a, b);
    c->data  = (char *) dst->data + feat0*dst->nb[0];
    c->nb[1] = dst->nb[1];
    c->nb[2] = dst->nb[2];
    c->nb[3] = dst->nb[3];

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, c);
    const enum ggml_status status = ggml_backend_graph_compute(cpu, graph);

    ggml_free(gctx);
    return status == GGML_STATUS_SUCCESS;
}

template <typename F>
static void ggml_xdna_parallel_for(ggml_backend_xdna_context * ctx, int64_t n, int64_t min_per_thread, F && fn) {
    int n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(ctx->n_threads, n/min_per_thread));
    if (n_threads <= 1) {
        fn(0, n);
        return;
    }
    for (int i = 1; i < n_threads; i++) {
        const int64_t start = i*n/n_threads;
        const int64_t end   = (i + 1)*n/n_threads;
        if (start < end) {
            ctx->tasks.push_back(std::async(std::launch::async, [=, &fn]() { fn(start, end); }));
        }
    }
    fn(0, n/n_threads);
    for (auto & t : ctx->tasks) {
        t.get();
    }
    ctx->tasks.clear();
}

// convert rows [row0, row0 + n_rows) of plane (i2, i3) of t to bf16, one row per dst_stride elements
static void ggml_xdna_rows_to_bf16(ggml_backend_xdna_context * ctx, const ggml_tensor * t,
                                   int64_t row0, int64_t n_rows, int64_t i2, int64_t i3,
                                   ggml_bf16_t * dst, int64_t dst_stride) {
    const int64_t ne0 = t->ne[0];
    const enum ggml_type type = t->type;
    const ggml_to_float_t to_float = ggml_get_type_traits(type)->to_float;

    ggml_xdna_parallel_for(ctx, n_rows, 8, [&](int64_t r0, int64_t r1) {
        std::vector<float> tmp;
        for (int64_t r = r0; r < r1; r++) {
            const char  * src = (const char *) t->data + (row0 + r)*t->nb[1] + i2*t->nb[2] + i3*t->nb[3];
            ggml_bf16_t * out = dst + r*dst_stride;
            switch (type) {
                case GGML_TYPE_BF16:
                    std::memcpy(out, src, ne0*sizeof(ggml_bf16_t));
                    break;
                case GGML_TYPE_F32:
                    ggml_fp32_to_bf16_row((const float *) src, out, ne0);
                    break;
                default:
                    tmp.resize(ne0);
                    if (type == GGML_TYPE_F16) {
                        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, tmp.data(), ne0);
                    } else {
                        to_float(src, tmp.data(), ne0);
                    }
                    ggml_fp32_to_bf16_row(tmp.data(), out, ne0);
                    break;
            }
        }
    });
}

// CPU_REPACK Q4_0 layout on x86 (block_q4_0x8, see make_block_q4_0x8 in ggml-cpu/repack.cpp): each
// group of 8 rows is stored as one block_q4_0x8 per 32-wide column block, holding the 8 rows'
// scales and their quants interleaved in 8-byte chunks, xor 0x88
struct ggml_xdna_block_q4_0   { ggml_fp16_t d;    uint8_t qs[16];  };
struct ggml_xdna_block_q4_0x8 { ggml_fp16_t d[8]; uint8_t qs[128]; };
static_assert(sizeof(ggml_xdna_block_q4_0)   == 18,  "unexpected block_q4_0 size");
static_assert(sizeof(ggml_xdna_block_q4_0x8) == 144, "unexpected block_q4_0x8 size");

// convert rows [row0, row0 + n_rows) (multiples of 8) of a CPU_REPACK Q4_0 tensor to bf16
static void ggml_xdna_repacked_q4_0_rows_to_bf16(ggml_backend_xdna_context * ctx, const ggml_tensor * t,
                                                 int64_t row0, int64_t n_rows, ggml_bf16_t * dst, int64_t dst_stride) {
    const int64_t ne0     = t->ne[0];
    const int64_t nblocks = ne0/32;
    const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_Q4_0)->to_float;

    ggml_xdna_parallel_for(ctx, n_rows/8, 2, [&](int64_t g0, int64_t g1) {
        std::vector<ggml_xdna_block_q4_0> rows(8*nblocks);   // the group's 8 rows as plain q4_0
        std::vector<float> tmp(ne0);
        for (int64_t g = g0; g < g1; g++) {
            const auto * src = (const ggml_xdna_block_q4_0x8 *) ((const char *) t->data + (row0 + 8*g)*t->nb[1]);
            for (int64_t x = 0; x < nblocks; x++) {
                for (int r = 0; r < 8; r++) {
                    rows[r*nblocks + x].d = src[x].d[r];
                }
                for (int c = 0; c < 16; c++) {
                    uint64_t q;
                    std::memcpy(&q, &src[x].qs[8*c], sizeof(q));
                    q ^= 0x8888888888888888ULL;
                    std::memcpy(&rows[(c % 8)*nblocks + x].qs[8*(c / 8)], &q, sizeof(q));
                }
            }
            for (int r = 0; r < 8; r++) {
                to_float(&rows[r*nblocks], tmp.data(), ne0);
                ggml_fp32_to_bf16_row(tmp.data(), dst + (8*g + r)*dst_stride, ne0);
            }
        }
    });
}

// CPU_REPACK Q4_K layout on x86 (block_q4_Kx8, see make_block_q4_Kx8 in ggml-cpu/repack.cpp): per
// 256-wide column block of an 8-row group, the rows' d/dmin, their quants interleaved in 8-byte
// chunks (no xor), and their 6-bit sub-block scales/mins regrouped as 12 bytes per sub-block
struct ggml_xdna_block_q4_K {
    ggml_fp16_t d;
    ggml_fp16_t dmin;
    uint8_t     scales[12];
    uint8_t     qs[128];
};
struct ggml_xdna_block_q4_Kx8 {
    ggml_fp16_t d[8];
    ggml_fp16_t dmin[8];
    uint8_t     scales[96];
    uint8_t     qs[1024];
};
static_assert(sizeof(ggml_xdna_block_q4_K)   == 144,  "unexpected block_q4_K size");
static_assert(sizeof(ggml_xdna_block_q4_Kx8) == 1152, "unexpected block_q4_Kx8 size");

// convert rows [row0, row0 + n_rows) (multiples of 8) of a CPU_REPACK Q4_K tensor to bf16
static void ggml_xdna_repacked_q4_K_rows_to_bf16(ggml_backend_xdna_context * ctx, const ggml_tensor * t,
                                                 int64_t row0, int64_t n_rows, ggml_bf16_t * dst, int64_t dst_stride) {
    const int64_t ne0     = t->ne[0];
    const int64_t nblocks = ne0/256;
    const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_Q4_K)->to_float;

    ggml_xdna_parallel_for(ctx, n_rows/8, 2, [&](int64_t g0, int64_t g1) {
        std::vector<ggml_xdna_block_q4_K> rows(8*nblocks);   // the group's 8 rows as plain q4_K
        std::vector<float> tmp(ne0);
        for (int64_t g = g0; g < g1; g++) {
            const auto * src = (const ggml_xdna_block_q4_Kx8 *) ((const char *) t->data + (row0 + 8*g)*t->nb[1]);
            for (int64_t x = 0; x < nblocks; x++) {
                const ggml_xdna_block_q4_Kx8 & in = src[x];
                for (int c = 0; c < 128; c++) {
                    std::memcpy(&rows[(c % 8)*nblocks + x].qs[8*(c / 8)], &in.qs[8*c], 8);
                }
                // 6-bit scale and min of sub-block s for row r
                uint8_t sc[8][8], mn[8][8];
                for (int s = 0; s < 8; s++) {
                    const uint8_t * p = in.scales + (s < 4 ? 12*s : 48 + 12*(s - 4));
                    for (int j = 0; j < 4; j++) {
                        sc[j][s]     = p[j] & 63;
                        mn[j][s]     = p[4 + j] & 63;
                        sc[j + 4][s] = (p[8 + j] & 15) | ((p[j] >> 6) << 4);
                        mn[j + 4][s] = (p[8 + j] >> 4) | ((p[4 + j] >> 6) << 4);
                    }
                }
                // re-pack them in block_q4_K's 12-byte form (the inverse of get_scale_min_k4)
                for (int r = 0; r < 8; r++) {
                    ggml_xdna_block_q4_K & b = rows[r*nblocks + x];
                    b.d    = in.d[r];
                    b.dmin = in.dmin[r];
                    for (int j = 0; j < 4; j++) {
                        b.scales[j]     = (uint8_t) (sc[r][j] | ((sc[r][j + 4] >> 4) << 6));
                        b.scales[j + 4] = (uint8_t) (mn[r][j] | ((mn[r][j + 4] >> 4) << 6));
                        b.scales[j + 8] = (uint8_t) ((sc[r][j + 4] & 15) | ((mn[r][j + 4] & 15) << 4));
                    }
                }
            }
            for (int r = 0; r < 8; r++) {
                to_float(&rows[r*nblocks], tmp.data(), ne0);
                ggml_fp32_to_bf16_row(tmp.data(), dst + (8*g + r)*dst_stride, ne0);
            }
        }
    });
}

// GGML_XDNA_W8_EMULATE=<group>: round the NPU's bf16 weights through int8, with one scale per row
// and <group> weights, to measure the accuracy of an int8-weight kernel before building one (0: off)
static int64_t ggml_xdna_w8_group() {
    static const int64_t group = []() {
        const char * env = ggml_xdna_getenv("GGML_XDNA_W8_EMULATE");
        return env != nullptr ? std::max<int64_t>(0, std::atoll(env)) : (int64_t) 0;
    }();
    return group;
}

static void ggml_xdna_w8_emulate(ggml_backend_xdna_context * ctx, ggml_bf16_t * dst, int64_t n_rows, int64_t ne0, int64_t dst_stride) {
    const int64_t group = ggml_xdna_w8_group();
    if (group <= 0) {
        return;
    }
    ggml_xdna_parallel_for(ctx, n_rows, 8, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) {
            ggml_bf16_t * row = dst + r*dst_stride;
            for (int64_t c0 = 0; c0 < ne0; c0 += group) {
                const int64_t c1 = std::min(ne0, c0 + group);
                float amax = 0.0f;
                for (int64_t c = c0; c < c1; c++) {
                    amax = std::max(amax, std::fabs(ggml_bf16_to_fp32(row[c])));
                }
                if (amax == 0.0f) {
                    continue;
                }
                const float scale = amax/127.0f;
                const float inv   = 127.0f/amax;
                for (int64_t c = c0; c < c1; c++) {
                    const float q = std::min(127.0f, std::max(-127.0f, std::nearbyint(ggml_bf16_to_fp32(row[c])*inv)));
                    row[c] = ggml_fp32_to_bf16(q*scale);
                }
            }
        }
    });
}

// weight rows of src0, plain or CPU_REPACK, to bf16
static void ggml_xdna_weight_rows_to_bf16(ggml_backend_xdna_context * ctx, const ggml_tensor * src0,
                                          int64_t row0, int64_t n_rows, int64_t i02, int64_t i03,
                                          ggml_bf16_t * dst, int64_t dst_stride) {
    if (ggml_xdna_is_cpu_repack(src0)) {
        GGML_ASSERT(row0 % 8 == 0 && n_rows % 8 == 0 && i02 == 0 && i03 == 0);
        switch (src0->type) {
            case GGML_TYPE_Q4_0: ggml_xdna_repacked_q4_0_rows_to_bf16(ctx, src0, row0, n_rows, dst, dst_stride); break;
            case GGML_TYPE_Q4_K: ggml_xdna_repacked_q4_K_rows_to_bf16(ctx, src0, row0, n_rows, dst, dst_stride); break;
            default: GGML_ABORT("%s: unsupported repacked type %s", __func__, ggml_type_name(src0->type));
        }
    } else {
        ggml_xdna_rows_to_bf16(ctx, src0, row0, n_rows, i02, i03, dst, dst_stride);
    }
    ggml_xdna_w8_emulate(ctx, dst, n_rows, src0->ne[0], dst_stride);
}

// copy an [n_rows x n_cols] window of a bf16 row-major matrix into a zero-padded [rows x cols] block
static void ggml_xdna_fill_block(ggml_bf16_t * blk, int64_t rows, int64_t cols,
                                 const ggml_bf16_t * src, int64_t src_stride, int64_t n_rows, int64_t n_cols) {
    if (n_rows < rows || n_cols < cols) {
        std::memset(blk, 0, rows*cols*sizeof(ggml_bf16_t));
    }
    for (int64_t r = 0; r < n_rows; r++) {
        std::memcpy(blk + r*cols, src + r*src_stride, n_cols*sizeof(ggml_bf16_t));
    }
}

// last-resort host GEMM on staged bf16 blocks, so a runtime NPU failure never corrupts results:
// c[t][r] = sum_i a[r][i]*b[t][i] for the valid rows of an [M x K] / [N x K] block pair
static void ggml_xdna_block_gemm_host(const ggml_bf16_t * a, const ggml_bf16_t * b, float * c,
                                      int64_t M, int64_t K, int64_t n_rows_a, int64_t n_rows_b) {
    for (int64_t t = 0; t < n_rows_b; t++) {
        for (int64_t r = 0; r < n_rows_a; r++) {
            float sum = 0.0f;
            const ggml_bf16_t * ar = a + r*K;
            const ggml_bf16_t * br = b + t*K;
            for (int64_t i = 0; i < K; i++) {
                sum += ggml_bf16_to_fp32(ar[i]) * ggml_bf16_to_fp32(br[i]);
            }
            c[t*M + r] = sum;
        }
    }
}

// bf16 weight rows [row0, row0 + n_rows) of plane (i02, i03) of src0, row stride ne00: contiguous
// bf16 weights are used in place; the leading rows of other model weights come from the shared cache
// (ggml_xdna_state::npu_weights), converted once and extended as needed; anything else is converted
// into the staging buffer. Called with st.mutex held, so ops never race each other on an entry.
static const ggml_bf16_t * ggml_xdna_npu_weights(ggml_backend_xdna_context * ctx, const ggml_tensor * src0,
                                                 int64_t row0, int64_t n_rows, int64_t i02, int64_t i03) {
    const int64_t n_k   = src0->ne[0];
    const char *  plane = (const char *) src0->data + i02*src0->nb[2] + i03*src0->nb[3];
    const char *  rows  = plane + row0*src0->nb[1];

    if (src0->type == GGML_TYPE_BF16 && src0->nb[1] == (size_t) n_k*sizeof(ggml_bf16_t)) {
        return (const ggml_bf16_t *) rows;
    }

    if (row0 == 0 && ggml_xdna_is_weight(src0)) {
        ggml_xdna_state & st  = ggml_xdna_get_state();
        const uint64_t    sig = ggml_xdna_weight_sig(plane, src0->ne[1]*src0->nb[1], src0->type, n_k);

        std::lock_guard<std::mutex> lock(st.cache_mutex);
        auto it = st.npu_weights.find(plane);
        if (it != st.npu_weights.end() && it->second.sig != sig) {
            // the address was reused by different weights: drop the stale copy
            st.npu_weight_bytes -= it->second.data.size()*sizeof(ggml_bf16_t);
            st.npu_weights.erase(it);
            it = st.npu_weights.end();
        }
        const int64_t have = it != st.npu_weights.end() ? it->second.n_rows : 0;
        if (have >= n_rows) {
            return it->second.data.data();
        }
        const size_t add = (size_t) (n_rows - have)*n_k*sizeof(ggml_bf16_t);
        if (st.npu_weight_bytes + add <= ggml_xdna_npu_cache_limit()) {
            try {
                auto & w = st.npu_weights[plane];
                w.data.resize((size_t) n_rows*n_k);
                ggml_xdna_weight_rows_to_bf16(ctx, src0, have, n_rows - have, i02, i03, w.data.data() + have*n_k, n_k);
                w.n_rows = n_rows;
                w.sig    = sig;
                st.npu_weight_bytes += add;
                ctx->npu_cold = true;   // this op paid for a conversion: its time says nothing about the NPU's speed
                return w.data.data();
            } catch (const std::bad_alloc &) {
                // out of memory: the entry keeps its rows, this call converts into the staging buffer
            }
        }
    }

    if ((int64_t) ctx->w_bf16.size() < n_rows*n_k) {
        ctx->w_bf16.resize(n_rows*n_k);
    }
    ggml_xdna_weight_rows_to_bf16(ctx, src0, row0, n_rows, i02, i03, ctx->w_bf16.data(), n_k);
    return ctx->w_bf16.data();
}

// GGML_XDNA_NPU_W8=1: the NPU reads model weights as int8, with one scale per row and
// GGML_XDNA_W8_GROUP weights that the host applies to each block's result. That halves the cache
// memory and the copy traffic of bf16 weights (the NPU's compute time is the same). Every kernel's
// K divides the group, so an NPU block never spans two scales.
static constexpr int64_t GGML_XDNA_W8_GROUP = 1024;

static bool ggml_xdna_npu_w8_enabled() {
    static const bool enabled = []() {
        const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_W8");
        return env != nullptr && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool ggml_xdna_npu_use_w8(const ggml_tensor * src0) {
    return ggml_xdna_npu_w8_enabled() && ggml_xdna_is_weight(src0) &&
           !(src0->type == GGML_TYPE_BF16 && src0->nb[1] == (size_t) src0->ne[0]*sizeof(ggml_bf16_t));
}

// the kernel for this weight: an int8-weight kernel when enabled and available, else a bf16 one
static ggml_xdna_kernel * ggml_xdna_select_npu_kernel(ggml_xdna_state & st, const ggml_tensor * src0,
                                                      int64_t n_feat, int64_t n_k, int64_t n_tok) {
    if (ggml_xdna_npu_use_w8(src0)) {
        if (ggml_xdna_kernel * k = ggml_xdna_select_kernel(st, n_feat, n_k, n_tok, true)) {
            return k;
        }
    }
    return ggml_xdna_select_kernel(st, n_feat, n_k, n_tok, false);
}

// weight rows [row0, row0 + n_rows) of plane (i02, i03) of src0 as int8 (row stride ne00) and their
// scales (row stride: groups per row), converted through bf16 a chunk of rows at a time
static void ggml_xdna_weight_rows_to_i8(ggml_backend_xdna_context * ctx, const ggml_tensor * src0,
                                        int64_t row0, int64_t n_rows, int64_t i02, int64_t i03,
                                        int8_t * q, float * scales) {
    const int64_t n_k   = src0->ne[0];
    const int64_t n_g   = (n_k + GGML_XDNA_W8_GROUP - 1)/GGML_XDNA_W8_GROUP;
    const int64_t chunk = 256;   // a multiple of the repacked layouts' 8-row groups
    std::vector<ggml_bf16_t> tmp;
    for (int64_t c0 = 0; c0 < n_rows; c0 += chunk) {
        const int64_t nr = std::min(chunk, n_rows - c0);
        tmp.resize((size_t) nr*n_k);
        ggml_xdna_weight_rows_to_bf16(ctx, src0, row0 + c0, nr, i02, i03, tmp.data(), n_k);
        ggml_xdna_parallel_for(ctx, nr, 8, [&](int64_t r0, int64_t r1) {
            for (int64_t r = r0; r < r1; r++) {
                const ggml_bf16_t * in  = tmp.data() + r*n_k;
                int8_t            * out = q + (c0 + r)*n_k;
                float             * sc  = scales + (c0 + r)*n_g;
                for (int64_t g = 0; g < n_g; g++) {
                    const int64_t k0 = g*GGML_XDNA_W8_GROUP;
                    const int64_t k1 = std::min(n_k, k0 + GGML_XDNA_W8_GROUP);
                    float amax = 0.0f;
                    for (int64_t k = k0; k < k1; k++) {
                        amax = std::max(amax, std::fabs(ggml_bf16_to_fp32(in[k])));
                    }
                    const float inv = amax > 0.0f ? 127.0f/amax : 0.0f;
                    for (int64_t k = k0; k < k1; k++) {
                        out[k] = (int8_t) std::min(127.0f, std::max(-127.0f, std::nearbyint(ggml_bf16_to_fp32(in[k])*inv)));
                    }
                    sc[g] = amax/127.0f;
                }
            }
        });
    }
}

// int8 counterpart of ggml_xdna_npu_weights: the leading rows of model weights come from the shared
// int8 cache, anything else is converted into the staging buffers; *scales gets their scales
static const int8_t * ggml_xdna_npu_weights_w8(ggml_backend_xdna_context * ctx, const ggml_tensor * src0,
                                               int64_t row0, int64_t n_rows, int64_t i02, int64_t i03,
                                               const float ** scales) {
    const int64_t n_k   = src0->ne[0];
    const int64_t n_g   = (n_k + GGML_XDNA_W8_GROUP - 1)/GGML_XDNA_W8_GROUP;
    const char *  plane = (const char *) src0->data + i02*src0->nb[2] + i03*src0->nb[3];

    if (row0 == 0 && ggml_xdna_is_weight(src0)) {
        ggml_xdna_state & st  = ggml_xdna_get_state();
        const uint64_t    sig = ggml_xdna_weight_sig(plane, src0->ne[1]*src0->nb[1], src0->type, n_k);

        std::lock_guard<std::mutex> lock(st.cache_mutex);
        auto it = st.npu_weights_w8.find(plane);
        if (it != st.npu_weights_w8.end() && it->second.sig != sig) {
            // the address was reused by different weights: drop the stale copy
            st.npu_weight_bytes -= it->second.q.size() + it->second.scales.size()*sizeof(float);
            st.npu_weights_w8.erase(it);
            it = st.npu_weights_w8.end();
        }
        const int64_t have = it != st.npu_weights_w8.end() ? it->second.n_rows : 0;
        if (have >= n_rows) {
            *scales = it->second.scales.data();
            return it->second.q.data();
        }
        const size_t add = (size_t) (n_rows - have)*(n_k + n_g*sizeof(float));
        if (st.npu_weight_bytes + add <= ggml_xdna_npu_cache_limit()) {
            try {
                auto & w = st.npu_weights_w8[plane];
                w.q.resize((size_t) n_rows*n_k);
                w.scales.resize((size_t) n_rows*n_g);
                ggml_xdna_weight_rows_to_i8(ctx, src0, have, n_rows - have, i02, i03, w.q.data() + have*n_k, w.scales.data() + have*n_g);
                w.n_rows = n_rows;
                w.sig    = sig;
                st.npu_weight_bytes += add;
                ctx->npu_cold = true;   // this op paid for a conversion: its time says nothing about the NPU's speed
                *scales = w.scales.data();
                return w.q.data();
            } catch (const std::bad_alloc &) {
                // out of memory: the entry keeps its rows, this call converts into the staging buffers
            }
        }
    }

    if ((int64_t) ctx->w_i8.size() < n_rows*n_k) {
        ctx->w_i8.resize(n_rows*n_k);
    }
    if ((int64_t) ctx->w_scales.size() < n_rows*n_g) {
        ctx->w_scales.resize(n_rows*n_g);
    }
    ggml_xdna_weight_rows_to_i8(ctx, src0, row0, n_rows, i02, i03, ctx->w_i8.data(), ctx->w_scales.data());
    *scales = ctx->w_scales.data();
    return ctx->w_i8.data();
}

// int8 versions of ggml_xdna_fill_block: into an int8 NPU block, or widened (exactly) into a bf16
// host-fallback block
static void ggml_xdna_fill_block_i8(int8_t * blk, int64_t rows, int64_t cols,
                                    const int8_t * src, int64_t src_stride, int64_t n_rows, int64_t n_cols) {
    if (n_rows < rows || n_cols < cols) {
        std::memset(blk, 0, rows*cols);
    }
    for (int64_t r = 0; r < n_rows; r++) {
        std::memcpy(blk + r*cols, src + r*src_stride, n_cols);
    }
}

static void ggml_xdna_fill_block_i8_bf16(ggml_bf16_t * blk, int64_t rows, int64_t cols,
                                         const int8_t * src, int64_t src_stride, int64_t n_rows, int64_t n_cols) {
    if (n_rows < rows || n_cols < cols) {
        std::memset(blk, 0, rows*cols*sizeof(ggml_bf16_t));
    }
    for (int64_t r = 0; r < n_rows; r++) {
        for (int64_t c = 0; c < n_cols; c++) {
            blk[r*cols + c] = ggml_fp32_to_bf16((float) src[r*src_stride + c]);
        }
    }
}

// compute dst rows [feat0, feat1) of a MUL_MAT on the NPU (host fallback if it fails). Blocks are
// pipelined over the kernel's two buffer slots: while the NPU computes block i, the host stages
// block i+1 and then accumulates the result of block i.
static void ggml_xdna_mul_mat_npu(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst, int64_t feat0, int64_t feat1) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_xdna_state & st = ggml_xdna_get_state();
    st.warm_stop = true;   // the NPU is in use: conversion on first use takes over from the warm-up

    const int64_t n_k    = ne00;           // reduction length
    const int64_t n_feat = feat1 - feat0;  // output features handled here (weight rows)
    const int64_t n_tok  = ne11;           // tokens (activation rows)

    ggml_xdna_kernel * kp = ggml_xdna_select_npu_kernel(st, src0, n_feat, n_k, n_tok);
    GGML_ASSERT(kp != nullptr && "supports_op admitted a MUL_MAT but no kernel is available");
    ggml_xdna_kernel & k = *kp;
    const bool w8 = k.w8;   // int8 weights, scaled by the host

    const int64_t kM = k.M, kK = k.K, kN = k.N;

    // the NPU is a single device and the kernel's slots are shared: hold the state lock for the op
    std::lock_guard<std::mutex> lock(st.mutex);

    const bool was_loaded = k.loaded;
    bool use_npu = ggml_xdna_kernel_load(st, k);
    if (use_npu && !was_loaded) {
        ctx->npu_cold = true;   // this op paid for loading the kernel: its time says nothing about the NPU's speed
    }
    if (!use_npu && !ctx->warned_fallback) {
        GGML_LOG_WARN("%s: NPU kernel unavailable, computing on the host instead\n", __func__);
        ctx->warned_fallback = true;
    }

    // the blocks of one plane, in the order their results are accumulated (k0 innermost)
    struct block { int64_t i0, m_rows, j0, n_rows, k0, kk; };
    std::vector<block> blocks;
    for (int64_t i0 = 0; i0 < n_feat; i0 += kM) {
        for (int64_t j0 = 0; j0 < n_tok; j0 += kN) {
            for (int64_t k0 = 0; k0 < n_k; k0 += kK) {
                blocks.push_back({ i0, std::min(kM, n_feat - i0), j0, std::min(kN, n_tok - j0), k0, std::min(kK, n_k - k0) });
            }
        }
    }

    if ((int64_t) ctx->x_bf16.size() < n_tok*n_k) {
        ctx->x_bf16.resize(n_tok*n_k);
    }

    std::vector<ggml_bf16_t> host_a, host_b;   // host-fallback staging, allocated on demand
    std::vector<float>       host_c;

    double  npu_us     = 0.0;
    int64_t npu_blocks = 0;

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    for (int64_t i13 = 0; i13 < ne13; i13++) {
        for (int64_t i12 = 0; i12 < ne12; i12++) {
            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            ggml_xdna_rows_to_bf16(ctx, src1, 0, n_tok, i12, i13, ctx->x_bf16.data(), n_k);

            // weights: bf16 rows, or int8 rows with one scale per row and GGML_XDNA_W8_GROUP weights
            const ggml_bf16_t * w  = nullptr;
            const int8_t      * wq = nullptr;
            const float       * ws = nullptr;
            if (w8) {
                wq = ggml_xdna_npu_weights_w8(ctx, src0, feat0, n_feat, i02, i03, &ws);
            } else {
                w = ggml_xdna_npu_weights(ctx, src0, feat0, n_feat, i02, i03);
            }
            const ggml_bf16_t * x = ctx->x_bf16.data();
            char * d_plane = (char *) dst->data + i12*nb2 + i13*nb3 + feat0*nb0;

            // stage a block's operands into an NPU slot ...
            auto fill = [&](ggml_xdna_kernel::slot & s, const block & bl) {
                if (w8) {
                    ggml_xdna_fill_block_i8(s.a8_map, kM, kK, wq + bl.i0*n_k + bl.k0, n_k, bl.m_rows, bl.kk);
                } else {
                    ggml_xdna_fill_block(s.a_map, kM, kK, w + bl.i0*n_k + bl.k0, n_k, bl.m_rows, bl.kk);
                }
                ggml_xdna_fill_block(s.b_map, kN, kK, x + bl.j0*n_k + bl.k0, n_k, bl.n_rows, bl.kk);
            };
            // ... or into the host-fallback buffers
            auto fill_host = [&](const block & bl) {
                if (w8) {
                    ggml_xdna_fill_block_i8_bf16(host_a.data(), kM, kK, wq + bl.i0*n_k + bl.k0, n_k, bl.m_rows, bl.kk);
                } else {
                    ggml_xdna_fill_block(host_a.data(), kM, kK, w + bl.i0*n_k + bl.k0, n_k, bl.m_rows, bl.kk);
                }
                ggml_xdna_fill_block(host_b.data(), kN, kK, x + bl.j0*n_k + bl.k0, n_k, bl.n_rows, bl.kk);
            };
            // C block is [kN x kM]: row t = token, col r = feature; int8 weights get their rows' scales
            // for this K block
            const int64_t n_g = (n_k + GGML_XDNA_W8_GROUP - 1)/GGML_XDNA_W8_GROUP;
            auto consume = [&](const float * c, const block & bl) {
                const float * sc = w8 ? ws + bl.i0*n_g + bl.k0/GGML_XDNA_W8_GROUP : nullptr;
                for (int64_t t = 0; t < bl.n_rows; t++) {
                    float       * drow = (float *) (d_plane + (bl.j0 + t)*nb1) + bl.i0;
                    const float * crow = c + t*kM;
                    if (w8) {
                        if (bl.k0 == 0) {
                            for (int64_t r = 0; r < bl.m_rows; r++) {
                                drow[r] = crow[r]*sc[r*n_g];
                            }
                        } else {
                            for (int64_t r = 0; r < bl.m_rows; r++) {
                                drow[r] += crow[r]*sc[r*n_g];
                            }
                        }
                    } else if (bl.k0 == 0) {
                        std::memcpy(drow, crow, bl.m_rows*sizeof(float));
                    } else {
                        for (int64_t r = 0; r < bl.m_rows; r++) {
                            drow[r] += crow[r];
                        }
                    }
                }
            };

            size_t i = 0;   // first block not yet accumulated into dst

            if (use_npu) {
                const auto t0 = std::chrono::steady_clock::now();
                try {
                    fill(k.slots[0], blocks[0]);
                    ggml_xdna_slot_start(k.slots[0]);
                    for (; i < blocks.size(); i++) {
                        auto & cur = k.slots[i % 2];
                        auto & nxt = k.slots[(i + 1) % 2];
                        if (i + 1 < blocks.size()) {
                            fill(nxt, blocks[i + 1]);
                            ggml_xdna_slot_start(nxt);
                        }
                        if (!ggml_xdna_slot_wait(cur)) {
                            throw std::runtime_error("kernel did not complete");
                        }
                        consume(cur.c_map, blocks[i]);
                    }
                    npu_us     += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
                    npu_blocks += (int64_t) blocks.size();
                } catch (const std::exception & e) {
                    GGML_LOG_WARN("%s: NPU dispatch failed (%s), finishing this op on the host\n", __func__, e.what());
                    use_npu = false;
                    for (auto & s : k.slots) {
                        try { s.run.wait(std::chrono::milliseconds(1000)); } catch (...) {}
                    }
                }
            }

            if (!use_npu && i < blocks.size()) {
                host_a.resize(kM*kK);
                host_b.resize(kN*kK);
                host_c.resize(kN*kM);
                for (; i < blocks.size(); i++) {
                    fill_host(blocks[i]);
                    ggml_xdna_block_gemm_host(host_a.data(), host_b.data(), host_c.data(), kM, kK, blocks[i].m_rows, blocks[i].n_rows);
                    consume(host_c.data(), blocks[i]);
                }
            }
        }
    }

    // effective per-block NPU time with the pipeline running, for the kernel picker; the first op
    // after loading includes one-time setup and is left out
    if (npu_blocks > 0 && k.n_ops++ > 0) {
        const double us = npu_us/npu_blocks;
        k.launch_us = k.launch_us > 0.0 ? 0.9*k.launch_us + 0.1*us : us;
    }
}

// load-time warm-up (see ggml_xdna_state): loads the kernels of the queued weights and converts the
// rows the NPU is expected to take, until the queue has been empty for a second, the cache is full,
// or the first NPU op starts
struct ggml_xdna_warm_stats {
    int    n_weights = 0;
    int    n_kernels = 0;
    size_t n_bytes   = 0;
    bool   full      = false;
};

static void ggml_xdna_warm_items(ggml_xdna_state & st, ggml_backend_xdna_context & wctx, ggml_xdna_warm_stats & stats) {
    int    & n_weights = stats.n_weights;
    int    & n_kernels = stats.n_kernels;
    size_t & n_bytes   = stats.n_bytes;
    bool   & full      = stats.full;

    for (;;) {
        ggml_xdna_warm_item item;
        float share;
        {
            std::unique_lock<std::mutex> lock(st.warm_mutex);
            st.warm_cv.wait_for(lock, std::chrono::seconds(1), [&]() { return !st.warm_queue.empty() || st.warm_stop; });
            if (st.warm_queue.empty() || st.warm_stop || full) {
                st.warm_queue.clear();
                st.warm_running = false;
                break;
            }
            item = st.warm_queue.front();
            st.warm_queue.erase(st.warm_queue.begin());
            share = st.warm_share;
        }

        const ggml_tensor & t      = item.t;
        const int64_t       n_k    = t.ne[0];
        const int64_t       n_feat = t.ne[1];

        // the rows the NPU is expected to take, rounded to its kernel's blocks like a fixed split
        const int64_t target = std::max<int64_t>(1, (int64_t) (n_feat*share));
        ggml_xdna_kernel * kp = ggml_xdna_select_npu_kernel(st, &t, target, n_k, item.n_tok);
        if (kp == nullptr) {
            continue;
        }
        int64_t rows = std::min((target + kp->M/2)/kp->M*kp->M, n_feat);
        if (rows > 0 && n_feat - rows < 64) {
            rows = n_feat;
        }
        if (rows == 0) {
            continue;
        }

        // the kernel for those rows, and the one the auto split plans the whole op with
        for (ggml_xdna_kernel * k : { ggml_xdna_select_npu_kernel(st, &t, rows, n_k, item.n_tok), ggml_xdna_select_npu_kernel(st, &t, n_feat, n_k, item.n_tok) }) {
            if (k != nullptr && !st.warm_stop) {
                std::lock_guard<std::mutex> lock(st.mutex);
                if (!k->loaded && !k->broken && ggml_xdna_kernel_load(st, *k)) {
                    n_kernels++;
                }
            }
        }

        if (t.type == GGML_TYPE_BF16 && t.nb[1] == (size_t) n_k*sizeof(ggml_bf16_t)) {
            continue;   // used in place
        }

        const bool    w8       = kp->w8;   // int8 weights when enabled and int8 kernels exist
        const bool    repacked = ggml_xdna_is_cpu_repack(&t);
        const int64_t n_i02    = repacked ? 1 : t.ne[2];
        const int64_t n_i03    = repacked ? 1 : t.ne[3];
        const int64_t n_g      = (n_k + GGML_XDNA_W8_GROUP - 1)/GGML_XDNA_W8_GROUP;
        const size_t  bytes    = w8 ? (size_t) rows*(n_k + n_g*sizeof(float)) : (size_t) rows*n_k*sizeof(ggml_bf16_t);
        for (int64_t i03 = 0; i03 < n_i03 && !st.warm_stop && !full; i03++) {
            for (int64_t i02 = 0; i02 < n_i02 && !st.warm_stop && !full; i02++) {
                const char * plane = (const char *) t.data + i02*t.nb[2] + i03*t.nb[3];
                {
                    std::lock_guard<std::mutex> lock(st.cache_mutex);
                    if ((w8 ? st.npu_weights_w8.count(plane) : st.npu_weights.count(plane)) != 0) {
                        continue;
                    }
                    if (st.npu_weight_bytes + bytes > ggml_xdna_npu_cache_limit()) {
                        full = true;
                        break;
                    }
                    st.npu_weight_bytes += bytes;   // reserved while converting
                }

                const uint64_t sig = ggml_xdna_weight_sig(plane, n_feat*t.nb[1], t.type, n_k);
                bool inserted = false;
                if (w8) {
                    std::vector<int8_t> q((size_t) rows*n_k);
                    std::vector<float>  sc((size_t) rows*n_g);
                    ggml_xdna_weight_rows_to_i8(&wctx, &t, 0, rows, i02, i03, q.data(), sc.data());

                    std::lock_guard<std::mutex> lock(st.cache_mutex);
                    auto ins = st.npu_weights_w8.try_emplace(plane);
                    inserted = ins.second;
                    if (inserted) {
                        ins.first->second.q      = std::move(q);
                        ins.first->second.scales = std::move(sc);
                        ins.first->second.n_rows = rows;
                        ins.first->second.sig    = sig;
                    } else {
                        st.npu_weight_bytes -= bytes;   // an op converted it first
                    }
                } else {
                    std::vector<ggml_bf16_t> data((size_t) rows*n_k);
                    ggml_xdna_weight_rows_to_bf16(&wctx, &t, 0, rows, i02, i03, data.data(), n_k);

                    std::lock_guard<std::mutex> lock(st.cache_mutex);
                    auto ins = st.npu_weights.try_emplace(plane);
                    inserted = ins.second;
                    if (inserted) {
                        ins.first->second.data   = std::move(data);
                        ins.first->second.n_rows = rows;
                        ins.first->second.sig    = sig;
                    } else {
                        st.npu_weight_bytes -= bytes;   // an op converted it first
                    }
                }
                if (inserted) {
                    n_weights++;
                    n_bytes += bytes;
                }
            }
        }
    }
}

static void ggml_xdna_warm_run() {
    ggml_xdna_state & st = ggml_xdna_get_state();

    ggml_backend_xdna_context wctx;   // only for ggml_xdna_parallel_for
    wctx.n_threads = std::max(1, (int) std::thread::hardware_concurrency()/2);

    const auto t_start = std::chrono::steady_clock::now();
    ggml_xdna_warm_stats stats;
    try {
        ggml_xdna_warm_items(st, wctx, stats);
    } catch (const std::exception & e) {
        // out of memory (Windows without a page file fails allocations outright) or no thread for
        // the conversion: stop warming, the remaining weights are converted on first use
        GGML_LOG_WARN("%s: warm-up stopped: %s\n", __func__, e.what());
        std::lock_guard<std::mutex> lock(st.warm_mutex);
        st.warm_queue.clear();
        st.warm_running = false;
    }

    if (stats.n_weights > 0 || stats.n_kernels > 0) {
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
        GGML_LOG_INFO("%s: loaded %d NPU kernel(s) and converted %d weight(s) (%.0f MB) in %.0f ms%s\n",
                      __func__, stats.n_kernels, stats.n_weights, stats.n_bytes/1048576.0, ms,
                      stats.full ? " (cache full)" : st.warm_stop ? " (stopped early: NPU in use or backend freed)" : "");
    }
}

// supports_op admitted a MUL_MAT on this weight: queue it for the warm-up (once per weight)
static void ggml_xdna_warm_enqueue(const ggml_tensor * src0, int64_t n_tok) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    if (st.warm_stop) {
        return;
    }
    // only weights backed by real memory: llama.cpp also builds graphs over a model loaded with
    // no_alloc (e.g. to fit the context to memory), whose weights sit in a size-0 dummy buffer
    // tagged as weights, with no data
    const char * base = src0->buffer != nullptr ? (const char *) ggml_backend_buffer_get_base(src0->buffer) : nullptr;
    const size_t size = src0->buffer != nullptr ? ggml_backend_buffer_get_size(src0->buffer) : 0;
    if (src0->data == nullptr || base == nullptr || size < ggml_nbytes(src0) ||
        (const char *) src0->data < base || (const char *) src0->data + ggml_nbytes(src0) > base + size) {
        return;
    }
    std::lock_guard<std::mutex> lock(st.warm_mutex);
    if (st.warm_stop || st.warm_share <= 0.0f || st.warm_seen.count(src0->data) != 0) {
        return;
    }
    st.warm_seen[src0->data] = true;
    st.warm_queue.push_back({ *src0, n_tok });
    if (st.warm_running) {
        st.warm_cv.notify_one();
        return;
    }
    if (st.warm_thread.joinable()) {
        st.warm_thread.join();   // a previous run that found its queue empty and has finished
    }
    st.warm_running = true;
    st.warm_thread  = std::thread(ggml_xdna_warm_run);
}

// the share of each op's rows the NPU is expected to take, for the warm-up (0: no warm-up):
// an explicit share, none next to a GPU worker, and about half with the auto split, which settles
// between 0.3 and 0.5 on the 8700G (any rows beyond the warmed ones are converted on first use)
static float ggml_xdna_warm_share(const ggml_backend_xdna_context * ctx) {
    if (const char * env = ggml_xdna_getenv("GGML_XDNA_NPU_WARM")) {
        if (std::atoi(env) == 0) {
            return 0.0f;
        }
    }
    if (ggml_xdna_npu_cache_limit() == 0) {
        return 0.0f;
    }
    if (ctx->share_fixed) {
        return ctx->share[GGML_XDNA_WORKER_NPU];
    }
    return ctx->vk != nullptr ? 0.0f : 0.5f;
}

static void ggml_xdna_warm_acquire(const ggml_backend_xdna_context * ctx) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    const float share = ggml_xdna_warm_share(ctx);
    std::lock_guard<std::mutex> lock(st.warm_mutex);
    if (st.n_backends++ == 0) {
        st.warm_share = share;
        st.warm_stop  = share <= 0.0f;
    }
}

// a backend is going away: when it is the last one, stop the warm-up and drop the shared cache
// (the model's weights may be freed next)
static void ggml_xdna_warm_release() {
    ggml_xdna_state & st = ggml_xdna_get_state();
    std::unique_lock<std::mutex> lock(st.warm_mutex);
    if (--st.n_backends > 0) {
        return;
    }
    st.warm_stop = true;
    st.warm_cv.notify_all();
    std::thread th = std::move(st.warm_thread);
    lock.unlock();
    if (th.joinable()) {
        th.join();
    }
    lock.lock();
    st.warm_running = false;
    st.warm_queue.clear();
    st.warm_seen.clear();
    lock.unlock();

    std::lock_guard<std::mutex> cache_lock(st.cache_mutex);
    st.npu_weights.clear();
    st.npu_weights_w8.clear();
    st.npu_weight_bytes = 0;
}

// shape key for the auto split; repacked and plain weights of the same shape run at different speeds
static std::tuple<int64_t, int64_t, int> ggml_xdna_shape_key(const struct ggml_tensor * src0) {
    return std::make_tuple(src0->ne[1], src0->ne[0], (int) src0->type + (ggml_xdna_is_cpu_repack(src0) ? 1000 : 0));
}

// auto split: choose how many rows each worker gets by predicted finish time. The NPU takes whole
// kernel blocks (none, some, or all rows); the rest is shared by the Vulkan and CPU workers in
// proportion to their speed. Speeds are learned per matrix shape; for a shape not seen yet the NPU
// is estimated from its kernel's per-block time and the other workers from their average MAC rate.
// Returns false until the Vulkan/CPU workers have been measured once (the caller then uses the
// default shares, which measures them).
static bool ggml_xdna_plan_auto(ggml_backend_xdna_context * ctx, ggml_xdna_state & st, const struct ggml_tensor * dst,
                                bool npu_ok, int64_t n[], double * t_pred) {
    const struct ggml_tensor * src0 = dst->src[0];
    const int64_t n_feat = src0->ne[1];
    const int64_t n_k    = src0->ne[0];
    const int64_t n_tok  = dst->src[1]->ne[1];

    // the Vulkan worker cannot read the CPU's repacked layout
    const bool vk_on  = ctx->vk  != nullptr && ctx->share[GGML_XDNA_WORKER_VK]  > 0.0f && !ggml_xdna_is_cpu_repack(src0);
    const bool cpu_on = ctx->cpu != nullptr && ctx->share[GGML_XDNA_WORKER_CPU] > 0.0f;

    const auto     it  = ctx->shape_perf_map.find(ggml_xdna_shape_key(src0));
    const double * thr = it != ctx->shape_perf_map.end() ? it->second.thr : nullptr;

    // ms per row on worker i, or -1 if it has never been measured
    auto ms_per_row = [&](int i) -> double {
        if (thr != nullptr && thr[i] > 0.0) {
            return (double) n_tok/thr[i];
        }
        if (ctx->mac_thr[i] > 0.0) {
            return (double) n_k*n_tok/ctx->mac_thr[i];
        }
        return -1.0;
    };

    const double r_vk  = vk_on  ? ms_per_row(GGML_XDNA_WORKER_VK)  : 0.0;
    const double r_cpu = cpu_on ? ms_per_row(GGML_XDNA_WORKER_CPU) : 0.0;
    if ((!vk_on && !cpu_on) || r_vk < 0.0 || r_cpu < 0.0) {
        return false;
    }
    // rows per ms of the Vulkan and CPU workers together
    const double rest_rate = (vk_on ? 1.0/r_vk : 0.0) + (cpu_on ? 1.0/r_cpu : 0.0);

    const ggml_xdna_kernel * kp = npu_ok ? ggml_xdna_select_npu_kernel(st, src0, n_feat, n_k, n_tok) : nullptr;
    const double npu_ms_per_row = thr != nullptr && thr[GGML_XDNA_WORKER_NPU] > 0.0 ? (double) n_tok/thr[GGML_XDNA_WORKER_NPU] : -1.0;
    auto t_npu = [&](int64_t c) -> double {
        if (npu_ms_per_row > 0.0) {
            return c*npu_ms_per_row;
        }
        const double blocks = (double) ((c + kp->M - 1)/kp->M) * ((n_k + kp->K - 1)/kp->K) * ((n_tok + kp->N - 1)/kp->N);
        return blocks*ggml_xdna_kernel_launch_us(*kp)/1000.0;
    };

    int64_t best_c = 0;
    double  best_t = (double) n_feat/rest_rate;
    if (kp != nullptr) {
        for (int64_t c = kp->M; ; c += kp->M) {
            c = std::min(c, n_feat);
            const double t = std::max(t_npu(c), (n_feat - c)/rest_rate);
            if (t < best_t) {
                best_t = t;
                best_c = c;
            }
            if (c == n_feat) {
                break;
            }
        }
    }

    // a worker left out is never measured again, so a single bad sample (a busy moment, some
    // first-use cost) would keep the NPU off this shape for good: every 16th op without it, give
    // it one block so its estimate can recover
    if (best_c == 0 && kp != nullptr && npu_ms_per_row > 0.0 && it != ctx->shape_perf_map.end()) {
        if (++it->second.npu_skips % 16 == 0) {
            best_c = std::min(kp->M, n_feat);
            best_t = std::max(t_npu(best_c), (n_feat - best_c)/rest_rate);
        }
    }

    const int64_t rest = n_feat - best_c;
    n[GGML_XDNA_WORKER_NPU] = best_c;
    n[GGML_XDNA_WORKER_VK]  = 0;
    if (vk_on && cpu_on) {
        n[GGML_XDNA_WORKER_VK] = std::min(rest, (int64_t) (rest*(1.0/r_vk)/rest_rate + 32) / 64 * 64);
    } else if (vk_on) {
        n[GGML_XDNA_WORKER_VK] = rest;
    }
    n[GGML_XDNA_WORKER_CPU] = rest - n[GGML_XDNA_WORKER_VK];
    *t_pred = best_t;
    return true;
}

static void ggml_backend_xdna_mul_mat(ggml_backend_xdna_context * ctx, struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(src0->type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == sizeof(float));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    const int64_t n_feat = ne01;

    ggml_xdna_init_shares(ctx);

    // split the output features into consecutive ranges: NPU, Vulkan, CPU.
    ggml_xdna_state & st = ggml_xdna_get_state();

    // the NPU part is rounded to whole blocks of the kernel it will run on - a partial block
    // costs the NPU as much as a full one - so a small share can round down to nothing on a
    // small op; an op too small to amortize the launch cost skips the NPU when others exist
    const double gflop = 2.0*n_feat*ne00*ne11/1e9;
    const bool npu_worthwhile = gflop >= ctx->npu_min_gflop || (ctx->vk == nullptr && ctx->cpu == nullptr);

    const float share_used[GGML_XDNA_N_WORKERS] = {
        ctx->share[GGML_XDNA_WORKER_NPU], ctx->share[GGML_XDNA_WORKER_VK], ctx->share[GGML_XDNA_WORKER_CPU] };

    int64_t n[GGML_XDNA_N_WORKERS] = { 0, 0, 0 };

    const char * mode    = "fixed";
    double       t_pred  = 0.0;
    bool         planned = false;
    if (!ctx->share_fixed && (ctx->vk != nullptr || ctx->cpu != nullptr)) {
        planned = ggml_xdna_plan_auto(ctx, st, dst, npu_worthwhile, n, &t_pred);
        mode    = planned ? "auto" : "explore";
    }

    if (!planned) {
        if (ctx->share[GGML_XDNA_WORKER_NPU] >= 1.0f || (ctx->vk == nullptr && ctx->cpu == nullptr)) {
            n[GGML_XDNA_WORKER_NPU] = n_feat;
        } else if (ctx->share[GGML_XDNA_WORKER_NPU] > 0.0f && npu_worthwhile) {
            const int64_t target = std::max<int64_t>(1, (int64_t) (n_feat*ctx->share[GGML_XDNA_WORKER_NPU]));
            const ggml_xdna_kernel * kp = ggml_xdna_select_npu_kernel(st, src0, target, ne00, ne11);
            const int64_t kM = kp ? kp->M : 512;
            n[GGML_XDNA_WORKER_NPU] = std::min((target + kM/2)/kM*kM, n_feat);
            if (n[GGML_XDNA_WORKER_NPU] > 0 && n_feat - n[GGML_XDNA_WORKER_NPU] < 64) {
                n[GGML_XDNA_WORKER_NPU] = n_feat;
            }
        }
        if (ctx->vk && ctx->share[GGML_XDNA_WORKER_VK] > 0.0f && !ggml_xdna_is_cpu_repack(src0)) {
            n[GGML_XDNA_WORKER_VK] = (int64_t) (n_feat*ctx->share[GGML_XDNA_WORKER_VK] + 32) / 64 * 64;
            n[GGML_XDNA_WORKER_VK] = std::min(n[GGML_XDNA_WORKER_VK], n_feat - n[GGML_XDNA_WORKER_NPU]);
        }
    }
    n[GGML_XDNA_WORKER_CPU] = n_feat - n[GGML_XDNA_WORKER_NPU] - n[GGML_XDNA_WORKER_VK];
    if (n[GGML_XDNA_WORKER_CPU] > 0 && (ctx->cpu == nullptr || n[GGML_XDNA_WORKER_CPU] < 16)) {
        // no CPU worker, or a sliver not worth a thread pool: hand it to the GPU, else the NPU
        n[ctx->vk && n[GGML_XDNA_WORKER_VK] > 0 ? GGML_XDNA_WORKER_VK : GGML_XDNA_WORKER_NPU] += n[GGML_XDNA_WORKER_CPU];
        n[GGML_XDNA_WORKER_CPU] = 0;
    }

    const int64_t f_npu0 = 0,                            f_npu1 = f_npu0 + n[GGML_XDNA_WORKER_NPU];
    const int64_t f_vk0  = f_npu1,                       f_vk1  = f_vk0  + n[GGML_XDNA_WORKER_VK];
    const int64_t f_cpu0 = f_vk1,                        f_cpu1 = n_feat;

    const auto t_start = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(); };
    double t[GGML_XDNA_N_WORKERS] = { 0.0, 0.0, 0.0 };
    bool   vk_ok = true, cpu_ok = true;

    ctx->npu_cold = false;
    std::thread npu_thread, vk_thread;
    if (n[GGML_XDNA_WORKER_NPU] > 0) {
        npu_thread = std::thread([&]() {
            ggml_xdna_mul_mat_npu(ctx, dst, f_npu0, f_npu1);
            t[GGML_XDNA_WORKER_NPU] = elapsed();
        });
    }
    if (n[GGML_XDNA_WORKER_VK] > 0) {
        vk_thread = std::thread([&]() {
            vk_ok = ggml_xdna_mul_mat_vk(ctx, dst, f_vk0, f_vk1);
            t[GGML_XDNA_WORKER_VK] = elapsed();
        });
    }
    if (n[GGML_XDNA_WORKER_CPU] > 0) {
        cpu_ok = ggml_xdna_mul_mat_cpu(ctx, dst, f_cpu0, f_cpu1);
        t[GGML_XDNA_WORKER_CPU] = elapsed();
    }
    if (npu_thread.joinable()) { npu_thread.join(); }
    if (vk_thread.joinable())  { vk_thread.join();  }

    // a worker that failed leaves its rows to the NPU (always available once we got here)
    if (!vk_ok) {
        GGML_LOG_WARN("%s: Vulkan worker failed, finishing its rows on the NPU\n", __func__);
        ggml_xdna_mul_mat_npu(ctx, dst, f_vk0, f_vk1);
        ctx->share[GGML_XDNA_WORKER_NPU] += ctx->share[GGML_XDNA_WORKER_VK];
        ctx->share[GGML_XDNA_WORKER_VK]   = 0.0f;
    }
    if (!cpu_ok) {
        ggml_xdna_mul_mat_npu(ctx, dst, f_cpu0, f_cpu1);
        ctx->share[GGML_XDNA_WORKER_NPU] += ctx->share[GGML_XDNA_WORKER_CPU];
        ctx->share[GGML_XDNA_WORKER_CPU]  = 0.0f;
    }

    if (!ctx->share_fixed && vk_ok && cpu_ok) {
        // learn each worker's speed for this matrix shape (rows*tokens per ms) and overall (MACs per ms)
        // the NPU's first run of a shape carries first-use costs, like a kernel load or conversion
        // (npu_cold), so neither is learned from
        auto & sp = ctx->shape_perf_map[ggml_xdna_shape_key(src0)];
        const bool npu_first = !sp.npu_seen;
        if (n[GGML_XDNA_WORKER_NPU] > 0) {
            sp.npu_seen = true;
        }
        for (int i = 0; i < GGML_XDNA_N_WORKERS; i++) {
            if (n[i] > 0 && t[i] > 0.0 && !(i == GGML_XDNA_WORKER_NPU && (ctx->npu_cold || npu_first))) {
                const double thr = (double) n[i]*ne11/(t[i]*1e3);
                sp.thr[i]       = sp.thr[i]       > 0.0 ? 0.7*sp.thr[i]       + 0.3*thr      : thr;
                ctx->mac_thr[i] = ctx->mac_thr[i] > 0.0 ? 0.9*ctx->mac_thr[i] + 0.1*thr*ne00 : thr*ne00;
            }
        }
    }

    if (ggml_xdna_debug()) {
        static int op_id = 0;
        GGML_LOG_INFO("xdna_split: op=%d feat=%" PRId64 " k=%" PRId64 " tok=%" PRId64 " rows=%" PRId64 "/%" PRId64 "/%" PRId64
                      " t_ms=%.3f/%.3f/%.3f share=%.3f/%.3f/%.3f -> %.3f/%.3f/%.3f mode=%s pred_ms=%.3f cold=%d\n",
                      op_id++, n_feat, ne00, ne11,
                      n[GGML_XDNA_WORKER_NPU], n[GGML_XDNA_WORKER_VK], n[GGML_XDNA_WORKER_CPU],
                      t[GGML_XDNA_WORKER_NPU]*1e3, t[GGML_XDNA_WORKER_VK]*1e3, t[GGML_XDNA_WORKER_CPU]*1e3,
                      share_used[GGML_XDNA_WORKER_NPU], share_used[GGML_XDNA_WORKER_VK], share_used[GGML_XDNA_WORKER_CPU],
                      ctx->share[GGML_XDNA_WORKER_NPU], ctx->share[GGML_XDNA_WORKER_VK], ctx->share[GGML_XDNA_WORKER_CPU],
                      mode, t_pred, (int) ctx->npu_cold);
    }
}

// -------------------------------------------------------------------------------------------------
// backend interface

static const char * ggml_backend_xdna_get_name(ggml_backend_t backend) {
    return "XDNA";

    GGML_UNUSED(backend);
}

static void ggml_backend_xdna_free(ggml_backend_t backend) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;
    ggml_xdna_warm_release();
    if (ctx->cpu) {
        ggml_backend_free(ctx->cpu);
    }
    for (auto & kv : ctx->vk_imports) {
        ggml_backend_buffer_free(kv.second);
    }
    if (ctx->vk_out != nullptr) {
        GGML_XDNA_PAGE_FREE(ctx->vk_out);   // after its import was released above
    }
    for (auto & kv : ctx->vk_weights) {
        ggml_backend_buffer_free(kv.second.buf);
        ggml_free(kv.second.wctx);
    }
    if (ctx->vk) {
        ggml_backend_free(ctx->vk);
    }
    delete ctx;
    delete backend;
}

static enum ggml_status ggml_backend_xdna_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_backend_xdna_mul_mat(ctx, node);
                break;

            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;

            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i xdna_backend_i = {
    /* .get_name                = */ ggml_backend_xdna_get_name,
    /* .free                    = */ ggml_backend_xdna_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_xdna_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_xdna_guid(void) {
    static ggml_guid guid = { 0x7c, 0x0f, 0x3a, 0x91, 0x5e, 0xd2, 0x4b, 0x86, 0xa4, 0x19, 0xc7, 0x62, 0x0e, 0xb5, 0x88, 0x31 };
    return &guid;
}

ggml_backend_t ggml_backend_xdna_init(void) {
    ggml_xdna_probe(ggml_xdna_get_state());

    ggml_backend_xdna_context * ctx = new ggml_backend_xdna_context;
    ggml_xdna_init_shares(ctx);    // creates the workers now, so the warm-up knows the NPU's share
    ggml_xdna_warm_acquire(ctx);

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_xdna_guid(),
        /* .iface   = */ xdna_backend_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_xdna_reg(), 0),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_xdna(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_xdna_guid());
}

void ggml_backend_xdna_set_n_threads(ggml_backend_t backend_xdna, int n_threads) {
    GGML_ASSERT(ggml_backend_is_xdna(backend_xdna));

    ggml_backend_xdna_context * ctx = (ggml_backend_xdna_context *) backend_xdna->context;
    ctx->n_threads = n_threads;
}

// device interface

static const char * ggml_backend_xdna_device_get_name(ggml_backend_dev_t dev) {
    return "XDNA";

    GGML_UNUSED(dev);
}

static const char * ggml_backend_xdna_device_get_description(ggml_backend_dev_t dev) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    ggml_xdna_probe(st);
    return st.description.c_str();

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    // operands live in host memory
    *free  = 0;
    *total = 0;

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_xdna_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;

    GGML_UNUSED(dev);
}

static void ggml_backend_xdna_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_xdna_device_get_name(dev);
    props->description = ggml_backend_xdna_device_get_description(dev);
    props->type        = ggml_backend_xdna_device_get_type(dev);
    ggml_backend_xdna_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
        /* .mmap_support          = */ true,
    };
}

static ggml_backend_t ggml_backend_xdna_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    return ggml_backend_xdna_init();

    GGML_UNUSED(dev);
    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_xdna_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(dev);
}

static ggml_backend_buffer_t ggml_backend_xdna_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);

    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
}

static bool ggml_backend_xdna_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT:
        {
            ggml_xdna_state & st = ggml_xdna_get_state();
            ggml_xdna_probe(st);
            if (!st.available || st.kernels.empty()) {
                return false;
            }

            const struct ggml_tensor * src0 = op->src[0];
            const struct ggml_tensor * src1 = op->src[1];

            // default back to CPU fast path
            // see: https://github.com/ggml-org/llama.cpp/issues/25565
            if (ggml_get_op_params_i32(op, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
                return false;
            }

            const int64_t n_k    = src0->ne[0];
            const int64_t n_feat = src0->ne[1];
            const int64_t n_tok  = src1->ne[1];

            // per-launch overhead makes small products a loss; decode-sized (n_tok == 1) never qualifies
            const int64_t min_dim = 32;

            const bool ok = ggml_is_contiguous(src0) &&
                   ggml_is_contiguous(src1) &&
                   src1->type == GGML_TYPE_F32 &&
                   op->type   == GGML_TYPE_F32 &&
                   (n_tok >= st.min_batch && n_k >= min_dim && n_feat >= min_dim) &&
                   (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 ||
                    ggml_get_type_traits(src0->type)->to_float != NULL) &&
                   // repacked weights: only the layouts the NPU side can convert back (Q4_0, Q4_K; 8-row groups)
                   (!ggml_xdna_is_cpu_repack(src0) ||
                    (ggml_xdna_repack_enabled() && ggml_n_dims(src0) == 2 && n_feat % 8 == 0 &&
                     ((src0->type == GGML_TYPE_Q4_0 && n_k % 32 == 0) || (src0->type == GGML_TYPE_Q4_K && n_k % 256 == 0))));

            if (ok && ggml_xdna_is_weight(src0)) {
                ggml_xdna_warm_enqueue(src0, n_tok);
            }

            // the scheduler asks again on every graph build: log each distinct decision once, so that
            // debug runs are not slowed down by thousands of repeated lines
            static std::mutex logged_mutex;
            static std::map<std::tuple<int, int, int64_t, int64_t, int64_t, int, bool>, bool> logged;
            bool log_it = false;
            if (ggml_xdna_debug()) {
                std::lock_guard<std::mutex> lock(logged_mutex);
                log_it = logged.emplace(std::make_tuple((int) src0->type, (int) ggml_xdna_is_cpu_repack(src0), n_k, n_feat, n_tok,
                                                        ggml_is_contiguous(src0) + 2*ggml_is_contiguous(src1), ok), true).second;
            }
            if (log_it) {
                GGML_LOG_INFO("%s: MUL_MAT %s[%" PRId64 ",%" PRId64 "] x %s[%" PRId64 ",%" PRId64 "] -> %s cont=%d/%d min_batch=%" PRId64 " : %s\n",
                              __func__, ggml_type_name(src0->type), n_k, n_feat, ggml_type_name(src1->type), src1->ne[0], n_tok,
                              ggml_type_name(op->type), ggml_is_contiguous(src0), ggml_is_contiguous(src1), st.min_batch,
                              ok ? "yes" : "no");
            }
            return ok;
        }

        default:
            return false;
    }

    GGML_UNUSED(dev);
}

static bool ggml_backend_xdna_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    // repacked weights are readable too: the NPU converts its rows back, the CPU worker keeps them
    return ggml_backend_buft_is_host(buft) || (ggml_xdna_repack_enabled() && ggml_xdna_buft_is_cpu_repack(buft));

    GGML_UNUSED(dev);
}

static const struct ggml_backend_device_i ggml_backend_xdna_device_i = {
    /* .get_name             = */ ggml_backend_xdna_device_get_name,
    /* .get_description      = */ ggml_backend_xdna_device_get_description,
    /* .get_memory           = */ ggml_backend_xdna_device_get_memory,
    /* .get_type             = */ ggml_backend_xdna_device_get_type,
    /* .get_props            = */ ggml_backend_xdna_device_get_props,
    /* .init_backend         = */ ggml_backend_xdna_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_xdna_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_xdna_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_xdna_device_supports_op,
    /* .supports_buft        = */ ggml_backend_xdna_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

static const char * ggml_backend_xdna_reg_get_name(ggml_backend_reg_t reg) {
    return "XDNA";

    GGML_UNUSED(reg);
}

static size_t ggml_backend_xdna_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_xdna_state & st = ggml_xdna_get_state();
    ggml_xdna_probe(st);
    return st.available ? 1 : 0;

    GGML_UNUSED(reg);
}

static ggml_backend_dev_t ggml_backend_xdna_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device ggml_backend_xdna_device = {
        /* .iface   = */ ggml_backend_xdna_device_i,
        /* .reg     = */ reg,
        /* .context = */ nullptr,
    };

    return &ggml_backend_xdna_device;

    GGML_UNUSED(reg);
    GGML_UNUSED(index);
}

static void * ggml_backend_xdna_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_set_n_threads") == 0) {
        return (void *) ggml_backend_xdna_set_n_threads;
    }
    return NULL;

    GGML_UNUSED(reg);
    GGML_UNUSED(name);
}

static const struct ggml_backend_reg_i ggml_backend_xdna_reg_i = {
    /* .get_name         = */ ggml_backend_xdna_reg_get_name,
    /* .get_device_count = */ ggml_backend_xdna_reg_get_device_count,
    /* .get_device       = */ ggml_backend_xdna_reg_get_device,
    /* .get_proc_address = */ ggml_backend_xdna_get_proc_address,
};

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
    static struct ggml_backend_reg ggml_backend_xdna_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_xdna_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_xdna_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
