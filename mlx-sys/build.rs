extern crate cmake;

use bindgen::RustTarget;
use cmake::Config;
use std::{
    env, fs,
    path::{Path, PathBuf},
};

/// Find a file by exact name anywhere under `root`.
///
/// FetchContent's layout under `_deps/<name>-{src,build}` varies per dependency
/// and per generator, and a static archive records no paths, so MLX's PRIVATE
/// dependencies have to be located rather than predicted.
fn find_file(root: &Path, name: &str) -> Option<PathBuf> {
    let entries = fs::read_dir(root).ok()?;
    let mut dirs = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            dirs.push(path);
        } else if path.file_name().is_some_and(|f| f == name) {
            return Some(path);
        }
    }
    dirs.iter().find_map(|d| find_file(d, name))
}

/// Emit a link-search for the directory holding `name`, and link it by stem.
fn link_found(root: &Path, name: &str, what: &str) {
    match find_file(root, name) {
        Some(path) => {
            let dir = path.parent().unwrap();
            let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
            println!("cargo:rustc-link-search=native={}", dir.display());
            println!("cargo:rustc-link-lib=static={stem}");
            println!("cargo:warning=mlx-sys: linking {what} from {}", path.display());
        }
        None => println!("cargo:warning=mlx-sys: {what} ({name}) not found under {}", root.display()),
    }
}

/// Rename a CuTe member alias that MSVC 19.44 cannot compile.
///
/// `cute/stride.hpp:299` names `Lambda::template seq<Shape>`. The enclosing
/// namespace also declares `template <int... Ints> struct seq;`; MSVC binds to
/// that and reports
///
///     error C3545: 'Ints': parameter pack expects a non-type template argument
///
/// Affects every CUTLASS release from v4.4.2 (MLX's pin) through v4.7.0.
/// Renaming the member alias is sufficient; the tree is substituted through
/// FetchContent so the change survives `cargo clean`.
fn patched_cutlass(out_dir: &Path) -> Option<PathBuf> {
    // Must track the GIT_TAG in mlx/backend/cuda/CMakeLists.txt.
    const CUTLASS_TAG: &str = "v4.4.2";

    let dir = out_dir.join("cutlass-msvc");
    let stride = dir.join("include/cute/stride.hpp");
    if !stride.exists() {
        let status = std::process::Command::new("git")
            .args([
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                CUTLASS_TAG,
                "https://github.com/NVIDIA/cutlass.git",
            ])
            .arg(&dir)
            .status()
            .ok()?;
        if !status.success() {
            println!("cargo:warning=mlx-sys: could not clone CUTLASS {CUTLASS_TAG}");
            return None;
        }
    }

    let src = fs::read_to_string(&stride).ok()?;
    let patched = src
        .replace(
            "typename Lambda::template seq<Shape>",
            "typename Lambda::template cute_compact_seq<Shape>",
        )
        .replace(
            "using seq = tuple_seq<Shape>;",
            "using cute_compact_seq = tuple_seq<Shape>;",
        )
        .replace(
            "using seq = tuple_rseq<Shape>;",
            "using cute_compact_seq = tuple_rseq<Shape>;",
        );
    if patched != src {
        fs::write(&stride, patched).ok()?;
    }
    Some(dir)
}

/// The MLX this crate builds. Must track the GIT_TAG of the `mlx`
/// FetchContent_Declare in src/mlx-c/CMakeLists.txt.
const MLX_TAG: &str = "v0.32.4";

/// Where a completed build records that every FetchContent dependency is
/// present in OUT_DIR, so later configures can skip downloading them.
///
/// MLX declares CCCL, nlohmann json and metal-cpp as URL archives with no
/// hash, and FetchContent downloads and re-extracts those on every configure.
/// Cargo configures again whenever this script reruns, and CMP0135 stamps the
/// extracted headers with the extraction time, so ninja sees every CUDA kernel
/// as out of date behind CCCL: about two and a half hours on Windows for an
/// edit to this file. With the sources already there the population step has
/// nothing to add, so it is switched off. Named for the MLX tag, because that
/// tag decides which versions of the dependencies MLX asks for.
fn fetchcontent_marker(out_dir: &Path) -> PathBuf {
    out_dir.join(format!("fetchcontent-populated-{MLX_TAG}"))
}

/// Oldest macOS an Apple build of this runs on.
///
/// It has to be stated rather than inherited, or the answer is whichever SDK
/// built the binary and the artefact only works on machines no older than the
/// one it was built on. See where this is used.
const MACOS_DEPLOYMENT_TARGET: &str = "14.0";

/// Clone MLX at `MLX_TAG` into OUT_DIR and apply this crate's patches to it.
///
/// CMake builds the result through `FETCHCONTENT_SOURCE_DIR_MLX`. The patches
/// match their anchors verbatim, indentation included, so an MLX that has moved
/// fails the build instead of being patched at the wrong line.
fn patched_mlx(out_dir: &Path) -> PathBuf {
    let dir = out_dir.join(format!("mlx-{MLX_TAG}"));
    let probe = dir.join("mlx/backend/cuda/jit_module.cpp");
    if !probe.exists() {
        // `git clone` refuses a non-empty destination, so an interrupted clone
        // would block every later build. Start over instead.
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap_or_else(|e| {
                panic!("mlx-sys: clearing the partial MLX clone at {}: {e}", dir.display())
            });
        }
        // core.autocrlf is a per-machine setting and a CRLF checkout would
        // defeat every anchor, so pin the line endings for this clone.
        let status = std::process::Command::new("git")
            .args([
                "-c",
                "core.autocrlf=false",
                "-c",
                "core.eol=lf",
                "clone",
                "-q",
                "--depth",
                "1",
                "--branch",
                MLX_TAG,
                "https://github.com/ml-explore/mlx.git",
            ])
            .arg(&dir)
            .status();
        let cloned = matches!(&status, Ok(s) if s.success());
        assert!(
            cloned && probe.exists(),
            "mlx-sys: could not clone MLX {MLX_TAG} into {}: {status:?}",
            dir.display()
        );
    }
    patch_no_device(&dir);
    patch_no_cusolver(&dir);
    patch_tma_on_windows(&dir);
    patch_ptx_cache(&dir);
    patch_host_cache(&dir);
    dir
}

/// Apply `edits` to each file under `dir`, once: a file that already holds
/// every replacement, and none of the text an empty replacement deletes, is
/// left alone.
fn apply(dir: &Path, files: &[(&str, &[(&str, &str)])], what: &str) {
    for (file, edits) in files {
        let path = dir.join(file);
        let src = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("mlx-sys: reading {}: {e}", path.display()))
            .replace("\r\n", "\n");
        let done = |(from, to): &(&str, &str)| {
            if to.is_empty() {
                !src.contains(from)
            } else {
                src.contains(to)
            }
        };
        if edits.iter().all(done) {
            continue;
        }
        let mut out = src;
        for (from, to) in edits.iter() {
            assert_eq!(
                out.matches(from).count(),
                1,
                "mlx-sys: no unique anchor for the {what} patch in {}; MLX has moved",
                path.display()
            );
            out = out.replacen(from, to, 1);
        }
        fs::write(&path, &out)
            .unwrap_or_else(|e| panic!("mlx-sys: writing {}: {e}", path.display()));
        println!("cargo:warning=mlx-sys: applied the {what} patch to {}", path.display());
    }
}

/// allocator.cpp: `std::malloc` and `std::free`.
const ALLOC_INCLUDE: &str = "#include <cassert>\n";
const ALLOC_INCLUDE_PATCHED: &str = "#include <cassert>\n#include <cstdlib>\n";

/// allocator.cpp: host buffers come from the C heap when CUDA has no device.
const ALLOC_MALLOC: &str = r#"inline void* unified_malloc(size_t size) {
  void* data = nullptr;
"#;
const ALLOC_MALLOC_PATCHED: &str = r#"// Patched in by mlx-sys: without a CUDA device, host buffers come from the
// C heap so the CPU backend still works.
bool has_cuda_device() {
  static bool found = gpu::is_available();
  return found;
}

inline void* unified_malloc(size_t size) {
  void* data = nullptr;
  if (!has_cuda_device()) {
    return std::malloc(size);
  }
"#;

const ALLOC_FREE: &str = r#"inline void unified_free(void* data) {
"#;
const ALLOC_FREE_PATCHED: &str = r#"inline void unified_free(void* data) {
  if (!has_cuda_device()) {
    std::free(data);
    return;
  }
"#;

/// allocator.cpp, in the `CudaAllocator` constructor.
const ALLOC_LIMITS: &str = r#"  size_t free;
  CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total_memory_));
  memory_limit_ = total_memory_ * 0.95;
  free_limit_ = total_memory_ - memory_limit_;
  max_pool_size_ = memory_limit_;
"#;
const ALLOC_LIMITS_PATCHED: &str = r#"  size_t free;
  if (has_cuda_device()) {
    CHECK_CUDA_ERROR(cudaMemGetInfo(&free, &total_memory_));
  } else {
    // The CPU-only allocator's limit when the RAM size is unknown.
    total_memory_ = size_t(1) << 33;
  }
  memory_limit_ = total_memory_ * 0.95;
  free_limit_ = total_memory_ - memory_limit_;
  // Without a device, freed host buffers go straight back to the heap.
  max_pool_size_ = has_cuda_device() ? memory_limit_ : 0;
"#;

/// device_info.cpp: `gpu::is_available()` is `true` in every CUDA build.
const GPU_AVAILABLE: &str = r#"bool is_available() {
  return true;
}

int device_count() {
  int count = 0;
  cudaGetDeviceCount(&count);
  return count;
}
"#;
const GPU_AVAILABLE_PATCHED: &str = r#"int device_count() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

bool is_available() {
  return device_count() > 0;
}
"#;

/// allocator.cpp: on Windows the limit reads the device's memory pool.
const ALLOC_WDDM: &str = r#"size_t CudaAllocator::get_memory_limit() {
#ifdef _WIN32
"#;
const ALLOC_WDDM_PATCHED: &str = r#"size_t CudaAllocator::get_memory_limit() {
  if (!has_cuda_device()) {
    return memory_limit_;
  }
#ifdef _WIN32
"#;

/// event.h: the `EventImpl` behind every `Event`, CPU streams included.
const EVENT_H_INCLUDES: &str = "#include <memory>\n";
const EVENT_H_INCLUDES_PATCHED: &str =
    "#include <condition_variable>\n#include <memory>\n#include <mutex>\n";

const EVENT_IMPL: &str = r##"  std::unique_ptr<cu::CopyableCudaEvent> cuda;
  std::unique_ptr<cu::AtomicEvent> atomic;

  bool is_created() const {
    return cuda || atomic;
  }
"##;
const EVENT_IMPL_PATCHED: &str = r##"  std::unique_ptr<cu::CopyableCudaEvent> cuda;
  std::unique_ptr<cu::AtomicEvent> atomic;

  // Patched in by mlx-sys: without a CUDA device, a host counter like the
  // CPU-only build's.
  struct Host {
    uint64_t value{0};
    std::mutex mtx;
    std::condition_variable cv;

    void wait(uint64_t target) {
      std::unique_lock lk(mtx);
      cv.wait(lk, [this, target] { return value >= target; });
    }
  };
  std::unique_ptr<Host> host;

  bool is_created() const {
    return cuda || atomic || host;
  }
"##;

/// event.cu
const EVENT_NEW: &str = r##"Event::Event(Stream s) : stream_(s) {
  event_ = std::make_shared<cu::EventImpl>();
}
"##;
const EVENT_NEW_PATCHED: &str = r##"Event::Event(Stream s) : stream_(s) {
  auto impl = std::make_shared<cu::EventImpl>();
  static bool no_device = gpu::device_count() == 0;
  if (no_device) {
    impl->host = std::make_unique<cu::EventImpl::Host>();
  }
  event_ = std::move(impl);
}
"##;

const EVENT_WAIT: &str = r##"void Event::wait() {
  check_error();
  auto& event = cast<cu::EventImpl>();
  assert(event.is_created());
"##;
const EVENT_WAIT_PATCHED: &str = r##"void Event::wait() {
  check_error();
  auto& event = cast<cu::EventImpl>();
  assert(event.is_created());
  if (event.host) {
    event.host->wait(value());
    check_error();
    return;
  }
"##;

const EVENT_WAIT_STREAM: &str = r##"void Event::wait(Stream s) {
  auto& event = cast<cu::EventImpl>();
  assert(event.is_created());
"##;
const EVENT_WAIT_STREAM_PATCHED: &str = r##"void Event::wait(Stream s) {
  auto& event = cast<cu::EventImpl>();
  assert(event.is_created());
  if (event.host) {
    scheduler::wait_event(s, *this, [value = value()](Event& self) {
      self.cast<cu::EventImpl>().host->wait(value);
    });
    return;
  }
"##;

const EVENT_SIGNAL: &str = r##"void Event::signal(Stream s) {
  auto& event = cast<cu::EventImpl>();
  event.ensure_created(s, value());
"##;
const EVENT_SIGNAL_PATCHED: &str = r##"void Event::signal(Stream s) {
  auto& event = cast<cu::EventImpl>();
  if (event.host) {
    scheduler::signal_event(s, *this, [value = value()](Event& self) {
      auto& host = *self.cast<cu::EventImpl>().host;
      {
        std::lock_guard lk(host.mtx);
        host.value = value;
      }
      host.cv.notify_all();
    });
    return;
  }
  event.ensure_created(s, value());
"##;

const EVENT_SIGNALED: &str = r##"bool Event::is_signaled() const {
  auto& event = cast<cu::EventImpl>();
"##;
const EVENT_SIGNALED_PATCHED: &str = r##"bool Event::is_signaled() const {
  auto& event = cast<cu::EventImpl>();
  if (event.host) {
    std::lock_guard lk(event.host->mtx);
    return event.host->value >= value();
  }
"##;

/// Let a CUDA build of MLX run on the CPU of a machine without a CUDA device.
///
/// A CUDA build serves CPU arrays and CPU streams through CUDA too: the
/// allocator calls `cudaMemGetInfo` and `cudaMallocManaged`, and every `Event`,
/// the ones inside a `Fence` included, is built on device 0. Without a device
/// each of these fails. After this patch `gpu::is_available()` answers whether
/// a device exists, which makes the CPU MLX's default device, and in that case
/// the allocator uses `malloc` and events use the CPU-only build's host counter.
fn patch_no_device(dir: &Path) {
    apply(
        dir,
        &[
            (
                "mlx/backend/cuda/allocator.cpp",
                &[
                    (ALLOC_INCLUDE, ALLOC_INCLUDE_PATCHED),
                    (ALLOC_MALLOC, ALLOC_MALLOC_PATCHED),
                    (ALLOC_FREE, ALLOC_FREE_PATCHED),
                    (ALLOC_LIMITS, ALLOC_LIMITS_PATCHED),
                    (ALLOC_WDDM, ALLOC_WDDM_PATCHED),
                ],
            ),
            ("mlx/backend/cuda/device_info.cpp", &[(GPU_AVAILABLE, GPU_AVAILABLE_PATCHED)]),
            (
                "mlx/backend/cuda/event.h",
                &[(EVENT_H_INCLUDES, EVENT_H_INCLUDES_PATCHED), (EVENT_IMPL, EVENT_IMPL_PATCHED)],
            ),
            (
                "mlx/backend/cuda/event.cu",
                &[
                    (EVENT_NEW, EVENT_NEW_PATCHED),
                    (EVENT_WAIT, EVENT_WAIT_PATCHED),
                    (EVENT_WAIT_STREAM, EVENT_WAIT_STREAM_PATCHED),
                    (EVENT_SIGNAL, EVENT_SIGNAL_PATCHED),
                    (EVENT_SIGNALED, EVENT_SIGNALED_PATCHED),
                ],
            ),
        ],
        "no-device CPU fallback",
    );
}

/// mlx/backend/cuda/CMakeLists.txt: the cuSOLVER sources and library.
const SOLVER_CHOLESKY: &str = "          ${CMAKE_CURRENT_SOURCE_DIR}/cholesky.cu\n";
const SOLVER_UTILS: &str = "          ${CMAKE_CURRENT_SOURCE_DIR}/cusolver_utils.cpp\n";
const SOLVER_LINK: &str = "# Use cuSOLVER.\ntarget_link_libraries(mlx PRIVATE CUDA::cusolver)\n";

/// eval.cpp
const SOLVER_INCLUDE: &str = "#include \"mlx/backend/cuda/cusolver_utils.h\"\n";
const SOLVER_INIT: &str = "  init_cusolver_handles_cache();\n";

/// primitives.cpp: the CPU-fallback list of linear algebra primitives.
const SOLVER_NO_GPU: &str = "NO_GPU(Inverse)\n";
const SOLVER_NO_GPU_PATCHED: &str = "NO_GPU(Cholesky)\nNO_GPU(Inverse)\n";

/// Leave cuSOLVER out of MLX.
///
/// MLX uses cuSOLVER for its GPU Cholesky alone. Linking it means shipping
/// libcusolver and the libcusparse it loads, about 340 MB, for an operation
/// stemd never runs. After this patch Cholesky runs on the CPU, as it does in
/// MLX v0.31.
fn patch_no_cusolver(dir: &Path) {
    apply(
        dir,
        &[
            (
                "mlx/backend/cuda/CMakeLists.txt",
                &[(SOLVER_CHOLESKY, ""), (SOLVER_UTILS, ""), (SOLVER_LINK, "")],
            ),
            ("mlx/backend/cuda/eval.cpp", &[(SOLVER_INCLUDE, ""), (SOLVER_INIT, "")]),
            ("mlx/backend/cuda/primitives.cpp", &[(SOLVER_NO_GPU, SOLVER_NO_GPU_PATCHED)]),
        ],
        "no-cuSOLVER",
    );
}

/// fp_quantize.cuh: the MXFP8 columnwise kernel takes two `CUtensorMap`s by value.
const TMA_KERNEL: &str = r#"template <
    typename T,
    bool USE_SR,
    int THREADS_PER_BLOCK,
    int COLS_PER_BLOCK,
    int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fp_quantize_columnwise_mxfp8(
"#;
const TMA_KERNEL_PATCHED: &str = r#"// Patched in by mlx-sys: MSVC cannot pass the over-aligned CUtensorMap by
// value, so Windows builds use the fallback kernel.
#if !defined(_WIN32)
template <
    typename T,
    bool USE_SR,
    int THREADS_PER_BLOCK,
    int COLS_PER_BLOCK,
    int ROWS_PER_BLOCK>
__global__ void __launch_bounds__(THREADS_PER_BLOCK)
    fp_quantize_columnwise_mxfp8(
"#;
const TMA_KERNEL_END: &str = r#"#endif // defined(CUTE_ARCH_TMA_SM90_ENABLED)
}

// TODO: add kernel with tma instructions
"#;
const TMA_KERNEL_END_PATCHED: &str = r#"#endif // defined(CUTE_ARCH_TMA_SM90_ENABLED)
}
#endif // !defined(_WIN32)

// TODO: add kernel with tma instructions
"#;

/// fp_quantize.cu: the host function that launches it, and its one caller.
const TMA_HOST: &str = "}\n\nvoid fp_quantize_columnwise_mxfp8(\n";
const TMA_HOST_PATCHED: &str = "}\n\n#if !defined(_WIN32)\nvoid fp_quantize_columnwise_mxfp8(\n";
const TMA_HOST_END: &str = "}\n\nvoid fp_quantize_columnwise(\n";
const TMA_HOST_END_PATCHED: &str = "}\n#endif // !defined(_WIN32)\n\nvoid fp_quantize_columnwise(\n";
const TMA_CALL: &str = r#"  if (use_tma) {
    fp_quantize_columnwise_mxfp8(
        w, wq, scales, group_size, bits, global_scale, enc, s);
  } else {
    fp_quantize_columnwise_fallback(
        w, wq, scales, group_size, bits, global_scale, enc, s);
  }
"#;
const TMA_CALL_PATCHED: &str = r#"#if !defined(_WIN32)
  if (use_tma) {
    fp_quantize_columnwise_mxfp8(
        w, wq, scales, group_size, bits, global_scale, enc, s);
    return;
  }
#else
  (void)use_tma;
#endif
  fp_quantize_columnwise_fallback(
      w, wq, scales, group_size, bits, global_scale, enc, s);
"#;

/// Leave the TMA quantize kernel out of Windows builds.
///
/// MSVC rejects a kernel parameter of over-aligned type passed by value
/// (`error C2719`), and the MXFP8 columnwise quantize kernel takes two
/// `CUtensorMap`s that way. Windows uses the fallback kernel, which MLX already
/// runs below compute capability 10.
fn patch_tma_on_windows(dir: &Path) {
    apply(
        dir,
        &[
            (
                "mlx/backend/cuda/quantized/fp_quantize.cuh",
                &[(TMA_KERNEL, TMA_KERNEL_PATCHED), (TMA_KERNEL_END, TMA_KERNEL_END_PATCHED)],
            ),
            (
                "mlx/backend/cuda/quantized/fp_quantize.cu",
                &[
                    (TMA_HOST, TMA_HOST_PATCHED),
                    (TMA_HOST_END, TMA_HOST_END_PATCHED),
                    (TMA_CALL, TMA_CALL_PATCHED),
                ],
            ),
        ],
        "Windows TMA",
    );
}

/// jit_module.cpp: the end of the anonymous namespace, before `JitModule`.
const JIT_ARCH_DIR: &str = "} // namespace\n\nJitModule::JitModule(\n";
const JIT_ARCH_DIR_PATCHED: &str = r#"// Patched in by mlx-sys: a cubin runs on one architecture only, so each one
// gets its own directory.
std::filesystem::path arch_cache_dir(Device& device) {
  const auto& base = ptx_cache_dir();
  if (base.empty()) {
    return base;
  }
  auto dir = base /
      fmt::format("sm_{}{}",
                  device.compute_capability_major(),
                  device.compute_capability_minor());
  std::error_code error;
  std::filesystem::create_directories(dir, error);
  return error ? std::filesystem::path() : dir;
}

} // namespace

JitModule::JitModule(
"#;

const JIT_READ: &str = r#"  // Try to load them from the file cache
  if (!read_cached_ptx(ptx_cache_dir(), module_name, ptx, ptx_kernels)) {
"#;
const JIT_READ_PATCHED: &str = r#"  // Try to load them from the file cache
  auto cache_dir = arch_cache_dir(device);
  if (!read_cached_ptx(cache_dir, module_name, ptx, ptx_kernels)) {
"#;

const JIT_WRITE: &str = r#"      write_cached_ptx(
          ptx_cache_dir(), module_name, ptx, ptx_kernels, source_code);
"#;
const JIT_WRITE_PATCHED: &str = r#"      write_cached_ptx(
          cache_dir, module_name, ptx, ptx_kernels, source_code);
"#;

/// Key MLX's on-disk kernel cache by GPU architecture.
///
/// MLX caches each JIT-compiled cubin under its kernel name alone, in a
/// directory shared by every GPU of the machine, so a GPU of another
/// architecture loads a cubin it cannot run and fails with
/// `Failed to load compiled <kernel> kernel`. After this patch the cache has
/// one `sm_<major><minor>` directory per architecture.
fn patch_ptx_cache(dir: &Path) {
    apply(
        dir,
        &[(
            "mlx/backend/cuda/jit_module.cpp",
            &[
                (JIT_ARCH_DIR, JIT_ARCH_DIR_PATCHED),
                (JIT_READ, JIT_READ_PATCHED),
                (JIT_WRITE, JIT_WRITE_PATCHED),
            ],
        )],
        "kernel cache",
    );
}

/// allocator.h: the second cache and its helpers.
const ALLOC_H_CACHE: &str = "  BufferCache<CudaBuffer> buffer_cache_;\n";
const ALLOC_H_CACHE_PATCHED: &str =
    "  BufferCache<CudaBuffer> buffer_cache_;\n  BufferCache<CudaBuffer> host_cache_;\n";
const ALLOC_H_HELPERS: &str = "  void free_cuda_buffer(CudaBuffer* buf);\n";
const ALLOC_H_HELPERS_PATCHED: &str = r#"  void free_cuda_buffer(CudaBuffer* buf);
  BufferCache<CudaBuffer>& cache_for(int device);
  void release_cached(size_t bytes);
"#;

/// allocator.cpp: the `CudaAllocator` constructor's initializer list.
const ALLOC_CTOR: &str = r#"    : buffer_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) { free_cuda_buffer(buf); }) {
"#;
const ALLOC_CTOR_PATCHED: &str = r#"    : buffer_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) { free_cuda_buffer(buf); }),
      host_cache_(
          page_size,
          [](CudaBuffer* buf) { return buf->size; },
          [this](CudaBuffer* buf) { free_cuda_buffer(buf); }) {
"#;
const ALLOC_REUSE: &str = "  CudaBuffer* buf = buffer_cache_.reuse_from_cache(size);\n";
const ALLOC_REUSE_PATCHED: &str = "  CudaBuffer* buf = cache_for(device).reuse_from_cache(size);\n";
const ALLOC_PRESSURE: &str = "      buffer_cache_.release_cached_buffers(mem_to_free);\n";
const ALLOC_PRESSURE_PATCHED: &str = "      release_cached(mem_to_free);\n";
const ALLOC_TRIM: &str =
    "    buffer_cache_.release_cached_buffers(get_cache_memory() - max_pool_size_);\n";
const ALLOC_TRIM_PATCHED: &str = "    release_cached(get_cache_memory() - max_pool_size_);\n";
const ALLOC_RECYCLE: &str = "    buffer_cache_.recycle_to_cache(buf);\n";
const ALLOC_RECYCLE_PATCHED: &str = "    cache_for(buf->device).recycle_to_cache(buf);\n";
const ALLOC_CACHE_SIZE: &str = "  return buffer_cache_.cache_size();\n";
const ALLOC_CACHE_SIZE_PATCHED: &str =
    "  return buffer_cache_.cache_size() + host_cache_.cache_size();\n";
const ALLOC_CLEAR: &str = "    buffer_cache_.clear();\n";
const ALLOC_CLEAR_PATCHED: &str = "    buffer_cache_.clear();\n    host_cache_.clear();\n";
const ALLOC_HELPERS: &str = "// This must be called with mutex_ acquired\n";
const ALLOC_HELPERS_PATCHED: &str = r#"// Patched in by mlx-sys: without managed memory a host buffer is pinned host
// memory that kernels read over PCIe, so host buffers get their own cache and
// a device allocation never reuses one.
BufferCache<CudaBuffer>& CudaAllocator::cache_for(int device) {
  return (device == -1 && !supports_managed_memory()) ? host_cache_
                                                      : buffer_cache_;
}

// Frees host buffers first. This must be called with mutex_ acquired.
void CudaAllocator::release_cached(size_t bytes) {
  size_t host = host_cache_.cache_size();
  if (host > 0) {
    host_cache_.release_cached_buffers(std::min(bytes, host));
  }
  if (bytes > host) {
    buffer_cache_.release_cached_buffers(bytes - host);
  }
}

// This must be called with mutex_ acquired
"#;

/// Keep pinned host buffers out of device allocations.
///
/// Without managed memory (Windows, WSL and Tegra), MLX moves a buffer the CPU
/// reads into pinned host memory and later recycles it into a device
/// allocation of the same size, so the kernels that use it read over PCIe and
/// a server gets slower with every job. After this patch host and device
/// buffers have separate caches.
fn patch_host_cache(dir: &Path) {
    apply(
        dir,
        &[
            (
                "mlx/backend/cuda/allocator.h",
                &[
                    (ALLOC_H_CACHE, ALLOC_H_CACHE_PATCHED),
                    (ALLOC_H_HELPERS, ALLOC_H_HELPERS_PATCHED),
                ],
            ),
            (
                "mlx/backend/cuda/allocator.cpp",
                &[
                    (ALLOC_CTOR, ALLOC_CTOR_PATCHED),
                    (ALLOC_REUSE, ALLOC_REUSE_PATCHED),
                    (ALLOC_PRESSURE, ALLOC_PRESSURE_PATCHED),
                    (ALLOC_TRIM, ALLOC_TRIM_PATCHED),
                    (ALLOC_RECYCLE, ALLOC_RECYCLE_PATCHED),
                    (ALLOC_CACHE_SIZE, ALLOC_CACHE_SIZE_PATCHED),
                    (ALLOC_CLEAR, ALLOC_CLEAR_PATCHED),
                    (ALLOC_HELPERS, ALLOC_HELPERS_PATCHED),
                ],
            ),
        ],
        "host cache",
    );
}

fn build_and_link_mlx_c() {
    // A build script runs on the host, so `cfg!(target_os = ...)` here would
    // describe the host, not the target. The target is only in the environment.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
    let is_apple = matches!(
        target_os.as_str(),
        "macos" | "ios" | "tvos" | "watchos" | "visionos"
    );
    let is_windows = target_os == "windows";
    let is_msvc = is_windows && target_env == "msvc";

    // mlx-rs depends on mlx-sys without `default-features = false`, so metal
    // and accelerate arrive switched on whatever the consuming manifest asks
    // for. Off Apple neither exists, so this is where they get dropped.
    let metal = cfg!(feature = "metal") && is_apple;
    let accelerate = cfg!(feature = "accelerate") && is_apple;
    // Either the cargo feature or MLX_BUILD_CUDA=1. The environment variable is
    // for consumers that reach mlx-sys through mlx-rs and so have no manifest in
    // which to name the feature.
    println!("cargo:rerun-if-env-changed=MLX_BUILD_CUDA");
    let cuda = (cfg!(feature = "cuda")
        || matches!(env::var("MLX_BUILD_CUDA").as_deref(), Ok("1") | Ok("ON") | Ok("on")))
        && !is_apple;

    // Everything the prebuilt key and link_cuda read, declared before the
    // prebuilt branch returns: once a build script prints one rerun line,
    // cargo reruns it only for the ones it printed.
    for var in [
        "MLX_PREBUILT_DIR",
        "MLX_PREBUILT_OUT",
        "MLX_CUDA_ARCHITECTURES",
        "CUDA_PATH",
        "CUDNN_PATH",
        "CXX",
    ] {
        println!("cargo:rerun-if-env-changed={var}");
    }
    let key = prebuilt_key(cuda, is_msvc);
    let prebuilt_os = target_os == "linux" || is_msvc;
    if let Some(dir) = env::var_os("MLX_PREBUILT_DIR") {
        assert!(prebuilt_os, "mlx-sys: MLX_PREBUILT_DIR is supported on Linux and Windows only");
        let dir = PathBuf::from(dir);
        let manifest = dir.join(PREBUILT_MANIFEST);
        let found = fs::read_to_string(&manifest)
            .unwrap_or_else(|e| panic!("mlx-sys: reading {}: {e}", manifest.display()));
        assert_eq!(
            found.trim(),
            key.trim(),
            "mlx-sys: the prebuilt MLX in {} was built for a different configuration. \
             Rebuild it, or unset MLX_PREBUILT_DIR to build from source.",
            dir.display()
        );
        println!("cargo:rerun-if-changed={}", manifest.display());
        println!("cargo:warning=mlx-sys: linking prebuilt MLX from {}", dir.display());
        link_mlx(&dir.join("lib"), None, is_apple, is_windows, metal, accelerate, cuda);
        return;
    }

    let mut config = Config::new("src/mlx-c");
    config.very_verbose(true);
    config.define("CMAKE_INSTALL_PREFIX", ".");
    // Nothing here needs mlx-c's own example programs, and on MSVC they are
    // just more surface to fail on.
    config.define("MLX_C_BUILD_EXAMPLES", "OFF");
    // CMake 4 refuses projects that declare compatibility below 3.5. Several
    // of MLX's FetchContent'd dependencies still do.
    config.define("CMAKE_POLICY_VERSION_MINIMUM", "3.5");

    if is_msvc {
        // rustc on MSVC always links the release, DLL-flavoured CRT. A Debug
        // MLX compiles against /MDd, and mixing the two is an outright link
        // error on _ITERATOR_DEBUG_LEVEL and _MSC_VER mismatches. So the C++
        // side is Release regardless of the Rust profile.
        config.profile("Release");
        config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
        // cmake-rs otherwise picks the Visual Studio generator, which is
        // multi-config: CMAKE_BUILD_TYPE is ignored and the artefacts land in
        // build/lib/Release rather than build/lib. Ninja is single-config,
        // matches how MLX itself is built on Windows, and is the only sane
        // generator once CUDA is in the picture.
        config.generator("Ninja");
    } else if cfg!(debug_assertions) {
        config.define("CMAKE_BUILD_TYPE", "Debug");
    } else {
        config.define("CMAKE_BUILD_TYPE", "Release");
    }

    if is_apple {
        // Two halves, and both have to be said or the artefact is still tied to
        // the machine that built it.
        //
        // This one is the C++. cmake-rs derives `-mmacosx-version-min` through
        // `cc`, which reads this variable and falls back to the SDK's own
        // version when it is unset, so the objects would otherwise carry the
        // builder's macOS whatever CMAKE_OSX_DEPLOYMENT_TARGET below says.
        // Setting it here rather than asking every caller to remember keeps a
        // plain `cargo build` producing the same floor as a release.
        //
        // Sound in a build script: this runs before anything else in the
        // process and there is no second thread to observe the write.
        env::set_var("MACOSX_DEPLOYMENT_TARGET", MACOS_DEPLOYMENT_TARGET);

        // And this one is the Metal shaders, pinned rather than inherited. MLX
        // asks the metal compiler instead of deciding:
        //
        //   if(NOT CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        //     set(XCRUN_FLAGS "-mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}")
        //   execute_process(... xcrun -sdk macosx metal ${XCRUN_FLAGS} ...
        //                   OUTPUT_VARIABLE MLX_METAL_VERSION)
        //
        // Left empty the flag is empty too, so the shaders come out built for
        // whichever SDK the machine doing the building happened to carry, and
        // MLX_METAL_JIT is OFF, so there is no runtime fallback to a version
        // the target can load. Meanwhile the Mach-O header keeps its own
        // `minos`, so the binary launches anywhere and fails only when it first
        // needs the GPU, well past the point where the log still says what it
        // was doing.
        //
        // That is not hypothetical. The first release was built against a macOS
        // 26.5 SDK and separated nothing on 26.2, while the same source built
        // on the reporter's own machine worked.
        //
        // 14.0 is MLX's own floor for MLX_BUILD_METAL=ON: it refuses an SDK
        // below that a few lines above the block quoted here.
        config.define("CMAKE_OSX_DEPLOYMENT_TARGET", MACOS_DEPLOYMENT_TARGET);
    }
    config.define("MLX_BUILD_METAL", if metal { "ON" } else { "OFF" });
    config.define("MLX_BUILD_ACCELERATE", if accelerate { "ON" } else { "OFF" });
    config.define("MLX_BUILD_CUDA", if cuda { "ON" } else { "OFF" });
    if cuda {
        let dir = patched_mlx(&PathBuf::from(env::var("OUT_DIR").unwrap()));
        config.define("FETCHCONTENT_SOURCE_DIR_MLX", &dir);
        println!("cargo:warning=mlx-sys: using patched MLX at {}", dir.display());
        // MLX's FindCUDNN.cmake only looks in the CUDA toolkit and in
        // C:/Program Files/NVIDIA/CUDNN/*. Point it at CUDNN_PATH instead; the
        // suffixes it appends are `include` and `lib/x64`, so CUDNN_PATH is the
        // root of a toolkit-shaped tree.
        if let Ok(cudnn) = env::var("CUDNN_PATH") {
            config.define("CUDNN_INCLUDE_PATH", &cudnn);
            config.define("CUDNN_LIBRARY_PATH", &cudnn);
        }
        // Otherwise MLX shells out to __nvcc_device_query, which needs a GPU
        // present at build time.
        if let Ok(arch) = env::var("MLX_CUDA_ARCHITECTURES") {
            config.define("MLX_CUDA_ARCHITECTURES", arch);
        }
        if is_msvc {
            // MLX declares CUTLASS with `SOURCE_SUBDIR include`, so CUTLASS's
            // top-level CMakeLists - the file that knows CuTe needs MSVC's
            // conformant preprocessor and a truthful __cplusplus - is never
            // processed. Without these, cute/stride.hpp dies with
            //   error C3545: 'Ints': parameter pack expects a non-type
            //   template argument
            // and takes every gemm .cu with it. Same shape of problem as the
            // delay-load options below: a CMake usage requirement that never
            // reaches our compile line. Lifted verbatim from
            // cutlass/CMakeLists.txt.
            // /permissive- is the load-bearing one: CuTe writes dependent
            // template aliases like
            //   using Seq = typename Lambda::template seq<Shape>;
            // (cute/stride.hpp:299) and MSVC's default non-conformant mode
            // binds the name at parse time instead of at instantiation, which
            // is where C3545 comes from. /Zc:__cplusplus and /Zc:preprocessor
            // are what CUTLASS's own CMakeLists sets and are needed too.
            for flag in ["/Zc:__cplusplus", "/Zc:preprocessor"] {
                config.cxxflag(flag);
            }
            config.define(
                "CMAKE_CUDA_FLAGS",
                "-Xcompiler=/Zc:__cplusplus -Xcompiler=/Zc:preprocessor",
            );
            if let Some(dir) = patched_cutlass(&PathBuf::from(env::var("OUT_DIR").unwrap())) {
                config.define("FETCHCONTENT_SOURCE_DIR_CUTLASS", &dir);
                println!("cargo:warning=mlx-sys: using MSVC-patched CUTLASS at {}", dir.display());
            }
        }
    }

    // Stated both ways, ON and OFF: the value lives in CMakeCache.txt between
    // configures, and an ON left there after the marker went would keep a
    // cold directory from ever fetching anything.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let populated = fetchcontent_marker(&out_dir);
    let warm = populated.exists() && out_dir.join("build/_deps").is_dir();
    config.define("FETCHCONTENT_FULLY_DISCONNECTED", if warm { "ON" } else { "OFF" });

    // build the mlx-c project
    let dst = config.build();
    // Only reached when cmake succeeded, so everything it fetches is in place.
    if let Err(e) = fs::write(&populated, "") {
        println!("cargo:warning=mlx-sys: could not write {}: {e}", populated.display());
    }
    let build_dir = dst.join("build");
    let lib_dir = build_dir.join("lib");

    link_mlx(&lib_dir, Some(&build_dir), is_apple, is_windows, metal, accelerate, cuda);

    if let Some(out) = env::var_os("MLX_PREBUILT_OUT") {
        assert!(prebuilt_os, "mlx-sys: MLX_PREBUILT_OUT is supported on Linux and Windows only");
        export_prebuilt(&lib_dir, &build_dir, Path::new(&out), &key, is_msvc);
    }
}

/// The file in a prebuilt MLX directory that records what it was built for.
const PREBUILT_MANIFEST: &str = "mlx-prebuilt.txt";

/// What a prebuilt MLX must match to stand in for a build from source.
///
/// Bump `recipe` when this script changes what goes into libmlx.a without a
/// change to the crate version or the MLX tag.
fn prebuilt_key(cuda: bool, is_msvc: bool) -> String {
    let (archs, nvcc, cudnn) = if cuda {
        let cuda_path = env::var("CUDA_PATH").map(PathBuf::from).unwrap_or_default();
        let cudnn_path = env::var("CUDNN_PATH").map(PathBuf::from).unwrap_or_default();
        let nvcc = format!("bin/nvcc{}", env::consts::EXE_SUFFIX);
        (
            env::var("MLX_CUDA_ARCHITECTURES").unwrap_or_default(),
            tool_version(&cuda_path.join(nvcc), "release"),
            cudnn_version(&cudnn_path.join("include")),
        )
    } else {
        Default::default()
    };
    // MSVC 14.x toolsets link libraries built by an older 14.x, and hosted
    // runners update theirs every month, so only the family is recorded.
    let cxx = if is_msvc {
        "msvc".to_string()
    } else {
        tool_version(Path::new(&env::var("CXX").unwrap_or_else(|_| "c++".into())), "")
    };
    format!(
        "recipe 6\nmlx-sys {}\nmlx {MLX_TAG}\ntarget {}\ncxx {}\ncuda {cuda}\n\
         nvcc {nvcc}\ncudnn {cudnn}\ncuda-architectures {archs}\n",
        env::var("CARGO_PKG_VERSION").unwrap(),
        env::var("TARGET").unwrap(),
        cxx,
    )
}

/// The first line of `tool --version` that contains `needle`, or "unknown".
fn tool_version(tool: &Path, needle: &str) -> String {
    std::process::Command::new(tool)
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .find(|l| l.contains(needle))
                .map(|l| l.trim().to_string())
        })
        .unwrap_or_else(|| "unknown".into())
}

/// The cuDNN version from cudnn_version.h under `include`, or "unknown".
fn cudnn_version(include: &Path) -> String {
    let Ok(header) = fs::read_to_string(include.join("cudnn_version.h")) else {
        return "unknown".into();
    };
    let field = |name: &str| {
        header
            .lines()
            .find_map(|l| l.strip_prefix(&format!("#define {name} ")))
            .map(|v| v.trim().to_string())
            .unwrap_or_default()
    };
    format!(
        "{}.{}.{}",
        field("CUDNN_MAJOR"),
        field("CUDNN_MINOR"),
        field("CUDNN_PATCHLEVEL")
    )
}

/// Copy the static libraries and the manifest into `out`. On Windows also
/// OpenBLAS.
fn export_prebuilt(lib_dir: &Path, build_dir: &Path, out: &Path, key: &str, is_msvc: bool) {
    let lib_out = out.join("lib");
    fs::create_dir_all(&lib_out)
        .unwrap_or_else(|e| panic!("mlx-sys: creating {}: {e}", lib_out.display()));
    let copy = |from: &Path| {
        fs::copy(from, lib_out.join(from.file_name().unwrap()))
            .unwrap_or_else(|e| panic!("mlx-sys: copying {}: {e}", from.display()));
    };
    if is_msvc {
        for name in ["mlx.lib", "mlxc.lib"] {
            copy(&lib_dir.join(name));
        }
        for name in ["libopenblas.lib", "libopenblas.dll"] {
            let from = find_file(build_dir, name)
                .unwrap_or_else(|| panic!("mlx-sys: no {name} under {}", build_dir.display()));
            copy(&from);
        }
    } else {
        for name in ["libmlx.a", "libmlxc.a"] {
            copy(&lib_dir.join(name));
        }
    }
    fs::write(out.join(PREBUILT_MANIFEST), key)
        .unwrap_or_else(|e| panic!("mlx-sys: writing the manifest in {}: {e}", out.display()));
    println!("cargo:warning=mlx-sys: exported prebuilt MLX to {}", out.display());
}

/// Emit the link lines for libmlx and libmlxc in `lib_dir` and everything they
/// depend on. `build_dir` is the CMake build tree, which Windows needs for the
/// libraries MLX downloads; a prebuilt keeps those in `lib_dir`.
fn link_mlx(
    lib_dir: &Path,
    build_dir: Option<&Path>,
    is_apple: bool,
    is_windows: bool,
    metal: bool,
    accelerate: bool,
    cuda: bool,
) {
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    // GNU ld resolves archives strictly left to right, so the consumer has to
    // be named before the library it consumes. ld64 and link.exe do not care,
    // so this order is the correct one everywhere.
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    if is_apple {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=objc");
        println!("cargo:rustc-link-lib=framework=Foundation");
    } else if is_windows {
        // No libstdc++ to name: MSVC object files carry /DEFAULTLIB directives
        // for the C++ runtime, so link.exe pulls it in by itself.
        //
        // MLX on Windows downloads a prebuilt OpenBLAS and links it PRIVATE.
        // A static archive keeps no record of that, so the final link has to
        // name it itself. A prebuilt keeps it beside libmlx.
        let root = build_dir.unwrap_or(lib_dir);
        link_found(root, "libopenblas.lib", "OpenBLAS import library");
        // MLX compiles dlfcn-win32 into itself and links Psapi PRIVATE for it.
        println!("cargo:rustc-link-lib=dylib=psapi");
        // mlx/backend/cuda/wddm.cpp reads the GPU memory budget through DXGI.
        if cuda {
            println!("cargo:rustc-link-lib=dylib=dxgi");
        }
        // mlx/backend/cpu/device_info.cpp reads the CPU name out of the
        // registry on Windows (RegOpenKeyExA and friends). CMake gets advapi32
        // from its default platform libraries; cargo's link line has no such
        // default.
        println!("cargo:rustc-link-lib=dylib=advapi32");
    } else {
        // libstdc++ rather than libc++, and no Objective-C runtime.
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // MLX links LAPACK and BLAS PRIVATE-ly. A static archive keeps no
        // record of that, so the final link has to name them itself.
        println!("cargo:rustc-link-lib=dylib=lapack");
        println!("cargo:rustc-link-lib=dylib=blas");
    }

    if metal {
        println!("cargo:rustc-link-lib=framework=Metal");
    }

    if accelerate {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    if cuda {
        link_cuda(is_windows, build_dir.is_none());
    }
}

/// MLX links the CUDA toolkit PRIVATE-ly, and on Windows it additionally hands
/// the consumer a translation unit and a pile of linker options as CMake
/// *usage requirements*. cargo consumes a static archive and emits its own link
/// line, so it never sees any of that. Reproduce it here.
/// `prebuilt` links the delay-load helper compiled with the prebuilt MLX.
fn link_cuda(is_windows: bool, prebuilt: bool) {
    let cuda_path = env::var("CUDA_PATH")
        .map(PathBuf::from)
        .expect("CUDA_PATH must be set to build with the cuda feature");

    if is_windows {
        println!("cargo:rustc-link-search=native={}", cuda_path.join("lib/x64").display());
        // delayimp.lib provides __delayLoadHelper2, which the /DELAYLOAD stubs
        // call. MLX asks for it with target_link_libraries(mlx PUBLIC ...).
        println!("cargo:rustc-link-lib=dylib=delayimp");

        // MLX emits one /DELAYLOAD per DLL in the toolkit's bin/x64, plus the
        // cuDNN ones. Delay-loading is what lets the helper below place the
        // DLLs at first use rather than at process start.
        let cuda_bin = cuda_path.join("bin/x64");
        let cudnn_root = env::var("CUDNN_PATH").ok().map(PathBuf::from);
        let cudnn_bin = cudnn_root.as_ref().map(|r| r.join("bin/x64"));

        let mut dlls: Vec<String> = Vec::new();
        let mut collect = |dir: &Path| {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().is_some_and(|e| e.eq_ignore_ascii_case("dll")) {
                        dlls.push(p.file_name().unwrap().to_string_lossy().into_owned());
                    }
                }
            }
        };
        collect(&cuda_bin);
        if let Some(dir) = &cudnn_bin {
            collect(dir);
        }
        if let Some(root) = &cudnn_root {
            println!(
                "cargo:rustc-link-search=native={}",
                root.join("lib/x64").display()
            );
        }
        dlls.sort();
        dlls.dedup();
        for dll in &dlls {
            println!("cargo:rustc-link-arg=/DELAYLOAD:{dll}");
        }
        // Unlike rustc-link-lib, rustc-link-arg does not propagate downstream:
        // cargo applies it to this package's own binaries, examples, tests and
        // cdylibs only. A consuming executable links the CUDA DLLs the ordinary
        // way and needs them findable at process start. To make it delay-load
        // them too, repeat these flags at the top of the workspace through
        // .cargo/config.toml rustflags or a build.rs in the binary crate.
        let flags: Vec<String> = dlls.iter().map(|d| format!("/DELAYLOAD:{d}")).collect();
        fs::write(
            PathBuf::from(env::var("OUT_DIR").unwrap()).join("delayload_flags.txt"),
            flags.join("\n"),
        )
        .ok();
        println!("cargo:warning=mlx-sys: {} DLLs delay-loaded", dlls.len());

        // MLX ships the delay-load helper as source rather than in libmlx.a:
        // __pfnDliNotifyHook2 must be defined in the final link unit, not pulled
        // from an archive. Compiled here, through a one-line shim that defines
        // its two directory macros, which is simpler than quoting paths on MSVC's
        // command line.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let mlx_src = out_dir.join("build/_deps/mlx-src");
        let delayload = mlx_src.join("mlx/backend/cuda/delayload.cpp");
        if prebuilt {
            // No helper: the packages put the CUDA DLLs next to the executable.
        } else if delayload.exists() {
            let mut shim = String::new();
            shim.push_str(&format!(
                "#define MLX_CUDA_BIN_DIR \"{}\"\n",
                cuda_bin.display().to_string().replace('\\', "/")
            ));
            if let Some(dir) = &cudnn_bin {
                shim.push_str(&format!(
                    "#define MLX_CUDNN_BIN_DIR \"{}\"\n",
                    dir.display().to_string().replace('\\', "/")
                ));
            }
            shim.push_str(&format!(
                "#include \"{}\"\n",
                delayload.display().to_string().replace('\\', "/")
            ));
            let shim_path = out_dir.join("mlx_delayload_shim.cpp");
            fs::write(&shim_path, shim).expect("write delayload shim");

            cc::Build::new()
                .cpp(true)
                .std("c++20")
                .file(&shim_path)
                .include(&mlx_src)
                .compile("mlx_delayload");
            println!("cargo:warning=mlx-sys: compiled delay-load helper");
        } else {
            println!(
                "cargo:warning=mlx-sys: delayload.cpp not found at {}",
                delayload.display()
            );
        }
    } else {
        println!("cargo:rustc-link-search=native={}", cuda_path.join("lib64").display());
        // cuDNN is not part of the CUDA toolkit and is frequently not on the
        // system linker path either: an unpacked wheel or tarball lives wherever
        // it was put, and CUDNN_PATH is how this build hears about it. Windows
        // already added it; without the same here, `-lcudnn` below resolves
        // against whatever the distribution happens to ship. On Debian 13 that
        // is 9.0.0, which predates cudnnBackendPopulateCudaGraph, so a build
        // configured correctly against a newer cuDNN still dies at the final
        // link with undefined symbols out of cudnn_frontend.
        if let Ok(cudnn) = env::var("CUDNN_PATH") {
            let root = PathBuf::from(cudnn);
            for dir in ["lib", "lib64"] {
                let path = root.join(dir);
                if path.is_dir() {
                    println!("cargo:rustc-link-search=native={}", path.display());
                }
            }
        }
    }

    // One entry per target_link_libraries(mlx PRIVATE CUDA::...) in
    // mlx/backend/cuda/CMakeLists.txt. Omitting cufft surfaces only at final
    // link, as eight unresolved cufft* symbols out of fft.cu.
    for lib in ["cudart", "cublas", "cublasLt", "cufft", "nvrtc", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    // CUDNN::cudnn_all. The two beyond `cudnn` are the ones FindCUDNN.cmake
    // marks required for cuDNN 9.
    for lib in ["cudnn", "cudnn_graph", "cudnn_engines_runtime_compiled"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
}

/// bindgen renders a C enum with the target ABI's underlying type. The Itanium
/// and SysV ABIs give an all-non-negative enum `unsigned int`; MSVC gives it
/// plain `int`. So `mlx_dtype` and friends come out `c_uint` on macOS and Linux
/// and `c_int` on Windows, and mlx-rs - which hardcodes the u32 spelling in its
/// `From<Dtype> for u32` impls and its `TryFrom` conversions - fails to compile
/// with about 45 signedness errors it has no business knowing about.
///
/// Every enumerator in mlx-c is small and non-negative, and int and unsigned
/// int agree on size, alignment and register class, so declaring them unsigned
/// on the Rust side is ABI-identical and makes the generated bindings the same
/// shape on all three platforms. bindgen has no knob for this, hence the
/// rewrite of its output. The `mlx_*_` spelling with the trailing underscore is
/// bindgen's name for the enum *tag*, so this touches mlx-c's own enums only.
fn unsign_enum_aliases(src: &str) -> (String, usize) {
    let mut n = 0;
    let out = src
        .lines()
        .map(|line| {
            let t = line.trim_start();
            if t.starts_with("pub type mlx_")
                && t.ends_with("_ = ::std::os::raw::c_int;")
                && t.contains("_ = ::std::os::raw::c_int;")
            {
                n += 1;
                line.replace("::std::os::raw::c_int;", "::std::os::raw::c_uint;")
            } else {
                line.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    (out, n)
}

fn main() {
    build_and_link_mlx_c();

    // generate bindings
    let bindings = bindgen::Builder::default()
        .rust_target(RustTarget::Stable_1_73)
        .header("src/mlx-c/mlx/c/mlx.h")
        .header("src/mlx-c/mlx/c/linalg.h")
        .header("src/mlx-c/mlx/c/error.h")
        .header("src/mlx-c/mlx/c/transforms_impl.h")
        .clang_arg("-Isrc/mlx-c")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut src = bindings.to_string();
    if env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default() == "msvc" {
        let (rewritten, n) = unsign_enum_aliases(&src);
        println!("cargo:warning=mlx-sys: made {n} MSVC enum aliases unsigned");
        src = rewritten;
    }
    fs::write(out_path.join("bindings.rs"), src).expect("Couldn't write bindings!");
}
