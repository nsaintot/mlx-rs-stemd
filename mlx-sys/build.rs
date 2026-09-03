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

/// Cherry-pick upstream a5a684d into MLX v0.31.2: CUDA RMSNorm computes the
/// wrong normaliser for a bounded window of row widths.
///
/// `rms_norm.cu:298` is the one dispatch arm asking for a 64-thread block
/// (`group_dim * n_groups` = 32 * 2) and two rows per block at once.
/// `BlockBroadcastReduce` reduces across the block through a `TempStorage` of
/// `T[BLOCK_DIM / WARP_SIZE]` = `T[2]`, so two rows' partial sums merge and each
/// is scaled by `rsqrt((ss(row 2k) + ss(row 2k+1)) / axis_size)`.
///
/// Affected window: `n_per_thread * 32 < axis_size <= n_per_thread * 32 * 2`,
/// with `n_per_thread = N_READS = 16 / sizeof(T)`: widths 129-256 at f32 and
/// 257-512 at f16.
///
/// Fixed upstream in v0.32.0. mlx-c 0.6.0 is the newest tag and pins v0.31.2,
/// so bumping MLX would mean regenerating this crate's bindings. Patched through
/// `FETCHCONTENT_SOURCE_DIR_MLX` so it survives `cargo clean`.
///
/// Panics rather than falling back: a missing patch is a wrong answer for one
/// band of shapes, not a compile error. `MLX_SKIP_RMSNORM_PATCH=1` opts out.
///
/// The anchors are matched verbatim, indentation included, so an MLX that has
/// moved fails the build instead of being patched at the wrong line.

/// Oldest macOS an Apple build of this runs on.
///
/// It has to be stated rather than inherited, or the answer is whichever SDK
/// built the binary and the artefact only works on machines no older than the
/// one it was built on. See where this is used.
const MACOS_DEPLOYMENT_TARGET: &str = "14.0";

/// rms_norm.cu:298-301 as v0.31.2 ships it.
const RMS_BROKEN: &str = r#"  } else if (axis_size <= n_per_thread * 32 * 2) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 2>(),
      std::integral_constant<int, 2>());
"#;

const RMS_FIXED: &str = r#"  } else if (axis_size <= n_per_thread * 32 * 2) {
    f(std::integral_constant<int, 32>{},
      std::integral_constant<int, 2>(),
      std::integral_constant<int, 1>());
"#;

/// rms_norm.cu:371, inside `RMSNorm::eval_gpu`.
const RMS_FWD_ANCHOR: &str = r#"            constexpr int block_dim = n_groups() * group_dim();
"#;

const RMS_FWD_GUARD: &str = r#"            constexpr int block_dim = n_groups() * group_dim();
            static_assert(block_dim <= 32 || groups_per_block() == 1);
"#;

/// rms_norm.cu:481, inside `RMSNormVJP::eval_gpu`. Note the operands are the
/// other way round there, which is what keeps the two anchors distinct.
const RMS_VJP_ANCHOR: &str = r#"              constexpr int block_dim = group_dim() * n_groups();
"#;

const RMS_VJP_GUARD: &str = r#"              constexpr int block_dim = group_dim() * n_groups();
              static_assert(block_dim <= 32 || groups_per_block() == 1);
"#;

fn patched_mlx(out_dir: &Path) -> PathBuf {
    // Must track the GIT_TAG of the `mlx` FetchContent_Declare in
    // src/mlx-c/CMakeLists.txt.
    const MLX_TAG: &str = "v0.31.2";

    let dir = out_dir.join("mlx-rmsnorm");
    let kernel = dir.join("mlx/backend/cuda/rms_norm.cu");
    if !kernel.exists() {
        // A directory without the kernel in it is a clone that was interrupted -
        // and `git clone` refuses a non-empty destination, so a half-finished
        // one would wedge every later build. Start over instead.
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap_or_else(|e| {
                panic!("mlx-sys: clearing the partial MLX clone at {}: {e}", dir.display())
            });
        }
        // core.autocrlf is a per-machine setting and a CRLF checkout would
        // defeat every anchor above, so pin the line endings for this clone.
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
            cloned && kernel.exists(),
            "mlx-sys: could not clone MLX {MLX_TAG} into {} to apply the CUDA \
             RMSNorm fix (upstream a5a684d). Building against the stock v0.31.2 \
             would give a library that is numerically wrong for last-axis \
             widths 129-256 in f32 and 257-512 in f16, so this is a hard stop. \
             Fix the network or the git on PATH, or set \
             MLX_SKIP_RMSNORM_PATCH=1 if you genuinely want stock MLX. \
             git said: {status:?}",
            dir.display()
        );
    }

    let raw = fs::read_to_string(&kernel)
        .unwrap_or_else(|e| panic!("mlx-sys: reading {}: {e}", kernel.display()));
    let src = raw.replace("\r\n", "\n");

    // Idempotent: a warm OUT_DIR already holds the patched file.
    if src.contains(RMS_FIXED) && src.contains(RMS_FWD_GUARD) && src.contains(RMS_VJP_GUARD) {
        return dir;
    }

    assert_eq!(
        src.matches(RMS_BROKEN).count(),
        1,
        "mlx-sys: {} does not contain the v0.31.2 RMSNorm dispatch arm this \
         patch is written against. MLX has moved; re-derive the cherry-pick of \
         a5a684d rather than shipping an unpatched kernel.",
        kernel.display()
    );
    assert_eq!(
        src.matches(RMS_FWD_ANCHOR).count(),
        1,
        "mlx-sys: no unique RMSNorm::eval_gpu block_dim in {}",
        kernel.display()
    );
    assert_eq!(
        src.matches(RMS_VJP_ANCHOR).count(),
        1,
        "mlx-sys: no unique RMSNormVJP::eval_gpu block_dim in {}",
        kernel.display()
    );

    let patched = src
        .replace(RMS_BROKEN, RMS_FIXED)
        .replace(RMS_FWD_ANCHOR, RMS_FWD_GUARD)
        .replace(RMS_VJP_ANCHOR, RMS_VJP_GUARD);
    fs::write(&kernel, &patched)
        .unwrap_or_else(|e| panic!("mlx-sys: writing {}: {e}", kernel.display()));
    assert!(
        patched.contains(RMS_FIXED)
            && patched.contains(RMS_FWD_GUARD)
            && patched.contains(RMS_VJP_GUARD),
        "mlx-sys: the RMSNorm patch did not take in {}",
        kernel.display()
    );
    println!("cargo:warning=mlx-sys: applied CUDA RMSNorm fix (upstream a5a684d) to {}", kernel.display());
    dir
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
        // Cherry-pick of upstream a5a684d onto the pinned v0.31.2. Every CUDA
        // build, not just MSVC: the bug is in the kernel, not the toolchain.
        // See `patched_mlx` for what it is and why it is not a version bump.
        println!("cargo:rerun-if-env-changed=MLX_SKIP_RMSNORM_PATCH");
        if env::var_os("MLX_SKIP_RMSNORM_PATCH").is_none() {
            let dir = patched_mlx(&PathBuf::from(env::var("OUT_DIR").unwrap()));
            config.define("FETCHCONTENT_SOURCE_DIR_MLX", &dir);
            println!(
                "cargo:warning=mlx-sys: using RMSNorm-patched MLX at {}",
                dir.display()
            );
        } else {
            println!(
                "cargo:warning=mlx-sys: MLX_SKIP_RMSNORM_PATCH is set - building STOCK MLX \
                 v0.31.2, whose CUDA RMSNorm returns the wrong normaliser for last-axis \
                 widths 129-256 (f32) and 257-512 (f16)"
            );
        }
        // MLX's FindCUDNN.cmake only looks in the CUDA toolkit and in
        // C:/Program Files/NVIDIA/CUDNN/*. Point it at CUDNN_PATH instead; the
        // suffixes it appends are `include` and `lib/x64`, so CUDNN_PATH is the
        // root of a toolkit-shaped tree.
        if let Ok(cudnn) = env::var("CUDNN_PATH") {
            println!("cargo:rerun-if-env-changed=CUDNN_PATH");
            config.define("CUDNN_INCLUDE_PATH", &cudnn);
            config.define("CUDNN_LIBRARY_PATH", &cudnn);
        }
        // Otherwise MLX shells out to __nvcc_device_query, which needs a GPU
        // present at build time.
        if let Ok(arch) = env::var("MLX_CUDA_ARCHITECTURES") {
            println!("cargo:rerun-if-env-changed=MLX_CUDA_ARCHITECTURES");
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

    // build the mlx-c project
    let dst = config.build();
    let build_dir = dst.join("build");

    println!("cargo:rustc-link-search=native={}/build/lib", dst.display());
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
        // MLX on Windows downloads a prebuilt OpenBLAS and links it PRIVATE,
        // and gets dlfcn.h from a vendored dlfcn-win32 that it also links
        // PRIVATE. A static archive keeps no record of either, so the final
        // link has to name them itself.
        link_found(&build_dir, "libopenblas.lib", "OpenBLAS import library");
        link_found(&build_dir, "dl.lib", "dlfcn-win32");
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
        link_cuda(is_windows);
    }
}

/// MLX links the CUDA toolkit PRIVATE-ly, and on Windows it additionally hands
/// the consumer a translation unit and a pile of linker options as CMake
/// *usage requirements*. cargo consumes a static archive and emits its own link
/// line, so it never sees any of that. Reproduce it here.
fn link_cuda(is_windows: bool) {
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
        if delayload.exists() {
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
