{
  description = "fishsense-fishdetector — DeepFish YOLO seg training and Fishial/SAM 3 comparison eval";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # The CUDA wheels are unfree; uv pulls them, not nixpkgs, but allowUnfree
          # is set defensively in case anything pulls cudaPackages transitively.
          config.allowUnfree = true;
        };

        # System libraries that pip wheels (opencv-python, torch, ORT, scikit-image)
        # dlopen at runtime. Pure nix paths — safe to put on LD_LIBRARY_PATH globally
        # because nix's glibc is forward-compatible with the wheels.
        wheelRuntimeLibs = with pkgs; [
          stdenv.cc.cc.lib   # libstdc++.so.6 — torch / ORT / nearly every C++ wheel
          zlib               # png decoders, model archives
          libGL              # opencv-python imread / cv2.imshow
          glib               # transitive opencv runtime
        ];
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # Python pinned to 3.13 (matches .python-version). uv resolves the
            # interpreter from this when creating .venv.
            python313

            # uv owns all Python dependency management — see pyproject.toml + uv.lock.
            uv

            # Rust toolchain for the maturin-built fishsense-core extension.
            # Building with --features cuda triggers ort/copy-dylibs to fetch the
            # CUDA-enabled ONNX Runtime binaries during `cargo build`.
            rustc
            cargo
            pkg-config

            # The Rust `opencv` crate (a fishsense-core dep) uses bindgen via
            # clang-sys to generate Rust bindings from OpenCV headers — needs
            # libclang at build time.
            llvmPackages.libclang.lib
            clang
            cmake

            # OpenCV system libraries that the `opencv` Rust crate links against.
            opencv4

            # OpenSSL dev libs + pkg-config files. fishsense-core's build.rs uses
            # reqwest (with default native-tls) to download ORT binaries, which
            # transitively requires openssl-sys at compile time.
            openssl
            openssl.dev

            # Convenience CLI for fetching gated SAM 3.1 weights without going
            # through uv (handy if you want to grab them before `uv sync`).
            python313Packages.huggingface-hub
          ];

          buildInputs = wheelRuntimeLibs;

          shellHook = ''
            # Expose nix-built wheel runtime libs globally — safe because they're
            # forward-compatible with anything else in the shell.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath wheelRuntimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Tell clang-sys / bindgen where libclang.so lives so the opencv
            # Rust crate can generate its FFI bindings during fishsense-core's
            # cargo build.
            export LIBCLANG_PATH="${pkgs.llvmPackages.libclang.lib}/lib"

            # NixOS surfaces the system NVIDIA userspace driver at a path that
            # *doesn't* shadow nix's glibc, so it's safe to add globally.
            if [ -d /run/opengl-driver/lib ]; then
              export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/run/opengl-driver/lib"
            fi

            # On non-NixOS Linux (Ubuntu / Debian / Fedora) the NVIDIA driver
            # lives in /usr/lib/x86_64-linux-gnu, which ALSO holds the system
            # glibc — putting that on LD_LIBRARY_PATH globally would break every
            # nix-built binary that needs nix's newer glibc (e.g. coreutils' rm).
            #
            # Instead we provide a `with-cuda` shell function that prepends the
            # NVIDIA dir for one command at a time. Use it whenever you need
            # CUDA-resolved libs:
            #
            #   with-cuda uv run jupyter lab
            #   with-cuda .venv/bin/python my_script.py
            #
            # uv sync itself does NOT need CUDA visibility — only runtime does.
            with-cuda() {
              local extra_path=""
              for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /run/opengl-driver/lib; do
                if [ -d "$d" ]; then
                  extra_path="$extra_path:$d"
                fi
              done
              LD_LIBRARY_PATH="''${LD_LIBRARY_PATH}$extra_path" "$@"
            }

            if [ ! -d .venv ]; then
              cat <<'HINT'

            First-time setup (build fishsense-core with CUDA):

              MATURIN_PEP517_ARGS='--features cuda' \
                uv sync --config-setting 'build-args=--features cuda'

            Run anything that needs CUDA libs at runtime via the `with-cuda` wrapper:

              with-cuda uv run jupyter lab
              with-cuda .venv/bin/python -c "import torch; print(torch.cuda.is_available())"

            SAM 3.1 weights (gated, ~3.3 GB; only needed for the eval notebook):

              huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
                --local-dir . --local-dir-use-symlinks False

            HINT
            fi
          '';
        };
      });
}
