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
          # The wheels themselves are unfree (CUDA), but they're managed by uv,
          # not nixpkgs. allowUnfree is set defensively so anything pulled
          # transitively from nixpkgs (e.g. cudaPackages helpers if you add
          # them later) doesn't error out.
          config.allowUnfree = true;
        };

        # System libraries that pip wheels (opencv-python, torch, ORT, scikit-image)
        # dlopen at runtime. The CUDA stack itself ships with torch via the
        # nvidia-* pip packages; the system NVIDIA driver is exposed below.
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

            # Convenience CLI for fetching gated SAM 3.1 weights without going
            # through uv (handy if you want to grab them before `uv sync`).
            python313Packages.huggingface-hub
          ];

          buildInputs = wheelRuntimeLibs;

          shellHook = ''
            # Expose wheel-dlopened system libs.
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath wheelRuntimeLibs}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Surface the system NVIDIA userspace driver libs (libcuda.so.1,
            # libnvidia-ml.so). The CUDA EP in fishsense-core and torch both
            # need these to talk to the kernel driver. On NixOS they live at
            # /run/opengl-driver/lib; on Ubuntu/Debian/Fedora they're under
            # /usr/lib/x86_64-linux-gnu (or /usr/lib64 on Fedora).
            for nvdir in /run/opengl-driver/lib /usr/lib/x86_64-linux-gnu /usr/lib64; do
              if [ -d "$nvdir" ]; then
                export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$nvdir"
              fi
            done

            if [ ! -d .venv ]; then
              cat <<'HINT'

            First-time setup (build fishsense-core with CUDA):

              MATURIN_PEP517_ARGS='--features cuda' \
                uv sync --config-setting 'build-args=--features cuda'

            SAM 3.1 weights (gated, ~3.3 GB; only needed for the eval notebook):

              huggingface-cli download facebook/sam3.1 sam3.1_multiplex.pt \
                --local-dir . --local-dir-use-symlinks False

            HINT
            fi
          '';
        };
      });
}
