# WSL2 deployment notes

This fork was deployed on Windows 11 with Ubuntu inside WSL2. Upstream supports Linux x86-64 with NVIDIA
Blackwell (`sm_120`) and native Windows x64 on a single GPU.
These notes describe the Linux build inside WSL2, not the native Windows port or an upstream WSL support
guarantee. The tested dual-GPU deployment continues to use WSL.

## Tested workstation

| Component | Deployment |
| --- | --- |
| Main GPU | RTX 5090 32 GB |
| Peer GPU | RTX 5060 Ti 16 GB; also drives the Windows display |
| Host | Ryzen 9 7950X, 96 GB RAM |
| WSL | 3.0.3.0, kernel 6.18.54.1; 70 GB cap, 24 virtual CPUs, no swap |
| Toolchain | CUDA Toolkit 13.0.88, CMake, Ninja, Release, architecture 120 |
| Python | Python 3.12 virtual environment |
| Model | `Qwen3.8-FN-NVFP4.basalt`, approximately 122 GiB |
| API | `http://127.0.0.1:18235/v1`; one slot, 262144-token context ceiling |

Check `nvidia-smi -L` before copying GPU indices. The tested cards used PCIe 5.0 x8 links, and CUDA P2P
access was unavailable. The peer expert tier still worked; a second card is not a guarantee of a speedup.

The installed pack uses the original
[NVIDIA NVFP4 checkpoint](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4), revision `fc694b54`,
Q6_K dense projections, and a self-distilled MTP draft layer. The main model is **not** the newer
`FP4-Local` QAD checkpoint. A repository update does not replace the model file.
See the [model format](model-file.md) and [pack model card](https://huggingface.co/jesdga/Qwen3.8-Flash-Next-Basalt).

## 1. Prepare WSL, the Windows driver, and the Linux toolkit

In Windows:

```powershell
wsl --version
wsl --list --verbose
wsl --update
```

Use WSL2 and an NVIDIA Windows driver with WSL CUDA support. Inside WSL, install the Linux CUDA toolkit
following [NVIDIA's WSL instructions](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
The tested package was `cuda-toolkit-13-0` from NVIDIA's repository. Do not install a Linux NVIDIA display
driver or a driver-bundling CUDA metapackage in WSL. Windows `nvcc.exe` is not the compiler for this build.

Inside Ubuntu:

```sh
sudo apt update
sudo apt install build-essential cmake ninja-build git python3-venv python3-dev curl util-linux
nvidia-smi -L
/usr/local/cuda-13.0/bin/nvcc --version
```

Adjust the toolkit path if using another supported CUDA 13 installation.

## 2. Allocate WSL RAM

Edit Windows `%UserProfile%\.wslconfig`, not Linux `/etc/wsl.conf`. Example for the tested 96 GB host:

```ini
[wsl2]
memory=70GB
processors=24
swap=0
```

These are this workstation's settings, not minimum requirements or a universal optimum. Leave RAM for
Windows, Docker and other applications. Swap is not a replacement for resident expert RAM.

After saving work, `wsl --shutdown` applies VM settings on the next launch. It stops **all** WSL distributions
and their services, including Docker workloads. It is unnecessary for an ordinary Basalt rebuild or restart.
Verify effective limits with `free -h` and `nproc`.
Reference: [Microsoft WSL configuration](https://learn.microsoft.com/en-us/windows/wsl/wsl-config).

## 3. Model disk: why F:, and why native ext4?

These are two separate choices:

- **Location / C: space:** the main reason for placing `models.vhdx` on **F:** is to keep the approximately
  122 GiB model, extracted cache and download temporaries from consuming the WSL distribution's root disk.
  That root VHDX commonly lives on C: by default. The code and venv can stay in `~/basalt`; the large model
  storage grows on F: instead. F: is this workstation's choice, not a required drive letter.
- **Filesystem / WSL I/O:** format the model VHDX as **ext4** and mount it natively inside WSL. This gives
  Basalt a Linux filesystem path such as `/mnt/wsl/basalt-models` instead of the Windows-drive route
  `/mnt/f`. Moving a file to F: alone does not change that I/O route.

An ext4 VHDX on F: still occupies real space on the Windows F: volume. A dynamic VHDX grows as files are
written, up to its virtual capacity. This deployment used a 384 GiB virtual capacity; that is not a reservation
of free host space. Leave room for the model, cache and temporary downloads on the host drive.

Build in a Linux directory such as `~/basalt`. If the distribution itself already lives on a sufficiently
large secondary SSD, storing models in its ext4 filesystem is also possible; a separate VHDX is not mandatory.

### Attach an existing model disk

From an elevated Windows terminal, for an existing whole-disk ext4 VHDX:

```powershell
wsl --mount --vhd "F:\WSL\BasaltModels\models.vhdx" --name basalt-models --type ext4
```

With WSL's default mount root, the mountpoint is `/mnt/wsl/basalt-models`.
For a new empty VHDX, first create it using Windows Disk Management, attach it using `--bare`, and identify
the new device with `lsblk -f`. Format **only the new empty disk** as ext4; formatting erases existing contents.
Do not copy a device name such as `/dev/sdd` from another machine. Follow
[Microsoft's disk/VHD guide](https://learn.microsoft.com/en-us/windows/wsl/wsl2-mount-disk)
for creation, formatting, partitioned disks and mounting by filesystem UUID.

After mounting, check it before creating the model directory:

```sh
test "$(findmnt -rn -M /mnt/wsl/basalt-models -o FSTYPE)" = ext4 || exit 1
sudo install -d -o "$(id -u)" -g "$(id -g)" /mnt/wsl/basalt-models/models
findmnt -T /mnt/wsl/basalt-models/models
df -h /mnt/wsl/basalt-models/models
```

Do not continue if the check fails: otherwise downloads could silently fill the distribution's root VHDX
and therefore C: on a default installation. An automated launcher should also verify the expected UUID.

The VHDX and files survive `wsl --shutdown`, but the attachment/mount must be restored before the next launch.
An `/etc/fstab` entry alone does not attach the Windows VHDX. Do not reformat it on subsequent starts.
Browse from Windows using `\\wsl.localhost\Ubuntu\mnt\wsl\basalt-models\models`, replacing the distro name as needed.

## 4. Build and download

Inside WSL:

```sh
git clone https://github.com/LesterLyu/basalt.git
cd basalt
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install huggingface_hub 'jsonschema>=4.23,<5'
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j 8
test "$(findmnt -rn -M /mnt/wsl/basalt-models -o FSTYPE)" = ext4 || exit 1
.venv/bin/hf download jesdga/Qwen3.8-Flash-Next-Basalt Qwen3.8-FN-NVFP4.basalt \
  --local-dir /mnt/wsl/basalt-models/models
```

Keep the build and server in the same distribution. The server uses the repository's `.venv` by default.
`jsonschema` enables complete JSON Schema validation. Restart an existing server after installing it.

The tested pack was 130971045087 bytes. Its SHA-256 was:

```text
264e2f6b00d01dd32f68a401c9d7ec8256f4b15b58cf23b23bfdfa8803d08ae3
```

This identifies this exact pack. A later publisher revision can legitimately have a different checksum;
verify it against that revision's metadata.

## 5. Example dual-GPU configuration

Save as `basalt_config.json` in the repository root:

```json
{
  "server": {
    "model": "/mnt/wsl/basalt-models/models/Qwen3.8-FN-NVFP4.basalt",
    "alias": "Qwen3.8-Flash-Next-NVFP4",
    "gpu": "0,1",
    "host": "127.0.0.1",
    "port": 18235,
    "parallel": 1,
    "vision": "peer",
    "temperature": 1.0,
    "top-p": 0.95,
    "top-k": 20,
    "min-p": 0.0,
    "presence-penalty": 0.0,
    "repetition-penalty": 1.0
  },
  "engine": {
    "expert-cache": "auto",
    "prefill": "auto",
    "spec": 7,
    "spec-min-p": 0.7,
    "max-context": 262144,
    "kv": "int8",
    "resident-experts": true,
    "vram-reserve-mib": 700,
    "peer-reserve-mib": 4096,
    "peer-exclusive": true,
    "adapt-every": 1,
    "adapt-decay": 0.85,
    "pcie-frac": 0.25
  }
}
```

The first GPU runs the main model. The second holds **and computes** peer experts; `vision: peer` also places
the image encoder there. It is not a vision-only configuration. The 4096 MiB peer reserve leaves headroom for
vision, display applications and other work; it does not guarantee capacity for any arbitrary new workload.

`peer-exclusive` avoids duplicating peer-only experts in RAM, reducing RAM use with a prefill tradeoff.
The 262144 context is a ceiling: elastic KV starts smaller and reclaims expert-cache slots as context grows.
Multiple slots share the context budget unless `slot-full-context` is enabled.
`spec-min-p` controls drafting and is separate from request sampling `min-p`.

`pcie-frac` is retained from the tested configuration, but is not a universal decode speed knob:
the device-planned MoE tail serves misses on the CPU. `kv-resident` was not enabled in the selected setup.
These values are a starting point for this hardware/pack, not a promise of optimal settings after every update.
See [configuration](configuration/index.md) and [running.md](running.md).

## 6. Locked memory: Linux limits versus CUDA limits

A high `ulimit -l` permits Linux `mlock`; it does **not** remove CUDA/WSL driver's pinned-memory limits.
NVIDIA documents limited pinned-system-memory availability in its
[WSL limitations](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#known-limitations-for-linux-cuda-applications).

This fork retains the resident-cache sequence:

1. Try mapped `cudaHostAlloc`.
2. If refused, allocate page-aligned ordinary memory and try `cudaHostRegister` **before populating it**.
3. If the whole range is refused, retry smaller prefixes in roughly 2 GiB steps. Each prefix ends on both
   a page boundary and a complete expert boundary, so CUDA expert copies cannot straddle memory types.
4. Lock the unregistered remainder using Linux `mlock`.

The earlier local deployment registered about 35.6 GiB of an approximately 40.4 GiB complement.
That is an observed result, not a fixed WSL limit. This rebase keeps upstream's unbuffered reads and
page-table headroom alongside the pre-population registration and prefix retries.

Raise the lock limit to **48 GiB** for only the current shell and its children:

```sh
sudo prlimit --pid "$$" --memlock=51539607552:51539607552
ulimit -l
```

Then launch Basalt as your normal user from that shell. The limit is per process, inherited by children;
it is not a shared 48 GiB allowance for the entire tree. This changes no global PAM policy and does not run
the engine as root. A different terminal or a systemd service needs its own setting.

Use the startup log's actual mapped-pinned and locked-resident figures. A successful `ulimit` change alone
does not prove that all expert memory became CUDA-pinned.

## 7. Start, inspect and restart

After mounting the model disk and setting the shell's lock limit, also place Basalt's extracted cache on F:.
The existing `BASALT_CACHE` setting otherwise defaults to `~/.cache/basalt` inside the distribution's root VHDX:

```sh
export BASALT_CACHE=/mnt/wsl/basalt-models/models/.basalt-cache
bin/basalt serve --config basalt_config.json
```

UI: <http://127.0.0.1:18235/>. API base: `http://127.0.0.1:18235/v1`.
Ctrl+C in the launching terminal stops the server. Inspect it from another terminal:

```sh
curl --fail http://127.0.0.1:18235/v1/models
curl --fail http://127.0.0.1:18235/metrics
nvidia-smi --query-gpu=index,name,memory.used,memory.free,power.draw --format=csv
```

Keep the listener on loopback; upstream requires a command-line API key if intentionally serving other machines.
Identify the existing listener when a port is busy rather than killing unrelated WSL/Docker services.

A running process keeps its old executable after a rebuild. Restart Basalt when ready to use the new engine;
a WSL shutdown is unnecessary.


See [Codex and maintenance](wsl2-operations.md) for client setup and update checks.
