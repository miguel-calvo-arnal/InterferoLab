# install/ — deployment payloads (not tracked in git)

This folder holds installers and bundles used when deploying InterferoLab to
lab machines. Only this README travels in git; the binaries stay on local
disk / private storage:

- `VC_redist.x64.exe` — Microsoft Visual C++ 2015–2022 Redistributable (x64),
  version 14.44.35211.0, SHA-256
  `cc0ff0eb1dc3f5188ae6300faef32bf5beeba4bdd6e8e445a9184072096b713b`
  (matches the hash Microsoft publishes at
  <https://aka.ms/vs/17/release/vc_redist.x64.exe>).
- Dated devkit/release archives, when present.

## Why the VC++ redistributable matters

The compiled backend (`analysis_backend.*.pyd`) is built with MSVC and needs
these DLLs at runtime:

    MSVCP140.dll      (C++ standard library)
    VCRUNTIME140.dll  (already in the bundle, shipped by Python)
    VCRUNTIME140_1.dll
    VCOMP140.DLL      (OpenMP runtime)

On a development machine they resolve from System32 because the Build Tools
are installed. On a clean PC they are missing and the app fails with
`ImportError: DLL load failed while importing analysis_backend`.

A redistributable **newer** than the toolset used to compile (MSBuild 17.14 /
VS 2022) also works — they are backwards compatible; an older one does not.

## How it is handled today (automated, app-local deployment)

`interferolab.spec` copies `MSVCP140.dll` and `VCOMP140.DLL` into the bundle
(`dist/win/InterferoLab/_internal/`) on every Windows build — the "app-local
deployment" Microsoft allows. The package is self-contained: target machines
need no installer and no admin rights. (`VCRUNTIME140*.dll` already ship with
Python, so they are not collected twice.)

Search order used by the spec:

1. `install/vcredist/x64/` — a copy pinned in the repo folder, if present
2. the local Visual Studio / Build Tools `Redist` folder, highest version
   first (same toolset that compiled the `.pyd`)
3. `C:\Windows\System32`

During the build PyInstaller prints where each DLL came from
(`MSVC runtime: MSVCP140.dll <- ...`). If a WARNING appears instead, the
package is NOT self-contained and the target machine will need the
redistributable.

To pin an exact version independent of the build machine, create
`install/vcredist/x64/` and copy the two DLLs from:

    C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Redist\
        MSVC\<version>\x64\Microsoft.VC143.CRT\MSVCP140.dll
        MSVC\<version>\x64\Microsoft.VC143.OpenMP\vcomp140.dll

Do NOT use the copies PySide6 ships in `_internal/PySide6/` or
`_internal/shiboken6/`; they can be older than what the `.pyd` requires and
would break loading.

## The installer as plan B

If app-local deployment ever fails, or a target machine has a broken/outdated
runtime, install once (idempotent — does nothing if an equal or newer version
is present):

    VC_redist.x64.exe /install /passive /norestart

To check whether it is already installed:

    reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Version
