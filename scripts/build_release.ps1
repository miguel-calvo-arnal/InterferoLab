# =====================================================================
# build_release.ps1 - Windows: C++ backend -> PyInstaller -> releases\*.zip
#
# Runs the whole release chain in one go:
#   1. Configure + build the C++ backend with MSVC + vcpkg (Release).
#   2. Copy the resulting .pyd next to backend\analysis\ (where the spec
#      globs it first; see interferolab.spec).
#   3. Import-check the module BEFORE the slow PyInstaller run, so an ABI
#      or Python-version mismatch fails in seconds instead of minutes.
#   4. Freeze with interferolab.spec into dist\win\ (the root of dist\
#      belongs to the Linux build; its symlinks cannot be deleted over
#      vboxsf).
#   5. Strip runtime residue and archive dist\win\InterferoLab into
#      releases\.
#
# Prerequisites (once per machine, see backend\analysis\compile_notes.txt):
#   - Visual Studio 2022 Build Tools with the VCTools workload
#   - vcpkg in C:\vcpkg with:  opencv4 tiff fftw3[threads]  (x64-windows)
#   - The Windows venv in .venv\ or .venv-win\ (the latter when the folder
#     is shared with Linux; -VenvDir <path> also works)
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\build_release.ps1
#   ...  -SkipBuild        reuse the .pyd already in backend\analysis\
#   ...  -VcpkgRoot D:\vcpkg   if vcpkg is not in C:\vcpkg
#   ...  -VenvDir <path>   explicit venv location
#   ...  -Force            overwrite the release archive if it exists
#
# NOTE: build\Release\ is NOT deleted: interferolab.spec takes the vcpkg
# DLLs the .pyd needs inside the bundle from there.
#
# The Linux counterpart is scripts/build_release.sh
# =====================================================================
[CmdletBinding()]
param(
    [switch] $SkipBuild,
    [switch] $Force,
    [string] $VcpkgRoot = $(if ($env:VCPKG_ROOT) { $env:VCPKG_ROOT } else { "C:\vcpkg" }),
    [string] $Generator = "Visual Studio 17 2022",
    [string] $VenvDir   = ""
)

$ErrorActionPreference = "Stop"

# An accidental click inside a classic console window enables text selection
# (QuickEdit) and FREEZES the running build until a key is pressed — which
# looks like the script randomly needing Enter to continue. Disable QuickEdit
# for this console session. Harmless if it fails (Windows Terminal, no console).
try {
    Add-Type -Namespace Win32 -Name ConsoleMode -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError=true)]
public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
[DllImport("kernel32.dll", SetLastError=true)]
public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
'@
    $hIn  = [Win32.ConsoleMode]::GetStdHandle(-10)  # STD_INPUT_HANDLE
    $mode = 0
    if ([Win32.ConsoleMode]::GetConsoleMode($hIn, [ref]$mode)) {
        # clear ENABLE_QUICK_EDIT_MODE (0x40); ENABLE_EXTENDED_FLAGS (0x80)
        # must be set for the change to take effect.
        [void][Win32.ConsoleMode]::SetConsoleMode($hIn, ($mode -band (-bnot 0x40)) -bor 0x80)
    }
} catch { }

$Root    = Split-Path -Parent $PSScriptRoot
# The Windows venv: .venv\ on a native checkout, or .venv-win\ when the
# folder is shared with Linux (there .venv\ is the Linux venv, with bin/
# instead of Scripts\, and the two cannot coexist in one directory).
if ($VenvDir) {
    $Py = Join-Path $VenvDir "Scripts\python.exe"
} else {
    $Py = Join-Path $Root ".venv\Scripts\python.exe"
    if (-not (Test-Path $Py)) { $Py = Join-Path $Root ".venv-win\Scripts\python.exe" }
}
$Src     = Join-Path $Root "backend\analysis"
$Build   = Join-Path $Src  "build"
$Name    = "InterferoLab"
$Stamp   = Get-Date -Format "yyyy-MM-dd"
$Archive = Join-Path $Root "releases\$Name-$Stamp-windows-x64.zip"

function Step($m) { Write-Host "`n== $m" -ForegroundColor Cyan }
function Die($m)  { Write-Host "ERROR: $m" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------------
Step "0/5  Preflight checks"
# ---------------------------------------------------------------------
if (-not (Test-Path $Py)) {
    Die ("no Windows venv found (.venv\ or .venv-win\).`n" +
         "       Create it with:  py -m venv .venv-win ; .venv-win\Scripts\pip install -r requirements.txt`n" +
         "       (or point at another one with -VenvDir <path>)")
}
if (-not (Test-Path (Join-Path $Root "interferolab.spec"))) { Die "interferolab.spec is missing" }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { Die "cmake is not on the PATH" }
$Toolchain = Join-Path $VcpkgRoot "scripts\buildsystems\vcpkg.cmake"
if (-not $SkipBuild -and -not (Test-Path $Toolchain)) {
    Die "cannot find vcpkg in '$VcpkgRoot' (use -VcpkgRoot <path> or set VCPKG_ROOT)"
}
if ((Test-Path $Archive) -and -not $Force) {
    Die "$Archive already exists (use -Force to overwrite it)"
}
Write-Host "  project : $Root"
Write-Host "  python  : $(& $Py -V)"
Write-Host "  vcpkg   : $VcpkgRoot"
Write-Host "  target  : $Archive"

# ---------------------------------------------------------------------
if (-not $SkipBuild) {
    Step "1/5  Building the C++ backend (MSVC + vcpkg, Release)"
    # pybind11_DIR is asked of the venv itself, to avoid depending on
    # user-specific absolute paths (see compile_notes.txt).
    $Pybind11Dir = (& $Py -m pybind11 --cmakedir).Trim()
    if (-not $Pybind11Dir) { Die "could not get pybind11 --cmakedir from the venv" }

    # Arguments as an array: interpolated inside quotes, a path with spaces
    # would split into several arguments and cmake would receive garbage.
    $cmakeArgs = @(
        "-S", $Src, "-B", $Build, "-G", $Generator, "-A", "x64",
        "-DPython3_EXECUTABLE=$Py",
        "-Dpybind11_DIR=$Pybind11Dir",
        "-DCMAKE_TOOLCHAIN_FILE=$Toolchain",
        "-DVCPKG_TARGET_TRIPLET=x64-windows",
        "-DCMAKE_BUILD_TYPE=Release"
    )
    & cmake @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Die "cmake configuration failed" }

    & cmake --build $Build --config Release -j
    if ($LASTEXITCODE -ne 0) { Die "backend build failed" }

    Step "2/5  Copying the .pyd to backend\analysis\"
    # Copy (not move): the spec globs backend\analysis\ first, and keeping
    # build\Release\ intact allows incremental rebuilds and preserves the
    # vcpkg DLLs the spec needs from there.
    $Built = Get-ChildItem -Path (Join-Path $Build "Release") -Filter "analysis_backend*.pyd" -ErrorAction SilentlyContinue
    if (-not $Built) { Die "cmake produced no analysis_backend*.pyd in $Build\Release" }
    Get-ChildItem -Path $Src -Filter "analysis_backend*.pyd" -ErrorAction SilentlyContinue | Remove-Item -Force
    foreach ($f in $Built) {
        Copy-Item $f.FullName -Destination $Src -Force
        Write-Host "  $($f.Name) -> backend\analysis\"
    }
} else {
    Step "1-2/5  Skipped (-SkipBuild): reusing the existing .pyd"
}

# ---------------------------------------------------------------------
Step "3/5  Verifying that the module imports"
# ---------------------------------------------------------------------
# The .pyd links against the vcpkg DLLs cmake leaves in build\Release\
# (see compile_notes.txt). Since Python 3.8 PATH no longer resolves DLLs:
# the directory must be declared with os.add_dll_directory, just like
# main.py does in development.
$check = @"
import os, sys
sys.path.insert(0, r'$Src')
for d in (r'$Build\Release', r'$Root\dist\win\InterferoLab\_internal'):
    if os.path.isdir(d):
        os.add_dll_directory(d)
import analysis_backend as a
print('  analysis_backend OK - reconstruction methods:', [m['id'] for m in a.get_reconstruction_methods()])
"@
& $Py -c $check
if ($LASTEXITCODE -ne 0) { Die "the compiled backend does NOT import; aborting before PyInstaller" }

# ---------------------------------------------------------------------
Step "4/5  Freezing with PyInstaller"
# ---------------------------------------------------------------------
# dist\ and build\ (PyInstaller's workpath) are also used by the Linux
# build, and through the vboxsf shared folder Windows cannot delete the
# symlinks the Linux PyInstaller leaves behind (Remove-Item fails with
# "The parameter is incorrect"). Windows therefore packages into its own
# subfolders, dist\win\ and build\win\.
$DistRoot = Join-Path $Root "dist\win"
$WorkRoot = Join-Path $Root "build\win"
$DistApp  = Join-Path $DistRoot $Name
if (Test-Path $DistApp) { Remove-Item $DistApp -Recurse -Force }   # no leftovers from a previous build
Push-Location $Root
try {
    & $Py -m PyInstaller interferolab.spec --noconfirm --distpath $DistRoot --workpath $WorkRoot
    if ($LASTEXITCODE -ne 0) { Die "PyInstaller failed" }
} finally { Pop-Location }
if (-not (Test-Path $DistApp)) { Die "PyInstaller did not produce $DistApp" }

# The app writes app_config.json and logs\ next to the executable on
# startup; if someone ran it from dist\, that residue must not ship.
Remove-Item (Join-Path $DistApp "app_config.json") -Force -ErrorAction SilentlyContinue
Remove-Item (Join-Path $DistApp "logs") -Recurse -Force -ErrorAction SilentlyContinue

# >>> sim-exclusion check
# The hardware simulator (sim\, with FAKE pylablib/pipython) must never ship.
# interferolab.spec already excludes it and aborts if it slips in; this
# second gate inspects the finished bundle (files AND the embedded PYZ).
$simCheck = @"
import os, sys
from PyInstaller.archive.readers import CArchiveReader, PKG_ITEM_PYZ
dist, exe = r'$DistApp', r'$DistApp\$Name.exe'
bad = []
for dirpath, dirnames, filenames in os.walk(dist):
    rel = os.path.relpath(dirpath, dist).replace(os.sep, '/')
    if 'sim' in rel.split('/'):
        bad.append(rel + '/')
    bad += [os.path.join(rel, f) for f in filenames
            if f in ('run_simulated.py', 'lab_timing_probe.py', 'hwprofile.py')]
arch = CArchiveReader(exe)
for name, entry in arch.toc.items():
    if entry[-1] != PKG_ITEM_PYZ:
        continue
    pyz = arch.open_embedded_archive(name)
    for mod in pyz.toc:
        if mod == 'sim' or mod.startswith('sim.'):
            bad.append('PYZ:' + mod)
        elif mod.split('.')[0] in ('pylablib', 'pipython'):
            code = pyz.extract(mod)
            doc = code.co_consts[0] if code.co_consts and isinstance(code.co_consts[0], str) else ''
            if '__simulated__' in code.co_names or doc.lstrip().startswith('[SIMULATION]'):
                bad.append('PYZ:' + mod + ' (FAKE driver)')
if bad:
    print('  simulator content in the bundle:', *bad[:20], sep='\n    ')
    sys.exit(1)
print('  OK - no simulator module or fake driver in the bundle')
"@
& $Py -c $simCheck
if ($LASTEXITCODE -ne 0) { Die "the bundle contains the hardware simulator (sim\); refusing to package" }
# <<< sim-exclusion check

# ---------------------------------------------------------------------
Step "5/5  Creating the archive in releases\"
# ---------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path (Join-Path $Root "releases") | Out-Null
if (Test-Path $Archive) { Remove-Item $Archive -Force }
# ZipFile instead of Compress-Archive: much faster with hundreds of files,
# and includeBaseDirectory=$true leaves "InterferoLab\" at the zip root,
# same as the Linux tar.gz.
Add-Type -AssemblyName System.IO.Compression.FileSystem
[System.IO.Compression.ZipFile]::CreateFromDirectory(
    $DistApp, $Archive, [System.IO.Compression.CompressionLevel]::Optimal, $true)

$distMB = [math]::Round((Get-ChildItem $DistApp -Recurse -File | Measure-Object Length -Sum).Sum / 1MB, 1)
$zipMB  = [math]::Round((Get-Item $Archive).Length / 1MB, 1)
Write-Host "`nDone" -ForegroundColor Green
Write-Host "  dist\win\$Name : $distMB MB"
Write-Host "  archive        : $Archive ($zipMB MB)"
