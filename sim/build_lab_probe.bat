@echo off
rem Build the STANDALONE lab timing probe on the Windows VM (one step: double-click).
rem Result: dist\win\lab_timing_probe\lab_timing_probe.exe  - copy the WHOLE folder to the lab PC.
rem Needs the Windows venv (.venv\ or .venv-win\, same as scripts\build_release.bat) with
rem pyinstaller, pylablib, pipython and numpy, and API\PI\E816_DLL_x64.dll in the repo.
setlocal
set "ROOT=%~dp0.."
set "PY=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%ROOT%\.venv-win\Scripts\python.exe"
if not exist "%PY%" (echo ERROR: no Windows venv in .venv\ or .venv-win\ & pause & exit /b 1)
pushd "%ROOT%"
"%PY%" -m PyInstaller sim\lab_timing_probe.spec --noconfirm --distpath dist\win --workpath build\win\probe
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (echo ERROR: PyInstaller failed & pause & exit /b %RC%)
echo.
echo Done: dist\win\lab_timing_probe\lab_timing_probe.exe
pause
