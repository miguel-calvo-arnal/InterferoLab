@echo off
rem Double-click launcher for build_release.ps1 (Windows).
rem %~dp0 is this .bat's folder, so it works from anywhere.
rem Arguments pass straight through: build_release.bat -SkipBuild -Force ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_release.ps1" %*
echo.
pause
