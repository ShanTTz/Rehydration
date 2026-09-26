@echo off
setlocal
set "PYTHONW_EXE=%~1"
set "ROOT=%~dp0.."

pushd "%ROOT%"
start "BDMTF cross-platform" "%PYTHONW_EXE%" "%ROOT%\scripts\run_cross_platform_background.py"
popd
endlocal
