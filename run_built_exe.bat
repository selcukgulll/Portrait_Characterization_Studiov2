@echo off
cd /d "%~dp0"
if not exist "dist\FootballPortraitStudio\FootballPortraitStudio.exe" (
  echo EXE bulunamadi. Once build_exe.bat calistir.
  pause
  exit /b 1
)
start "" "dist\FootballPortraitStudio\FootballPortraitStudio.exe"
