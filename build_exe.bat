@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo HATA: .venv bulunamadi.
  echo Once install_gpu.bat calistir.
  pause
  exit /b 1
)

call .venv\Scripts\activate.bat

echo Eski build ve dist klasorleri temizleniyor...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name FootballPortraitStudio ^
  --collect-all diffusers ^
  --collect-all transformers ^
  --collect-all rembg ^
  --collect-all onnxruntime ^
  --collect-all torch ^
  --add-data "config.example.json;." ^
  app.py

if errorlevel 1 (
  echo.
  echo EXE olusturma basarisiz oldu.
  pause
  exit /b 1
)

set "EXE_PATH=%~dp0dist\FootballPortraitStudio\FootballPortraitStudio.exe"
set "PYDLL_PATH=%~dp0dist\FootballPortraitStudio\_internal\python312.dll"

if not exist "%EXE_PATH%" (
  echo HATA: Beklenen EXE bulunamadi:
  echo %EXE_PATH%
  pause
  exit /b 1
)

if not exist "%PYDLL_PATH%" (
  echo UYARI: python312.dll beklenen yerde bulunamadi:
  echo %PYDLL_PATH%
  echo EXE calismayabilir.
)

echo.
echo BASARILI.
echo Calistirilacak EXE SADECE sudur:
echo %EXE_PATH%
echo.
echo build klasorundeki hicbir exe veya dosyayi calistirma.
echo dist\FootballPortraitStudio klasorunun tamamini birlikte tut.
echo.
start "" explorer.exe "%~dp0dist\FootballPortraitStudio"
pause
