@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo Python bulunamadi. Python 3.11 x64 kur ve "Add Python to PATH" sec.
  pause
  exit /b 1
)

if not exist .venv (
  py -3.11 -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

echo.
echo NVIDIA CUDA 12.4 PyTorch kuruluyor...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
  echo CUDA paketi kurulamadi. CPU paketi deneniyor...
  python -m pip install torch torchvision
)

python -m pip install -r requirements.txt

if not exist config.json copy /Y config.example.json config.json >nul

echo.
echo Kurulum tamamlandi.
echo run_app.bat ile baslat.
pause
