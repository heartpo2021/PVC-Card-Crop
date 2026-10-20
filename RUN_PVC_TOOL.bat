@echo off
cd /d "%~dp0"
python -m pip install pypdfium2 pillow numpy opencv-python-headless tkinterdnd2
if errorlevel 1 (echo Installation failed. Check Python and internet connection. & pause & exit /b 1)
python pvc_card_tool.py
if errorlevel 1 pause
