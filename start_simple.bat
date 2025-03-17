@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

:: 直接启动Python程序
python -u gui_test.py

pause 