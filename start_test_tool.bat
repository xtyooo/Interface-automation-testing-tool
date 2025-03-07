@echo off
chcp 936 >nul
echo 正在启动接口自动化测试工具...

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python安装，请安装Python 3.6或更高版本。
    pause
    exit /b 1
)

:: 检查并安装必要的库
echo 正在检查必要的库...
python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装pandas...
    pip install pandas
)

python -c "import openpyxl" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装openpyxl...
    pip install openpyxl
)

:: 创建必要的目录
if not exist output mkdir output
if not exist analysis_results mkdir analysis_results

:: 启动GUI应用
echo 启动测试工具...
python gui_test.py

if %errorlevel% neq 0 (
    echo 程序异常退出，错误代码: %errorlevel%
    pause
) 