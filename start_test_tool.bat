@echo off
:: 设置控制台代码页为UTF-8
chcp 65001 >nul

:: 设置Python环境变量
set PYTHONIOENCODING=utf-8

echo 正在启动接口自动化测试工具...

:: 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未检测到Python安装，请安装Python 3.6或更高版本
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

python -c "import transformers" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装transformers...
    pip install transformers
)

python -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装torch...
    pip install torch
)

python -c "import sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo 正在安装scikit-learn...
    pip install scikit-learn
)

:: 创建必要的目录
if not exist output mkdir output
if not exist analysis_results mkdir analysis_results

:: 直接启动Python程序
echo 启动测试工具...
python -u gui_test.py

if %errorlevel% neq 0 (
    echo 程序异常退出，错误代码: %errorlevel%
    pause
)

reg add HKCU\Console /v CodePage /t REG_DWORD /d 65001 /f 