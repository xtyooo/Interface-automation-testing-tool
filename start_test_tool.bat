@echo off
chcp 936 >nul
echo ���������ӿ��Զ������Թ���...

:: ���Python�Ƿ�װ
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ����: δ��⵽Python��װ���밲װPython 3.6����߰汾��
    pause
    exit /b 1
)

:: ��鲢��װ��Ҫ�Ŀ�
echo ���ڼ���Ҫ�Ŀ�...
python -c "import pandas" >nul 2>&1
if %errorlevel% neq 0 (
    echo ���ڰ�װpandas...
    pip install pandas
)

python -c "import openpyxl" >nul 2>&1
if %errorlevel% neq 0 (
    echo ���ڰ�װopenpyxl...
    pip install openpyxl
)

:: ������Ҫ��Ŀ¼
if not exist output mkdir output
if not exist analysis_results mkdir analysis_results

:: ����GUIӦ��
echo �������Թ���...
python gui_test.py

if %errorlevel% neq 0 (
    echo �����쳣�˳����������: %errorlevel%
    pause
) 