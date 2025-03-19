@echo off
chcp 65001 > nul
cls

:: 显示标题
echo ================================
echo    接口自动化测试工具 - 启动中
echo ================================
echo.

:: 定义加载动画字符
set "loading=⣾⣽⣻⢿⡿⣟⣯⣷"

:: 显示加载动画
for /l %%i in (1,1,30) do (
    for %%l in (%loading%) do (
        <nul set /p =\r正在启动中... %%l
        timeout /t 0 /nobreak > nul
    )
)

:: 清除加载动画
<nul set /p =\r                                            \r

:: 启动Python程序
python gui_test.py

:: 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo 程序运行出现错误，请检查以上错误信息。
    pause
) 