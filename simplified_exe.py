"""
简化版打包指南

要打包此应用，您可以使用auto-py-to-exe工具，它提供了一个图形界面来配置和创建exe文件。

按以下步骤操作：

1. 安装auto-py-to-exe：
   pip install auto-py-to-exe

2. 运行auto-py-to-exe：
   auto-py-to-exe

3. 在打开的GUI中：
   - 选择"Script Location"为 app_gui.py
   - 选择"One File" 选项
   - 选择"Window Based" 选项
   - 添加以下文件到"Additional Files"：
     - env.example -> .
   - 设置输出目录为：dist
   - 点击"CONVERT .PY TO .EXE" 按钮

4. 打包完成后，在dist目录中找到生成的exe文件

也可以直接使用PyInstaller命令行：
pyinstaller --onefile --windowed --add-data "env.example;." app_gui.py
"""

import os
import sys
import subprocess

def install_auto_py_to_exe():
    print("安装auto-py-to-exe...")
    subprocess.run([sys.executable, "-m", "pip", "install", "auto-py-to-exe"])
    print("安装完成！")

def launch_auto_py_to_exe():
    print("启动auto-py-to-exe图形界面...")
    subprocess.run([sys.executable, "-m", "auto_py_to_exe"])

def run_direct_pyinstaller():
    print("直接使用PyInstaller打包...")
    cmd = [
        sys.executable, 
        "-m", 
        "PyInstaller", 
        "--onefile", 
        "--windowed", 
        "--add-data", 
        "env.example;.", 
        "app_gui.py"
    ]
    subprocess.run(cmd)
    print("打包完成！")

if __name__ == "__main__":
    print("视频分析AI对话系统打包工具")
    print("------------------------")
    print("1. 安装auto-py-to-exe")
    print("2. 启动auto-py-to-exe图形界面")
    print("3. 直接使用PyInstaller打包")
    print("4. 退出")
    
    choice = input("请选择操作 (1-4): ")
    
    if choice == "1":
        install_auto_py_to_exe()
    elif choice == "2":
        launch_auto_py_to_exe()
    elif choice == "3":
        run_direct_pyinstaller()
    elif choice == "4":
        print("退出程序")
    else:
        print("无效选择") 