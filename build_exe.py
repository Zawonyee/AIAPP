import os
import subprocess
import shutil
import sys

def build_exe():
    print("开始打包应用程序...")
    
    # 确保安装了必要的依赖
    print("检查并安装依赖...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
    # 创建构建目录
    if not os.path.exists("dist"):
        os.makedirs("dist")
    
    # 使用PyInstaller打包
    print("正在使用PyInstaller打包应用...")
    
    # 使用Python -m 方式运行PyInstaller
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=视频分析AI对话系统",
        "--windowed",  # 使用GUI模式，不显示控制台
        "--onefile",   # 打包成单个exe文件
        "--icon=NONE", # 可以替换为你的图标路径
        "--add-data=env.example;.",  # 包含配置文件示例
        "app_gui.py"    # 主程序入口
    ]
    
    # 执行打包命令
    subprocess.run(cmd)
    
    print("打包完成！")
    print(f"可执行文件位于: {os.path.abspath('dist/视频分析AI对话系统.exe')}")

if __name__ == "__main__":
    build_exe() 