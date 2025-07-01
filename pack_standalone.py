import os
import sys
import subprocess

def build_exe():
    """使用PyInstaller打包应用"""
    print("开始打包独立应用...")
    
    # 检查app_standalone.py是否存在
    if not os.path.exists('app_standalone.py'):
        print("错误: app_standalone.py文件不存在!")
        return False
    
    # 确保env.example文件存在
    if not os.path.exists('env.example'):
        print("警告: env.example文件不存在，创建默认版本")
        with open('env.example', 'w', encoding='utf-8') as f:
            f.write("# API配置\n")
            f.write("DASHSCOPE_API_KEY=your_api_key_here\n")
            f.write("MODEL_NAME=qwen-vl-max-0809\n")
    
    # 构建PyInstaller命令
    cmd = [
        sys.executable,
        '-m',
        'PyInstaller',
        '--onefile',
        '--windowed',
        '--clean',
        '--noconfirm',
        '--name=视频分析AI对话系统',
        '--add-data', 'env.example;.',
        'app_standalone.py'
    ]
    
    print("执行打包命令...")
    print(f"命令: {' '.join(cmd)}")
    
    # 执行打包命令
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # 输出结果
        if result.returncode == 0:
            print("打包成功!")
            print(f"可执行文件位于: {os.path.abspath('dist/视频分析AI对话系统.exe')}")
            return True
        else:
            print("打包失败!")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"打包过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        success = build_exe()
        if success:
            print("打包完成!")
        else:
            print("打包失败，请检查错误信息。")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    input("按Enter键退出...") 