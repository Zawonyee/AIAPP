import os
import sys
import shutil
import subprocess

def create_bundle_script():
    """创建一个独立的脚本文件，将main.py的内容集成到app_gui中"""
    print("正在创建打包用独立脚本...")
    
    # 读取main.py内容
    with open('main.py', 'r', encoding='utf-8') as f:
        main_content = f.read()
    
    # 读取app_gui.py内容
    with open('app_gui.py', 'r', encoding='utf-8') as f:
        app_content = f.read()
    
    # 替换导入语句
    app_content = app_content.replace('import main', '# main module integrated below')
    
    # 创建一个新的combined_app.py
    with open('combined_app.py', 'w', encoding='utf-8') as f:
        f.write("# Combined application with integrated main module\n\n")
        # 先写入main.py内容作为模块
        f.write("# === Begin main.py content ===\n")
        f.write(main_content)
        f.write("\n# === End main.py content ===\n\n")
        # 再写入app_gui.py内容
        f.write("# === Begin app_gui.py content ===\n")
        f.write(app_content)
        f.write("\n# === End app_gui.py content ===\n")
    
    print("创建完成: combined_app.py")
    return 'combined_app.py'

def build_exe(script_path):
    """使用PyInstaller打包应用"""
    print(f"正在打包 {script_path}...")
    
    # 确保存在env.example文件
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
        '--add-data', 'env.example;.',
        # 添加额外的隐藏导入
        '--hidden-import', 'PyQt5.QtCore',
        '--hidden-import', 'PyQt5.QtGui',
        '--hidden-import', 'PyQt5.QtWidgets',
        '--hidden-import', 'cv2',
        '--hidden-import', 'numpy',
        '--hidden-import', 'openai',
        script_path
    ]
    
    # 执行打包命令
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 输出结果
    if result.returncode == 0:
        print("打包成功!")
        print(f"可执行文件位于: {os.path.abspath('dist/' + os.path.splitext(os.path.basename(script_path))[0] + '.exe')}")
    else:
        print("打包失败!")
        print("错误信息:")
        print(result.stderr)
    
    return result.returncode == 0

if __name__ == "__main__":
    try:
        # 创建合并后的脚本
        combined_script = create_bundle_script()
        
        # 打包合并后的脚本
        success = build_exe(combined_script)
        
        # 如果成功，删除临时文件
        if success and os.path.exists(combined_script):
            print(f"正在清理临时文件: {combined_script}")
            os.remove(combined_script)
            
        print("所有操作完成!")
        
    except Exception as e:
        print(f"发生错误: {str(e)}")
    
    input("按Enter键退出...") 