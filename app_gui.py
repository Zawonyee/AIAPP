import os
import sys
import time
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QFileDialog, QTextEdit, QLineEdit, QMessageBox,
                            QProgressBar, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap
import main

class VideoProcessThread(QThread):
    progress_update = pyqtSignal(str)
    processing_complete = pyqtSignal()
    
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        
    def run(self):
        try:
            # 设置环境变量
            os.environ["VIDEO_PATH"] = self.video_path
            
            # 创建追踪器并开始处理
            self.tracker = main.VideoObjectTracker()
            self.progress_update.emit("正在分析视频...")
            
            # 手动执行视频处理部分，但不执行interactive_query部分
            try:
                cap = cv2.VideoCapture(self.tracker.video_path)
                if not cap.isOpened():
                    self.progress_update.emit(f"无法打开视频文件: {self.tracker.video_path}")
                    return

                # 只需要获取视频的第一帧和最后一帧
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.progress_update.emit(f"视频共有 {total_frames} 帧")
                
                # 获取第一帧
                ret, first_frame = cap.read()
                if not ret:
                    self.progress_update.emit("无法读取视频第一帧")
                    return
                
                # 获取最后一帧
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
                ret, last_frame = cap.read()
                if not ret:
                    self.progress_update.emit("无法读取视频最后一帧")
                    return
                
                # 分析这两帧中的物体
                self.progress_update.emit("正在分析视频首帧...")
                self.tracker.first_frame_objects = self.tracker.analyze_frame(first_frame)
                
                self.progress_update.emit("正在分析视频尾帧...")
                self.tracker.last_frame_objects = self.tracker.analyze_frame(last_frame)
                
                # 打印检测到的物体
                all_objects = set(list(self.tracker.first_frame_objects.keys()) + list(self.tracker.last_frame_objects.keys()))
                if all_objects:
                    self.progress_update.emit(f"检测到的物体: {', '.join(all_objects)}")
                else:
                    self.progress_update.emit("未检测到任何物体")
                
            finally:
                if 'cap' in locals() and cap is not None:
                    cap.release()
            
            # 处理完成
            self.processing_complete.emit()
            
        except Exception as e:
            self.progress_update.emit(f"处理出错: {str(e)}")

class VideoAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = None
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("视频分析与AI对话系统")
        self.setMinimumSize(800, 600)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 顶部标题
        title_label = QLabel("视频分析与AI对话系统")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)
        
        # 视频上传区域
        upload_layout = QHBoxLayout()
        self.video_path_label = QLabel("未选择视频")
        upload_button = QPushButton("选择视频文件")
        upload_button.clicked.connect(self.select_video)
        upload_layout.addWidget(self.video_path_label)
        upload_layout.addWidget(upload_button)
        main_layout.addLayout(upload_layout)
        
        # 分析按钮
        process_button = QPushButton("开始分析")
        process_button.clicked.connect(self.process_video)
        main_layout.addWidget(process_button)
        
        # 进度显示
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(100)
        main_layout.addWidget(self.progress_text)
        
        # 对话区域
        chat_layout = QVBoxLayout()
        
        # 聊天历史
        self.chat_history = QTextEdit()
        self.chat_history.setReadOnly(True)
        self.chat_history.setMinimumHeight(200)
        chat_layout.addWidget(self.chat_history)
        
        # 输入区域
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("输入问题...")
        self.input_field.returnPressed.connect(self.send_query)
        send_button = QPushButton("发送")
        send_button.clicked.connect(self.send_query)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(send_button)
        chat_layout.addLayout(input_layout)
        
        main_layout.addLayout(chat_layout)
        
        # 设置中心组件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def select_video(self):
        file_dialog = QFileDialog()
        video_path, _ = file_dialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi *.mov)")
        
        if video_path:
            self.video_path_label.setText(video_path)
            self.add_progress_message(f"已选择视频: {os.path.basename(video_path)}")
    
    def process_video(self):
        video_path = self.video_path_label.text()
        
        if video_path == "未选择视频":
            QMessageBox.warning(self, "警告", "请先选择一个视频文件!")
            return
            
        # 禁用按钮防止重复处理
        buttons = self.findChildren(QPushButton)
        for button in buttons:
            button.setEnabled(False)
        
        # 创建并启动处理线程
        self.processing_thread = VideoProcessThread(video_path)
        self.processing_thread.progress_update.connect(self.add_progress_message)
        self.processing_thread.processing_complete.connect(self.processing_finished)
        self.processing_thread.start()
        
        self.add_progress_message("开始处理视频...")
    
    def processing_finished(self):
        self.add_progress_message("视频处理完成！现在您可以提问关于视频内容的问题。")
        self.tracker = self.processing_thread.tracker
        
        # 确保tracker被正确初始化
        if not self.tracker:
            try:
                self.add_progress_message("重新初始化视频处理器...")
                self.tracker = main.VideoObjectTracker()
                self.tracker.video_path = self.video_path_label.text()
                self.tracker.first_frame_objects = {}
                self.tracker.last_frame_objects = {}
            except Exception as e:
                self.add_progress_message(f"初始化失败: {str(e)}")
        
        # 重新启用按钮
        buttons = self.findChildren(QPushButton)
        for button in buttons:
            button.setEnabled(True)
        
    def send_query(self):
        query = self.input_field.text().strip()
        
        if not query:
            return
            
        if not self.tracker:
            QMessageBox.warning(self, "警告", "请先处理视频!")
            return
            
        # 显示用户问题
        self.add_chat_message(f"您: {query}", "user")
        self.input_field.clear()
        
        # 处理查询并获取响应
        try:
            if "在" in query and "移动" in query:
                response = self.tracker.describe_movement(query)
                if not response:
                    response = "无法描述移动情况，请尝试其他问题。"
            else:
                response = self.tracker.general_object_query(query)
                if not response:
                    # 使用轨迹分析尝试回答
                    potential_objects = []
                    words = query.lower().split()
                    for word in words:
                        if len(word) > 1 and word not in ["在哪", "哪里", "位置", "移动", "什么", "有没有", "的", "是", "了", "最后"]:
                            potential_objects.append(word)
                    
                    if potential_objects:
                        obj = potential_objects[0]
                        is_first_frame = "初始" in query or "开始" in query
                        is_last_frame = "最后" in query or "现在" in query or "当前" in query
                        
                        # 检查物体是否在初始或最后帧中
                        if is_first_frame and obj in self.tracker.first_frame_objects:
                            response = f"{obj}在视频开始时出现在画面中。"
                        elif is_last_frame and obj in self.tracker.last_frame_objects:
                            response = f"{obj}在视频结束时出现在画面中。"
                        elif obj in self.tracker.first_frame_objects or obj in self.tracker.last_frame_objects:
                            response = f"{obj}在视频中被检测到。"
                        else:
                            response = f"未在视频中检测到{obj}。"
                    else:
                        response = "请具体询问视频中的某个物体。"
                        
            self.add_chat_message(f"AI: {response}", "ai")
        except Exception as e:
            self.add_chat_message(f"AI: 处理您的问题时出错: {str(e)}", "error")
    
    def add_progress_message(self, message):
        self.progress_text.append(message)
        # 自动滚动到底部
        scrollbar = self.progress_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        
    def add_chat_message(self, message, msg_type):
        # 不同类型的消息可以有不同的样式
        if msg_type == "user":
            self.chat_history.append(f"<p style='color: #007bff;'>{message}</p>")
        elif msg_type == "ai":
            self.chat_history.append(f"<p style='color: #28a745;'>{message}</p>")
        elif msg_type == "error":
            self.chat_history.append(f"<p style='color: #dc3545;'>{message}</p>")
        else:
            self.chat_history.append(f"<p>{message}</p>")
            
        # 自动滚动到底部
        scrollbar = self.chat_history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoAnalysisApp()
    window.show()
    sys.exit(app.exec_()) 