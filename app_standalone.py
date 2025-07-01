import os
import sys
import time
import cv2
import json
import base64
import logging
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
import pymysql
from dotenv import load_dotenv
from openai import OpenAI
from colorama import init, Fore, Style
from tqdm import tqdm
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QFileDialog, QTextEdit, QLineEdit, QMessageBox,
                            QProgressBar, QScrollArea)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap

# 加载环境变量
load_dotenv()

# 配置日志系统
logging.basicConfig(
    level=os.getenv("DEBUG_LEVEL", "info").upper(),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("object_tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ===== 从main.py集成的VideoObjectTracker类 =====
class VideoObjectTracker:
    def __init__(self):
        # 初始化配置
        self.load_config()
        self.setup_database()
        self.setup_tracking()
        
        # 状态变量
        self.frame_count = 0
        self.trajectories = defaultdict(lambda: deque(maxlen=int(os.getenv("TRACK_HISTORY", "30"))))
        self.object_cache = {}
        self.last_roi = {}  # 存储每个物体最后的ROI
        
        # 添加内存管理
        self.max_cache_size = int(os.getenv("MAX_CACHE_SIZE", "1024")) * 1024 * 1024  # MB转字节
        self.current_cache_size = 0
        
        # 特征点匹配器初始化
        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last_features = {}
        
        # 初始化空的帧对象列表
        self.first_frame_objects = {}
        self.last_frame_objects = {}

    def load_config(self):
        """加载环境变量配置"""
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        self.model_name = os.getenv("MODEL_NAME", "qwen-vl-max-0809")
        self.video_path = os.getenv("VIDEO_PATH", "input.mp4")
        self.output_dir = os.getenv("OUTPUT_DIR", "./results")
        os.makedirs(self.output_dir, exist_ok=True)

        # 性能参数
        self.frame_skip = int(os.getenv("FRAME_SKIP", "3"))
        self.motion_thresh = int(os.getenv("MOTION_THRESHOLD", "1000"))
        self.min_area = int(os.getenv("MIN_OBJECT_AREA", "500"))
        self.api_interval = float(os.getenv("API_CALL_INTERVAL", "1.1"))
        self.last_api_call = 0

        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=int(os.getenv("API_TIMEOUT", "10"))
        )
        
        # 添加GPU加速支持
        if os.getenv("USE_GPU", "false").lower() == "true":
            try:
                # 检测CUDA可用性
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.use_gpu = True
                    logger.info("已启用GPU加速")
                else:
                    self.use_gpu = False
                    logger.warning("未检测到支持CUDA的设备，已禁用GPU加速")
            except:
                self.use_gpu = False
                logger.warning("OpenCV未编译GPU支持，已禁用GPU加速")
        else:
            self.use_gpu = False
        
    def setup_database(self):
        """初始化数据库连接和表结构"""
        self.db_conn = None
        if os.getenv("DB_HOST"):
            try:
                self.db_conn = pymysql.connect(
                    host=os.getenv("DB_HOST"),
                    user=os.getenv("DB_USER"),
                    password=os.getenv("DB_PASSWORD"),
                    database=os.getenv("DB_NAME"),
                    port=int(os.getenv("DB_PORT", "3306")),
                    cursorclass=pymysql.cursors.DictCursor
                )
                logger.info("数据库连接成功")
                
                # 创建表（如果不存在）
                with self.db_conn.cursor() as cursor:
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS objects (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        name VARCHAR(255) NOT NULL,
                        first_seen DATETIME NOT NULL,
                        last_seen DATETIME NOT NULL,
                        count INT DEFAULT 1,
                        UNIQUE KEY (name)
                    )
                    """)
                    
                    # 检查是否需要重置数据库
                    if os.getenv("RESET_DB", "false").lower() == "true":
                        cursor.execute("TRUNCATE TABLE objects")
                        logger.info("数据库表已重置")
                    
                    self.db_conn.commit()
                    logger.info("确保数据库表已创建")
                    
            except Exception as e:
                logger.error(f"数据库连接/初始化失败: {e}")
                self.db_conn = None
        else:
            self.db_conn = None
            logger.warning("未配置数据库参数")

    def setup_tracking(self):
        """初始化追踪参数 - 移除可视化相关设置"""
        self.background = None
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        # 仅在需要保存结果时初始化视频编解码器
        if os.getenv("SAVE_TRACKING_VIDEO", "false").lower() == "true":
            self.fourcc = cv2.VideoWriter_fourcc(*os.getenv("OUTPUT_CODEC", "mp4v"))
            self.output_video = None
        else:
            self.output_video = None

    def analyze_frame(self, frame):
        """简化的帧分析函数，只返回当前帧中的物体"""
        try:
            objects = {}
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 运动检测部分可以移除，因为我们只关注单帧中的物体
            contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) < self.min_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                roi = frame[y:y+h, x:x+w]

                if roi.shape[0] < 20 or roi.shape[1] < 20:
                    continue
                
                # 识别物体
                obj_name = self.identify_object(roi)
                if obj_name:
                    objects[obj_name] = (x + w//2, y + h//2)  # 只记录物体的中心位置

            return objects

        except Exception as e:
            logger.error(f"帧分析异常: {e}")
            return {}

    def identify_object(self, roi):
        """精简版的物体识别与匹配函数，不进行名称标准化"""
        try:
            # 特征提取
            kp, des = self.orb.detectAndCompute(roi, None)
            if des is None or len(des) < 5:
                return None
            
            # 1. 基于特征点的物体匹配
            best_match = None
            best_match_score = 0
            
            for obj_name, (prev_des, _) in self.last_features.items():
                if prev_des is not None and len(prev_des) >= 5:
                    try:
                        matches = self.bf.match(des, prev_des)
                        matches = sorted(matches, key=lambda x: x.distance if x.distance > 0 else float('inf'))
                        
                        if len(matches) > 5:
                            match_scores = [1.0/m.distance if m.distance > 0 else 0 for m in matches[:10]]
                            match_score = sum(match_scores)
                            if match_score > best_match_score and match_score > 0.5:
                                best_match = obj_name
                                best_match_score = match_score
                    except Exception as e:
                        logger.debug(f"特征匹配出错: {e}")
                        continue
            
            # 如果找到匹配的物体
            if best_match:
                self.last_roi[best_match] = roi
                self.last_features[best_match] = (des, kp)
                return best_match
                
            # 2. 缓存查找
            obj_hash = hash(roi.tobytes())
            if obj_hash in self.object_cache:
                obj_name = self.object_cache[obj_hash]
                self.last_roi[obj_name] = roi
                self.last_features[obj_name] = (des, kp)
                return obj_name

            # 当没有找到匹配且启用了调试模式，返回一个通用名
            return "物体"  # 简化版直接返回，避免API调用

        except Exception as e:
            logger.error(f"物体识别/匹配失败: {e}")
        return None

    def api_call_with_retry(self, image_str, prompt, max_tokens=10):
        """精简版API调用"""
        max_retries = 2  # 减少重试次数，加快失败恢复
        retry_delay = 1.0
        
        # 检查是否配置了API密钥
        if not self.api_key:
            logger.warning("未配置API密钥，无法进行API调用")
            return None
        
        for attempt in range(max_retries):
            try:
                # 构建消息内容
                message_content = []
                
                # 只有当图像数据非空时才添加图像
                if image_str:
                    message_content.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_str}"}}
                    )
                
                # 添加文本内容
                message_content.append({"type": "text", "text": prompt})
                
                # 发送API请求
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{
                        "role": "user",
                        "content": message_content
                    }],
                    temperature=0.1,
                    max_tokens=max_tokens
                )
                return response
            except Exception as e:
                logger.warning(f"API调用失败(尝试{attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        return None  # 简化的错误处理

    def store_object_to_db(self, obj_name):
        """存储物体到数据库"""
        if not self.db_conn:
            return
            
        try:
            with self.db_conn.cursor() as cursor:
                sql = """INSERT INTO objects (name, first_seen, last_seen, count) 
                         VALUES (%s, NOW(), NOW(), 1)
                         ON DUPLICATE KEY UPDATE 
                         last_seen = NOW(), count = count + 1"""
                cursor.execute(sql, (obj_name,))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"数据库存储失败: {e}")

    def describe_movement(self, query):
        """新增的移动轨迹描述函数"""
        try:
            # 提取查询中的物体名称
            potential_objects = []
            words = query.lower().split()
            for word in words:
                if len(word) > 1 and word not in ["在哪", "哪里", "位置", "移动", "从哪", "轨迹", "的", "是", "了"]:
                    potential_objects.append(word)
            
            if not potential_objects:
                return "请明确指出您想查询的物体"
            
            # 对每个物体进行轨迹分析
            results = []
            for obj in potential_objects:
                # 检查物体是否在初始和最后位置都被检测到
                initial_pos = self.first_frame_objects.get(obj)
                final_pos = self.last_frame_objects.get(obj)
                
                if not initial_pos and not final_pos:
                    results.append(f"未检测到「{obj}」")
                    continue
                elif not initial_pos:
                    results.append(f"「{obj}」只在视频结束时被检测到")
                    continue
                elif not final_pos:
                    results.append(f"「{obj}」只在视频开始时被检测到")
                    continue
                
                # 简化版描述
                if initial_pos == final_pos:
                    results.append(f"「{obj}」在视频过程中保持静止")
                else:
                    results.append(f"「{obj}」从视频开始的位置移动到了结束的位置")
            
            return "\n".join(results) if results else "无法描述移动情况"
        
        except Exception as e:
            logger.error(f"移动描述失败: {e}")
            return "描述移动轨迹时出错，请重试"

    def general_object_query(self, query):
        """简化的通用物体查询，只关注初始和最后位置"""
        try:
            # 提取查询中的物体名称
            potential_objects = []
            words = query.lower().split()
            for word in words:
                if len(word) > 1 and word not in ["在哪", "哪里", "位置", "移动", "什么", "有没有", "的", "是", "了", "最后"]:
                    potential_objects.append(word)
            
            if not potential_objects:
                return "请明确指出您想查询的物体"

            # 判断是否查询最后位置
            is_last_position = any(word in query for word in ["最后", "现在", "当前"])
            
            # 简化版处理
            results = []
            for obj in potential_objects:
                if is_last_position:
                    if obj in self.last_frame_objects:
                        results.append(f"在视频结束时，「{obj}」出现在画面中")
                    else:
                        results.append(f"在视频结束时，未检测到「{obj}」")
                else:
                    if obj in self.first_frame_objects:
                        results.append(f"在视频开始时，「{obj}」出现在画面中")
                    else:
                        results.append(f"在视频开始时，未检测到「{obj}」")
            
            return "\n".join(results) if results else "未找到相关信息"
            
        except Exception as e:
            logger.error(f"通用物体查询失败: {e}", exc_info=True)
            return "物体查询出错，请重试"

# ===== GUI应用部分 =====
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
            self.tracker = VideoObjectTracker()
            self.tracker.video_path = self.video_path
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
        
        # 直接在主线程处理视频以简化代码
        self.add_progress_message("开始处理视频...")
        
        try:
            # 创建追踪器
            self.tracker = VideoObjectTracker()
            self.tracker.video_path = video_path
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.add_progress_message(f"无法打开视频文件: {video_path}")
                return

            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.add_progress_message(f"视频共有 {total_frames} 帧")
            
            # 获取第一帧
            ret, first_frame = cap.read()
            if not ret:
                self.add_progress_message("无法读取视频第一帧")
                return
            
            # 获取最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
            if not ret:
                self.add_progress_message("无法读取视频最后一帧")
                return
            
            # 分析这两帧中的物体
            self.add_progress_message("正在分析视频首帧...")
            self.tracker.first_frame_objects = self.tracker.analyze_frame(first_frame)
            
            self.add_progress_message("正在分析视频尾帧...")
            self.tracker.last_frame_objects = self.tracker.analyze_frame(last_frame)
            
            # 打印检测到的物体
            all_objects = set(list(self.tracker.first_frame_objects.keys()) + list(self.tracker.last_frame_objects.keys()))
            if all_objects:
                self.add_progress_message(f"检测到的物体: {', '.join(all_objects)}")
            else:
                self.add_progress_message("未检测到任何物体")
            
            # 处理完成
            self.add_progress_message("视频处理完成！现在您可以提问关于视频内容的问题。")
            
        except Exception as e:
            self.add_progress_message(f"处理出错: {str(e)}")
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            
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