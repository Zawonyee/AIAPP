import os
import cv2
import time
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

    def process_video(self):
        """重新设计的视频处理流程"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件: {self.video_path}")
                return

            # 只需要获取视频的第一帧和最后一帧
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 获取第一帧
            ret, first_frame = cap.read()
            if not ret:
                logger.error("无法读取视频第一帧")
                return
            
            # 获取最后一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, last_frame = cap.read()
                if not ret:
                logger.error("无法读取视频最后一帧")
                return
            
            # 分析这两帧中的物体
            self.first_frame_objects = self.analyze_frame(first_frame)
            self.last_frame_objects = self.analyze_frame(last_frame)
            
            # 直接进入交互查询
            self.interactive_query()

        except Exception as e:
            logger.error(f"视频处理异常: {e}", exc_info=True)
        finally:
            cap.release()

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

            # 3. API识别
            current_time = time.time()
            if current_time - self.last_api_call < self.api_interval:
                return None

            _, img_encoded = cv2.imencode('.jpg', roi)
            img_str = base64.b64encode(img_encoded).decode('utf-8')

            prompt = """请识别图中最明显的物体，只回答物体的名称。
            请用一个简短的词语描述物体，不要添加任何解释。"""
            
            try:
                response = self.api_call_with_retry(img_str, prompt, max_tokens=10)
                
                if response and response.choices:
                obj_name = response.choices[0].message.content.strip()
                    # 不再进行名称标准化
                    
                self.object_cache[obj_hash] = obj_name
                self.last_api_call = time.time()
                
                # 存储到数据库
                if self.db_conn:
                    self.store_object_to_db(obj_name)
                
                    self.last_roi[obj_name] = roi
                    self.last_features[obj_name] = (des, kp)
                return obj_name
            except Exception as e:
                logger.error(f"API识别失败: {e}")
                
            return None

        except Exception as e:
            logger.error(f"物体识别/匹配失败: {e}")
        return None

    def api_call_with_retry(self, image_str, prompt, max_tokens=10):
        """精简版API调用"""
        max_retries = 2  # 减少重试次数，加快失败恢复
        retry_delay = 1.0
        
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

    def record_trajectory(self, obj_name, position):
        """仅记录轨迹数据，不进行可视化绘制"""
        try:
            self.trajectories[obj_name].append((self.frame_count, *position))
            
            # 存储关键帧（可选，仅调试用）
            if os.getenv("SAVE_DEBUG_DATA", "false").lower() == "true":
                debug_dir = os.path.join(self.output_dir, "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                if obj_name in self.last_roi:
                    cv2.imwrite(
                        os.path.join(debug_dir, f"{obj_name}_{self.frame_count}.jpg"),
                        self.last_roi[obj_name]
                    )
        except Exception as e:
            logger.error(f"轨迹记录失败: {e}")

    def save_trajectories(self):
        """保存轨迹数据"""
        try:
            traj_path = os.path.join(self.output_dir, "trajectories.json")
            with open(traj_path, 'w') as f:
                json.dump({
                    obj: list(frames) 
                    for obj, frames in self.trajectories.items()
                }, f)
            logger.info(f"轨迹数据已保存到: {traj_path}")
        except Exception as e:
            logger.error(f"保存轨迹数据失败: {e}")

    def interactive_query(self):
        """改进的交互查询系统"""
        init()  # 初始化colorama
        
        def print_color(text, color=Fore.WHITE):
            print(f"{color}{text}{Style.RESET_ALL}")
        
        print_color("\n=== 智能场景助手 ===", Fore.CYAN)
        
        # 获取当前视频中实际检测到的物体
        object_list = list(self.first_frame_objects.keys()) + list(self.last_frame_objects.keys())
        object_list = list(set(object_list))  # 去重
        
        if object_list:
            print_color(f"当前场景检测到的物体：{', '.join(object_list)}", Fore.YELLOW)
        else:
            print_color("当前场景未检测到物体", Fore.YELLOW)
        
        print("\n您可以询问视频中的任何物体")
        print("指令: objects(物体列表) q(退出) 或直接询问任何物体")

        while True:
            try:
                query = input("\n> ").strip()
                
                if query.lower() in ['q', '退出', 'exit', 'quit']:
                    break

                if query.lower() == 'objects':
                    print_color(f"当前场景检测到的物体：", Fore.YELLOW)
                    for obj in object_list:
                        print(f"- {obj}")
                    continue
                
                # 处理轨迹查询
                if any(w in query for w in ["移动", "从哪", "轨迹"]):
                    self.describe_movement(query)
                else:
                    # 处理位置查询
                    self.general_object_query(query)
                    
            except KeyboardInterrupt:
                print("\n再见！")
                break
            except Exception as e:
                logger.error(f"查询处理出错: {e}")
                print("系统处理出错，请重试")

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
                print("请明确指出您想查询的物体")
                return
            
            # 对每个物体进行轨迹分析
            for obj in potential_objects:
                # 检查物体是否在初始和最后位置都被检测到
                initial_pos = self.first_frame_objects.get(obj)
                final_pos = self.last_frame_objects.get(obj)
                
                if not initial_pos or not final_pos:
                    print(f"无法完整追踪「{obj}」的移动轨迹")
                    continue
                
                # 获取视频首尾帧
                cap = cv2.VideoCapture(self.video_path)
                ret_start, start_frame = cap.read()
                
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret_end, end_frame = cap.read()
                cap.release()
                
                if not (ret_start and ret_end):
                    print(f"无法读取视频帧来分析「{obj}」的移动")
                    continue
                
                # 创建拼接图像
                h1, w1 = start_frame.shape[:2]
                combined_image = np.zeros((h1, w1*2 + 10, 3), dtype=np.uint8)
                combined_image[:, :w1] = start_frame
                combined_image[:, w1:w1+10] = 255  # 白色分隔线
                combined_image[:, w1+10:] = end_frame
                
                # 编码图像
                _, img_encoded = cv2.imencode('.jpg', combined_image)
                img_str = base64.b64encode(img_encoded).decode('utf-8')
                
                # 构建提示
                prompt = f"""分析这张拼接图像，左边是起始场景，右边是结束场景。
                请描述{obj}是如何移动的，从左图的位置移动到右图的位置。
                用一句话描述移动过程，例如"从桌子移动到了椅子上"。
                请参考周围的物体或环境特征来描述位置变化。
                不要使用左右等相对方位词。"""
                
                response = self.api_call_with_retry(img_str, prompt, max_tokens=100)
                
                if response and response.choices:
                    result = response.choices[0].message.content.strip()
                    print(f"「{obj}」的移动轨迹: {result}")
                else:
                    print(f"无法描述「{obj}」的移动轨迹")
        
        except Exception as e:
            logger.error(f"移动描述失败: {e}")
            print("描述移动轨迹时出错，请重试")

    def describe_position(self, obj_name, initial=False, return_text=False):
        """优化的位置描述函数 - 增强位置识别的准确性"""
        try:
            if obj_name not in self.trajectories:
                response = f"无法找到{obj_name}的信息"
                print(f"⚠️ {response}")
                return response if return_text else None
            
            # 获取轨迹数据
            traj_data = list(self.trajectories[obj_name])
            
            if not traj_data:
                response = f"{obj_name}的数据为空"
                print(f"⚠️ {response}")
                return response if return_text else None
            
            # 获取关键帧索引和位置
            if initial:
                # 获取前3个位置的平均值以提高初始位置的准确性
                start_points = traj_data[:min(3, len(traj_data))]
                target_frame_idx = start_points[0][0]
                avg_x = sum(p[1] for p in start_points) / len(start_points)
                avg_y = sum(p[2] for p in start_points) / len(start_points)
                pos_x, pos_y = int(avg_x), int(avg_y)
                pos_type = "初始位置"
            else:
                # 获取最后3个位置的平均值以提高最终位置的准确性
                end_points = traj_data[-min(3, len(traj_data)):]
                target_frame_idx = end_points[-1][0]
                avg_x = sum(p[1] for p in end_points) / len(end_points)
                avg_y = sum(p[2] for p in end_points) / len(end_points)
                pos_x, pos_y = int(avg_x), int(avg_y)
                pos_type = "当前位置"
            
            # 获取帧图像并进行预处理
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                response = f"无法读取视频帧"
                print(f"⚠️ {response}")
                return response if return_text else None
            
            # 获取目标区域的上下文信息
            context_size = 100  # 扩大上下文区域
            h, w = frame.shape[:2]
            x1 = max(0, pos_x - context_size)
            y1 = max(0, pos_y - context_size)
            x2 = min(w, pos_x + context_size)
            y2 = min(h, pos_y + context_size)
            
            # 裁剪上下文区域
            context_frame = frame[y1:y2, x1:x2]
            
            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', context_frame)
            img_str = base64.b64encode(img_encoded).decode('utf-8')
            
            # 构建更精确的提示
            prompt = f"""请仔细观察图像中{obj_name}的位置。

要求：
1. 描述{obj_name}相对于周围环境的具体位置
2. 使用"在...上/旁/附近"等明确的位置关系词
3. 参考周围的物体、家具或场景特征
4. 如果看到人，可以用"在某人旁边"等方式描述
5. 避免使用"左边、右边"等相对方位词
6. 不要提及坐标或技术细节

请用一句话描述{obj_name}的具体位置。"""

            # 使用带重试机制的API调用
            try:
                response = self.api_call_with_retry(img_str, prompt, max_tokens=60)
                
                if not response:
                    err_msg = f"无法描述{obj_name}的位置，API调用失败"
                    print(f"⚠️ {err_msg}")
                    return err_msg if return_text else None
                    
                result = response.choices[0].message.content.strip()
                
                # 验证位置描述的质量
                if len(result) < 10 or not any(word in result.lower() for word in ["在", "旁", "边", "附近", "上", "下", "中"]):
                    # 如果描述质量不佳，尝试使用完整帧重新识别
                    _, full_img_encoded = cv2.imencode('.jpg', frame)
                    full_img_str = base64.b64encode(full_img_encoded).decode('utf-8')
                    
                    backup_response = self.api_call_with_retry(full_img_str, prompt, max_tokens=60)
                    if backup_response and backup_response.choices:
                        result = backup_response.choices[0].message.content.strip()
                
                output = f"{obj_name}的{pos_type}：\n{result}"
                print(output)
                
                return output if return_text else None
                
                except Exception as e:
                logger.error(f"位置描述API调用失败: {e}")
                response = f"描述出错，请重试"
                print(f"⚠️ {response}")
                return response if return_text else None
            
        except Exception as e:
            logger.error(f"位置描述失败: {e}", exc_info=True)
            response = f"描述出错，请重试"
            print(f"⚠️ {response}")
            return response if return_text else None

    def describe_trajectory(self, obj_name, return_text=False):
        """精简的轨迹描述函数 - 不依赖背景物体列表"""
        try:
            if obj_name not in self.trajectories or len(self.trajectories[obj_name]) < 2:
                response = f"{obj_name}没有明显移动"
                print(response)
                return response if return_text else None
            
            # 获取起点和终点
            traj_data = list(self.trajectories[obj_name])
            start_frame_idx, start_x, start_y = traj_data[0]
            end_frame_idx, end_x, end_y = traj_data[-1]
            
            # 读取视频帧
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
            ret_start, start_frame = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_idx)
            ret_end, end_frame = cap.read()
            cap.release()
            
            if not (ret_start and ret_end):
                response = f"无法读取视频帧"
                print(response)
                return response if return_text else None
            
            # 创建拼接图像
            h1, w1 = start_frame.shape[:2]
            combined_image = np.zeros((h1, w1*2 + 10, 3), dtype=np.uint8)
            # 左侧放起始帧
            combined_image[:, :w1] = start_frame
            # 中间放分隔线
            combined_image[:, w1:w1+10] = 255
            # 右侧放结束帧
            combined_image[:, w1+10:] = end_frame
            
            # 编码图像
            _, img_encoded = cv2.imencode('.jpg', combined_image)
            img_str = base64.b64encode(img_encoded).decode('utf-8')
            
            # 构建提示
            prompt = f"""分析这张拼接图像，左边是起始场景，右边是结束场景。
            {obj_name}从左图中的位置({start_x}, {start_y})移动到右图中的位置({end_x}, {end_y})。
            用一句话描述{obj_name}的移动情况，比如"从桌子移动到了椅子上"。
            可以参考图中的任何物体作为参照物。
            不要提及坐标、帧号等技术细节。"""

            # API调用
            response = self.api_call_with_retry(img_str, prompt, max_tokens=50)
            
            if not response:
                result = f"{obj_name}从初始位置移动到了新的位置"
                            else:
                result = response.choices[0].message.content
            
            output = f"{obj_name}的移动轨迹：{result}"
            print(output)
            
            return output if return_text else None

            except Exception as e:
            logger.error(f"轨迹描述失败: {e}")
            response = f"无法描述{obj_name}的移动轨迹"
            print(response)
            return response if return_text else None

    def manage_cache(self):
        """管理缓存大小，防止内存溢出"""
        # 计算当前缓存大小
        roi_size = sum(roi.nbytes for roi in self.last_roi.values() if isinstance(roi, np.ndarray))
        
        # 如果缓存过大，清理最旧的ROI
        if roi_size > self.max_cache_size * 0.8:  # 使用80%作为清理阈值
            # 按时间排序，保留最新的
            sorted_keys = sorted(self.last_roi.keys(), 
                                key=lambda k: max([p[0] for p in self.trajectories[k]]) if self.trajectories[k] else 0)
            
            # 删除最早的ROI直到缓存降到合理大小
            while roi_size > self.max_cache_size * 0.5 and sorted_keys:  # 降到50%
                oldest_key = sorted_keys.pop(0)
                if isinstance(self.last_roi.get(oldest_key), np.ndarray):
                    roi_size -= self.last_roi[oldest_key].nbytes
                    del self.last_roi[oldest_key]
                
            logger.info(f"缓存已清理，当前大小: {roi_size/(1024*1024):.2f}MB")

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
                print("请明确指出您想查询的物体")
                return

            # 判断是否查询最后位置
            is_last_position = any(word in query for word in ["最后", "现在", "当前"])
            
            # 获取视频帧
            cap = cv2.VideoCapture(self.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 只获取初始或最后位置的帧
            frame_idx = int(frame_count * 0.98) if is_last_position else int(frame_count * 0.02)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("无法从视频中获取图像")
                return
            
            # 对每个物体进行查询
            for obj in potential_objects:
                # 编码图像
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(img_encoded).decode('utf-8')
                
                # 构建提示词
                prompt = f"""请仔细观察图像，描述{obj}的{'最后' if is_last_position else '初始'}位置。

要求：
1. 如果能看到{obj}，请详细描述它的具体位置和周围环境
2. 使用"在...上/旁/附近"等明确的位置关系词
3. 描述时参考周围的物体、家具或场景特征
4. 如果看到人，可以用"在某人旁边"等方式描述
5. 避免使用"左边、右边"等相对方位词
6. 如果没有看到{obj}，请直接回答"未发现{obj}"

请用一到两句话描述。"""
                
                response = self.api_call_with_retry(img_str, prompt, max_tokens=100)
                
                if response and response.choices:
                    result = response.choices[0].message.content.strip()
                    
                    # 检查是否找到物体
                    if "未发现" not in result.lower() and "没有" not in result.lower():
                        position_type = "最后" if is_last_position else "初始"
                        print(f"关于「{obj}」的{position_type}位置: {result}")
                    else:
                        print(f"在视频{position_type}场景中没有发现「{obj}」")
            
        except Exception as e:
            logger.error(f"通用物体查询失败: {e}", exc_info=True)
            print("物体查询出错，请重试")

    def analyze_all_objects(self):
        """对视频场景进行通用物体分析"""
        try:
            # 获取视频中分散的几个帧
            cap = cv2.VideoCapture(self.video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # 选择3个均匀分布的帧进行分析
            frames = []
            for i in range(3):
                frame_idx = max(0, min(frame_count-1, int(frame_count * (i+1) / 4)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frames.append((frame_idx, frame))
            cap.release()
            
            if not frames:
                print("无法从视频中获取图像")
                return []
            
            print("正在分析视频场景中的所有物体...")
            all_objects = []
            
            for frame_idx, frame in frames:
                # 编码图像
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_str = base64.b64encode(img_encoded).decode('utf-8')
                
                # 构建提示
                prompt = """请详细列出这个图像场景中可见的所有物体。
                以简单的列表形式返回物体名称，每行一个物体。
                把相似或相同的物体归类（如"三本书"而不是"书1, 书2, 书3"）。
                请按照物体在场景中的明显程度排序。
                只列出确实可见的物体，不要猜测。"""
                
                response = self.api_call_with_retry(img_str, prompt, max_tokens=200)
                
                if response and response.choices:
                    result = response.choices[0].message.content
                    
                    # 解析结果为物体列表
                    objects = [line.strip() for line in result.split('\n') if line.strip()]
                    all_objects.extend(objects)
            
            # 去重
            unique_objects = list(set(all_objects))
            # 按照出现频率排序
            sorted_objects = sorted(
                [(obj, all_objects.count(obj)) for obj in unique_objects],
                key=lambda x: x[1],
                reverse=True
            )
            
            return [obj for obj, _ in sorted_objects]
            
        except Exception as e:
            logger.error(f"场景物体分析失败: {e}", exc_info=True)
            print("物体分析出错，请重试")
            return []

if __name__ == "__main__":
    try:
        tracker = VideoObjectTracker()
        tracker.process_video()
    except Exception as e:
        logger.error(f"程序运行异常: {e}", exc_info=True)
    finally:
        cv2.destroyAllWindows()