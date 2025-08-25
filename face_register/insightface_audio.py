import gradio as gr
import cv2
import numpy as np
from datetime import datetime
import pandas as pd
from pathlib import Path
import os
import time
import json
from insightface.app import FaceAnalysis



# 全局变量
known_face_embeddings = []
known_face_names = []
face_display_info = {}  # 存储人脸显示信息的字典
status_display_info = {"status": "", "time": "", "display_until": 0}
# DISPLAY_DURATION = 15  # 显示持续时间(秒)
DETECTION_INTERVAL = 5  # 检测间隔(秒)
last_detection_time = 0  # 上次检测时间
today_visitors = set()  # 使用集合来避免重复计数
sign_in_records = []  # 存储所有签到记录
known_face_image_paths = []  # 存储人脸库图片路径
success_display_info = {"image": None, "display_until": 0}  # 存储识别结果的显示信息
success_FILE_PATH = "/root/face/face_register/success.mp3"
false_FILE_PATH = "/root/face/face_register/false.mp3"


face_app = FaceAnalysis(
    name='antelopev2',
    root='~/.insightface/models' , # 确保路径正确
    providers=['CUDAExecutionProvider']  # 使用GPU加速
)
face_app.prepare(ctx_id=0, det_size=(320, 320))


# 从face_database.json加载人脸特征编码和姓名
def load_face_database(json_path):
    global known_face_embeddings, known_face_names, known_face_image_paths
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 假设data是一个字典列表
        if isinstance(data, list):
            for entry in data:
                if 'facecode' in entry and 'cfgfilepath' in entry:
                    # 从文件路径中提取姓名 (格式为.../学号_姓名.json)
                    try:
                        filename = os.path.basename(entry['cfgfilepath'])
                        name = filename.split('_')[1].replace('.json', '')
                        known_face_names.append(name)
                        known_face_embeddings.append(np.array(entry['facecode']))

                        # 构建对应的图片路径
                        json_path = entry['cfgfilepath']
                        if 'students' in json_path:
                            # 对应图片路径: .../students/[学部]/[入学年份]/[班级]/images/{sid}_{name}.jpg
                            dir_path = os.path.dirname(json_path)
                            img_path = os.path.join(dir_path, 'images', filename.replace('.json', '.jpg'))
                        elif 'teachers' in json_path:
                            # 对应图片路径: .../teachers/images/{tid}_{name}.jpg
                            dir_path = os.path.dirname(json_path)
                            img_path = os.path.join(dir_path, 'images', filename.replace('.json', '.jpg'))
                        
                        known_face_image_paths.append(img_path)

                    except Exception as e:
                        print(f"处理条目时出错: {e}")
        
        print(f"成功加载 {len(known_face_names)} 人的人脸数据")
    except Exception as e:
        print(f"加载人脸数据库时出错: {e}")

# 加载人脸数据库
face_database_path = "face_data/db/face_database.json"
load_face_database(face_database_path)


def get_face_image(name):
    """根据姓名获取人脸库中的照片"""
    if name in known_face_names:
        idx = known_face_names.index(name)
        img_path = known_face_image_paths[idx]
        try:
            # 读取图片并调整为统一大小
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (300, 300))  # 调整为统一大小
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
                return img
        except Exception as e:
            print(f"读取人脸图片时出错: {e}")
    return None


def save_to_json(name, check_in_time):
    """保存签到信息到JSON文件"""
    global sign_in_records
    record_file = "图书馆签到表.json"
    
    # 初始化额外信息
    department = ""
    enrollment_year = ""
    class_name = ""
    sid = ""  # 学号/工号
    identity = "老师"  # 默认为老师
    
    # 在已知人脸列表中查找匹配的人脸
    try:
        if name in known_face_names:
            idx = known_face_names.index(name)
            # 获取对应的文件路径
            with open(face_database_path, 'r') as f:
                data = json.load(f)
                if isinstance(data, list) and idx < len(data):
                    filepath = data[idx].get('cfgfilepath', '')
                    
                    # 从文件名提取学号/工号 (格式: {sid}_{name}.json)
                    filename = os.path.basename(filepath)
                    sid = filename.split('_')[0]
                    
                    # 解析路径信息
                    parts = filepath.split('/')
                    if 'students' in parts:
                        identity = "学生"
                        # 格式: face_data/students/{department}/{enrollment_year}/{class_name}/{sid}_{name}.json
                        try:
                            students_idx = parts.index('students')
                            department = parts[students_idx + 1]
                            enrollment_year = parts[students_idx + 2]
                            class_name = parts[students_idx + 3]
                        except IndexError:
                            pass
                    elif 'teachers' in parts:
                        identity = "老师"
    except Exception as e:
        print(f"解析路径信息时出错: {e}")
    
    # 创建新记录
    new_record = {
        "姓名": name,
        "身份": identity,
        "学号/工号": sid,
        "院系": department if identity == "学生" else "",
        "入学年份": enrollment_year if identity == "学生" else "",
        "班级": class_name if identity == "学生" else "",
        "签到时间": check_in_time,
        "签到日期": datetime.now().strftime("%Y-%m-%d")
    }
    
    # 读取现有记录或创建新文件
    records = []
    try:
        if Path(record_file).exists():
            with open(record_file, 'r', encoding='utf-8') as f:
                records = json.load(f)
                if not isinstance(records, list):
                    records = []  # 如果文件内容不是列表，则重置
    except Exception as e:
        print(f"读取现有JSON文件出错: {e}")
        records = []
    
    # 添加新记录
    records.append(new_record)
    
    # 保存到JSON文件
    try:
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")
    sign_in_records.append(new_record)
    return new_record


def recognize_faces(image):
    global face_display_info, status_display_info, last_detection_time, today_visitors, success_display_info

    # 初始化返回的音频为None
    audio_output = None

    if image is None:
        return None, "等待输入...", "", "0", None, None, None

    image = image.copy()
    current_time = time.time()

    # 初始化返回的人脸库图片
    face_lib_image = None
    name = "未注册"

    # 清理过期的人脸信息
    face_display_info = {
        name: info for name, info in face_display_info.items() 
        if current_time <= info["display_until"]
    }
    
    # 重置状态信息如果过期
    if current_time > status_display_info["display_until"]:
        status_display_info = {"status": "请将人脸对准摄像头...", "time": "", "display_until": 0}
        # 当状态信息过期时，也清除识别结果图像
        success_display_info = {"image": None, "display_until": 0}

    # 检查是否达到检测间隔
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = current_time  # 更新最后检测时间

    # 清理过期的识别结果
    current_time = time.time()
    if current_time > success_display_info["display_until"]:
        face_lib_image = None  # 过期后清空识别结果
    else:
        face_lib_image = success_display_info["image"]
       
    # 人脸检测
    faces = face_app.get(image)

    has_new_recognition = False  # 标记是否有新的识别结果

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]                 
        face_embedding = face.embedding
        
        # 与已知人脸比对
        if known_face_embeddings:
            # 计算相似度
            similarities = [
                np.dot(face_embedding, known_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(known_emb))
                for known_emb in known_face_embeddings
            ]

            best_match_idx = np.argmax(similarities)
            best_similarity = similarities[best_match_idx]
            
            if best_similarity > 0.5:
                name = known_face_names[best_match_idx]
                today_visitors.add(name)

                # 如果是新检测到的人脸
                if name not in face_display_info:
                    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    status_html = f"""
                <div style='color: green; font-size: 20px; font-weight: bold;'>
                    {name} 签到成功!
                </div>
                """
                    status_display_info = {
                        "status": status_html,
                        "time": current_time_str,
                        "display_until": current_time 
                    }
                    
                    record = save_to_json(name, current_time_str)

                   # 获取并保存识别结果图像
                    face_lib_image = get_face_image(name)
                    success_display_info = {
                        "image": face_lib_image,
                        "display_until": current_time  
                    }  
                    # 设置音频输出
                    audio_output = success_FILE_PATH         

                # 更新人脸显示信息
                face_display_info[name] = {
                    "display_until": current_time ,
                    "location": (y1, x2, y2, x1),  # top, right, bottom, left
                    "embedding": face_embedding
                }
            else:
                name = "未注册"
                # 使用HTML格式的文本，设置红色和较大字体
                status_html = """
                <div style='color: red; font-size: 20px; font-weight: bold;'>
                    未注册
                </div>
                """
                status_display_info = {
                    "status": status_html,
                    "time": "",
                    "display_until": current_time 
                }
                # 设置音频输出
                audio_output = false_FILE_PATH
 
    visitor_count = len(today_visitors)
    return (
        status_display_info["status"], 
        status_display_info["time"], 
        str(visitor_count),
        success_display_info["image"],
        audio_output
    )

def run():
    css = """
    .my-group {max-width: 600px !important; max-height: 600px !important;}
    .my-column {display: flex !important; justify-content: center !important; align-items: center !important;}
    """

    with gr.Blocks(css=css) as demo:
        # gr.HTML("<h1 style='text-align: center'>图书馆人脸识别系统</h1>")
        gr.HTML("""
        <div style="background: linear-gradient(to right, #4facfe, #00f2fe); 
                    padding: 20px; 
                    border-radius: 10px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 20px;">
            <h1 style="color: white; text-align: center;">图书馆人脸识别系统</h1>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("到馆显示"):
                with gr.Row():
                    with gr.Column(scale=3):
                        realtime_input = gr.Image(label="摄像头画面", sources="webcam", streaming=True, type="numpy")
                        
                    with gr.Column(scale=1):
                        with gr.Group():
                            realtime_status = gr.HTML(label="签到状态", value="<div>请将人脸对准摄像头...</div>")
                            realtime_time = gr.Textbox(label="签到时间")
                            visitor_count = gr.Textbox(label="今日到馆人数", value="0")
                            audio_player = gr.Audio(visible=True, autoplay=True)

                stream_processor = realtime_input.stream(
                    fn=recognize_faces,
                    inputs=realtime_input,
                    outputs=[
                        realtime_status, 
                        realtime_time, 
                        visitor_count, 
                        audio_player
                        ],
                    show_progress=False
                )
        with gr.Tabs() as tabs:    
            with gr.Tab("到馆详情", id="records_tab"):

                # 刷新按钮
                with gr.Row():
                    refresh_btn = gr.Button("点击刷新查看签到记录")

                # 签到详情表格
                with gr.Row():
                    record_table = gr.DataFrame(
                        headers=["姓名", "身份", "学号/工号", "院系", "入学年份", "班级", "签到时间", "签到日期"],
                        value=[],
                        interactive=False
                    )
                
                # 刷新表格数据
                def update_table():
                    return pd.DataFrame(sign_in_records) if sign_in_records else pd.DataFrame([])
                
                refresh_btn.click(
                    fn=update_table,
                    outputs=record_table
                )
                
                # 初始加载数据
                demo.load(
                    fn=update_table,
                    outputs=record_table
                )
         # 监听标签页切换事件
        def on_tab_change(evt: gr.SelectData):
            if evt.value == "realtime_tab":
                # 重新启动流
                nonlocal stream_processor
                if stream_processor is not None:
                    stream_processor.cancel()
                stream_processor = realtime_input.stream(
                    fn=recognize_faces,
                    inputs=realtime_input,
                    outputs=[
                            realtime_status, 
                             realtime_time, 
                             visitor_count, 
                             audio_player
                             ],
                    show_progress=False
                )

        tabs.select(fn=on_tab_change)

        demo.launch(
            server_name="0.0.0.0", 
            server_port=7860, 
            ssl_certfile="../cert.pem", 
            ssl_keyfile="../key.pem",
            ssl_verify=False,
            # prevent_thread_lock=True
        )

if __name__ == "__main__":
    demo = run()
