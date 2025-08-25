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
DETECTION_INTERVAL = 5  # 检测间隔(秒)
last_detection_time = 0  # 上次检测时间
today_visitors = set()  # 使用集合来避免重复计数
sign_in_records = []  # 存储所有签到记录
known_face_image_paths = []  # 存储人脸库图片路径
success_FILE_PATH = "/root/face/face_register/success.mp3"
false_FILE_PATH = "/root/face/face_register/false.mp3"

face_app = FaceAnalysis(
    name='antelopev2',
    root='~/.insightface/models',
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
    global face_display_info, status_display_info, last_detection_time, today_visitors

    # 初始化返回的音频为None
    audio_output = None

    if image is None:
        return None, "", "", "0", None
    
    start_time = time.time()  # 记录开始时间
    image = image.copy()
    current_time = time.time()

    name = "未注册"
    detailed_info = ""  # 初始化详细信息

    # 清理过期的人脸信息
    face_display_info = {
        name: info for name, info in face_display_info.items() 
        if current_time <= info["display_until"]
    }
    
    # 重置状态信息如果过期
    if current_time > status_display_info["display_until"]:
        status_display_info = {"status": '<span style="color: #0000CC; font-size: 30px; font-weight: bold;">请微笑面对摄像头...</span>', "time": "", "display_until": 0}

    # 检查是否达到检测间隔
    if current_time - last_detection_time >= DETECTION_INTERVAL:
        last_detection_time = current_time  # 更新最后检测时间
       
    # 人脸检测
    faces = face_app.get(image)

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
                    
                    # 获取详细信息
                    record = save_to_json(name, current_time_str)
                    
                    # 构建详细信息HTML
                    detailed_info = f"""
                    <div class="student-info" style="background-color: white; 
                    border-radius: 10px; 
                    text-indent: 1em;
                    font-size: 25px; 
                    margin-top: 10px;">
                        <div class="info-item" style="margin-bottom: 25px;">
                            <span class="info-label"><b>姓名:</b></span>
                            <span class="info-value" id="name">{record['姓名']}</span>
                        </div>
                        <div class="info-item" style="margin-bottom: 25px;">
                            <span class="info-label"><b>学号:</b></span>
                            <span class="info-value" id="id">{record['学号/工号']}</span>
                        </div>
                        <div class="info-item" style="margin-bottom: 25px;">
                            <span class="info-label"><b>班级:</b></span>
                            <span class="info-value" id="class">{record['班级'] if record['班级'] else ""}</span>
                        </div>
                        <div class="info-item" style="margin-bottom: 25px;">
                            <span class="info-label"><b>时间:</b></span>
                            <span class="info-value" id="time">{current_time_str}</span>
                        </div>
                    </div>
                    """


                    status_html = f"""
                    <div style='
                    border-radius: 10px; 
                    text-indent: 1em; 
                    color: blue; 
                    font-size: 30px; 
                    font-weight: bold;'>
                        {name} 签到成功!
                    </div>
                    """
                    status_display_info = {
                        "status": status_html,
                        "time": detailed_info,  # 这里存储详细信息
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
                <div style='color: red; text-indent: 1em; font-size: 30px; font-weight: bold;'>
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
        status_display_info["time"],  # 这里返回详细信息
        str(visitor_count),
        audio_output
    )

def run():
    # 简单的 passthrough 函数
    # def passthrough(frame):
    #     return frame

    # # 页面加载时自动开启摄像头并录制
    # auto_webcam_js = """
    # () => {
    #     function clickCamBtn() {
    #         const camBtn = document.querySelector('button.svelte-qbrfs'); // 摄像头按钮
    #         if (camBtn) {
    #             camBtn.click();
    #             console.log("自动开启摄像头");
    #             setTimeout(clickRecordBtn, 500); // 延迟 500ms 再点录制按钮
    #             clearInterval(camTimer);
    #         }
    #     }

    #     function clickRecordBtn() {
    #         const recordBtn = document.querySelector('button.svelte-10cpz3p'); // 录制按钮
    #         if (recordBtn) {
    #             recordBtn.click();
    #             console.log("自动开始录制");
    #         } else {
    #             // 如果按钮还没加载出来，继续等
    #             setTimeout(clickRecordBtn, 300);
    #         }
    #     }

    #     const camTimer = setInterval(clickCamBtn, 300); // 每 300ms 检查一次摄像头按钮
    # }
    # """

    # js=auto_webcam_js, 

    with gr.Blocks(theme=gr.themes.Soft(), css="""
    .camera-container {
        position: relative;
        width: 100%;
        height: 400px;
        margin-bottom: 20px;
    }
   
    .scan-animation {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 10px;
        background: linear-gradient(90deg, rgba(76, 110, 245, 0) 0%, rgba(76, 110, 245, 0.8) 50%, rgba(76, 110, 245, 0) 100%);
        animation: scan 2s linear infinite;
        z-index: 2;
        pointer-events: none;
    }
    @keyframes scan {
        0% { top: 0; }
        50% { top: calc(100% - 8px); }
        100% { top: 0; }
    }
                   
    .camera-image {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        border-radius: 20px;       
    }
                   
    .overlay-container {
        position: relative;
        width: 100%;
    }
                   
    # /* 隐藏停止按钮 */
    # button[aria-label="capture photo"] {
    #     display: none !important;
    # }
    """) as demo:

        gr.HTML("""
        <div style="background: linear-gradient(to right, #0000CC, #0000CC); 
                    padding: 20px; 
                    border-radius: 20px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;">
            <img src="" style="height: 80px; margin-right: 20px;">
            <h1 style="color: white; text-align: center; font-size:50px;">笑脸签到</h1>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab(""):
                with gr.Row():
                    with gr.Column():
                        # 摄像头区域 - 使用自定义容器
                        with gr.Group():
                            gr.HTML("""
                            <div class="overlay-container">
                                <div class="camera-container">
                                    <div class="camera-frame">            
                                        <div class="scan-animation"></div>
                                    </div>
                                </div>
                            </div>
                            """)
                            realtime_input = gr.Image(
                                label="摄像头画面", 
                                sources="webcam", 
                                # streaming=True, 
                                every=5.0,
                                type="numpy",
                                elem_classes=["camera-image"]
                            )
                            
                    with gr.Column():
                        with gr.Group():                           
                            realtime_status = gr.HTML(
                                label="签到状态", 
                                value="<div style='color: #0000CC; text-indent: 1em; font-size: 30px; font-weight: bold;'>请微笑面对摄像头...</div>"
                            )
                            realtime_details = gr.HTML(label="签到详情", value="")
                            
                            with gr.Row():
                                visitor_count = gr.Textbox(
                                    label="今日签到人数", 
                                    value="0", 
                                    scale=2
                                )
                                audio_player = gr.Audio(
                                    label="语音提示",
                                    visible=True, 
                                    autoplay=True,
                                    show_download_button=False,
                                    show_share_button=False,
                                    waveform_options={"show_recording_waveform": False}, 
                                    scale=1
                                )
                            
                stream_processor = realtime_input.stream(
                    fn=recognize_faces,
                    inputs=realtime_input,
                    outputs=[
                        realtime_status, 
                        realtime_details,
                        visitor_count, 
                        audio_player
                    ],
                    show_progress=False
                )
        
        with gr.Tabs() as tabs:    
            with gr.Tab("签到详情", id="records_tab"):
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
                        realtime_details,
                        visitor_count, 
                        audio_player
                    ],
                    show_progress=False
                )

        tabs.select(fn=on_tab_change)

        demo.launch(
            server_name="0.0.0.0", 
            server_port=7861, 
            ssl_certfile="../cert.pem", 
            ssl_keyfile="../key.pem",
            ssl_verify=False
        )

if __name__ == "__main__":
    run()