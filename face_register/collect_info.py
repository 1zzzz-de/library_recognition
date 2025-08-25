import gradio as gr

import json
import os
from datetime import datetime
import cv2
import time

import uuid

from face_data.facedatabase import FaceDatabase


# 设定根目录
base_dir = 'face_data'
# 确保目录存在
os.makedirs(base_dir, exist_ok=True)

# 设定数据库文件的创建路径
school_face_db_path = f'{base_dir}/db/face_database.json'
school_face_db = FaceDatabase(school_face_db_path)

def face_detect(image, conf):
    from insightface.app import FaceAnalysis

    # 初始化人脸分析
    # app = FaceAnalysis(name='antelopev2')
    app = FaceAnalysis(
    name='antelopev2',
    root='~/.insightface/models'  # 确保路径正确
)
    app.prepare(ctx_id=0, det_size=(640,640))  # 使用 GPU，ctx_id=-1 使用 CPU

    result_face_image_path=""
    face_coding =[]
    if conf is None:
        gr.Warning("请设置“置信度”!")
        return None
    if image is None:
        gr.Warning("请上传图片!")
        return None
    print(image)
    img1 = cv2.imread(image) 
    if img1 is None:
        print("Failed to load image. Please check the file path.")
    else:
        print(f"Image loaded successfully with shape: {img1.shape}")
     
    #  # 设置新的尺寸
    # new_width = 800
    # new_height = 600

    # # 调整图片大小
    # resized_image = cv2.resize(img1, (new_width, new_height))
    t1=time.time()
    results = app.get(img1)
    t2=time.time()
    print("shi",t2-t1)
    # print(results)

    if len(results) > 0:
        # 获取图像的高度和宽度
        height, width = img1.shape[:2]
        # print("height,with",height,width )
        # print(results[0].bbox)
        
        # 从 bbox 中提取坐标
        x1, y1, x2, y2 = map(int, results[0].bbox) #左,上,右,下

        # 确保坐标值在图像范围内
        x1, y1 = max(0, x1-50), max(0, y1-50)
        x2, y2 = min(width, x2+50), min(height, y2+50)

        face_image = img1[y1:y2, x1:x2]

        result_face_image_path = f'{base_dir}/face_detected/r-{uuid.uuid4()}.jpg'
        print(result_face_image_path)
        cv2.imwrite(result_face_image_path, face_image)
        
        face_coding = results[0].embedding
        if face_coding is not None:
            face_coding = face_coding.tolist()
        else:
            face_coding = '没有生成人脸编码'
    else:
        print("results is null")
                
    return result_face_image_path, face_coding, face_coding

def submit_student_info(school, role, department, enrollment_year, class_name, name, sid, gender, face_detect_output,face_encoding_output):
    # 检查输入是否为None或空
    if not all([school, role, department, enrollment_year, class_name, name, sid, gender, role]):
        return "请确保所有字段都已填写，并上传有效的照片。"
    
    if face_detect_output is None:
        return "photo is None"
    # 创建班级目录
    class_dir = os.path.join(base_dir, "students", department, enrollment_year, class_name)
    ret = os.makedirs(class_dir, exist_ok=True)
    print("create class dir",class_dir,ret)

    # 保存照片
    photo_dir = os.path.join(class_dir, "images")
    os.makedirs(photo_dir, exist_ok=True)
    
    # 使用cv2读取和保存照片
    img = cv2.imread(face_detect_output)
    if img is None:
        return "上传的照片无效，请确保文件路径正确。"
    
    photo_path = os.path.join(photo_dir, f"{sid}_{name}.jpg")
    cv2.imwrite(photo_path, img)
    
    # 准备保存的信息
    info = {
        "school": school,
        "department": department,
        "enrollment_year": enrollment_year,
        "class": class_name,
        "name": name,
        "sid": sid,
        "gender": gender,
        "role": role,
        "photo": photo_path,
        "face_coding": face_encoding_output
    }
    
    # 保存为JSON文件
    json_file_path = os.path.join(class_dir, f"{sid}_{name}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(info, json_file, ensure_ascii=False, indent=4)
    
    #创建班级人脸数据库
    #检查班级数据库文件是否存在，存在则添加，不存在先创建再添加
    # 保存照片
    class_face_db_filepath = os.path.join(class_dir, "class_face_db.json")
    # # 判断文件路径是否存在
    # if not os.path.exists(class_face_db_filepath):
        # 如果不存在，使用 open() 函数创建一个班级人脸库
        
    class_face_db = FaceDatabase(class_face_db_filepath)
    class_face_db.add_entry(face_encoding_output, json_file_path)
    
    #创建学校人脸数据库   
    school_face_db.add_entry(face_encoding_output, json_file_path) 
    return f"信息已提交:\n学校: {school}\n学部: {department}\n入学时间: {enrollment_year}\n班级: {class_name}\n姓名: {name}\n学号: {sid}\n性别: {gender}\n角色: {role}\n照片路径: {photo_path}\n人脸编码:{face_encoding_output}"

def submit_teacher_info(school, role, department, name, sid, gender, subject, face_detect_output,face_encoding_output):
 # 检查输入是否为None或空
    if not all([school, role, department, name, sid, gender,subject]):
        return "请确保所有字段都已填写，并上传有效的照片。"
    
    if face_detect_output is None:
        return "photo is None"
    # 创建班级目录
    teacher_dir = os.path.join(base_dir, "teachers")
    ret = os.makedirs(teacher_dir, exist_ok=True)
    print("create teacher dir",teacher_dir,ret)

    # 保存照片
    photo_dir = os.path.join(teacher_dir, "images")
    os.makedirs(photo_dir, exist_ok=True)
    
    # 使用cv2读取和保存照片
    img = cv2.imread(face_detect_output)
    if img is None:
        return "上传的照片无效，请确保文件路径正确。"
    
    photo_path = os.path.join(photo_dir, f"{sid}_{name}.jpg")
    cv2.imwrite(photo_path, img)
    
    # 准备保存的信息
    info = {
        "school": school,
        "department": department,
        "name": name,
        "sid": sid,
        "gender": gender,
        "role": role,
        "photo": photo_path,
        "face_coding": face_encoding_output,
        "subject": subject
    }
    
    # 保存为JSON文件
    json_file_path = os.path.join(teacher_dir, f"{sid}_{name}.json")
    with open(json_file_path, "w") as json_file:
        json.dump(info, json_file, ensure_ascii=False, indent=4)
    
    school_face_db.add_entry(face_encoding_output, json_file_path)
    return f"信息已提交:\n学校: {school}\n学部: {department}\n姓名: {name}\n学号: {sid}\n性别: {gender}\n角色: {role}\n照片路径: {photo_path}\n人脸编码:{face_encoding_output}"

def run():
    # 创建Gradio界面
    with gr.Blocks() as demo:
        gr.Markdown("### 人脸信息采集系统")
        gr.Markdown("## 输入照片信息")
        register_face_codings = gr.State()
        with gr.Row():
            with gr.Group():
                photo_input = gr.Image(label="上传照片", type="filepath", height=400, width=400)
                slider_input = gr.Slider(minimum=0, maximum=100, step=1, label="置信度", value=50)
    
            with gr.Group():
                face_detect_output = gr.Image(label="人脸检测", type="filepath", height=350, width=400)
                face_encoding_output = gr.Textbox(label="人脸编码", max_lines=2)
                
        with gr.Group():
            school = gr.Textbox(label="学校")
            role = gr.Radio(label="角色", choices=["学生", "老师"], value="学生")
            
        @gr.render(inputs=role)
        def show_personInfo(role_value):
            if role_value == "学生":
                with gr.Group():
                    department = gr.Radio(label="学部", choices=["高中", "初中", "小学"], value="高中") 
                    with gr.Row():   
                        enrollment_year = gr.Dropdown(label="入学时间", choices=[str(year) for year in range(2020, datetime.now().year + 1)])
                        class_name = gr.Textbox(label="班级")
                    with gr.Row():
                        name = gr.Textbox(label="姓名")
                        sid = gr.Textbox(label=f"{role_value} ID")
                        gender = gr.Radio(label="性别", choices=["男", "女"])
                submit_button.click(submit_student_info, inputs=[school, role, department, enrollment_year, class_name, name, sid, gender, face_detect_output,register_face_codings], outputs=output) 
            else:
                with gr.Group():
                    department = gr.CheckboxGroup(["高中", "初中", "小学"], label="学部", info="老师的任教年级?")
                    with gr.Row():
                        name = gr.Textbox(label="姓名")
                        sid = gr.Textbox(label=f"{role_value} ID")
                        gender = gr.Radio(label="性别", choices=["男", "女"]) 
                    subject = gr.Textbox(label="任教课程")
            
                submit_button.click(submit_teacher_info, inputs=[school, role, department, name, sid, gender, subject, face_detect_output,register_face_codings], outputs=output) 

        photo_input.input(face_detect, inputs=[photo_input,slider_input], outputs=[face_detect_output, face_encoding_output,register_face_codings])
        slider_input.change(face_detect, inputs=[photo_input,slider_input], outputs=[face_detect_output, face_encoding_output,register_face_codings])
        
        submit_button = gr.Button("提交")
        output = gr.Textbox(label="输出信息", interactive=False)
    return demo


if __name__ == "__main__":
    demo = run()
    demo.launch(server_name="0.0.0.0")
