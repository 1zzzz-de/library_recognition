图书馆人脸识别签到系统，使用InsightFace模型进行人脸检测和识别，通过Gradio构建用户界面。

系统要求：
Python 3.7+/
CUDA兼容的GPU（推荐）/
摄像头设备

安装依赖：
pip install gradio opencv-python numpy pandas insightface-app

配置文件：
cert.pem - SSL证书文件/
key.pem - SSL密钥文件

注意事项：
首次运行会自动创建签到记录文件/
系统默认使用GPU加速，如无GPU可修改为CPU模式/
确保摄像头权限已开启/
人脸识别阈值设置为0.5，可根据需要调整
