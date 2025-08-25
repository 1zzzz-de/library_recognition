import json
import os

class FaceDatabase:
    def __init__(self, db_path):
        """
        初始化数据库文件路径，如果文件不存在则创建一个空的 JSON 文件。
        """
        self.db_path = db_path
        if not os.path.exists(db_path):
            with open(db_path, 'w') as f:
                json.dump([], f)
        else:
            pass
            print(f"Database file '{db_path}' already exists.")

    def add_entry(self, facecode, cfgfilepath):
        """
        向数据库文件添加一条记录，包括 facecode 和 cfgfilepath。
        """
        # 读取现有数据
        with open(self.db_path, 'r') as f:
            data = json.load(f)
        
        # 创建新条目
        new_entry = {
            "facecode": facecode,
            "cfgfilepath": cfgfilepath
        }
        
        # 添加新条目到数据中
        data.append(new_entry)
        
        # 写回文件
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=4)
        
        print("Entry added:", new_entry)
