from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import psycopg2
from dotenv import load_dotenv

# Load biến môi trường từ .env
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

# Kết nối PostgreSQL với xử lý lỗi
def connect_db():
    try:
        conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT)
        return conn
    except Exception as e:
        print(f"Lỗi kết nối DB: {e}")
        return None

# ====== TẠO BẢNG NEWS ======
def create_table():
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS news (
                    id SERIAL PRIMARY KEY,
                    content TEXT NOT NULL,
                    predicted_label TEXT NOT NULL
                );
            """)
            conn.commit()
            cursor.close()
            conn.close()
            print("Bảng 'news' đã được tạo hoặc đã tồn tại.")
        except Exception as e:
            print(f"Lỗi khi tạo bảng: {e}")

# ====== LOAD MÔ HÌNH ======
PHOBERT_MODEL_NAME = "vinai/phobert-base"
NUM_CLASSES = 4

print("Đang tải mô hình PhoBERT...")
tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL_NAME)
phobert_model = AutoModel.from_pretrained(PHOBERT_MODEL_NAME)
print("PhoBERT đã tải xong!")

class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        logits = self.fc4(x)
        probs = torch.softmax(logits, dim=1)
        return logits, probs

print(" Đang tải mô hình phân loại...")
input_size = 768
model = ConvNet(input_size, NUM_CLASSES)
try:
    model.load_state_dict(torch.load("tindochai_model.pth", map_location=torch.device('cpu')))
    model.eval()
    print("Mô hình phân loại đã tải xong!")
except Exception as e:
    print(f"Lỗi tải mô hình: {e}")

label_mapping = {
    0: "Tin thường",
    1: "Tin vu khống, bôi nhọ (Tin độc hại)",
    2: "Tin kêu gọi chống đối (Tin độc hại)",
    3: "Tin thúc đẩy tư tưởng cực đoan (Tin độc hại)"
}

def predict_label(text):
    if not text.strip():
        return "Nội dung trống"

    encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=256)
    with torch.no_grad():
        model_output = phobert_model(**encoded_input)
    
    text_feature = model_output.last_hidden_state[:, 0, :].numpy()
    feature_tensor = torch.tensor(text_feature, dtype=torch.float32)
    
    logits, probs = model(feature_tensor)
    predicted_class = torch.argmax(probs, dim=1).item()
    
    return label_mapping.get(predicted_class, "Không xác định")

app = Flask(__name__)
CORS(app)  # 🚀 Fix lỗi CORS

# @app.route('/predict', methods=['POST'])
# def classify_text():
#     data = request.json
#     text = data.get("text", "")

#     if not text:
#         return jsonify({"error": "Hãy nhập tin tức! "}), 400

#     predicted_label = predict_label(text)

#     conn = connect_db()
#     if conn:
#         try:
#             cursor = conn.cursor()
#             cursor.execute("INSERT INTO news (content, predicted_label) VALUES (%s, %s) RETURNING id;", 
#                            (text, predicted_label))
#             message_id = cursor.fetchone()[0]
#             conn.commit()
#             cursor.close()
#             conn.close()
#             return jsonify({"id": message_id, "label": predicted_label})
#         except Exception as e:
#             return jsonify({"error": f"Lỗi database: {e}"}), 500
#     else:
#         return jsonify({"error": "Không thể kết nối database"}), 500

@app.route('/predict', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Hãy nhập tin tức! "}), 400

    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()

            # Kiểm tra xem tin đã tồn tại trong bảng chưa
            cursor.execute("SELECT id, predicted_label FROM news WHERE content = %s;", (text,))
            existing = cursor.fetchone()

            if existing:
                # Nếu đã tồn tại, trả về kết quả dự đoán cũ
                news_id, label = existing
                cursor.close()
                conn.close()
                return jsonify({"id": news_id, "label": label})
            else:
                # Nếu chưa có, dự đoán và lưu vào DB
                predicted_label = predict_label(text)
                cursor.execute("INSERT INTO news (content, predicted_label) VALUES (%s, %s) RETURNING id;",
                               (text, predicted_label))
                message_id = cursor.fetchone()[0]
                conn.commit()
                cursor.close()
                conn.close()
                return jsonify({"id": message_id, "label": predicted_label})
        except Exception as e:
            return jsonify({"error": f"Lỗi database: {e}"}), 500
    else:
        return jsonify({"error": "Không thể kết nối database"}), 500


@app.route('/get-news', methods=['GET'])
def get_news():
    conn = connect_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, predicted_label FROM news ORDER BY id ASC;")
            news_list = cursor.fetchall()
            cursor.close()
            conn.close()

            news_data = [
                {"id": row[0], "content": row[1], "label": row[2]} for row in news_list
            ]
            return jsonify({"news": news_data})
        except Exception as e:
            return jsonify({"error": f"Lỗi database: {e}"}), 500
    else:
        return jsonify({"error": "Không thể kết nối database"}), 500

if __name__ == '__main__':
    create_table()  # Tự động tạo bảng khi khởi chạy
    app.run(debug=True)
