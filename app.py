import os
import csv
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

app = Flask(__name__)
CORS(app)

# === Path Dasar
base_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(base_dir, 'log_interaksi.txt')
dataset_file = os.path.join(base_dir, 'chat_dataset.xlsx')
user_input_file = os.path.join(base_dir, 'interaksi_user.xlsx')

vectorizer = TfidfVectorizer()
nb_model = MultinomialNB()

# === Fungsi Tambahan
def bersihkan_teks(teks):
    return teks.lower().strip()

def simpan_prompt_user(prompt):
    new_data = pd.DataFrame([[prompt]], columns=['pesan'])
    if os.path.exists(user_input_file):
        existing = pd.read_excel(user_input_file)
        updated = pd.concat([existing, new_data], ignore_index=True)
    else:
        updated = new_data
    updated.to_excel(user_input_file, index=False)

def simpan_dataset_baru(user_msg, ai_reply):
    new_data = pd.DataFrame([[user_msg, ai_reply]], columns=['pesan', 'balasan'])
    if os.path.exists(dataset_file):
        existing = pd.read_excel(dataset_file)
        if not ((existing['pesan'] == user_msg) & (existing['balasan'] == ai_reply)).any():
            updated = pd.concat([existing, new_data], ignore_index=True)
            updated.to_excel(dataset_file, index=False)
    else:
        new_data.to_excel(dataset_file, index=False)

def catat_interaksi(aksi, isi):
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {aksi}: {isi}\n")

def latih_model():
    try:
        if not os.path.exists(dataset_file):
            return
        data = pd.read_excel(dataset_file)
        if 'pesan' not in data or 'balasan' not in data:
            return
        data = data.dropna(subset=['pesan', 'balasan'])
        data['pesan'] = data['pesan'].apply(bersihkan_teks)
        X = vectorizer.fit_transform(data['pesan'])
        y = data['balasan']
        nb_model.fit(X, y)
    except Exception as e:
        print("[ERROR]", str(e))

def deteksi_intent(pesan):
    pesan = pesan.lower()
    if "prediksi harga" in pesan or "harga rumah" in pesan:
        return {"balasan": "Silakan isi ukuran rumah di kolom Prediksi Harga Rumah di bawah 👇", "redirect": "prediksi-harga"}
    if "email spam" in pesan or "deteksi spam" in pesan:
        return {"balasan": "Silakan isi email Anda di kolom Deteksi Spam Email di bawah 👇", "redirect": "cek-spam"}
    return None

# === ROUTES
@app.route('/')
def index():
    return "API AI Multi-Fungsi Aktif"

@app.route('/status')
def status():
    return jsonify({"status": "Aktif", "versi": "1.3", "fitur": ["Chat", "Prediksi Harga", "Deteksi Spam", "Analisa Gambar"]})

@app.route('/chat', methods=['POST'])
def chat_ai():
    try:
        data = request.get_json()
        pesan = bersihkan_teks(data.get("pesan", ""))
        catat_interaksi("Chat AI", pesan)
        simpan_prompt_user(pesan)

        # Deteksi fitur
        intent = deteksi_intent(pesan)
        if intent:
            return jsonify(intent)

        # Latih model jika belum
        if not hasattr(vectorizer, 'vocabulary_') or not hasattr(nb_model, 'classes_'):
            latih_model()

        if hasattr(vectorizer, 'vocabulary_') and hasattr(nb_model, 'classes_'):
            X_input = vectorizer.transform([pesan])
            jawaban = nb_model.predict(X_input)[0]
        else:
            jawaban = "Model belum cukup belajar. Coba lagi nanti ya."

        simpan_dataset_baru(pesan, jawaban)
        return jsonify({"balasan": jawaban})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/prediksi-harga', methods=['POST'])
def prediksi_harga():
    try:
        data = request.get_json()
        ukuran_str = str(data.get('ukuran', '')).strip()

        if not ukuran_str.isdigit():
            return jsonify({"error": "Ukuran harus berupa angka dan tidak boleh kosong."})

        ukuran = int(ukuran_str)
        catat_interaksi("Prediksi Harga", f"Ukuran: {ukuran}")

        X = np.array([[50], [60], [70], [80], [90], [100]])
        y = np.array([150000, 180000, 210000, 240000, 270000, 300000])

        model = LinearRegression()
        model.fit(X, y)
        prediksi = model.predict([[ukuran]])[0]

        def format_harga(value):
            if value >= 1_000_000_000:
                return f"Rp. {value / 1_000_000_000:.1f} triliun"
            elif value >= 1_000_000:
                return f"Rp. {value / 1_000_000:.1f} milyar"
            elif value >= 1_000:
                return f"Rp. {value / 1_000:.1f} juta"
            else:
                return f"Rp. {value:.0f} ribu"

        return jsonify({"hasil": f"Perkiraan harga rumah ukuran {ukuran} m² adalah {format_harga(prediksi)}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/cek-spam', methods=['POST'])
def prediksi_spam():
    try:
        data = request.get_json()
        email_input = data['email']
        catat_interaksi("Deteksi Spam", email_input[:100])

        filepath = os.path.join(base_dir, 'datalatihan_AI.xlsx')
        if not os.path.exists(filepath):
            return jsonify({"error": "File data latih tidak ditemukan."})

        data = pd.read_excel(filepath)
        spam_vectorizer = TfidfVectorizer()
        X = spam_vectorizer.fit_transform(data["emails"])
        y = data["labels"]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
        model = MultinomialNB()
        model.fit(X_train, y_train)

        pred_input = spam_vectorizer.transform([email_input])
        pred = model.predict(pred_input)[0]
        hasil = "Ya, ini terdeteksi sebagai spam." if pred == 1 else "Tidak, ini bukan spam."
        return jsonify({"hasil": hasil})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/analisa-gambar', methods=['POST'])
def analisa_gambar():
    try:
        file = request.files['gambar']
        if not file:
            return jsonify({"error": "Tidak ada file gambar."})

        img_path = os.path.join(base_dir, 'tmp', file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)
        catat_interaksi("Analisa Gambar", file.filename)

        train_dir = os.path.join(base_dir, 'train')
        val_dir = os.path.join(base_dir, 'validation')

        train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            train_dir, target_size=(150, 150), batch_size=2, class_mode='categorical')
        val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
            val_dir, target_size=(150, 150), batch_size=2, class_mode='categorical')

        model = Sequential([
            Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
            MaxPooling2D(2,2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(len(train_gen.class_indices), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_gen, steps_per_epoch=5, epochs=3, validation_data=val_gen, validation_steps=2)

        img = load_img(img_path, target_size=(150, 150))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        label = list(train_gen.class_indices.keys())[np.argmax(pred)]

        return jsonify({"hasil": f"Gambar ini kemungkinan besar adalah: {label}"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)})

# Inisialisasi Model
latih_model()

#if __name__ == "__main__":
#    app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)