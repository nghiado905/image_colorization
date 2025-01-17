import os
import datetime
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify

# Tạo Flask app
app = Flask(__name__)

# Thư mục lưu trữ ảnh tải lên và ảnh dự đoán
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load các model generator và classifier
generators = {
    'Coast': tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\Coast\\Coast.keras"),
    'Desert': tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\Desert\\Desert.keras"),
    'Forest': tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\Forest\\Forest.keras"),
    'Glacier': tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\Glacier\\Glacier.keras"),
    'Mountain': tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\Mountain\\Mountain.keras"),
}

classifier = tf.keras.models.load_model("D:\\Private\\CODE\\kaggle\\image_colorization\\save_model\\prediction\\model.keras")

# Hàm tiền xử lý ảnh
def get_image_to_predict(path, size):
    try:
        # Load ảnh và lưu lại kích thước gốc
        with Image.open(path) as img:
            original_size = img.size
            
            # Resize ảnh thành kích thước cố định (128x128)
            img_resized = img.resize((size, size))
            img_array = np.asarray(img_resized.convert('L')).reshape((size, size, 1)) / 255.0

        return img_array.reshape(1, size, size, 1), original_size
    except Exception as e:
        print(f"Error loading or preprocessing image: {e}")
        return None, None

# Hàm lưu ảnh dự đoán
def save_prediction_image(y, output_path, original_size):
    try:
        colored_image = (y[0] * 255).numpy().astype('uint8')
        image = Image.fromarray(colored_image)

        # Resize ảnh về kích thước gốc
        image = image.resize(original_size, Image.LANCZOS)
        image.save(output_path)
    except Exception as e:
        print(f"Error saving predicted image: {e}")

# Hàm phân loại ảnh
def predict_label(img, path):
    try:
        img = tf.keras.utils.load_img(path, target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)  # Tạo batch

        predictions = classifier.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        label_index = np.argmax(score)

        labels = ['Coast', 'Desert', 'Forest', 'Glacier', 'Mountain']
        return labels[label_index]
    except Exception as e:
        print(f"Error during classification: {e}")
        return None

# Route chính
@app.route('/')
def index():
    return render_template('index.html')

# Route upload và dự đoán
@app.route('/upload', methods=['POST'])
def prediction():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Lưu ảnh tải lên
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_filename = os.path.join(UPLOAD_FOLDER, f'photo_{timestamp}.png')
    file.save(photo_filename)

    # Tiền xử lý ảnh và lấy kích thước gốc
    img, original_size = get_image_to_predict(photo_filename, 128)

    if img is not None:
        label = predict_label(img, photo_filename)

        if label:
            generator = generators.get(label)
            print(label)

            if generator:
                # Sinh ảnh dự đoán
                y = generator(img)
                
                # Lưu ảnh dự đoán
                prediction_filename = os.path.join(UPLOAD_FOLDER, f'predicted_{timestamp}.png')
                save_prediction_image(y, prediction_filename, original_size)

                # Trả về URL cho ảnh gốc và ảnh dự đoán
                photo_url = f'/static/uploads/photo_{timestamp}.png'
                prediction_url = f'/static/uploads/predicted_{timestamp}.png'

                return jsonify({
                    'photo_url': photo_url,
                    'prediction_url': prediction_url
                })
            else:
                return jsonify({'error': f'No generator found for label {label}.'}), 400
        else:
            return jsonify({'error': 'Failed to classify the image.'}), 400
    else:
        return jsonify({'error': 'Failed to process the image.'}), 400

# Chạy ứng dụng Flask
if __name__ == '__main__':
    app.run(debug=True)
