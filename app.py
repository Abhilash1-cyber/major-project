from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import cv2
import base64
import os

app = Flask(__name__)

# Constants
IMAGE_SIZE = 224
MODEL_PATH = "unet_dr_model.h5"
CLASS_NAMES = ['DR', 'No_DR']
LAST_CONV_LAYER_NAME = "conv2d_7"

# Load the model globally
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    grad_model = None


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)

        while isinstance(preds, (tuple, list)):
            preds = preds[0]

        preds = tf.convert_to_tensor(preds, dtype=tf.float32)

        if pred_index is None:
            pred_index = tf.argmax(preds[0])

        if len(preds.shape) == 1:
            preds = tf.expand_dims(preds, 0)

        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def save_and_display_gradcam(img, heatmap, alpha=0.4):
    img = np.array(img)

    heatmap = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    superimposed_img = jet * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

    return Image.fromarray(superimposed_img)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or grad_model is None:
        return jsonify({"error": "Model is not loaded."}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        img_bytes = file.read()

        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))

        img_array = image.img_to_array(original_img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])

        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])

        # Grad-CAM
        heatmap = make_gradcam_heatmap(img_array, grad_model, predicted_class_index)
        cam_image = save_and_display_gradcam(original_img, heatmap)

        buffered = io.BytesIO()
        cam_image.save(buffered, format="JPEG")
        cam_b64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            "prediction": predicted_class_name,
            "confidence": confidence,
            "heatmap": f"data:image/jpeg;base64,{cam_b64}",
            "success": True
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


# ✅ IMPORTANT: Render-compatible run
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Render provides PORT
    app.run(host='0.0.0.0', port=port)