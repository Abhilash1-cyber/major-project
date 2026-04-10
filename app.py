from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import cv2
import base64

app = Flask(__name__)

# Constants based on training notebook
IMAGE_SIZE = 224
MODEL_PATH = "unet_dr_model.h5"
CLASS_NAMES = ['DR', 'No_DR']
LAST_CONV_LAYER_NAME = "conv2d_7"

# Load the model globally
try:
    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
    
    # Create the model required for Grad-CAM
    # It inputs the same as the main model, but outputs both the final conv layer and the final prediction
    grad_model = tf.keras.models.Model(
        model.inputs, 
        [model.get_layer(LAST_CONV_LAYER_NAME).output, model.output]
    )
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    grad_model = None

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    # Then we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        
        while isinstance(preds, tuple) or isinstance(preds, list):
            preds = preds[0]
            
        preds = tf.convert_to_tensor(preds, dtype=tf.float32)
            
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        if len(preds.shape) == 1:
            preds = tf.expand_dims(preds, 0)
            
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4):
    # Convert PIL Image to OpenCV format (numpy array, RGB)
    img = np.array(img)
    
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # OpenCV uses BGR, but we want RGB for displaying in PIL, so convert jet array
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)

    # Resize the colormapped heatmap to the size of the original image
    jet = cv2.resize(jet, (img.shape[1], img.shape[0]))

    # Superimpose the heatmap on original image
    superimposed_img = jet * alpha + img
    
    # Clip the values to 0-255 and convert to uint8
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    # Convert numpy array back to PIL image
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
        # Read the image file
        img_bytes = file.read()
        
        # Keep Original Image for overlay
        original_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        original_img = original_img.resize((IMAGE_SIZE, IMAGE_SIZE))
        
        # Convert to numpy array and rescale (as done in ImageDataGenerator with rescale=1./255)
        img_array = image.img_to_array(original_img)
        img_array = img_array / 255.0
        
        # Expand dimensions to add batch size (shape becomes: 1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        predicted_class_index = int(np.argmax(prediction, axis=1)[0])
        
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(prediction[0][predicted_class_index])
        
        # === GRAD-CAM INTEGRATION ===
        heatmap = make_gradcam_heatmap(img_array, grad_model, predicted_class_index)
        cam_image = save_and_display_gradcam(original_img, heatmap)
        
        # Convert the CAM image to base64 so we can send it directly to the HTML
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

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)
