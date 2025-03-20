from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="acne_model_efficientnet.tflite")
interpreter.allocate_tensors()

# Define class labels
class_labels = ['Mild', 'Moderate', 'Moderate Severe', 'Severe', 'Clear Skin']

# Treatment Plans for Each Acne Type
treatment_plans = {
    "Mild": ["""
        <h2>Dermatologist-Recommended Treatments</h2>
        <h3>Topical Medications:</h3>
        <p><strong>1. Retinoids (Comedolytic & anti-inflammatory)</strong></p>
        <ul>
            <li>Adapalene 0.1% (Differin) – OTC option</li>
            <li>Tretinoin 0.025%-0.05% (Retin-A) – Prescription</li>
            <li>Tazarotene 0.05%-0.1% (Tazorac) – Stronger option, Rx only</li>
        </ul>
        <p><strong>2. Benzoyl Peroxide (BPO) 2.5%-5%</strong> – Reduces bacteria & prevents resistance</p>
        <p><strong>3. Salicylic Acid 2%</strong> – Helps exfoliate and unclog pores</p>
        <p><strong>4. Azelaic Acid 10%-20%</strong> – Anti-inflammatory & brightens post-acne marks</p>
        
        <h3>Healthy Preventive Measures:</h3>
        <ul>
            <li>Use a gentle cleanser (pH-balanced, non-comedogenic)</li>
            <li>Apply an oil-free, non-comedogenic moisturizer</li>
            <li>Use sunscreen (non-comedogenic, mineral-based preferred)</li>
            <li>Avoid touching face & picking at acne</li>
            <li>Reduce dairy and high-glycemic foods</li>
        </ul>
    """],
    
    "Moderate": ["""
        <h2>Dermatologist-Recommended Treatments</h2>
        <h3>Topical Medications:</h3>
        <ul>
            <li><strong>Retinoid</strong> (Adapalene, Tretinoin, or Tazarotene)</li>
            <li><strong>Benzoyl Peroxide 5%-10%</strong></li>
            <li><strong>Topical Antibiotics:</strong> Clindamycin 1%, Erythromycin 2%, Dapsone 5%-7.5%</li>
        </ul>
        
        <h3>Oral Medications (if needed):</h3>
        <ul>
            <li><strong>Oral Antibiotics:</strong> Doxycycline 50-100mg, Minocycline 50-100mg</li>
            <li><strong>Hormonal Therapy:</strong> Spironolactone 50-100 mg/day, Combined Oral Contraceptives (COCs)</li>
        </ul>
        
        <h3>Healthy Preventive Measures:</h3>
        <ul>
            <li>Reduce stress, get enough sleep</li>
            <li>Avoid high-sugar foods</li>
            <li>Incorporate omega-3 fatty acids</li>
        </ul>
    """],

    "Severe": ["""
        <h2>Dermatologist-Recommended Treatments</h2>
        <h3>Primary Treatment:</h3>
        <p><strong>Oral Isotretinoin (Accutane)</strong> – 0.5-1 mg/kg/day for 5-6 months</p>
        
        <h3>Adjunctive Therapies:</h3>
        <ul>
            <li><strong>Oral Antibiotics:</strong> Doxycycline, Minocycline</li>
            <li><strong>Intralesional Corticosteroid Injections</strong> – For large nodules/cysts</li>
            <li><strong>Hormonal Therapy:</strong> Spironolactone, Oral Contraceptives (Yaz, Diane-35)</li>
        </ul>
        
        <h3>Advanced Procedures (For Scarring & Severe Cases):</h3>
        <ul>
            <li>Chemical Peels (Salicylic, TCA)</li>
            <li>Laser Therapy (Fraxel, CO2, Nd:YAG)</li>
            <li>Microneedling with PRP</li>
            <li>Dermal Fillers (For Atrophic Scars)</li>
            <li>Surgical Excision (For Persistent Cysts)</li>
        </ul>
    """],

    "Clear Skin": ["""
        <h2>Natural Skin Care & Acne Prevention</h2>
        <h3>1. Morning Routine:</h3>
        <ul>
            <li>Wash your face with lukewarm water</li>
            <li>Use raw honey or oat flour paste as a natural cleanser</li>
            <li>Apply aloe vera gel for hydration</li>
            <li>Use rose water as a natural toner</li>
        </ul>

        <h3>2. Healthy Diet:</h3>
        <ul>
            <li>Eat antioxidant-rich foods (berries, spinach, fish)</li>
            <li>Avoid dairy and high-glycemic foods</li>
        </ul>

        <h3>3. Stress Management & Lifestyle:</h3>
        <ul>
            <li>Practice yoga, meditation, deep breathing</li>
            <li>Get 7-9 hours of sleep daily</li>
            <li>Stay hydrated and drink green tea</li>
        </ul>
    """]
}

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


@app.route('/')
def index():
    return render_template('a.html')

@app.route('/reports')
def reports():
    return render_template('reports.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process image
        img_array = preprocess_image(filepath)
        
        # Get model input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], img_array)
        
        # Run inference
        interpreter.invoke()
        
        # Get output tensor
        pred = interpreter.get_tensor(output_details[0]['index'])
        
        # Get predicted class
        predicted_class = np.argmax(pred, axis=1)[0]
        result = class_labels[predicted_class]
        treatment = treatment_plans[result]  # Get treatment plan

        return render_template('index.html', result=result, treatment=treatment, image_url=filepath)
    
    return render_template('index.html', result=None, treatment=None, image_url=None)

if __name__ == '__main__':
    print("\n=== Server Startup Checks ===")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Python Version: {os.sys.version}")
    app.run(debug=False, host='0.0.0.0', port=5000)
