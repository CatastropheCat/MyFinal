# app.py
import flask
import pandas as pd
import numpy as np
import joblib

app = flask.Flask(__name__, template_folder='templates')

# --- โหลดโมเดลและส่วนประกอบที่เกี่ยวข้อง ---
# หมายเหตุ: ตรวจสอบให้แน่ใจว่าชื่อไฟล์ตรงกับที่บันทึกไว้จาก Notebook
MODEL_FILENAME = 'optimized_xgboost.joblib' # หรือชื่อโมเดลที่ดีที่สุดที่เลือก
PREPROCESSOR_FILENAME = 'preprocessor_accident_severity.joblib'
LABEL_ENCODER_FILENAME = 'target_label_encoder_accident_severity.joblib'
PROCESSED_FEATURE_NAMES_FILENAME = 'processed_feature_names.joblib' # เพิ่มการโหลดไฟล์นี้

try:
    model = joblib.load(MODEL_FILENAME)
    preprocessor = joblib.load(PREPROCESSOR_FILENAME)
    label_encoder = joblib.load(LABEL_ENCODER_FILENAME)
    processed_feature_names = joblib.load(PROCESSED_FEATURE_NAMES_FILENAME) # โหลดรายชื่อ feature หลัง OHE
    print("Model, preprocessor, label encoder, and processed feature names loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading files: {e}. Please ensure all .joblib files are in the correct directory.")
    model, preprocessor, label_encoder, processed_feature_names = None, None, None, None
except Exception as e:
    print(f"An unexpected error occurred during loading: {e}")
    model, preprocessor, label_encoder, processed_feature_names = None, None, None, None

numerical_cols = [
    'latitude', 'longitude', 'Holiday', 'incident_hour',
    'incident_dayofweek', 'incident_month', 'is_weekend'
]

categorical_cols = [
    'weather_condition', 'road_description', 'slope_description', 'agency'
]

# expected_input_columns ไม่จำเป็นต้องใช้โดยตรงแล้ว ถ้า preprocessor ถูกสร้างด้วยชื่อคอลัมน์
# และ input_df ที่สร้างขึ้นก็ใช้ชื่อคอลัมน์จาก numerical_cols และ categorical_cols


# --- ฟังก์ชันสำหรับแสดงผลระดับความรุนแรง ---
def get_severity_meaning(encoded_severity, le):
    """แปลงค่าที่เข้ารหัสกลับเป็นความหมายของระดับความรุนแรง"""
    try:
        original_severity = le.inverse_transform([encoded_severity])[0]
        # **สำคัญ:** ปรับการตีความหมายให้ตรงกับการกำหนด valid_severities ใน Notebook
        if original_severity == 0: return "บาดเจ็บเล็กน้อยมาก (Very Low Severity)"
        elif original_severity == 1: return "บาดเจ็บเล็กน้อย (Low Severity)"
        elif original_severity == 2: return "บาดเจ็บปานกลาง/รุนแรง (Medium Severity)"
        elif original_severity == 3: return "เสียชีวิต (High Severity)"
        else: return f"ไม่ทราบความหมาย (Original: {original_severity})"
    except Exception as e:
        print(f"Error decoding severity: {e}")
        return "ไม่สามารถถอดรหัสได้"

@app.route('/')
def main():
    # แสดงหน้าเว็บหลัก (index.html)
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not all([model, preprocessor, label_encoder, processed_feature_names]):
        return flask.jsonify({'error': 'Model or associated files not loaded properly. Check server logs.'}), 500
    try:
        form_data = flask.request.form.to_dict()
        print(f"Received form data: {form_data}")
        
        input_data_dict = {}
        for col in numerical_cols: # Loop สำหรับ numerical_cols
            value_str = form_data.get(col)
            if value_str is None or value_str == '':
                input_data_dict[col] = [np.nan] 
            else:
                try:
                    input_data_dict[col] = [float(value_str)]
                except ValueError: # <-- except block นี้
                    return flask.jsonify({'error': f"Invalid value for numerical feature {col}: {value_str}"}), 400 # บรรทัด 87 ★★★
        
        # ★★★ ปัญหาอาจจะอยู่ที่การย่อหน้าของ for loop ถัดไปนี้ ★★★
        for col in categorical_cols: # Loop สำหรับ categorical_cols
            input_data_dict[col] = [str(form_data.get(col, ''))] 

        input_df = pd.DataFrame(input_data_dict, columns=numerical_cols + categorical_cols)
        
        print(f"DataFrame created for preprocessing:\n{input_df}")

        # --- Preprocess ข้อมูล ---
        processed_input_array = preprocessor.transform(input_df)
        print(f"Shape of processed input array: {processed_input_array.shape}")
        
        # สร้าง DataFrame จาก processed_input_array โดยใช้ processed_feature_names
        # เพื่อให้แน่ใจว่า input ของ model มี feature names ที่ถูกต้อง
        if len(processed_feature_names) != processed_input_array.shape[1]:
            error_msg = (f"Mismatch in feature names: Expected {len(processed_feature_names)} "
                         f"features based on loaded names, but preprocessor output "
                         f"{processed_input_array.shape[1]} features.")
            print(error_msg)
            return flask.jsonify({'error': error_msg}), 500
            
        processed_input_df = pd.DataFrame(processed_input_array, columns=processed_feature_names)
        print(f"Processed input DataFrame for model:\n{processed_input_df.head()}")

        # --- ทำนายผล ---
        # โมเดล XGBoost ที่ train ด้วย scikit-learn wrapper สามารถรับ DataFrame ได้
        prediction_encoded = model.predict(processed_input_df)
        predicted_severity_encoded = int(prediction_encoded[0])

        # --- ถอดรหัสผลการทำนาย ---
        predicted_severity_meaning = get_severity_meaning(predicted_severity_encoded, label_encoder)

        print(f"Encoded Prediction: {predicted_severity_encoded}, Meaning: {predicted_severity_meaning}")

        return flask.jsonify({
            'predicted_severity_encoded': predicted_severity_encoded,
            'predicted_severity_meaning': predicted_severity_meaning
        })

    except Exception as e: # <--- except block หลักของ predict (ควรจะอยู่ในระดับเดียวกับ try ด้านบน)
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()
        return flask.jsonify({'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    import os
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True)
