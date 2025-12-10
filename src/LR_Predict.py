# Prediction.py
# -----------------------------
# Choosen Model
import joblib
import pandas as pd

# Load trained Logistic Regression model
logreg_model = joblib.load("../models/logreg_model.pkl")

# Columns ที่ต้องกรอก พร้อมคำอธิบาย input เป็น 0/1
input_cols = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'sudden weight loss',
              'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring',
              'Irritability', 'partial paresis', 'Alopecia']

col_display = {
    'Age': 'Age (อายุ) - ตัวเลขเต็ม',
    'Gender': 'Gender (เพศ) - Female=0 / Male=1',
    'Polyuria': 'Polyuria (ปัสสาวะบ่อย) - No=0 / Yes=1',
    'Polydipsia': 'Polydipsia (กระหายน้ำบ่อย) - No=0 / Yes=1',
    'sudden weight loss': 'Sudden weight loss (น้ำหนักลดทันที) - No=0 / Yes=1',
    'weakness': 'Weakness (อ่อนเพลีย) - No=0 / Yes=1',
    'Polyphagia': 'Polyphagia (หิวบ่อย) - No=0 / Yes=1',
    'Genital thrush': 'Genital thrush (ติดเชื้อบริเวณอวัยวะสืบพันธุ์) - No=0 / Yes=1',
    'visual blurring': 'Visual blurring (สายตาพร่ามัว) - No=0 / Yes=1',
    'Irritability': 'Irritability (หงุดหงิดง่าย) - No=0 / Yes=1',
    'partial paresis': 'Partial paresis (อ่อนแรงบางส่วน) - No=0 / Yes=1',
    'Alopecia': 'Alopecia (ผมร่วง) - No=0 / Yes=1'
}

# -----------------------------
# รับค่าจากผู้ใช้ทีละตัว
input_values = []
print("Please enter values for the following features (ใช้ 0/1 ตามคำอธิบาย):")

for col in input_cols:
    while True:
        val = input(f"{col_display[col]}: ").strip()
        try:
            val_int = int(val)
            if col == 'Age':
                input_values.append(val_int)
                break
            elif val_int in [0, 1]:
                input_values.append(val_int)
                break
            else:
                print("กรุณากรอกค่า 0 หรือ 1 ตามคำอธิบาย")
        except ValueError:
            print("กรุณากรอกตัวเลขเท่านั้น")

# -----------------------------
# สรุปค่าที่รับมา
input_dict = dict(zip(input_cols, input_values))
print("\nInput summary / สรุปค่าที่กรอก:")
for k, v in input_dict.items():
    print(f"{col_display[k]}: {v}")

# -----------------------------
# สร้าง DataFrame และ predict
input_df = pd.DataFrame([input_values], columns=input_cols)

pred_class = logreg_model.predict(input_df)[0]
pred_proba = logreg_model.predict_proba(input_df)[0,1]

pred_text = "Positive (เป็นผู้ป่วย)" if pred_class == 1 else "Negative (ไม่เป็นผู้ป่วย)"

print("\n=== Prediction Result / ผลลัพธ์การทำนาย ===")
print(f"Predicted class: {pred_text}")
print(f"Probability of positive class: {pred_proba:.4f}")
