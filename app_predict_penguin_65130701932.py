import streamlit as st
import pickle
import numpy as np
import pandas as pd

# โหลดโมเดลและตัวแปลง
@st.cache
def load_model():
    with open('model_penguin_65130701932.pkl', 'rb') as file:
        model, species_encoder, island_encoder, sex_encoder = pickle.load(file)
    return model, species_encoder, island_encoder, sex_encoder

model, species_encoder, island_encoder, sex_encoder = load_model()

# สร้างส่วนอินพุตสำหรับผู้ใช้
st.title("Penguin Species Prediction")
st.write("Enter the penguin features:")

# สร้างฟอร์มสำหรับป้อนข้อมูล
species = st.selectbox("Species", species_encoder.classes_)
island = st.selectbox("Island", island_encoder.classes_)
sex = st.selectbox("Sex", sex_encoder.classes_)
bill_length_mm = st.number_input("Bill Length (mm)", min_value=30.0, max_value=100.0, step=0.1)
bill_depth_mm = st.number_input("Bill Depth (mm)", min_value=10.0, max_value=60.0, step=0.1)
flipper_length_mm = st.number_input("Flipper Length (mm)", min_value=150, max_value=250, step=1)
body_mass_g = st.number_input("Body Mass (g)", min_value=2000, max_value=7000, step=10)


# คำนวณค่าพยากรณ์เมื่อผู้ใช้คลิกปุ่ม
if st.button("Predict"):
    # เตรียมข้อมูลสำหรับการทำนาย
    x_new = np.array([[bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g]])

    # สร้าง DataFrame สำหรับข้อมูลที่ได้รับจากผู้ใช้
    x_new = pd.DataFrame(x_new, columns=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])

    # แปลงข้อมูลที่เป็นหมวดหมู่ (เช่น species, island, sex) เป็นตัวเลข
    x_new["species"] = species_encoder.transform([species])
    x_new["island"] = island_encoder.transform([island])
    x_new["sex"] = sex_encoder.transform([sex])

    # ทำนายผลลัพธ์
    y_pred_new = model.predict(x_new)
    result = species_encoder.inverse_transform(y_pred_new)

    # แสดงผลลัพธ์
    st.write(f"Predicted Species: {result[0]}")
