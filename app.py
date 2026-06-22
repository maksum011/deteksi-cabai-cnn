import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model = load_model("model_cabai.keras")

st.title("🌶️ Deteksi Kesehatan Buah Cabai")

uploaded_file = st.file_uploader(
    "Upload gambar cabai",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(
        img,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    img = img.resize((224, 224))

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)

    st.subheader("Hasil Deteksi")

    if pred[0][0] > 0.5:
        st.error("❌ Cabai Tidak Sehat")
        st.write(f"Keyakinan: {pred[0][0]*100:.2f}%")
    else:
        st.success("✅ Cabai Sehat")
        st.write(f"Keyakinan: {(1-pred[0][0])*100:.2f}%")

    st.write(f"Nilai Prediksi: {pred[0][0]:.4f}")
