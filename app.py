import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import time

# ==========================
# Konfigurasi Halaman
# ==========================
st.set_page_config(
    page_title="AI Vision Cabai",
    page_icon="🌶️",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = load_model("model_cabai.keras")

# ==========================
# Sidebar
# ==========================
with st.sidebar:

    st.title("🌶️ AI Vision")

    st.markdown("---")

    st.info("""
### Tentang Sistem

Aplikasi ini menggunakan
**Convolutional Neural Network (CNN)**

untuk mendeteksi kondisi
buah cabai.

Kategori:

🟢 Cabai Sehat

🔴 Cabai Tidak Sehat
""")

    st.markdown("---")

    st.success("Model CNN berhasil dimuat")

    st.markdown("---")

    st.write("👨‍💻 Developer")
    st.write("Muh. Maksum")
    st.write("Sistem Informasi")
    st.write("Universitas Al Asyariah Mandar")

# ==========================
# Judul
# ==========================
st.title("🌶️ AI Vision for Chili Detection")

st.markdown("""
### Deteksi Kesehatan Buah Cabai Menggunakan CNN

Aplikasi ini mampu mendeteksi kondisi kesehatan buah cabai berdasarkan gambar yang diunggah.
""")

st.info("📌 Upload gambar cabai dengan format JPG, JPEG, atau PNG.")

# ==========================
# Upload
# ==========================
uploaded_file = st.file_uploader(
    "📤 Upload Gambar Cabai",
    type=["jpg", "jpeg", "png"]
)

# ==========================
# Jika gambar diupload
# ==========================
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    # --------------------------
    # Kolom kiri
    # --------------------------
    with col1:

        st.subheader("🖼️ Gambar Asli")

        st.image(
            img,
            use_container_width=True
        )

    # --------------------------
    # Prediksi
    # --------------------------
    img_resize = img.resize((224,224))

    img_array = image.img_to_array(img_resize)

    img_array = np.expand_dims(img_array,axis=0)

    img_array = img_array/255.0

    with st.spinner("🤖 AI sedang menganalisis gambar..."):

        time.sleep(1)

        pred = model.predict(img_array,verbose=0)

    nilai = pred[0][0]

    # --------------------------
    # Kolom kanan
    # --------------------------
    with col2:

        st.subheader("🔍 Hasil Analisis AI")

        if nilai > 0.5:

            confidence = nilai*100

            st.error("## 🔴 CABAI TIDAK SEHAT")

        else:

            confidence = (1-nilai)*100

            st.success("## 🟢 CABAI SEHAT")

        st.metric(
            label="Confidence",
            value=f"{confidence:.2f}%"
        )

        st.progress(int(confidence))

        st.write("### Nilai Prediksi")

        st.code(f"{nilai:.4f}")

    # ==========================
    # Kesimpulan
    # ==========================
    st.markdown("---")

    st.subheader("📊 Kesimpulan")

    if nilai > 0.5:

        st.error("""
Model CNN mendeteksi bahwa gambar termasuk kategori **Cabai Tidak Sehat**.

Cabai memiliki indikasi kerusakan sehingga kualitasnya menurun.
""")

    else:

        st.success("""
Model CNN mendeteksi bahwa gambar termasuk kategori **Cabai Sehat**.

Cabai masih berada dalam kondisi baik.
""")

# ==========================
# Footer
# ==========================
st.markdown("---")

st.caption("""
Developed by **sitti hafsyah**

Program Studi Sistem Informasi

Universitas Al Asyariah Mandar

2026
""")
