import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model

# Load model
@st.cache_resource
def load_model():
    return keras.models.load_model("vehicle_classification_model.h5")

model = load_model()

# Replace with your classes
class_names = [
    "aircraft",
    "boat",
    "bus",
    "car",
    "motorcycle",
    "truck"
]

st.set_page_config(
    page_title="Image Classification",
    layout="wide"
)

st.title("🖼️ CNN Image Classification")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    # Dynamic input size
    _, h, w, c = model.input_shape

    img = image.resize((w, h))

    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]

    pred_idx = np.argmax(prediction)
    class_name = class_names[pred_idx]
    confidence = prediction[pred_idx]

    # Two-column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.markdown("### Prediction")

        st.markdown(
            f"""
            <h1 style='
                text-align:center;
                color:#2E86C1;
                font-size:60px;
                font-weight:bold;'>
                {class_name}
            </h1>
            """,
            unsafe_allow_html=True
        )

        st.metric(
            label="Confidence",
            value=f"{confidence:.2%}"
        )

        st.markdown("---")
        st.subheader("Top Predictions")

        top3 = np.argsort(prediction)[::-1][:3]

        for idx in top3:
            st.write(
                f"**{class_names[idx]}** : {prediction[idx]:.2%}"
            )

    st.markdown("---")

    st.subheader("Class Probabilities")

    chart_data = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    st.bar_chart(chart_data)
