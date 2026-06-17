import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Configure page
st.set_page_config(
    page_title="Colorectal Cancer Polyp Detection and Classification",
    page_icon="🩺",
    layout="wide"
)

# Title and description
st.title("🩺 Colorectal Cancer Polyp Detection and Classification System")
st.markdown("""
This application uses a **YOLOv8 deep learning model** to automatically detect and classify
colorectal polyps from colonoscopy images.
""")

# Load trained YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your model path if necessary

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload a Colonoscopy Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

    # Run YOLOv8 inference
    results = model.predict(
        source=image_np,
        conf=0.25,
        save=False
    )

    # Generate annotated image
    annotated_image = results[0].plot()

    with col2:
        st.subheader("Detection Results")
        st.image(
            annotated_image,
            channels="BGR",
            use_container_width=True
        )

    # Display prediction details
    st.subheader("Prediction Details")

    boxes = results[0].boxes

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]

            st.write(f"""
            **Detection {i+1}**
            - **Polyp Type:** {class_name}
            - **Confidence Score:** {confidence:.2%}
            """)
    else:
        st.warning("No colorectal polyps detected in the uploaded image.")

# Footer
st.markdown("---")
st.markdown(
    "**Technologies Used:** Python | YOLOv8 | PyTorch | Streamlit | OpenCV"
)
