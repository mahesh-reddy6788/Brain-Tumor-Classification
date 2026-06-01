import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import os

st.set_page_config(page_title="Brain Tumor Detection 🧠", layout="centered")
st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image and get the tumor prediction instantly 💕")

# -------------------------
# Load model safely (Functional API model)
# -------------------------
@st.cache_resource
def load_brain_tumor_model():
    model = tf.keras.models.load_model("brain_tumor_model1.keras")
    return model

model = load_brain_tumor_model()

# Labels
labels = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)
    
    # Preprocess
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction
    preds = model.predict(img_array)
    predicted_class = labels[np.argmax(preds)]
    confidence = np.max(preds) * 100
    
    st.success(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence:.2f}%")

# -------------------------
# Show training metrics (if available)
# -------------------------
metrics_file = "history.pkl"

if os.path.exists(metrics_file):
    with open(metrics_file,"rb") as f:
        history_dict = pickle.load(f)
    
    show_metrics = st.checkbox("Show training metrics & confusion matrix")
    
    if show_metrics:
        # Accuracy & Loss
        fig, ax = plt.subplots(1,2, figsize=(12,4))
        ax[0].plot(history_dict['accuracy'], label='train_acc')
        ax[0].plot(history_dict['val_accuracy'], label='val_acc')
        ax[0].set_title('Accuracy')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Accuracy')
        ax[0].legend()
        
        ax[1].plot(history_dict['loss'], label='train_loss')
        ax[1].plot(history_dict['val_loss'], label='val_loss')
        ax[1].set_title('Loss')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        
        st.subheader("Training Accuracy & Loss")
        st.pyplot(fig)
        
        # Confusion matrix (optional)
        if os.path.exists("y_true.npy") and os.path.exists("y_pred.npy"):
            y_true = np.load("y_true.npy")
            y_pred = np.load("y_pred.npy")
            
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=labels)
            
            fig2, ax2 = plt.subplots(figsize=(6,6))
            disp.plot(ax=ax2, cmap=plt.cm.Blues, colorbar=False)
            st.subheader("Confusion Matrix")
            st.pyplot(fig2)
