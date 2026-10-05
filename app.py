import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Setup the Page Configuration (Must be the first Streamlit command)
st.set_page_config(page_title="Brain Tumor MRI Classifier", page_icon="🧠", layout="wide")

# 2. Inject Premium Custom CSS for Styling
st.markdown("""
<style>
    /* Dark theme colors and fonts */
    .stApp {
        background-color: #0e1117;
        font-family: 'Inter', sans-serif;
    }
    
    /* Gradient Title */
    .title-text {
        font-size: 3.5rem !important;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    
    /* Subtitle */
    .subtitle-text {
        text-align: center;
        color: #a0aec0;
        font-size: 1.2rem;
        margin-top: 10px;
        margin-bottom: 40px;
    }
    
    /* Prediction Box styling */
    .prediction-box {
        padding: 30px;
        border-radius: 15px;
        background: linear-gradient(145deg, #1e2532, #161b22);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        text-align: center;
        margin-top: 20px;
        border: 1px solid #2d3748;
    }
    
    .pred-class {
        color: #48bb78;
        font-size: 3rem;
        font-weight: bold;
        margin-top: 10px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
</style>
""", unsafe_allow_html=True)

# 3. Cache the Model Loading so it doesn't reload on every user interaction
@st.cache_resource
def load_model():
    # Load our finetuned RadImageNet model
    return tf.keras.models.load_model('saved_models/radimagenet_finetuned.h5')

model = load_model()

# 4. Define our target classes
class_names = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# --- HEADER SECTION ---
st.markdown('<h1 class="title-text">Brain Tumor MRI Classification</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle-text">Upload a brain MRI scan to instantly detect and classify tumor types using a deep learning model pre-trained on medical imaging.</p>', unsafe_allow_html=True)

st.write("---")

# --- MAIN LAYOUT (2 Columns) ---
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📸 1. Upload MRI Scan")
    uploaded_file = st.file_uploader("Choose a JPG or PNG image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        # Display the uploaded image beautifully
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded MRI Scan', use_container_width=True)

with col2:
    st.subheader("🔬 2. Analysis & Prediction")
    
    if uploaded_file is not None:
        with st.spinner('Analyzing scan using RadImageNet weights...'):
            # --- Preprocessing Pipeline ---
            # 1. Resize to 224x224 (what our model expects)
            img = image.resize((224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # 2. Rescale pixel values from [0, 255] to [0, 1] 
            # (Because we used Rescaling(1./255) in our notebook!)
            # img_array = img_array / 255.0
            
            # 3. Add the batch dimension (so shape becomes 1, 224, 224, 3)
            img_array = tf.expand_dims(img_array, 0)
            
            # --- Prediction ---
            predictions = model.predict(img_array)
            
            # Our model outputs raw logits (from_logits=True), so we must apply softmax
            # to convert them into standard percentage probabilities.
            score = tf.nn.softmax(predictions[0]) 
            
            confidence = np.max(score) * 100
            predicted_class = class_names[np.argmax(score)]
            
            # --- Display Results ---
            st.markdown(f"""
            <div class="prediction-box">
                <h3 style="color: #a0aec0; margin-bottom: 0;">Detected Classification:</h3>
                <div class="pred-class">{predicted_class.replace('_', ' ')}</div>
                <h4 style="color: #e2e8f0; margin-top: 15px;">Confidence Score: {confidence:.2f}%</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.write("")
            st.write("### Confidence Breakdown")
            
            # Display detailed progress bars for all classes
            for i, class_name in enumerate(class_names):
                class_score = score[i] * 100
                st.write(f"**{class_name.replace('_', ' ').title()}**: {class_score:.2f}%")
                st.progress(int(class_score))
                
    else:
        # Placeholder state before upload
        st.info("👈 Please upload an MRI scan on the left panel to begin the analysis.")
        st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #1e2532; border-radius: 10px; margin-top: 20px;">
            <h2 style="color: #a0aec0;">Awaiting Image...</h2>
            <p style="color: #718096;">The neural network is standing by.</p>
        </div>
        """, unsafe_allow_html=True)
