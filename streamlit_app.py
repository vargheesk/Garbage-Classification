
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image, ImageOps

# Set page configuration
st.set_page_config(
    page_title="EcoSort: AI Garbage Classification",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "MNC Theme"
st.markdown("""
    <style>
    /* Main Background and Text */
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #2E7D32; /* Eco Green */
        font-weight: 600;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Reduce Sidebar Top Padding */
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Sidebar Buttons (Navigation) */
    div.stButton > button {
        width: 100%;
        background-color: transparent;
        color: #424242;
        border: 1px solid #e0e0e0;
        text-align: left;
        padding-left: 15px;
        transition: all 0.2s;
    }
    
    div.stButton > button:hover {
        background-color: #e8f5e9;
        color: #2E7D32;
        border-color: #2E7D32;
    }
    
    div.stButton > button:focus {
        background-color: #2E7D32;
        color: white;
        border-color: #2E7D32;
    }

    /* Action Button (Upload/Predict context if needed) */
    /* We assume specific keys or classes for main action buttons if strictly needed, 
       but stButton affects all. We can inline style the specific functionality ones if needed, 
       or accept the unified look. */

    /* Info/Markup Sections */
    .info-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #2E7D32;
        margin-bottom: 20px;
    }
    
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# ----------------- SESSION STATE NAVIGATION ----------------- #
if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

def navigate_to(page):
    st.session_state['page'] = page

# ----------------- SIDEBAR NAVIGATION ----------------- #
with st.sidebar:
    st.title("♻️ EcoSort AI")
    st.markdown("---")
    
    # Navigation Buttons
    st.markdown("### 🧭 Navigation")
    
    if st.button("🏠 Home (Classifier)", use_container_width=True):
        navigate_to('Home')
        
    if st.button("🛠️ Methodology", use_container_width=True):
        navigate_to('Methodology')
        
    if st.button("📊 Classification Classes", use_container_width=True):
        navigate_to('Classes')
        
    if st.button("📚 Libraries & Tools", use_container_width=True):
        navigate_to('Libraries')
        
    st.markdown("---")
    st.markdown("### 🔗 External Links")
    st.link_button("📂 View GitHub Repo", "https://github.com/vargheesk/Garbage-Classification")
    st.link_button("🌐 Developer Portfolio", "https://vargheeskutty-eldhose.vercel.app/")
    
    st.markdown("---")
    st.caption("v1.0.0 | Luminar Python Project")


# ----------------- MODEL LOGIC ----------------- #
CLASSES = {
    0: 'Battery', 1: 'Biological', 2: 'Brown Glass', 3: 'Cardboard',
    4: 'Clothes', 5: 'Green Glass', 6: 'Metal', 7: 'Paper',
    8: 'Plastic', 9: 'Shoes', 10: 'Trash', 11: 'White Glass'
}

@st.cache_resource
def load_classification_model():
    """Loads the pre-trained Keras model by reconstructing architecture and loading weights."""
    try:
        # Load the model directly (this is safer and includes architecture + weights)
        model = tf.keras.models.load_model('Grabage_model.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """Resizes and rescales the image for the model."""
    img_array = np.array(image)
    if img_array.ndim == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    img_resized = cv2.resize(img_array, (224, 224))
    img_reshaped = img_resized.reshape(1, 224, 224, 3)
    img_reshaped = img_reshaped.astype('float32') / 255.0
    return img_reshaped


# ----------------- MAIN PAGE CONTENT ----------------- #

if st.session_state['page'] == 'Home':
    st.title("♻️ EcoSort: Intelligent Waste Classification")
    st.markdown("### Leveraging AI for a Cleaner Planet")
    
    st.markdown("""
    <div class="info-box">
        <p><b>Welcome!</b> Use this tool to classify waste items. Upload an image, and our AI will tell you what it is.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Choose an image (JPG/PNG)...", type=["jpg", "jpeg", "png"])
        
        st.markdown("---")
        st.write("**Or try a sample image:**")
        
        # Sample image paths
        sample_images = {
            "Textile_waste.jpg": "Textile Sample",
            "plastic-waste-headrer.jpg": "Plastic Sample",
            "images.jpg": "Misc Sample"
        }
        
        cols = st.columns(3)
        selected_sample = None
        
        for idx, (img_path, label) in enumerate(sample_images.items()):
            with cols[idx % 3]:
                try:
                    # Open and resize image for uniform display
                    img = Image.open(img_path)
                    img_fixed = ImageOps.fit(img, (300, 200), Image.Resampling.LANCZOS)
                    
                    st.image(img_fixed, use_container_width=True)
                    if st.button("Use", key=f"btn_{idx}", use_container_width=True):
                        st.session_state['selected_sample'] = img_path
                except Exception:
                    st.error(f"Missing {img_path}")

        # Determine which image to process
        image_to_process = None
        
        if uploaded_file is not None:
            image_to_process = Image.open(uploaded_file)
            st.session_state['selected_sample'] = None # Reset sample if user uploads
        elif st.session_state.get('selected_sample'):
            image_to_process = Image.open(st.session_state['selected_sample'])
            
        if image_to_process:
            st.image(image_to_process, caption='Selected Image', use_container_width=True)
            
            # Predict
            with st.spinner('Analyzing...'):
                model = load_classification_model()
                if model:
                    processed_img = preprocess_image(image_to_process)
                    predictions = model.predict(processed_img)
                    predicted_class_idx = np.argmax(predictions)
                    confidence = np.max(predictions) * 100
                    predicted_label = CLASSES.get(predicted_class_idx, "Unknown")
                    
                    st.session_state['last_pred'] = predicted_label
                    st.session_state['last_conf'] = confidence
                    st.session_state['last_probs'] = predictions[0]

    with col2:
        st.subheader("🔍 Analysis Results")
        
        # Check if we have a valid selection (upload or sample)
        has_image = uploaded_file is not None or st.session_state.get('selected_sample') is not None
        
        if not has_image:
            st.info("Awaiting image upload or sample selection...")
            st.markdown("""
            <div style="text-align: center; opacity: 0.5;">
                <img src="https://cdn-icons-png.flaticon.com/512/860/860329.png" width="150">
                <p>No image analysis yet.</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif 'last_pred' in st.session_state:
             st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 25px; border-radius: 10px; border: 2px solid #2196F3; text-align: center;">
                    <h5 style="color: #1565C0; margin:0; text-transform: uppercase; letter-spacing: 1px;">Detected Category</h5>
                    <h1 style="color: #0D47A1; margin: 10px 0; font-size: 3rem;">{st.session_state['last_pred']}</h1>
                </div>
                """, unsafe_allow_html=True)


elif st.session_state['page'] == 'Methodology':
    st.title("🛠️ Methodology & Documentation")
    st.markdown("### How EcoSort Works")
    
    st.markdown("""
    <div class="card">
        <h3>1. Data Acquisition</h3>
        <p>The model was trained on the <b>Garbage Classification Dataset</b> from Kaggle, consisting of thousands of images across 12 distinct categories.</p>
    </div>
    
    <div class="card">
        <h3>2. Preprocessing Pipeline</h3>
        <ul>
            <li><b>Resizing:</b> All input images are resized to a fixed dimension of 224 by 224 pixels to match the input layer of the neural network.</li>
            <li><b>RGB Conversion:</b> Images are ensured to have 3 color channels (Red, Green, Blue).</li>
            <li><b>Normalization:</b> Pixel intensity values (0-255) are scaled down to a range of <b>0 to 1</b> by dividing by 255.0. This helps the model converge faster during training/inference.</li>
        </ul>
    </div>
    
    <div class="card">
        <h3>3. Model Architecture (VGG16)</h3>
        <p>We utilized <b>Transfer Learning</b> with the VGG16 architecture.</p>
        <ul>
            <li><b>Base Model:</b> VGG16 pre-trained on the ImageNet dataset (feature extractor).</li>
            <li><b>Custom Head:</b> A custom classification head was added:
                <ul>
                    <li><code>Flatten</code> layer to serialize the feature map.</li>
                    <li><code>Dense</code> layer with 128 neurons and ReLU activation.</li>
                    <li><code>Output Dense</code> layer with 12 neurons and Softmax activation (for probability distribution across 12 classes).</li>
                </ul>
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


elif st.session_state['page'] == 'Classes':
    st.title("📊 Classification Categories")
    st.markdown("The AI model can identify the following **12 types** of waste:")
    
    col1, col2, col3 = st.columns(3)
    
    # helper for card display
    def class_card(name, emoji):
        return f"""
        <div class="card" style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem;">{emoji}</div>
            <h4 style="margin-top: 10px;">{name}</h4>
        </div>
        """
    
    with col1:
        st.markdown(class_card("Battery", "🔋"), unsafe_allow_html=True)
        st.markdown(class_card("Cardboard", "📦"), unsafe_allow_html=True)
        st.markdown(class_card("Metal", "⚙️"), unsafe_allow_html=True)
        st.markdown(class_card("Shoes", "👟"), unsafe_allow_html=True)

    with col2:
        st.markdown(class_card("Biological", "🥬"), unsafe_allow_html=True)
        st.markdown(class_card("Clothes", "👕"), unsafe_allow_html=True)
        st.markdown(class_card("Paper", "📄"), unsafe_allow_html=True)
        st.markdown(class_card("Trash", "🗑️"), unsafe_allow_html=True)

    with col3:
        st.markdown(class_card("Brown Glass", "🟤"), unsafe_allow_html=True)
        st.markdown(class_card("Green Glass", "🟢"), unsafe_allow_html=True)
        st.markdown(class_card("Plastic", "🥤"), unsafe_allow_html=True)
        st.markdown(class_card("White Glass", "⚪"), unsafe_allow_html=True)


elif st.session_state['page'] == 'Libraries':
    st.title("📚 Libraries & Tools")
    st.markdown("This project relies on the following powerful open-source libraries:")
    
    st.markdown("""
    <div class="card">
        <h4>🧠 TensorFlow & Keras</h4>
        <p>Used for building, training, and running the Deep Learning model (<b>VGG16</b>).</p>
    </div>
    
    <div class="card">
        <h4>🔢 NumPy</h4>
        <p>Essential for numerical operations and handling array manipulations for image data.</p>
    </div>
    
    <div class="card">
        <h4>👁️ OpenCV (cv2)</h4>
        <p>Used for image processing tasks such as resizing and colorspace conversion.</p>
    </div>
    
    <div class="card">
        <h4>🚀 Streamlit</h4>
        <p>The framework used to build this interactive web application entirely in Python.</p>
    </div>
    
    <div class="card">
        <h4>🖼️ Pillow (PIL)</h4>
        <p>Python Imaging Library, used for opening and manipulating image files from the uploader.</p>
    </div>
    """, unsafe_allow_html=True)

