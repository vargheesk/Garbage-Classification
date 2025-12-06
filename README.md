# ♻️ EcoSort: Intelligent Waste Classification

**EcoSort** is an AI-powered application designed to automate the classification of waste materials. Built with **Streamlit** and **TensorFlow**, it leverages a fine-tuned **VGG16** Deep Learning model to accurately identify garbage into 12 distinct categories.

This project aims to assist in proper waste segregation, promoting recycling efficiency and environmental sustainability.

## 🌟 Features
*   **Real-time Classification**: Instantly identify waste types from uploaded images.
*   **Sample Testing**: Try the model with built-in sample images (Textile, Plastic, etc.).
*   **MNC-Themed UI**: A professional, clean, and responsive user interface.
*   **Comprehensive Documentation**: In-app methodology detailed explanation of the model and preprocessing.
*   **12-Class Support**: Covers a wide range of common waste items.

## 📊 Classification Categories
The model can detect the following classes:
1.  Battery 🔋
2.  Biological 🥬
3.  Brown Glass 🟤
4.  Cardboard 📦
5.  Clothes 👕
6.  Green Glass 🟢
7.  Metal ⚙️
8.  Paper 📄
9.  Plastic 🥤
10. Shoes 👟
11. Trash 🗑️
12. White Glass ⚪

## 🛠️ Tech Stack
*   **Language**: Python
*   **Framework**: Streamlit
*   **Deep Learning**: TensorFlow / Keras (VGG16 Architecture)
*   **Image Processing**: OpenCV (cv2), Pillow (PIL), NumPy

## 💾 Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/vargheesk/Garbage-Classification.git
    cd Garbage-Classification
    ```

2.  **Create a Virtual Environment** (Optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Ensure you have `streamlit`, `tensorflow`, `numpy`, `opencv-python`, and `pillow` installed.*

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run streamlit_app.py
```

The app will open in your default browser at `http://localhost:8501`.

## 🧠 Model Details
*   **Architecture**: VGG16 (Transfer Learning)
*   **Input Size**: 224 x 224 pixels
*   **Training**: Trained on the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) from Kaggle.
*   **Preprocessing**: RGB conversion, resizing, and pixel normalization (1./255).

## 👨‍💻 Developer
Developed for **Luminar Python Project**.

*   [**GitHub Profile**](https://github.com/vargheeskutty)
*   [**Portfolio**](https://vargheeskutty-eldhose.vercel.app/)

---
*Reduce, Reuse, Recycle!* 🌍
