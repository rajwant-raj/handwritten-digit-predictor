# 🧠 Handwritten Digit Recognition using CNN (Enhanced Canvas Version)

An upgraded version of the Handwritten Digit Recognition project (handwritten digit predictor) that uses a Convolutional Neural Network (CNN) trained on the MNIST dataset to identify handwritten digits (0–9).
This new version introduces an interactive drawing canvas, allowing users to draw digits directly instead of uploading images.

Built with TensorFlow, Streamlit, and Streamlit-Draw-Canvas for a smooth real-time experience.



## ✨ What’s New in This Version

🖊️ Draw Instead of Uploading:
Users can now draw digits directly on a canvas using their mouse or touch input.

⚡ Instant Prediction:
As soon as the digit is drawn, the app processes it through the CNN model and shows the prediction with confidence levels.

🎨 Improved User Experience:
Simplified interface for quick testing, better suited for demonstrations or real-time recognition.

🔗 Linked to Previous Version:
This project is an enhanced version of the Handwritten Digit Recognition (Image Upload Version)
.
It reuses the same trained model (my_model.keras) but introduces a new Streamlit interface for interactive drawing.



## 🚀 Features

+ 🧠 Predicts handwritten digits (0–9) in real-time using CNN

+ 🖊️ Interactive canvas for drawing input

+ 📊 Displays prediction confidence and probability chart

+ ✅ Simple and modern Streamlit UI



---


## 📦 Project Overview

+ 🔍 Dataset: MNIST (Keras version)

+ 🧠 Model: CNN with ~98.6% accuracy

+ 🌐 Interface: Streamlit + Streamlit Draw Canvas

+ 📊 Visualization: Confidence bar chart + probability table


---


## 📁 Folder Structure

```

handwritten-digit-recognition/
├── streamlit_app.py                 # Streamlit web app with drawing canvas
├── my_model.keras                        # Trained CNN model (same as previous)
├── handwritten-digit-recognition.ipynb  # Colab notebook (training + evaluation)
├── handwritten-digit-recognition.py     # Python file version of notebook
├── ui-interface.png                     # Screenshot of canvas interface
├──prediction_sample1.png                # Screenshot of sample test run
├──
├── requirements.txt                    # Python dependencies
└── README.md

```

---

## 🧠 Model Architecture

+ 2x Conv2D Layers

+ 2x MaxPooling2D Layers

+ 1x Dense Layer (128 units)

+ 1x Output Layer (Softmax, 10 units)


---


## 📦 Dependencies

+ streamlit
+ tensorflow
+ numpy
+ Pillow
+ streamlit-draw-canvas

Install all dependencies with: 


 ```
pip install -r requirements.txt


```

---




# 🚀 How to Run
## 🔧 Setup

```
git clone https://github.com/your-username/digit-recognition-dl.git
cd digit-recognition-dl
pip install -r requirements.txt

```

## ▶️ Run the Streamlit App

```

streamlit run streamlit_app_canvas.py

```

Then open your browser at http://localhost:8501
and start drawing digits on the canvas!



---


## 📊 Results

+ ✅ Test Accuracy: ~98.6%
  
+ 📉 Visual Outputs:

+ Prediction confidence chart

+ Canvas input preview

+ Probability distribution of all 10 digits



---


## 🔗 Previous Version

If you prefer the image upload version, check out the original project here:  [👉 Handwritten Digit Recognition (Image Upload)](https://github.com/rajwant-raj/handwritten-digit-recognition.git)



---


# 👤 Author

Developed by: Rajwant Raj

GitHub: github.com/rajwant-raj

LinkedIn: linkedin.com/in/rajwant-raj-350519369


---


## ❤️ Made For

A deep learning internship project (at Scalezix) demonstrating CNN-based digit recognition with a modern Streamlit UI — now upgraded for real-time digit drawing.




---







