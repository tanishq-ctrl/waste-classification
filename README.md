# 🗑️ Waste Management Classification System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<div align="left">
  <img src="https://img.shields.io/badge/Accuracy-98%25-success"/>
  <img src="https://img.shields.io/badge/Status-Active-success"/>
</div>

---

A deep learning-based web application that classifies different types of waste materials using computer vision. The system helps in proper waste segregation by identifying whether an item belongs to categories like cardboard, glass, metal, paper, plastic, or trash.

![Waste Management demo]()
<div align="center">
  <h3>🎯 Categories</h3>
  <code>Cardboard</code> • <code>Glass</code> • <code>Metal</code> • <code>Paper</code> • <code>Plastic</code> • <code>Trash</code>
</div>

## ✨ Features

- 🚀 Real-time waste classification using deep learning
- 🌐 Web-based user interface for easy interaction
- 📸 Support for common image formats (PNG, JPG, JPEG)
- ⚡ Instant classification results with visual feedback

## 🛠️ Technology Stack

- **Backend**: 
  - ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) 
  - ![Flask](https://img.shields.io/badge/Flask-000000?style=flat&logo=flask&logoColor=white)
- **Deep Learning**: 
  - ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
  - ![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
- **Frontend**: 
  - ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)
  - ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)

## 🧠 Model Architecture

The waste classification model is built using transfer learning with MobileNetV2 as the base model:

1. 🏗️ **Base Model**: Pre-trained MobileNetV2 on ImageNet
2. 🔄 **Fine-tuning**: Last 50 layers unfrozen for training
3. ➕ **Additional Layers**:
   ```
   ├── Global Average Pooling
   ├── Dense Layer (128 units, ReLU)
   ├── Dropout (0.6)
   └── Output Layer (6 units, Softmax)
   ```

## 📊 Model Training

- 🖼️ **Input Image Size**: 128x128 pixels
- 📦 **Batch Size**: 32
- 🎯 **Training Strategy**:
  ```
  ├── Data augmentation (rotation, shift, shear, zoom, flip)
  ├── Learning rate scheduling with exponential decay
  ├── L2 regularization
  └── Class weight balancing
  ```
- 📈 **Training Results**:
  - Training Accuracy: ![98%](https://img.shields.io/badge/98%25-success)
  - Validation Accuracy: ![75%](https://img.shields.io/badge/75%25-yellow)

## 📁 Project Structure

```
waste_management/
├── 🌐 app.py                 # Flask application
├── 🛠️ utils.py              # Utility functions
├── 📓 WASTE_MANAGEMENT.ipynb # Model training notebook
├── 📂 static/
│   ├── 🎨 css/
│   │   └── style.css        # Custom styling
│   └── 📤 uploads/          # Image upload directory
└── 📂 templates/
    ├── 🏠 index.html        # Home page
    └── 📊 result.html       # Results page
```

## 🚀 Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd waste_management
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5000`

## 📱 Usage

1. 🌐 Access the web interface through your browser
2. 📤 Upload an image of the waste item you want to classify
3. ✨ Click submit to get the classification result
4. 📊 View the predicted category and confidence score

## 🔬 Model Training Process

The model was trained using transfer learning on MobileNetV2:

1. 📥 **Data Preparation**:
   ```
   ├── Dataset split: 80% training, 20% validation
   ├── Image resizing to 128x128 pixels
   └── Data augmentation for better generalization
   ```

2. ⚙️ **Training Configuration**:
   ```
   ├── Optimizer: Adam with learning rate scheduling
   ├── Loss function: Categorical Cross-entropy
   ├── Metrics: Accuracy
   └── Epochs: 50
   ```

3. 🎯 **Performance Optimization**:
   ```
   ├── Dropout for reducing overfitting
   ├── L2 regularization
   └── Class weight balancing
   ```

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is licensed under the [MIT License](LICENSE)

---

<div align="center">
  Made with ❤️ for a cleaner 🌍
</div> 
