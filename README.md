# Brain_Tumor_Detection

# Brain Tumor Detection using Deep Learning

This project uses Convolutional Neural Networks (CNNs) to detect brain tumors in MRI scans. It classifies images as **Tumor** or **No Tumor**, aiming to assist radiologists in quick and accurate diagnosis.

---

## Objectives

- Automatically detect tumors in brain MRI scans
- Reduce diagnostic time and errors
- Provide a deployable deep learning model
- Enable real-time tumor detection

---

## Tech Stack

- Python
- TensorFlow, Keras
- NumPy, Pandas
- OpenCV, Pillow
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab (for training)

---

## Dataset

- The dataset consists of MRI brain images labeled as:
  - `yes`: contains tumor
  - `no`: no tumor
- Located in:  
  `/content/drive/MyDrive/Brain_Tumor_DataSet/datasets/`

---

## Preprocessing

- Resize images to 224x224 pixels
- Normalize pixel values
- One-hot encode labels (Tumor = 1, No Tumor = 0)
- Split into training (80%) and testing (20%) sets

---

## Data Augmentation

To improve generalization and reduce overfitting:
- Rotation (up to 90°)
- Zoom, shear, shift
- Horizontal and vertical flip

Implemented using `ImageDataGenerator`.

---

## Visualizations

- Accuracy and loss curves
- Confusion matrix heatmap
- Flowchart of model pipeline

---

## Model Architecture

```python
model = Sequential([
    Input(shape=(224, 224, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])
