# Face Detection using TensorFlow

This project demonstrates face detection using a custom-built TensorFlow model. The model is trained on a dataset of images collected and annotated using OpenCV and LabelMe.

## Project Overview

This project covers the entire process of building a face detection model, from data collection and augmentation to model training and real-time detection.  Key steps include:

1. **Data Acquisition:**  Images are captured using a webcam and annotated with bounding boxes using the LabelMe annotation tool.
2. **Data Augmentation:** Albumentations library is used to augment the training data, increasing the dataset size and improving model robustness.
3. **Model Architecture:**  A custom model is built using TensorFlow's Functional API, leveraging a pre-trained VGG16 model as a feature extractor. The model consists of two branches: one for classification (face or no face) and one for bounding box regression.
4. **Model Training:** The model is trained using a custom training loop and Adam optimizer.  Training progress is monitored using TensorBoard.
5. **Prediction and Evaluation:** The trained model is used to make predictions on a test set, and the performance is visually evaluated.
6. **Real-time Detection:** The model is integrated into a real-time face detection application using OpenCV, enabling live detection from a webcam feed.

## Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/pdobariya1/Face_Detection_Using_Tensorflow.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
## Data Collection and Annotation

1. Run the `Face_Detection.ipynb` notebook. The initial section will guide you through collecting images using your webcam.
2.  Use the `labelme` tool (installed via `pip install labelme`) to annotate the collected images with bounding boxes around the faces.  The script will automatically move the generated JSON annotation files to the correct subfolders.

## Data Augmentation and Training

The notebook then performs data augmentation using Albumentations. The augmented data is used to train the custom TensorFlow model.  The training progress and results are visualized using TensorBoard and Matplotlib.

## Real-time Face Detection

After training, the notebook demonstrates real-time face detection using OpenCV, displaying the results from the webcam feed.

## Running the Notebook

The entire workflow, from data collection to real-time detection, is contained within the `Face_Detection.ipynb` notebook. Execute the notebook cells sequentially to reproduce the results.


## Directory Structure

```
Face_Detection_Using_Tensorflow/
├── Face_Detection.ipynb
└── requirements.txt
└── data/
    ├── images/
    └── labels/
└── aug_data/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/

```

Remember to create the `data` and `aug_data` directories before running the notebook.  The `aug_data` directory will be created by the script.
