# Ultrasound-based Stage Classification of CKD

This project focuses on the classification of chronic kidney disease (CKD) stages using ultrasound images. We utilize deep learning models, specifically ResNet and EfficientNet, to perform the classification task. Additionally, the project incorporates Explainable AI (XAI) techniques to provide insights into the model's decision-making process.

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Model Training](#model-training)
- [XAI Support](#xai-support)
- [Results](#results)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Introduction
Chronic kidney disease (CKD) is a significant health concern that can lead to kidney failure. Early detection and classification of CKD stages are crucial for effective treatment. This project aims to leverage deep learning techniques to automate the classification process using ultrasound images.

## Technologies Used
- Python
- PyTorch
- Jupyter Notebook
- Explainable AI (XAI) techniques

## Getting Started
To get a local copy of the project up and running, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/magnabenita/Ultrasound-based-Stage-Classification-of-CKD-using-ResNet-and-EfficientNet-with-XAI-support.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Ultrasound-based-Stage-Classification-of-CKD-using-ResNet-and-EfficientNet-with-XAI-support
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
To use the models for CKD stage classification, run the following command in a Jupyter Notebook or Python script:
```python
# Example code to load the model and classify an image
import torch
from model import load_model, classify_image

model = load_model('path_to_model.pth')
result = classify_image(model, 'path_to_ultrasound_image.jpg')
print("Predicted CKD Stage:", result)
```

## Model Training
To train the models, use the provided training scripts. Ensure you have the dataset in the correct format. Run the training script as follows:
```bash
python train.py --data_dir path_to_data --model_type resnet
```

## XAI Support
The project includes XAI techniques to interpret the model's predictions. Use the following command to visualize the model's decision-making process:
```python
# Example code for XAI visualization
from xai import explain_prediction

explanation = explain_prediction(model, 'path_to_ultrasound_image.jpg')
visualize_explanation(explanation)
```

## Results
The performance of the models is evaluated using standard metrics. Results are provided in the `results` directory, including accuracy, precision, recall, and F1-score.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## Acknowledgments
- [ResNet](https://arxiv.org/abs/1512.03385)
- [EfficientNet](https://arxiv.org/abs/1905.11946)
- [Explainable AI Techniques](https://arxiv.org/abs/2001.00149)
- [Dataset](https://universe.roboflow.com/ckd)
