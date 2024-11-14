# ML Model CI/CD Pipeline

This repository contains a simple CNN model for MNIST classification with a complete CI/CD pipeline.

## Local Setup

1. Clone the repository: 
```
git clone https://github.com/theschoolofai/S5MNIST_CICD_Test.git
cd YOUR_REPO
```
2. Create a virtual environment and activate it:
```
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Train the model:
```
python train.py
```

5. Run tests:   
```
pytest tests/
```
## CI/CD Pipeline

The pipeline automatically:
1. Trains the model
2. Validates model architecture
3. Checks model accuracy
4. Saves the trained model as an artifact

## Model Details

- Input: 28x28 grayscale images
- Architecture: 2 Conv layers + 2 FC layers
- Output: 10 classes (digits 0-9)
- Parameters: < 100,000
- Target Accuracy: > 80%
