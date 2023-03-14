# Toxicity_prediction
 
## **Model**

The model is a deep neural network built using PyTorch. It uses fully connected layers to extract features from the Morgan fingerprints of each chemical compounds and classify their toxicity. The model is trained using binary cross-entropy loss and optimized with Adam.

## **Usage**

To train the model, simply run the **`train.py`** script with the appropriate arguments. The trained model will be saved in the **`models`** directory. To evaluate the model, run the **`evalu.py`** script with the appropriate arguments. The script will load the trained model and evaluate it on the test set, printing out various evaluation metrics.

## **Requirements**

The code is written in Python 3 and requires the following libraries:

- PyTorch
- NumPy
- Pandas
- Scikit-learn
- rdkit
