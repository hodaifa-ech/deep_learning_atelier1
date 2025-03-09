# Atelier 1 Deep Learning

Written by ID LAHCEN El Mahdi
& Supervised by Prof. EL ACHAAK Lotfi

# Deep Learning Project: Stock Market Prediction and Classification

This project encompasses two distinct parts: Regression analysis applied to stock market data for price prediction and a deep learning classification project. Both leverage the power of deep learning to extract insights and build predictive models.

## Part 1: Stock Market Regression Analysis

This section focuses on predicting stock closing prices using historical data and deep learning regression techniques.

dataset link: https://www.kaggle.com/datasets/dgawlik/nyse

### Overview

The goal is to develop a regression model that accurately predicts the closing price of a stock based on its historical performance.  This involves data preprocessing, model training, evaluation, and visualization to understand model performance and stock trends.

### Dataset

The dataset consists of historical stock market data, with the following key features:

*   **date:** The date of the stock record.  (YYYY-MM-DD Format)
*   **symbol:** The stock ticker symbol (e.g., AAPL, GOOG).
*   **open:** The opening price of the stock for the day (in USD).
*   **close:** The closing price of the stock for the day (in USD) - **Target Variable**.
*   **low:** The lowest price of the stock during the day (in USD).
*   **high:** The highest price of the stock during the day (in USD).
*   **volume:** The number of shares traded during the day.

The data is commonly available from online financial data providers. Data cleaning, such as handling missing values, is crucial before training the model.

### Implementation

1.  **Data Acquisition and Preprocessing:**
    *   Loading the stock market data from a CSV.
    *   Handling missing values using techniques like imputation (filling with the mean or median).
    *   Ensuring correct data types (e.g., date as datetime).
    *   Feature scaling using `StandardScaler` to normalize the input features.

2.  **Model Training:**
    *   Splitting the data into training, validation, and testing sets.
    *   Defining a deep learning regression model using TensorFlow or PyTorch. A multi-layer perceptron (MLP) is a suitable choice.
    *   Using Mean Squared Error (MSE) as the loss function and Adam as the optimizer.
    *   Training the model on the training data and monitoring performance on the validation set.
    *   Implementing early stopping to prevent overfitting.

3.  **Evaluation:**
    *   Evaluating the trained model on the test set using metrics such as:
        *   **Mean Squared Error (MSE):**  Average squared difference between predicted and actual values.
        *   **Root Mean Squared Error (RMSE):** Square root of the MSE, providing a more interpretable error value in the original unit.
        *   **R-squared (R²):** Proportion of variance in the dependent variable that can be predicted from the independent variables.
    *   Visualizing predicted vs. actual closing prices on a plot.

### Results

The model was trained on historical stock data. The performance was evaluated based on the test set.

The visualizations show general trends and the model's ability to capture price movements.

**Potential Improvements:**

*   Incorporate more features, such as technical indicators.
*   Experiment with different model architectures, such as LSTMs.
*   Fine-tune hyperparameters using techniques like grid search.

## Part 2: Deep Learning Classification Project

This section covers the implementation of a deep learning model for a classification task.

dataset link: https://www.kaggle.com/datasets/shivamb/machine-predictive-
maintenance-classification

### Overview

This project aims to develop a deep learning model to classify data into distinct categories.  The notebook provides a comprehensive workflow, from data preparation to model evaluation and interpretation.

### Dataset

*   **Description:**  The dataset consists of labeled data suitable for classification.  Examples include image datasets (like MNIST or CIFAR-10) or text datasets for sentiment analysis.
*   **Preprocessing:**
    *   **Normalization:** Scaling pixel values (for images) to a range between 0 and 1.
    *   **Data Splitting:** Dividing the data into training, validation, and test sets.
    *   **Data Augmentation (if applicable):** Applying transformations to the training data (e.g., rotations, flips) to increase the dataset size.

### Model Architecture

*   **Type:** Convolutional Neural Network (CNN).
*   **Layers:**  The model typically includes convolutional layers, max-pooling layers, and fully connected (dense) layers.  ReLU activation functions are commonly used. The final layer uses a softmax activation for multi-class classification.
*   **Optimizer:** Adam.
*   **Loss Function:** Categorical cross-entropy.
*   **Evaluation Metrics:** Accuracy, precision, recall, and F1-score.

### Training and Evaluation

*   **Training Process:** The model is trained using a specific batch size and number of epochs.
*   **Validation:** Validation data is used to monitor performance and prevent overfitting using techniques like early stopping.
*   **Results:** The model's performance is evaluated on the test set, reporting accuracy, precision, recall and F1-score.

**Potential Improvements:**

*   Add more layers or increase the number of units in existing layers.
*   Use a different model architecture (e.g., ResNet).
*   Try different optimizers or learning rates.
*   Implement data augmentation techniques.
