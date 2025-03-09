# Atelier 1 

**Developed by:** Echffani Hodaifa
**Supervised by:** Prof. EL ACHAAK Lotfi

**Executive Summary:** This project demonstrates the application of deep learning techniques to two distinct but related problems: predicting stock prices using regression and classifying data using a deep neural network. The goal is to showcase the power and versatility of deep learning for extracting meaningful insights and creating predictive models across different domains. This project provides a foundation for understanding and applying deep learning to real-world challenges in finance and beyond.

## Part 1: Stock Market Regression Analysis - Predicting Future Stock Prices

**Objective:** To develop a robust deep learning model that accurately forecasts stock closing prices based on historical market data, providing valuable insights for potential investment strategies and risk management.

### Project Overview:

This section focuses on building a regression model to predict stock closing prices using historical data. This involves data preprocessing, model training, evaluation, and visualization to understand model performance and stock trends. Successful prediction can provide a basis for informed investment decisions.

### Dataset:

*   **Source:** Publicly available dataset from Kaggle: [https://www.kaggle.com/datasets/dgawlik/nyse](https://www.kaggle.com/datasets/dgawlik/nyse)
*   **Description:** The dataset consists of historical stock market data with the following key features:
    *   `date`: The date of the stock record (YYYY-MM-DD).
    *   `symbol`: The stock ticker symbol (e.g., AAPL, GOOG).
    *   `open`: The opening price of the stock for the day (in USD).
    *   `close`: The closing price of the stock for the day (in USD) - **Target Variable**.
    *   `low`: The lowest price of the stock during the day (in USD).
    *   `high`: The highest price of the stock during the day (in USD).
    *   `volume`: The number of shares traded during the day.
*   **Data Considerations:** This data represents a time-series; therefore, temporal relationships must be considered during the modeling process.

### Implementation Details:

#### Data Acquisition and Preprocessing:

*   Loading the stock market data from a CSV file.
*   **Handling Missing Values:** Addressing missing data through appropriate imputation techniques (e.g., mean, median, or more sophisticated methods like interpolation). Justification for the chosen imputation method should be provided.
*   **Data Type Conversion:** Ensuring data types are correctly formatted (e.g., date as datetime objects for time-series analysis).
*   **Feature Engineering:** Consider creating new features based on existing data to potentially improve model performance. Examples include:
    *   **Moving Averages:** Calculating moving averages over different time windows (e.g., 5-day, 20-day) to smooth out price fluctuations and identify trends.
    *   **Relative Strength Index (RSI):** A momentum indicator that measures the magnitude of recent price changes to evaluate overbought or oversold conditions in the price of a stock or other asset.
    *   **Moving Average Convergence Divergence (MACD):** Trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.
*   **Feature Scaling:** Applying `StandardScaler` or `MinMaxScaler` to normalize the input features, ensuring all features contribute equally to the model training process. Explain the rationale for choosing a particular scaling method.

#### Model Training:

*   **Data Splitting:** Dividing the dataset into training, validation, and testing sets. A typical split might be 70% training, 15% validation, and 15% testing. Consider using time-based splitting to preserve the temporal order of the data.
*   **Model Architecture:** Defining a deep learning regression model using TensorFlow or PyTorch. A multi-layer perceptron (MLP) is a suitable choice as a starting point. However, explore more advanced architectures:
    *   **Recurrent Neural Networks (RNNs):** Especially Long Short-Term Memory (LSTM) or Gated Recurrent Units (GRUs), which are well-suited for time-series data due to their ability to capture temporal dependencies.
    *   **Convolutional Neural Networks (CNNs):** Can be used to extract patterns from the time-series data, particularly when combined with feature engineering.
*   **Loss Function and Optimizer:** Using Mean Squared Error (MSE) as the loss function and Adam or RMSprop as the optimizer.
*   **Training Process:** Training the model on the training data and monitoring performance on the validation set.
*   **Early Stopping:** Implementing early stopping with a patience parameter to prevent overfitting by monitoring the validation loss.
*   **Hyperparameter Tuning:** Experimenting with different hyperparameters (e.g., learning rate, batch size, number of layers, number of neurons per layer) using techniques like grid search or random search to optimize model performance.

#### Evaluation:

*   Evaluating the trained model on the test set using metrics such as:
    *   **Mean Squared Error (MSE):** Average squared difference between predicted and actual values.
    *   **Root Mean Squared Error (RMSE):** Square root of the MSE, providing a more interpretable error value in the original unit.
    *   **R-squared (R²):** Proportion of variance in the dependent variable that can be predicted from the independent variables.
    *   **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual values, providing a more robust measure of error compared to MSE when outliers are present.
*   Visualizing predicted vs. actual closing prices on a plot. Create additional visualizations to analyze residuals and identify potential biases in the model's predictions.

### Results and Discussion:

*   Provide a clear and concise summary of the model's performance on the test set, including the values of the evaluation metrics (MSE, RMSE, R², MAE).
*   Discuss the strengths and weaknesses of the model based on the evaluation results and visualizations.
*   Analyze the model's predictions and identify potential reasons for any discrepancies between predicted and actual prices.

### Potential Improvements and Future Directions:

*   **Feature Engineering:**
    *   Incorporate more technical indicators (e.g., Bollinger Bands, Fibonacci Retracements).
    *   Include sentiment analysis from news articles or social media to capture market sentiment.
    *   Add macroeconomic indicators (e.g., GDP growth, interest rates, inflation) to account for broader economic factors.
*   **Model Architecture:**
    *   Experiment with more complex model architectures like Transformers, which have shown excellent performance in sequence modeling tasks.
    *   Use ensemble methods (e.g., Random Forests, Gradient Boosting) to combine multiple models and improve prediction accuracy.
*   **Hyperparameter Optimization:**
    *   Employ more sophisticated hyperparameter optimization techniques like Bayesian optimization to efficiently search the hyperparameter space.
*   **Regularization:**
    *   Apply regularization techniques (e.g., L1, L2 regularization, dropout) to prevent overfitting and improve the model's generalization ability.
*   **Data Preprocessing:**
    *   Explore different data scaling methods and their impact on model performance.
    *   Investigate the use of wavelet transforms for denoising the time-series data.
*   **Backtesting:** Implement a backtesting strategy to evaluate the model's performance on historical data and simulate real-world trading scenarios.
*   **Risk Management:** Incorporate risk management strategies into the model to limit potential losses.

## Part 2: Deep Learning Classification Project - Predictive Maintenance Classification

**Objective:** To develop a deep learning model that can accurately classify machine conditions to predict potential failures and improve maintenance scheduling.

### Project Overview:

This project aims to build a deep learning model for classifying data into distinct categories. The notebook provides a comprehensive workflow, from data preparation to model evaluation and interpretation.

### Dataset:

*   **Source:** Publicly available dataset from Kaggle: [https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)
*   **Description:** This dataset contains sensor readings, machine parameters, and failure labels, enabling the development of a predictive maintenance model. The goal is to classify whether a machine is likely to fail based on these features.
*   **Key Features:** Analyze the dataset to identify the most relevant features for predicting machine failure. This may involve feature importance analysis.

### Implementation Details:

#### Data Acquisition and Preprocessing:

*   **Data Loading:** Loading the data from CSV files.
*   **Handling Missing Values:** Address missing data using appropriate imputation techniques.
*   **Feature Selection:** Identify and select the most relevant features for classification. This can involve techniques like:
    *   **Correlation Analysis:** Removing highly correlated features to reduce redundancy.
    *   **Feature Importance:** Using tree-based models (e.g., Random Forest, Gradient Boosting) to determine feature importance.
    *   **Univariate Feature Selection:** Selecting features based on statistical tests (e.g., chi-squared test, ANOVA).
*   **Data Scaling:** Apply appropriate scaling techniques (e.g., StandardScaler, MinMaxScaler) to normalize the input features.
*   **Data Splitting:** Divide the data into training, validation, and testing sets.

#### Model Architecture:

*   **Model Type:** Convolutional Neural Network (CNN) is a good starting point. However, consider other architectures:
    *   **Multi-Layer Perceptron (MLP):** A simple feedforward neural network that can be used as a baseline model.
    *   **Recurrent Neural Networks (RNNs):** Useful if the data contains temporal dependencies.
    *   **Hybrid Models:** Combining different architectures to leverage their respective strengths.
*   **Layers:** The model typically includes convolutional layers, max-pooling layers, and fully connected (dense) layers. ReLU activation functions are commonly used. The final layer uses a softmax activation for multi-class classification.
*   **Optimizer:** Adam.
*   **Loss Function:** Categorical cross-entropy.

#### Training and Evaluation:

*   **Training Process:** Train the model using a specific batch size and number of epochs.
*   **Validation:** Use validation data to monitor performance and prevent overfitting using techniques like early stopping.
*   **Evaluation Metrics:** Evaluate the model's performance on the test set, reporting:
    *   **Accuracy:** Overall correctness of the model's predictions.
    *   **Precision:** Ability of the model to correctly identify positive cases.
    *   **Recall:** Ability of the model to identify all positive cases.
    *   **F1-Score:** Harmonic mean of precision and recall, providing a balanced measure of performance.
    *   **Confusion Matrix:** Visualize the model's performance by showing the number of true positives, true negatives, false positives, and false negatives.
    *   **AUC-ROC Curve:** Plot the Receiver Operating Characteristic (ROC) curve and calculate the Area Under the Curve (AUC) to assess the model's ability to discriminate between classes.

### Results and Discussion:

*   Provide a clear and concise summary of the model's performance on the test set, including the values of the evaluation metrics (accuracy, precision, recall, F1-score, AUC).
*   Discuss the strengths and weaknesses of the model based on the evaluation results and visualizations.
*   Analyze the confusion matrix to identify potential areas for improvement.

### Potential Improvements and Future Directions:

*   **Data Augmentation:** Implement data augmentation techniques (e.g., random rotations, flips, zooms) to increase the size and diversity of the training data.
*   **Model Architecture:**
    *   Explore different CNN architectures (e.g., ResNet, Inception).
    *   Experiment with different numbers of layers and units in the model.
    *   Add dropout layers to prevent overfitting.
*   **Hyperparameter Tuning:**
    *   Tune the learning rate, batch size, and other hyperparameters using techniques like grid search or random search.
*   **Ensemble Methods:**
    *   Combine multiple models using ensemble methods (e.g., voting, stacking) to improve prediction accuracy.
*   **Explainable AI (XAI):** Use techniques like SHAP or LIME to explain the model's predictions and identify the most important features.
*   **Cost-Sensitive Learning:** Incorporate the cost of misclassification into the model training process to minimize the overall cost of errors. This is particularly important in predictive maintenance where the cost of a false negative (missing a failure) may be much higher than the cost of a false positive (unnecessary maintenance).
*   **Online Learning:** Implement an online learning algorithm to continuously update the model as new data becomes available.

## Conclusion:

This project demonstrates the application of deep learning techniques to stock market prediction and predictive maintenance classification. By exploring different model architectures, data preprocessing techniques, and evaluation metrics, this project provides a solid foundation for further research and development in these areas. The potential improvements and future directions outlined in this document offer a roadmap for enhancing the performance and applicability of these models in real-world scenarios.
