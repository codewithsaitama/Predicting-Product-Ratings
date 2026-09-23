# Predicting-Product-Ratings
Machine learning project that predicts product ratings using customer experience factors with Scikit-learn and Random Forest, featuring an interactive Streamlit dashboard.
# Product Rating Prediction using Scikit-learn

A machine learning project that predicts product ratings on a scale of 1 to 5 using product quality, value for money, delivery experience, customer support, and verified purchase information.

The project includes model training, evaluation, single-product prediction, batch catalog prediction, feature importance analysis, and an interactive Streamlit web application.

## Project Overview

Customer ratings are influenced by several aspects of the overall purchasing experience. This project uses a supervised machine learning approach to estimate a product's expected rating based on key customer experience metrics.

The model is built using Scikit-learn's `RandomForestRegressor` and trained on a synthetic dataset containing 500 product records.

### Input Features

The model uses the following features:

* `product_quality` — Product quality score from 1 to 10
* `value_for_money` — Perceived value for money score from 1 to 10
* `delivery_rating` — Delivery and packaging score from 1 to 10
* `customer_support` — Customer support responsiveness score from 1 to 10
* `verified_purchase` — Whether the review is associated with a verified purchase, represented as 0 or 1

### Target

* `product_rating` — Predicted product rating between 1 and 5

## Machine Learning Model

The project uses:

```text
RandomForestRegressor
n_estimators = 200
random_state = 42
```

The dataset is split into:

```text
80% Training Data
20% Testing Data
```

The model is evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R² Score

The current application reports the following evaluation results:

```text
R² Score: 0.8172
MAE: 0.12
RMSE: approximately 0.22
```

These metrics are based on the current dataset and train/test split and should not be interpreted as real-world performance.

## Dataset

The project uses a synthetic dataset containing 500 product records.

Dataset columns:

| Feature           | Description                            |
| ----------------- | -------------------------------------- |
| product_quality   | Product quality score from 1–10        |
| value_for_money   | Value for money score from 1–10        |
| delivery_rating   | Delivery and packaging score from 1–10 |
| customer_support  | Customer support score from 1–10       |
| verified_purchase | Verified purchase indicator            |
| product_rating    | Target product rating from 1–5         |

The dataset is included for educational and demonstration purposes.

## Streamlit Application

The project includes an interactive Streamlit dashboard with three main sections.

### 1. Product Experience Estimator

Users can enter product and service ratings through interactive controls and generate an estimated product rating.

The application provides:

* Product quality input
* Value for money input
* Delivery and packaging input
* Customer support input
* Verified purchase selection
* Predicted rating
* Rating classification
* Satisfaction insights
* Raw feature display

### 2. Batch Catalog Rating Prediction

Users can upload a CSV containing multiple products and generate predictions for the entire catalog.

The application can:

* Upload a product catalog
* Validate required columns
* Predict ratings for multiple products
* Assign rating tiers
* Display catalog statistics
* Filter predictions by rating tier
* Download the predicted catalog as a CSV file

Required CSV columns:

```text
product_quality
value_for_money
delivery_rating
customer_support
verified_purchase
```

### 3. Model Diagnostics

The dashboard also provides model information and feature importance visualization.

The feature importance chart is generated automatically during model training and saved as:

```text
feature_importance.png
```

## Project Structure

```text
Product_Rating_Prediction_Sklearn/
│
├── app.py
├── predict.py
├── train_model.py
├── product_rating_model.pkl
├── feature_importance.png
├── requirements.txt
│
└── data/
    └── product_ratings.csv
```

### File Description

| File                       | Description                                             |
| -------------------------- | ------------------------------------------------------- |
| `app.py`                   | Streamlit web application                               |
| `train_model.py`           | Trains, evaluates, and saves the machine learning model |
| `predict.py`               | Runs a prediction on a sample product                   |
| `product_rating_model.pkl` | Saved trained Random Forest model                       |
| `feature_importance.png`   | Feature importance visualization                        |
| `data/product_ratings.csv` | Synthetic training dataset                              |
| `requirements.txt`         | Python dependencies                                     |

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Product_Rating_Prediction_Sklearn.git
```

Move into the project directory:

```bash
cd Product_Rating_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

To train the model from scratch:

```bash
python train_model.py
```

The training script will:

1. Load the product rating dataset.
2. Separate features and target variables.
3. Split the data into training and testing sets.
4. Train a Random Forest regression model.
5. Calculate MAE, RMSE, and R² Score.
6. Save the trained model as `product_rating_model.pkl`.
7. Generate the feature importance chart.

## Run a Sample Prediction

After training the model:

```bash
python predict.py
```

The script loads the trained model and predicts the rating for a sample product.

Example output:

```text
Predicted Product Rating: 5.00 / 5
```

The exact output may vary if the model or dataset is changed.

## Run the Streamlit Application

Start the web application using:

```bash
streamlit run app.py
```

Streamlit will provide a local URL where the application can be accessed in your browser.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest Regression
* Joblib
* Matplotlib
* Streamlit

## Machine Learning Workflow

```text
Dataset
   |
   v
Data Loading
   |
   v
Feature / Target Separation
   |
   v
Train-Test Split
   |
   v
Random Forest Regression
   |
   v
Model Evaluation
   |
   +------> MAE
   |
   +------> RMSE
   |
   +------> R² Score
   |
   v
Save Trained Model
   |
   v
Prediction
   |
   v
Streamlit Application
```

## Example Prediction

Example input:

```text
Product Quality:       9
Value for Money:       8
Delivery Rating:       9
Customer Support:      8
Verified Purchase:     Yes
```

The trained model uses these features to estimate the expected product rating.

## Important Note

This project uses a synthetic dataset and is intended for learning, experimentation, and demonstration purposes.

The model's performance on this dataset does not represent its expected performance on real-world e-commerce data. For production use, the model should be trained and validated using a sufficiently large and representative real-world dataset.

## Future Improvements

Potential improvements include:

* Training on real customer review datasets
* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Feature engineering
* Handling missing values and outliers
* Comparing Random Forest with other regression algorithms
* Adding model explainability using SHAP
* Adding prediction confidence or uncertainty estimates
* Deploying the Streamlit application online
* Adding automated model retraining
* Connecting the application to a database
* Adding authentication and user management
* Creating REST API endpoints for predictions

## License

This project is intended for educational and demonstration purposes. Add an appropriate open-source license if you plan to distribute or reuse the project publicly.
