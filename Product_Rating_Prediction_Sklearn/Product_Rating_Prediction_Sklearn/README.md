# Product Rating Prediction using Python and Scikit-learn

## Project Overview
This machine learning project predicts a product rating from 1 to 5 using product quality, value for money, delivery rating, customer support, and verified purchase information.

## Technology
- Python
- Pandas
- Scikit-learn
- Random Forest Regressor
- Joblib
- Matplotlib

## Files
- `train_model.py` - trains and evaluates the model
- `predict.py` - predicts a rating for a sample product
- `product_rating_model.pkl` - trained model
- `data/product_ratings.csv` - sample synthetic dataset
- `feature_importance.png` - feature importance chart
- `requirements.txt` - required libraries

## Run
```bash
pip install -r requirements.txt
python train_model.py
python predict.py
streamlit run app.py
```

## Output
The program predicts a product rating between 1 and 5.

Note: The dataset is synthetic and intended for educational purposes only.
