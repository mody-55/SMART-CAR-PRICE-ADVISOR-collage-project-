# Smart Car Price Advisor

A beginner-level Machine Learning project that predicts the expected price of a used car and compares it with the current advertisement price.

## Project Idea

The user enters the car information and the current advertisement price.

The model predicts the expected price and the application gives a simple recommendation:

- Good Deal
- Fair Price
- Overpriced

## Dataset

Use the selected Kaggle Car Price Dataset.

Place the CSV file here:

dataset/car_data.csv

The CSV should contain these columns:

- Brand
- Model
- Year
- Engine_Size
- Fuel_Type
- Transmission
- Mileage
- Doors
- Owner_Count
- Price

Price is the target column.

## Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Machine Learning

We compare three simple regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The project uses:

- train_test_split
- Pipeline
- ColumnTransformer
- StandardScaler
- OneHotEncoder
- mean_squared_error
- R2 score

The model with the highest R2 score is saved.

## Preprocessing

Numerical columns are scaled using StandardScaler.

Categorical columns are encoded using OneHotEncoder.

The preprocessing and model are kept together inside a Pipeline.

## How To Run

Install the libraries:

```bash
pip install -r requirements.txt
```

Train the models:

```bash
python train.py
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Smart-Car-Price-Advisor/
│
├── dataset/
│   ├── car_data.csv
│   └── README.txt
│
├── model/
│   └── car_price_model.pkl
│
├── train.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Important

Run `train.py` before `app.py`.

The trained model will be created automatically inside the model folder.
