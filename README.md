# 🚕 Taxi Fare Prediction

A machine learning regression project that predicts the **total fare (`total_amount`)** of a New York City yellow taxi trip from trip details such as distance, passenger count, surcharges, tips, and payment type. Several regression models are trained, tuned, and compared, and the best one is used to generate predictions for the test set.

## 📁 Repository Structure

```
Taxi-Fare-Prediction/
├── solution.ipynb   # Full workflow: EDA, preprocessing, modelling, tuning, prediction
├── train.csv        # Training data (10,000 rows, includes total_amount target)
├── test.csv         # Test data (1,500 rows, no target)
└── README.md
```

## 📊 Dataset

The data looks like NYC TLC yellow taxi trips from June 2023.

| Property | Train | Test |
|---|---|---|
| Rows | 10,000 | 1,500 |
| Columns | 17 | 17 (`id` instead of `total_amount`) |
| Target | `total_amount` | to be predicted |

**Features**

| Column | Description |
|---|---|
| `VendorID` | Taxi vendor / provider code |
| `tpep_pickup_datetime`, `tpep_dropoff_datetime` | Trip start and end timestamps |
| `passenger_count` | Number of passengers |
| `trip_distance` | Distance of the trip |
| `RatecodeID` | Rate code for the trip |
| `store_and_fwd_flag` | Whether the record was stored before being sent to the vendor |
| `PULocationID`, `DOLocationID` | Pickup and drop-off zone IDs |
| `payment_type` | Payment method (e.g. Credit Card, Cash) |
| `extra`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `congestion_surcharge`, `Airport_fee` | Fare components and surcharges |
| `total_amount` | **Target**: total fare charged |

## 🔄 Workflow

### 1. Data Understanding
- Inspected data types, shape, and descriptive statistics.
- Spotted data quality issues, such as zero passenger counts, zero trip distances, negative fare components, and extreme tips.

### 2. Data Cleaning
- **Missing values:** 366 nulls in train and 55 in test, in each of `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, and `Airport_fee`.
  - Numeric columns were imputed with the **median**.
  - Categorical columns were imputed with the **most frequent** value.
  - Imputers were fit on train only and applied to test (no leakage).
- **Duplicates:** none found.
- **Outliers:** checked with boxplots. The heavy outliers in `trip_distance`, `tip_amount`, and `tolls_amount` were **kept**, because there were too many to drop safely.

### 3. Exploratory Data Analysis
- **Fare distribution:** most fares fall between roughly 0 and 100, with a few high-value outliers.
- **Fare vs. passenger count:** average fare rises with passenger count, except for 6 passengers, which likely reflects very few samples.
- **Fare vs. distance:** average fare climbs sharply as trip distance increases.

### 4. Preprocessing
- `StandardScaler` on numeric features: `passenger_count`, `trip_distance`, `congestion_surcharge`, `Airport_fee`, `extra`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, `RatecodeID`, `VendorID`.
- `OneHotEncoder` on categorical features: `payment_type`, `store_and_fwd_flag`.
- Timestamp columns were dropped before modelling.
- **70/30 train/validation split** (`random_state=42`).

### 5. Modelling
Seven regressors were trained and evaluated with the **R² score** on the validation set.

| Model | R² (before tuning) | R² (after tuning) |
|---|---|---|
| **Random Forest Regressor** | **0.927** | 0.890 |
| Decision Tree Regressor (`max_depth=5`) | 0.853 | n/a |
| Linear Regression | 0.852 | n/a |
| Ridge Regression | 0.852 | n/a |
| K-Nearest Neighbors | 0.850 | 0.867 |
| SVR (linear kernel) | 0.838 | 0.838 |
| AdaBoost Regressor | 0.818 | n/a |

Hyperparameter tuning used `GridSearchCV` (5-fold CV, `scoring="r2"`) for Random Forest, SVR, and KNN.

### 6. Results
The **untuned Random Forest Regressor** scored best (**R² ≈ 0.927**) and was used to predict fares for `test.csv`. The tuned Random Forest scored lower on the validation set (0.890), so the base model was kept for the final predictions.

## 🛠️ Tech Stack

- Python 3
- pandas, NumPy
- scikit-learn
- Matplotlib
- Jupyter Notebook

## 🚀 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Tanishq-lalani/Taxi-Fare-Prediction.git
   cd Taxi-Fare-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install pandas numpy scikit-learn matplotlib jupyter
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook solution.ipynb
   ```
   Make sure `train.csv` and `test.csv` are in the same folder as the notebook.

## 📤 Output

The notebook builds a submission DataFrame with two columns:

| id | total_amount |
|---|---|
| 0 | 18.35 |
| 1 | 27.27 |
| ... | ... |

## 🔮 Possible Improvements

- Engineer features from the timestamps, such as trip duration, hour of day, and day of week.
- Use `PULocationID` and `DOLocationID` (zone IDs), which are currently unused.
- Review the use of `tip_amount` as a feature, since it is part of the total fare and may not be known ahead of the trip.
- Handle invalid values (zero distance, negative fees) explicitly.
- Try gradient boosting models such as XGBoost or LightGBM.
- Evaluate with additional metrics like RMSE and MAE, alongside R².

## 👤 Author

**Tanishq Lalani**
GitHub: [@Tanishq-lalani](https://github.com/Tanishq-lalani)
