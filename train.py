import os, random, argparse, datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm
import joblib
import xgboost as xgb
from tqdm.notebook import tqdm


# Custom progress bar callback
class ProgressBarCallback(xgb.callback.TrainingCallback):
    def __init__(self, expected_iterations):
        super().__init__()
        self.progress_bar = tqdm(total=expected_iterations, desc="Training XGBoost")

    def after_iteration(self, model, epoch, evals_log):
        self.progress_bar.update(1)

    def close(self):
        self.progress_bar.close()

def fixed_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def Data_Load(seed, path):
    # Load and preprocess training data
    train_data = []
    train_files = os.listdir(os.path.join(path, 'TRAIN_DATA'))
    for file in train_files:
        df = pd.read_csv(f"./TRAIN_DATA/{file}")
        train_data.append(df)
    train_data = pd.concat(train_data, ignore_index=True)


    # Feature engineering and preprocessing
    train_data["일시"] = pd.to_datetime(train_data["일시"], format="%m-%d %H:%M", errors='coerce')
    train_data["day_of_year"] = train_data["일시"].dt.dayofyear
    train_data["hour"] = train_data["일시"].dt.hour
    train_data.drop("일시", axis=1, inplace=True)
    
    return train_data


def Data_Preprocessing(data, seed, path):
    null_rows = data["day_of_year"].isna()

    for i in data[null_rows].index:
        data['day_of_year'] = i // 24

    null_rows = data["hour"].isna()

    for i in data[null_rows].index:
        data['hour'] = (i + 1) % 24
    
    # One-hot encoding for the '측정소' column
    encoder = OneHotEncoder(sparse=False)
    encoded_station = encoder.fit_transform(data["측정소"].values.reshape(-1, 1))
    station_columns = [f"station_{i}" for i in range(encoded_station.shape[1])]
    encoded_station_df = pd.DataFrame(encoded_station, columns=station_columns)
    data = pd.concat([data.drop("측정소", axis=1), encoded_station_df], axis=1)


    def create_rolling_windows(data, input_hours, output_hours):
        X, y = [], []
        for i in range(0, len(data) - input_hours - output_hours + 1):
            X.append(data[i:i + input_hours])
            y.append(data[i + input_hours:i + input_hours + output_hours, 0])  # Assuming PM2.5 is the first column
        return np.array(X), np.array(y)

    input_hours = 48
    output_hours = 72
    PM25_data = data[["PM2.5"]].values
    X, y = create_rolling_windows( PM25_data, input_hours, output_hours)
    X = X.reshape(X.shape[0], -1)
    
    return X, y


def train(X, y, seed, ratio):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
    
    params = {
    "objective": "reg:squarederror",
    "eval_metric": "mae",
    "eta": 0.1,
    "max_depth": 6,
    "n_estimators": 100,
    "random_state": 42,
    "verbosity": 0,
    }

    # Prepare the DMatrix objects for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Set up a watchlist for monitoring the learning process
    watchlist = [(dtrain, 'train'), (dtest, 'test')]

    num_boost_round = 1000
    progress_bar_callback = ProgressBarCallback(expected_iterations=num_boost_round)

    # Train the XGBoost model with custom progress bar
    model = xgb.train(
        params,
        dtrain,
        num_boost_round,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=False,
        callbacks=[progress_bar_callback],
    )

    # Close the progress bar
    progress_bar_callback.close()
    
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    print(f"Train Mean Absolute Error (MAE): {mae:.2f}")
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Val Mean Absolute Error (MAE): {mae:.2f}")
    
    return model



def save_model(model, path):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file = os.path.join(path, f"xgboost_model_{timestamp}.pkl") 
    joblib.dump(model, model_file)
    print(f"Model saved. Save to: {model_file}")
    
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default=os.getcwd(), help='input your absolute path.')
    parser.add_argument('--seed', type=int, default=2023, help='input your seed number.')
    parser.add_argument('--ratio', type=float, default=0.2, help='input your train test split ratio.')
    CFG = parser.parse_args()
    
    path = CFG.path; seed = CFG.seed; ratio = CFG.ratio
    print(f"your absolute path: {path}")
    
    fixed_seed(seed)
    print(f"your seed number: {seed}")
    
    data = Data_Load(seed, path)
    X, y = Data_Preprocessing(data, seed, path)
    model = train(X, y, seed, ratio)
    save_model(model, path)
    
if __name__ == "__main__":
    main()
    
    
    

