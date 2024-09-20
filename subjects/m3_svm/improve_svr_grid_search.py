import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error

from dataset_man import dataset_manager

auto_insurance_data = dataset_manager.load_dataset('m3.svm.20240914.auto-insurance', read_file_options={'names': ['n_claims', 'total_payment']})

df = pd.DataFrame(auto_insurance_data)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(df['n_claims'].values.reshape(-1, 1))
y = scaler_y.fit_transform(df['total_payment'].values.reshape(-1, 1))

test_size = 0.3
random_state = 1
is_shuffle = True
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

# Define the parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.5, 1],
    'kernel': ['rbf', 'linear', 'poly']
}

# Initialize SVR
svr = SVR()

# Initialize GridSearchCV
grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best Parameters
print("Best Parameters:", grid_search.best_params_)

# Best Estimator
best_svr = grid_search.best_estimator_

# Predict on Test Set
y_pred_scaled = best_svr.predict(X_val)

# Inverse Transform Predictions
# y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
# y_test_original = scaler_y.inverse_transform(y_val.reshape(-1, 1)).ravel()

# Evaluate

mae = mean_absolute_error(y_val, y_pred_scaled)
mse = mean_squared_error(y_val, y_pred_scaled)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
