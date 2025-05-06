import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("weatherAUS.csv")

# Drop columns with too many missing values
df.dropna(axis=1, thresh=len(df) * 0.5, inplace=True)

# Drop rows with any remaining nulls
df.dropna(inplace=True)

# Encode target
df['RainTomorrow'] = df['RainTomorrow'].map({'No': 0, 'Yes': 1})

# Balance the dataset
class_counts = df['RainTomorrow'].value_counts()
min_class = class_counts.min()
df_balanced = pd.concat([
    df[df['RainTomorrow'] == 0].sample(n=min_class, random_state=42),
    df[df['RainTomorrow'] == 1].sample(n=min_class, random_state=42)
])

# Drop unnecessary columns
df_balanced.drop(columns=['Date', 'Location', 'RainToday'], inplace=True, errors='ignore')

# Separate features/target
X = df_balanced.drop('RainTomorrow', axis=1)
y = df_balanced['RainTomorrow']

# One-hot encode
X = pd.get_dummies(X)
feature_names = X.columns.tolist()  # Save for Flask

# Impute and scale
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = SVR(kernel='rbf')
model.fit(X_train, y_train)

# Save model and preprocessing
joblib.dump(model, 'weather_model.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_names, 'features.pkl')

print("✅ Model and preprocessors saved successfully.")
