import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ✅ Load dataset (Ensure correct path)
DATA_PATH = "data/soybean_data.csv"
df = pd.read_csv(DATA_PATH)

# ✅ Print basic dataset info for debugging
print("Dataset Preview:\n", df.head())
print("\nDataset Summary:\n", df.info())

# 🔍 Print column names to check for typos
print("\n🔍 Column Names:", df.columns.tolist())

# ✅ Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# ✅ Drop unnecessary columns (if any exist)
columns_to_drop = ["Parameters", "Random"]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors="ignore", inplace=True)

# 🔍 Print cleaned column names
print("\n✅ Cleaned Column Names:", df.columns.tolist())

# ✅ Encode categorical variables (if any)
label_encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# ✅ Define features (X) and target variables (y)
target_columns = ["Seed Yield per Unit Area (SYUA)", "Protein Content (PCO)"]  # Correct column names
if all(col in df.columns for col in target_columns):
    y = df[target_columns]  # Target variables
    X = df.drop(columns=target_columns)  # Exclude target columns from features
else:
    raise ValueError(f"❌ Error: Required target columns {target_columns} are missing from the dataset!")

# ✅ Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ✅ Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ✅ Train a Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ✅ Make predictions
y_pred = model.predict(X_test)

# ✅ Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ✅ Save the trained model and scaler
joblib.dump(model, "model/soybean_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")

print("\n✅ Model and Scaler saved successfully!")
