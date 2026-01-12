import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("Housing.csv")

# Feature selection
X = df[["area", "bedrooms", "bathrooms", "stories", "parking"]]
y = df["price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Save trained model
with open("house_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as house_model.pkl")
