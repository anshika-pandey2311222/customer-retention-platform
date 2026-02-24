import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("Bank Customer Churn Prediction.csv")

# Drop ID
df = df.drop(columns=["customer_id"])

# Split X and y
X = df.drop("churn", axis=1)
y = df["churn"]

# One-hot encode
X = pd.get_dummies(X, columns=["country", "gender"], drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save files
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model trained and saved successfully!")