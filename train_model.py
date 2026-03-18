import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset (IMPORTANT)
data = pd.read_csv("diabetes.csv", sep='\t')

print(data.head())  # check

# Features & Target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Model trained & saved")
from sklearn.metrics import accuracy_score, confusion_matrix

# Predictions
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("✅ Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)