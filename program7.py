import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
df = pd.read_csv("brightness.csv")

# Check for NaN values and handle them
print("Missing values in dataset:\n", df.isna().sum())

# Encode the class labels
le = LabelEncoder()
df['class'] = le.fit_transform(df['class'])

# Verify the data types
print("Data types:\n", df.dtypes)

# Split the dataset into features (X) and labels (y)
x = df.drop(columns=['class'], axis=1)
y = df['class']

# Check the feature data types
print("Feature data types:\n", x.dtypes)

# Ensure all feature columns are numeric
x = x.apply(pd.to_numeric, errors='coerce')

# Split the dataset into training and testing sets
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train the KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(xtr, ytr)

# Predict the labels for the test set
y_pred = knn_model.predict(xte)

# Calculate the accuracy of the model
acc = accuracy_score(yte, y_pred)
print("Accuracy:", acc)

# Classification reports
print("Classification Report:\n", classification_report(yte, y_pred))

# Predict the class for a new input
Brightness = float(input("Enter the Brightness Value: "))
Saturation = float(input("Enter the Saturation Value: "))
user_input = np.array([[Brightness, Saturation]])
user_pred = knn_model.predict(user_input)

# Map the prediction to the class name
class_names = le.inverse_transform([0, 1])
predicted_class = class_names[user_pred[0]]
print(f"The value belongs to {predicted_class} Class")
