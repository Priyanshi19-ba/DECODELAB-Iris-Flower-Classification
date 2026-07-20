# Smart Iris Flower Classification Model

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Loaded Successfully!")
print("Total Samples:", len(X))
print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# Create Model
model = DecisionTreeClassifier(random_state=42)

# Train Model
model.fit(X_train, y_train)

print("\nModel Training Completed!")

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Predict New Flower
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nSample Flower Prediction:")
print("Predicted Class:", iris.target_names[prediction[0]])

print("\nThank You for Using Smart Iris Classification Model!")
