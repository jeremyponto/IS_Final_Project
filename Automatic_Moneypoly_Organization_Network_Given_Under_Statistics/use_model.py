# Import load from joblib to use the trained model.
from joblib import load

# Use the existing trained model from a joblib file.
model = load("model.joblib")
# Make the model predict the class feature value given a data.
class_feature_prediction = model.predict([[1000, 200, 1600, 3, 20]])
# Display the class feature prediction of the model.
print(class_feature_prediction[0])