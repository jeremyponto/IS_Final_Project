# Import pandas as pd to read and display the dataset and its amount of data.
import pandas as pd
# Import train_test_split from sklearn.model_selection to split train data and test data.
from sklearn.model_selection import train_test_split
# Import DecisionTreeClassifier from sklearn.tree to make a model which can make decisions using a tree,
# plot_tree from sklearn.tree to visualize the tree generated by the model, and
# export_graphviz from sklearn.tree to export the generated tree.
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
# Import accuracy_score from sklearn.metrics to see how well the model can predict.
from sklearn.metrics import accuracy_score
# Import StringIO from six to create a dot file in which the tree will be exported.
from six import StringIO
# Import graph_from_dot_data from pydotplus to get the tree from the dot file.
from pydotplus import graph_from_dot_data
# Import Image from IPython.display to create a png file which contains the tree.
from IPython.display import Image
# Import dump from joblib to export the trained model.
from joblib import dump

# Read the dataset given.
data = pd.read_csv("buy_property.csv")
# Display the contents and the amount of data of the dataset.
print(data.head(data.shape[0]))
print()

# List the features of the dataset.
features = ["money_left", "property_cost", "opponent_average", "go_distance", "opponent_house_distance"]
# Split the dataset into feature_data which will be the condition of the class feature.
feature_data = data[features]
# Display the contents and the amount of data of feature data.
print(feature_data.head(feature_data.shape[0]))
print()
# Split the dataset into class_feature_data whose value will be based on the features.
class_feature_data = data.buy_property
# Display the contents and the amount of data of class feature data.
print(class_feature_data.head(class_feature_data.shape[0]))
print()

# Split the datasets into 80% train data and 20% test data randomly.
(feature_train_data,
 feature_test_data,
 class_feature_train_data,
 class_feature_test_data) = train_test_split(feature_data,
                                             class_feature_data,
                                             test_size = 0.2)
# Display the contents and the amount of data of feature train data.
print(feature_train_data.head(feature_train_data.shape[0]))
print()
# Display the contents and the amount of data of feature test data.
print(feature_test_data.head(feature_test_data.shape[0]))
print()
# Display the contents and the amount of data of class feature train data.
print(class_feature_train_data.head(class_feature_train_data.shape[0]))
print()
# Display the contents and the amount of data of class feature test data.
print(class_feature_test_data.head(class_feature_test_data.shape[0]))
print()

# Create a decision tree classifier model with a maximum depth of 3
# which uses the information gain criterion for the tree formation.
model = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
model2 = DecisionTreeClassifier(criterion = "gini", max_depth = 3)
# Train the model with the train data.
model = model.fit(feature_train_data, class_feature_train_data)
model2 = model2.fit(feature_train_data, class_feature_train_data)
# Visualize the tree after it has been generated by the model.
plot_tree(model)
plot_tree(model2)
# Make the model predict the class feature values based on the feature test data.
class_feature_predictions = model.predict(feature_test_data)
class_feature_predictions2 = model2.predict(feature_test_data)
# Display the class feature predictions of the model.
print("Class feature predictions:", class_feature_predictions)
# Display the accuracy score to see how well the model can predict.
print("Accuracy score entropy:", accuracy_score(class_feature_test_data, class_feature_predictions))
print("Accuracy score gini:", accuracy_score(class_feature_test_data, class_feature_predictions2))

# Create a dot file in which the tree will be exported.
dot_data = StringIO()
# Export the tree to the dot file.
export_graphviz(model2,
                out_file = dot_data,
                filled = True,
                rounded = True,
                special_characters = True,
                feature_names = features,
                class_names = ['0', '1'])
# Get the tree from the dot file.
graph = graph_from_dot_data(dot_data.getvalue())
# Insert the tree to a png file.
graph.write_png("buy_property.png")
# Create a png file which contains the tree.
Image(graph.create_png())

# Export the trained model to a joblib file.
dump(model, "model.joblib")