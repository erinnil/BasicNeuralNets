import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Filter for classes 3 (dresses), 5 (sandals), and 7 (sneakers)
train_data = train_data[train_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Filter for classes 3 (dresses), 5 (sandals), and 7 (sneakers)
test_data = test_data[test_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier(hidden_layer_sizes=(8), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Determine model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Layer sizes: {layer_size_str}")

# Predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test) * 100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Filter for classes 3 (dresses) and 7 (sneakers)
train_data = train_data[train_data['class'].isin([3, 7])]

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Filter for classes 3 (dresses) and 7 (sneakers)
test_data = test_data[test_data['class'].isin([3, 7])]

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier(hidden_layer_sizes=(8), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Calculate and print accuracy for each class and overall test accuracy
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test) * 100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Filter for classes 3 (dresses) and 7 (sneakers)
train_data = train_data[train_data['class'].isin([3, 7])]

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Filter for classes 3 (dresses) and 7 (sneakers)
test_data = test_data[test_data['class'].isin([3, 7])]

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier(hidden_layer_sizes=(8), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test) * 100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()  # Newline for next row

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Filter for classes 3 (dresses), 5 (sandals), and 7 (sneakers)
train_data = train_data[train_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Filter for classes 3 (dresses), 5 (sandals), and 7 (sneakers)
test_data = test_data[test_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier(hidden_layer_sizes=(8), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test) * 100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()  # Newline for next row

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Commented out the filtering for specific classes
# train_data = train_data[train_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Commented out the filtering for specific classes
# test_data = test_data[test_data['class'].isin([3, 5, 7])]

# Separate the data (features) and the classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier(hidden_layer_sizes=(8), random_state=42, tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] * 100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test) * 100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training * 100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
class_ids = sorted(total_counts.keys())

# For better formatting
print("Confusion Matrix:")
print(f"{'':9s}", end='')
for label in class_ids:
    print(f"Class {label:2d} ", end='')
print()  # Newline for next row

for i, row in enumerate(conf_matrix):
    print(f"Class {class_ids[i]}:", " ".join(f"{num:8d}" for num in row))def show_pixel_grid(row_number):
    # Assuming you have a function to fetch images based on the row number
    image_data = get_image_data(row_number)  # Implement this function
    if image_data is not None:
        plt.imshow(image_data)  # Using matplotlib to show the image
        plt.title(f"Clothing Article Row: {row_number}")
        plt.show()
    else:
        print(f"No data available for row {row_number}")

# To display different clothing articles
for row in range(number_of_articles):  # replace number_of_articles with the actual count
    show_pixel_grid(row)def display_class_images(class_number):
    # Load and filter images based on the class_number
    class_images = get_images_by_class(class_number)  # Implement this function
    for img in class_images:
        plt.imshow(img)
        plt.title(f"Class: {class_number}")
        plt.show()

# Try different class numbers
for class_number in range(number_of_classes):  # replace number_of_classes with the actual count
    display_class_images(class_number)import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import  confusion_matrix, classification_report
from collections import defaultdict

# Load the training dataset
train_data = pd.read_csv('fashion_mnist_20bal_train.csv')

# Separate the data (features) and the  classes
X_train = train_data.drop('class', axis=1)  # Features (all columns except the first one)
X_train = X_train / 255.0
y_train = train_data['class']   # Target (first column)

# Load the testing dataset
test_data = pd.read_csv('fashion_mnist_20bal_test.csv')

# Separate the data (features) and the  classes
X_test = test_data.drop('class', axis=1)  # Features (all columns except the first one)
X_test = X_test / 255.0
y_test = test_data['class']   # Target (first column)

# Create the neural network model
neural_net_model = MLPClassifier( hidden_layer_sizes=(8),random_state=42,tol=0.005)

# Train the model
neural_net_model.fit(X_train, y_train)

# Determine model architecture
layer_sizes = [neural_net_model.coefs_[0].shape[0]]  # Start with the input layer size
layer_sizes += [coef.shape[1] for coef in neural_net_model.coefs_]  # Add sizes of subsequent layers
layer_size_str = " x ".join(map(str, layer_sizes))
print(f"Layer sizes: {layer_size_str}")

# predict the classes from the training and test sets
y_pred_train = neural_net_model.predict(X_train)
y_pred = neural_net_model.predict(X_test)

# Create dictionaries to hold total and correct counts for each class
correct_counts = defaultdict(int)
total_counts = defaultdict(int)
overall_correct = 0

# Count correct test predictions for each class
for true, pred in zip(y_test, y_pred):
    total_counts[true] += 1
    if true == pred:
        correct_counts[true] += 1
        overall_correct += 1

# For comparison, count correct _training_ set predictions
total_counts_training = 0
correct_counts_training = 0
for true, pred in zip(y_train, y_pred_train):
    total_counts_training += 1
    if true == pred:
        correct_counts_training += 1

# Calculate and print accuracy for each class and overall test accuracy
for class_id in sorted(total_counts.keys()):
    accuracy = correct_counts[class_id] / total_counts[class_id] *100
    print(f"Accuracy for class {class_id}: {accuracy:3.0f}%")
print(f"----------")
overall_accuracy = overall_correct / len(y_test)*100
print(f"Overall Test Accuracy: {overall_accuracy:3.1f}%")
overall_training_accuracy = correct_counts_training / total_counts_training*100
print(f"Overall Training Accuracy: {overall_training_accuracy:3.1f}%")
