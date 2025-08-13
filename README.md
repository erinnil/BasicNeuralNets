# Fashion MNIST Neural Network Classifier

### 📌 Description

This project trains and evaluates a **neural network** using `scikit-learn`’s `MLPClassifier` to classify clothing images from the **Fashion-MNIST** dataset.
The dataset is loaded from CSV files, normalized, and used to train a small multi-layer perceptron (MLP) for classification.
The model’s accuracy is reported per class and overall, along with a confusion matrix for detailed performance analysis.

The code supports:

* **All 10 Fashion-MNIST classes**
* **Filtered training/testing** for specific classes (e.g., Dresses, Sandals, Sneakers)
* **Per-class accuracy reporting**
* **Training vs. testing accuracy comparison**
* **Confusion matrix display**
* (Optional) Image display functions for visual inspection

---

### 📂 Dataset Requirements

You’ll need two CSV files in the same folder as the script:

1. `fashion_mnist_20bal_train.csv` — Balanced training set
2. `fashion_mnist_20bal_test.csv` — Balanced testing set

Each CSV should contain:

* **First column:** `class` (integer labels: 0–9, where each number represents a clothing category)
* **Remaining columns:** Pixel values (0–255) for each image

---

### ⚙️ How It Works

1. **Load Data**

   * Reads the CSV files into Pandas DataFrames.
   * Optionally filters for specific classes (uncomment filtering lines if needed).

2. **Preprocessing**

   * Separates features (`X`) and labels (`y`).
   * Normalizes pixel values to the range \[0, 1].

3. **Model Setup**

   * Creates an `MLPClassifier` with:

     * One hidden layer of 8 neurons.
     * A fixed random seed (`random_state=42`) for reproducibility.
     * Early stopping tolerance (`tol=0.005`).

4. **Training**

   * Fits the model to the training data.

5. **Evaluation**

   * Predicts labels for both training and testing sets.
   * Calculates per-class and overall accuracy.
   * Compares training vs. testing accuracy to detect overfitting.
   * Displays a confusion matrix.

6. **Optional**

   * Functions are provided for displaying images from the dataset based on row number or class.

---

### 📊 Output Example

After running, you’ll see output like:

```
Layer sizes: 784 x 8 x 10
Accuracy for class 3: 92%
Accuracy for class 5: 95%
Accuracy for class 7: 93%
----------
Overall Test Accuracy: 93.3%
Overall Training Accuracy: 97.5%
Confusion Matrix:
          Class  3   Class  5   Class  7
Class  3:       96        2        3
Class  5:        1       98        1
Class  7:        2        1       97
```

---

### ▶️ How to Run

1. Install dependencies:

   ```bash
   pip install pandas scikit-learn matplotlib
   ```
2. Place the dataset CSV files in the same directory as the script.
3. Run the script:

   ```bash
   python fashion_mnist_nn.py
   ```
4. (Optional) Uncomment filtering lines if you want to train on only certain classes.

---

### 📌 Notes

* **Hidden Layer Size:** `(8)` is intentionally small for demonstration purposes; increase for better accuracy.
* **Data Filtering:** Change `train_data['class'].isin([...])` to focus on specific categories.
* **Visualization:** Requires `matplotlib`.
