import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


data = pd.read_csv("customer_purchase_tree_data.csv")

print("Loaded dataset shape:", data.shape)

print("\nDataset information:")
data.info()

print("\nMissing values per column:")
print(data.isna().sum())

print("\nDuplicate rows:", data.duplicated().sum())

invalid_records = data[
    ~data["Age"].between(18, 70)
    | ~data["Income"].between(20000, 150000)
    | ~data["Website_Visits"].between(1, 20)
    | ~data["Time_Spent"].between(1, 60)
    | ~data["Previous_Purchases"].between(0, 10)
    | ~data["Product_Views"].between(1, 30)
    | ~data["Purchased"].isin([0, 1])
]

print("\nInvalid records:", len(invalid_records))

if not invalid_records.empty:
    print(invalid_records.head())


features = [
    "Age",
    "Income",
    "Website_Visits",
    "Time_Spent",
    "Previous_Purchases",
    "Product_Views"
]

target = "Purchased"

X = data[features]
y = data[target]

print("\nFeatures:", features)
print("Target:", target)
print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nClass distribution:")
print(pd.DataFrame({
    "Train": y_train.value_counts().sort_index(),
    "Test": y_test.value_counts().sort_index()
}))


model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=15,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed")
print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())

root_feature_index = model.tree_.feature[0]
root_feature = features[root_feature_index]
root_threshold = model.tree_.threshold[0]

print(f"Root question: {root_feature} <= {root_threshold:.2f}?")

y_pred = model.predict(X_test)

results = X_test.copy()
results["Actual"] = y_test
results["Predicted"] = y_pred

print("\nFirst 10 test predictions:")
print(results.head(10))

correct_predictions = (y_test == y_pred).sum()

print("\nTotal test customers:", len(y_test))
print("Correct predictions:", correct_predictions)
print(f"Test accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

matrix = confusion_matrix(y_test, y_pred)

print("\nConfusion matrix")
print(pd.DataFrame(
    matrix,
    index=["Actual Not Purchased", "Actual Purchased"],
    columns=["Predicted Not Purchased", "Predicted Purchased"]
))


plt.figure(figsize=(26, 13))

plot_tree(
    model,
    feature_names=features,
    class_names=["Not Purchased", "Purchased"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Customer Purchase Decision Tree")
plt.tight_layout()
plt.show()


new_customers = pd.DataFrame({
    "Age": [22, 34, 45, 29, 58, 41, 25, 63, 37, 50],
    "Income": [30000, 70000, 110000, 45000, 130000, 85000, 25000, 95000, 60000, 140000],
    "Website_Visits": [2, 10, 18, 5, 19, 12, 3, 15, 8, 20],
    "Time_Spent": [5, 25, 50, 12, 55, 35, 8, 45, 20, 58],
    "Previous_Purchases": [0, 3, 7, 1, 9, 4, 0, 6, 2, 10],
    "Product_Views": [3, 12, 25, 7, 28, 18, 5, 22, 10, 30]
})

new_predictions = model.predict(new_customers)

new_results = new_customers.copy()
new_results["Predicted"] = new_predictions
new_results["Purchase_Result"] = new_results["Predicted"].map({
    0: "Not Purchased",
    1: "Purchased"
})

print("\nPredictions for 10 new customers:")
print(new_results.to_string(index=False))