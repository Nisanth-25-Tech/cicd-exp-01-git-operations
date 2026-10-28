
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class KNNClassifier:
    def __init__(self, k=3):
        self.model = KNeighborsClassifier(n_neighbors=k)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        print("KNN model trained successfully!")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model Accuracy: {accuracy * 100:.2f}%")
        return accuracy


if __name__ == "__main__":
    # Features: [Hours studied, Practice tests completed]
    X = [
        [1, 1], [2, 1], [2, 2], [3, 2],
        [6, 5], [7, 6], [8, 6], [8, 7],
        [4, 3], [5, 4]
    ]

    # 0 = Fail, 1 = Pass
    y = [0, 0, 0, 0, 1, 1, 1, 1, 0, 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    knn = KNNClassifier(k=3)
    knn.train(X_train, y_train)

    predictions = knn.predict(X_test)
    print("Actual labels:    ", y_test)
    print("Predicted labels: ", predictions)

    knn.evaluate(X_test, y_test)

    # Predict a new student's result
    result = knn.predict([[7, 5]])
    print("New student prediction:",
          "Pass" if result[0] == 1 else "Fail")
