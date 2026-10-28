
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


class AdvancedKNN:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier())
        ])
        self.best_k = None

    def train(self, X_train, y_train):
        parameters = {
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
            "knn__metric": ["euclidean", "manhattan"]
        }

        search = GridSearchCV(
            self.model,
            parameters,
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        search.fit(X_train, y_train)
        self.model = search.best_estimator_
        self.best_k = self.model.named_steps[
            "knn"
        ].n_neighbors

        print("Training completed!")
        print("Best parameters:", search.best_params_)
        print(f"Best CV accuracy: {search.best_score_:.2%}")

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)

        print("\nTest Accuracy:",
              f"{accuracy_score(y_test, predictions):.2%}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            target_names=["Malignant", "Benign"],
            zero_division=0
        ))


if __name__ == "__main__":
    # Load a built-in dataset for demonstration
    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    knn = AdvancedKNN()
    knn.train(X_train, y_train)
    knn.evaluate(X_test, y_test)

    # Predict the class of a new sample
    prediction = knn.predict(X_test[:1])
    print("\nFirst test sample prediction:",
          data.target_names[prediction[0]])
