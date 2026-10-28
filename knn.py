from sklearn.neighbors import KNeighborsClassifier

class AIMLModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)

    def train(self, X, y):
        self.model.fit(X, y)
        print("Model trained successfully!")

    def predict(self, X):
        return self.model.predict(X)


# Demo execution
if __name__ == "__main__":
    # Features: [Hours studied, Practice tests completed]
    X = [
        [1, 1],
        [2, 1],
        [3, 2],
        [6, 5],
        [7, 6],
        [8, 7]
    ]

    # Labels: 0 = Fail, 1 = Pass
    y = [0, 0, 0, 1, 1, 1]

    ai_model = AIMLModel()
    ai_model.train(X, y)

    result = ai_model.predict([[7, 5]])
    print("Prediction:", "Pass" if result[0] == 1 else "Fail")