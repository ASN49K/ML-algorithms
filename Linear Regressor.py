class LinearRegressor:
    def predict_values(self, X):
        return X @ self.w + self.b

    def loss_mse(self, X):
        return np.mean((self.y - self.predict_values(X))**2)

    def calculate_grad_w(self):
        diff = self.predict_values(self.X) - self.y
        return (2 / self.n) * self.X.T @ diff

    def calculate_grad_b(self):
        diff = self.predict_values(self.X) - self.y
        return 2 * np.mean(diff)

    def __init__(self, X, y, learning_rate=0.005):
        self.n, self.m = X.shape
        self.X = np.array(X)
        self.y = np.array(y)
        self.w = np.random.uniform(0, 1, self.m)
        self.b = np.random.uniform(0, 1)
        print(self.w.max(),self.w.min())

        last = self.loss_mse(self.X)
        print(f"init loss(mse): {last}")
        delta = 100
        min_delta = 1e-3
        it = 0

        while it < 100000:
            it += 1
            self.w = self.w - learning_rate * self.calculate_grad_w()
            self.b = self.b - learning_rate * self.calculate_grad_b()
            now = self.loss_mse(self.X)
            delta = last - now
            last = now
            if it%1000==1:
                print(f"Loss after {it} gradient steps: {it} {delta} {now}")
