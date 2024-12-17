import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
class SyntheticDataGenerator:
    def __init__(self, n=1000, d_x=5, n_sources=5): #n=1000, d_x=20, n_sources=5
        self.n = n  # Total records
        self.d_x = d_x  # Dimension of x
        self.n_sources = n_sources  # Number of sources
        self.n_source_records = n // n_sources  # Records per source
          # Number of replications # DATA-1 or DATA-2
        self.lambda_func = lambda x: np.log(1 + np.exp(x))  # Softplus function
        self.phi_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid function

        self.sources_data = {f"Source_{i}": [] for i in range(n_sources)}

    def generate_data(self):
        b0, c0 = 6, 30
        a0 = 0.6
        a1 = np.random.normal(0, np.sqrt(2), self.d_x)#
        b1 = np.random.normal(10, np.sqrt(2), self.d_x)#
        c1 = np.random.normal(15, np.sqrt(2), self.d_x)#
        x = np.random.uniform(-1, 1, (self.n, self.d_x))

        w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

        # Outcomes
        y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
        y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
        y = np.where(w == 0, y_0, y_1)

        data = np.column_stack((w,y,y_1,y_0,x))


        return a1, data

