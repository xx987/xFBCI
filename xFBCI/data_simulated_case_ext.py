import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
class SyntheticDataGenerator:
    def __init__(self, flag,  n=500, d_x=20, n_sources=5):
        self.n = n  # Total records
        self.d_x = d_x  # Dimension of x
        self.n_sources = n_sources  # Number of sources
        self.n_source_records = n // n_sources  # Records per source
          # Number of replications # DATA-1 or DATA-2
        self.lambda_func = lambda x: np.log(1 + np.exp(x))  # Softplus function
        self.phi_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid function
        self.flag = flag

        # Define common parameters


        # Storage for sources' data across replications
        self.sources_data = {f"Source_{i}": [] for i in range(n_sources)}

    def generate_data(self):



        if self.flag==0:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-100, 100, (self.n, self.d_x))
            x = scaler.fit_transform(x)
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = self.lambda_func(b0 + x @ b1)#np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = self.lambda_func(c0 + x @ c1)
            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag==1:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-100, 100, (self.n, self.d_x))#np.random.uniform(-1, 1, (self.n, self.d_x))#
            x = scaler.fit_transform(x)
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))
            log_reg = LogisticRegression()
            log_reg.fit(x, w)
        elif self.flag ==2:
            a0, a1 = 0.5, np.random.normal(0, 2,self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            #x = np.random.uniform(-1, 1, (self.n, self.d_x))
            x = np.random.normal(0, 1, (self.n, self.d_x))

            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)


            data = np.column_stack((w, y, y_1, y_0, x))

            log_reg = LogisticRegression()
            log_reg.fit(x, w)

        elif self.flag ==3:


            x = np.random.uniform(-100, 100, (self.n, self.d_x))
            x = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

            log_reg = LogisticRegression()
            log_reg.fit(x, w)


        elif self.flag == 4:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.normal(0, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

            log_reg = LogisticRegression()
            log_reg.fit(x, w)
        return a1, data
