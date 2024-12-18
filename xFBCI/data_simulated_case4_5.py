import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.special import expit
scaler = StandardScaler()
import random

class SyntheticDataGenerator:
    def __init__(self, flag, n=300, d_x=5, n_sources=5):
        self.n = n  # Total records
        self.d_x = d_x  # Dimension of x
        self.n_sources = n_sources  # Number of sources
        self.n_source_records = n // n_sources  # Records per source
        self.lambda_func = lambda x: np.log(1 + np.exp(x))  # Softplus function
        self.phi_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid function
        self.flag = flag

        self.sources_data = {f"Source_{i}": [] for i in range(n_sources)}

    def generate_data(self):

        if self.flag == 0:
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
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

        elif self.flag == 1:
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
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

        elif self.flag == 2:
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
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


        elif self.flag == 3:

            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 1, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 2.0, np.random.normal(3, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 4.0, np.random.normal(3, 3, self.d_x)
            # x = scaler.fit_transform(x)
            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))




        elif self.flag == 4:
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 1, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 2.0, np.random.normal(3, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 4.0, np.random.normal(3, 3, self.d_x)
            # x = scaler.fit_transform(x)
            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag == 5:
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 1, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 2.0, np.random.normal(3, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 4.0, np.random.normal(3, 3, self.d_x)
            # x = scaler.fit_transform(x)
            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag == 6:
            x = np.random.normal(4, 2, (self.n, self.d_x))
            # x = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -10.0, np.random.beta(10, 5, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 6.0, np.random.normal(5, 5, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 8.0, np.random.normal(5, 5, self.d_x)
            # Treatment assignment
            # w = np.random.binomial(1,  np.clip(self.phi_func(a0 + x @ a1),0.05, 0.95))#w = np.random.binomial(1, self.phi_func((a0 + x @ a1)/100))

            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag == 7:
            x = np.random.normal(4, 2, (self.n, self.d_x))
            # x = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -10.0, np.random.beta(10, 5, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 6.0, np.random.normal(5, 5, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 8.0, np.random.normal(5, 5, self.d_x)
            # Treatment assignment
            # w = np.random.binomial(1,  np.clip(self.phi_func(a0 + x @ a1),0.05, 0.95))#w = np.random.binomial(1, self.phi_func((a0 + x @ a1)/100))

            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag == 8:

            x = np.random.normal(4, 2, (self.n, self.d_x))
            # x = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -10.0, np.random.beta(10, 5, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 6.0, np.random.normal(5, 5, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 8.0, np.random.normal(5, 5, self.d_x)
            # Treatment assignment
            # w = np.random.binomial(1,  np.clip(self.phi_func(a0 + x @ a1),0.05, 0.95))#w = np.random.binomial(1, self.phi_func((a0 + x @ a1)/100))

            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)

            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag == 9:
            x = np.random.normal(4, 2, (self.n, self.d_x))
            #x = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -10.0, np.random.beta(10, 5, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 6.0, np.random.normal(5, 5, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 8.0, np.random.normal(5, 5, self.d_x)
            # Treatment assignment
            # w = np.random.binomial(1,  np.clip(self.phi_func(a0 + x @ a1),0.05, 0.95))#w = np.random.binomial(1, self.phi_func((a0 + x @ a1)/100))

            linear_combination = x @ a1 + a0
            probabilities = expit(linear_combination)


            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))




        return a1, data
