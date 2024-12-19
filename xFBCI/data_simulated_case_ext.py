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
            x = np.random.uniform(0, 30, (self.n, self.d_x))
            x_D = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -3, np.random.normal(0, 3, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 4, np.random.normal(4, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 6, np.random.normal(4, 3, self.d_x)
            linear_combination = x_D @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_0[np.isinf(y_0)] = 5
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
            y_1[np.isinf(y_1)] = 5
            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag == 1:
            x = np.random.uniform(0, 30, (self.n, self.d_x))
            x_D = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -3, np.random.normal(0, 3, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 4, np.random.normal(4, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 6, np.random.normal(4, 3, self.d_x)
            linear_combination = x_D @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_0[np.isinf(y_0)] = 5
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
            y_1[np.isinf(y_1)] = 5
            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag == 2:
            x = np.random.uniform(0, 30, (self.n, self.d_x))
            x_D = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -3, np.random.normal(0, 3, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 4, np.random.normal(4, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 6, np.random.normal(4, 3, self.d_x)
            linear_combination = x_D @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_0[np.isinf(y_0)] = 5
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
            y_1[np.isinf(y_1)] = 5
            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag == 3:

            x = np.random.uniform(0, 30, (self.n, self.d_x))
            x_D = scaler.fit_transform(x)
            # Treatment assignment
            a0, a1 = -2, np.random.normal(0, 3, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 4, np.random.normal(4, 3, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 6, np.random.normal(4, 3, self.d_x)
            linear_combination = x_D @ a1 + a0
            probabilities = expit(linear_combination)

            # 根据概率生成二元响应变量 W
            w = np.random.binomial(1, probabilities)
            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_0[np.isinf(y_0)] = 5
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
            y_1[np.isinf(y_1)] = 5
            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))




        elif self.flag == 4:
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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

        elif self.flag == 7:
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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


        elif self.flag == 8:

            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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

        elif self.flag == 9:
            x = np.random.normal(2, 2, (self.n, self.d_x))
            # Treatment assignment
            a0, a1 = -5, np.random.beta(5, 15, self.d_x)  # Treatment assignment coefficients
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




        return a1, data
