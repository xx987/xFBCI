import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
class SyntheticDataGenerator:
    def __init__(self, flag,  n=300, d_x=10, n_sources=5):
        self.n = n  # Total records
        self.d_x = d_x  # Dimension of x
        self.n_sources = n_sources  # Number of sources
        self.n_source_records = n // n_sources  # Records per source
        self.lambda_func = lambda x: np.log(1 + np.exp(x))  # Softplus function
        self.phi_func = lambda x: 1 / (1 + np.exp(-x))  # Sigmoid function
        self.flag = flag

        self.sources_data = {f"Source_{i}": [] for i in range(n_sources)}

    def generate_data(self):



        if self.flag==0:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))


            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1),1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1),1)


            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag==1:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag ==2:
            a0, a1 = 0.5, np.random.normal(0, 2,self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            #x = np.random.uniform(-1, 1, (self.n, self.d_x))
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag ==3:


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



        elif self.flag == 4:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag==5:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag==6:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag ==7:
            a0, a1 = 0.5, np.random.normal(0, 2,self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            #x = np.random.uniform(-1, 1, (self.n, self.d_x))
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag ==8:


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

        elif self.flag == 9:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag==10:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag==11:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.uniform(-1, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            # Outcomes
            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag ==12:
            a0, a1 = 0.5, np.random.normal(0, 2,self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            #x = np.random.uniform(-1, 1, (self.n, self.d_x))
            x = np.random.normal(0, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag ==13:


            x = np.random.normal(0, 1, (self.n, self.d_x))
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



        elif self.flag == 14:
            a0, a1 = 0.5, np.random.normal(0, 2, self.d_x)  # Treatment assignment coefficients
            b0, b1 = 1.0, np.random.normal(0, 1, self.d_x)  # Coefficients for Y(0)
            c0, c1 = 2.0, np.random.normal(0, 1, self.d_x)
            x = np.random.normal(0, 1, (self.n, self.d_x))
            # Treatment assignment
            w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

            y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
            y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)

            y = np.where(w == 0, y_0, y_1)

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))


        elif self.flag==15:
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

        elif self.flag==16:
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

        elif self.flag ==17:
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

            # Observed data
            data = np.column_stack((w, y, y_1, y_0, x))

        elif self.flag ==18:


            x = np.random.normal(0, 1, (self.n, self.d_x))
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

        elif self.flag == 19:
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


        return a1, data