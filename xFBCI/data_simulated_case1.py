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

        # Define common parameters


        # Storage for sources' data across replications
        self.sources_data = {f"Source_{i}": [] for i in range(n_sources)}

    def generate_data(self):
        b0, c0 = 6, 30
        a0 = 0.6
        a1 = np.array([0]*self.d_x)#np.random.normal(0, np.sqrt(2), self.d_x)#
        b1 = np.array([10]*self.d_x)#np.random.normal(10, np.sqrt(2), self.d_x)#
        c1 = np.array([15]*self.d_x)#np.random.normal(15, np.sqrt(2), self.d_x)#
        x = np.random.uniform(-1, 1, (self.n, self.d_x))

        #x = scaler.fit_transform(x)
        # Treatment assignment
        w = np.random.binomial(1, self.phi_func(a0 + x @ a1))

        # Outcomes
        y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 0)
        y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 0)
        #y_0 = np.random.normal(self.lambda_func(b0 + x @ b1), 1)
        #y_1 = np.random.normal(self.lambda_func(c0 + x @ c1), 1)
        y = np.where(w == 0, y_0, y_1)

        # Observed data
        data = np.column_stack((w,y,y_1,y_0,x))

        log_reg = LogisticRegression()
        log_reg.fit(x, w)
        coeffice = log_reg.coef_
        #print(a1, "A1a1a1a1a")
        #print(coeffice,"coefficecoefficecoefficecoeffice")
        #print(np.sqrt(np.mean((coeffice - a1) ** 2)))




        return a1, data




# Example usage
"""
result = []
for _ in range(10):

    if __name__ == "__main__":
        # Initialize generator for DATA-1
        generator = SyntheticDataGenerator()

        generator.generate_data()


        # Access data for Source 1, Replication 1
        #source_1_rep_1_data = generator.get_multiple_sources_data(source_nums=[0,1,2,3,4], replication_num=1)
        #print("Source 1, Replication 1 Data:", source_1_rep_1_data)

        #ate_1 = np.mean(np.abs(source_1_rep_1_data['Source_0'][:,0] - source_1_rep_1_data['Source_0'][:,1]))

        all_d = generator.get_source_data(source_num=1,replication_num=10)
    #y0 = (1-all_d[:,1])*all_d[:,0]+all_d[:,1]*all_d[:,1]


    dataf = np.load('data-synthetic-largescale.npz', allow_pickle=True)
    nd = dataf['data_lst_Delta'][0][1]


    #y0 = (1- all_d[:,0])*all_d[:,1] + all_d[:,0]*all_d[:,2]
    #y1 = all_d[:,0]*all_d[:,1] + (1-all_d[:,0])*all_d[:,2]
    result.append(np.mean(all_d[:,2]- all_d[:,3]))
"""