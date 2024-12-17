import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class TreatmentEffectEstimator:
    def __init__(self, X, W, Y, coeff=None):
        self.X = X
        self.W = W
        self.Y = Y
        self.coeff = coeff
        self.propensity_scores = None
        self.data = None

    def calculate_propensity_scores(self):
        if self.coeff is None:
            # Use logistic regression to calculate propensity scores
            log_reg = LogisticRegression()
            log_reg.fit(self.X, self.W)
            self.propensity_scores = log_reg.predict_proba(self.X)[:, 1]
        else:
            # Use provided coefficients to calculate propensity scores
            linear_part = self.X @ self.coeff
            self.propensity_scores = 1 / (1 + np.exp(-linear_part))

        # Store the data in a DataFrame
        self.data = pd.DataFrame({
            'W': self.W,
            'Y': self.Y,
            'e': self.propensity_scores
        })

    def estimate_att(self):

        treated = self.W == 1
        control = self.W == 0

        # Nearest neighbor matching
        def match_nearest_neighbor(scores, treated, control):
            matches = []
            for i in np.where(treated)[0]:
                # Find control unit with the closest propensity score
                best_match = np.argmin(np.abs(scores[i] - scores[control]))
                matches.append((i, np.where(control)[0][best_match]))
            return matches

        matched_pairs = match_nearest_neighbor(self.data['e'], treated, control)

        # Step 3: Compute Average Treatment Effect
        ate_estimate = np.mean([
            self.Y[pair[0]] - self.Y[pair[1]] for pair in matched_pairs
        ])

        return ate_estimate




# Usage:
# estimator = TreatmentEffectEstimator(X_normalized, W, test_df['revenue'], coeff)
# estimator.calculate_propensity_scores()
# att = estimator.estimate_att()
# print("Average Treatment Effect on the Treated (ATT):", att)


