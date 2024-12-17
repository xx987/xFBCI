import jax
import jax.numpy as jnp
import numpy as np
import optax
from copy import deepcopy
from treatment_effect_compute import TreatmentEffectEstimator as TEE
from data_simulated_case2_3 import SyntheticDataGenerator as sy
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.linear_model import LogisticRegression

# 1. Define Logistic Regression Model using JAX

def W_k_update(w_k, lr_r, F_k,steps_up,X,w):
    for _ in range(steps_up):
        grads = jax.grad(lambda x: F_k(x,X,w))(w_k)
        #print(grads,'看看grads1')
        w_k = w_k - lr_r* grads

    return w_k


def V_k_update(v_k, lr_r, lam, w_k, F_k, steps_v_k, X, w):
    grad_loss = jax.grad(F_k)
    #loss = F_k(v_k, X, w)
    for _ in range(steps_v_k):
        grads = jax.grad(lambda x: F_k(x, X, w))(v_k)

        v_k -= lr_r * grads + lam* (v_k - w_k)
    return v_k


def sever_updat(w_t, sum_dlt):
    sum_arr = np.array(sum_dlt)
    w_t_new = w_t + np.mean(sum_arr,axis=0)
    return w_t_new

def F_k(w, X, y):
    logits = jnp.dot(X, w)
    predictions = 1 / (1 + jnp.exp(-logits))  # Sigmoid
    loss = -jnp.mean(y * jnp.log(predictions + 1e-8) + (1 - y) * jnp.log(1 - predictions + 1e-8))

    return loss

K = 30
n = 5
learning_rate_k = 0.002
learning_rate_l = 0.001
lam = 0

steps_up = 400
step_v_k = 400



all_round_rmse_fed = []
all_round_rmse_indi = []
all_round_rmse_center = []
all_round_ate_fed = []
all_round_ate_indi = []
all_round_ate_center = []
all_true_ate = []

min_fed_set = []
min_indi_set = []
use_fixed_data = []
para_all = []


for i in range(n):
    generator = sy()#flag=i)
    all_information = generator.generate_data()
    parap_s, data_all = all_information[0], all_information[1][200*i:1000]
    # data_all = generator.simulate()[1]
    para_all.append(parap_s)
    use_fixed_data.append(data_all)



initial_w = np.random.normal(0, 1, size=(5,))
initial_v = []
for i in range(n):

    initial_v.append(np.random.normal(0, 1, size=(5,)))#, size=(20,)))


for _ in range (K):

    min_fed_all = 0

    record_delta = []
    each_communi_rmse_fed = []
    each_communi_rmse_indi = []
    each_communi_ate_fed = []
    each_communi_ate_indi = []
    for i in range(n):


        #W_k_update(w_k, lr_r, F_k, steps_up, X, w)
        new_w= W_k_update(initial_w, learning_rate_k, F_k,steps_up,use_fixed_data[i][:, 4:],use_fixed_data[i][:, 0])
        new_v = V_k_update(initial_v[i], learning_rate_l, lam, new_w, F_k,steps_up,use_fixed_data[i][:, 4:],use_fixed_data[i][:, 0])
        #initial_w[i] = new_w
        initial_v[i] = new_v

        rmse_person = np.sqrt(np.mean((new_v - para_all[i]) ** 2))
        each_communi_rmse_fed.append(np.sqrt(np.mean((new_v - para_all[i]) ** 2)))
        print(para_all[i],'true parameters')
        print(new_v, "estiamted personalize")
        print(np.sqrt(np.mean((new_v - para_all[i]) ** 2)), f'this is {i} client')

        update_w_k = new_w - initial_w
        record_delta.append(update_w_k)


        TEG_ditto= TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
                       coeff=new_v)
        TEG_ditto.calculate_propensity_scores()
        # calculate_propensity_scores(
        treat_effG = TEG_ditto.estimate_att()
        "fed ATE computing here########################################"
        each_communi_ate_fed.append(treat_effG)

        true_ate_i_client = np.abs(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]))
        min_fed_ate = true_ate_i_client - treat_effG
        min_fed_all += min_fed_ate ** 2

    initial_w= sever_updat(initial_w, record_delta)
    #print(each_communi_rmse_fed,'each_communi_rmse_fed')
    fed_Armse = np.mean(each_communi_rmse_fed)
    ind_Armse = np.mean(each_communi_rmse_indi)
    fed_ate = np.mean(each_communi_ate_fed)
    ind_ate = np.mean(each_communi_ate_indi)
min_fed_set.append(np.sqrt(min_fed_all / n))
all_round_rmse_fed.append(fed_Armse)
all_round_rmse_indi.append(ind_Armse)
all_round_ate_fed.append(fed_ate)

print(all_round_rmse_fed)
print(min_fed_set)

