
import numpy as np
from treatment_effect_compute import TreatmentEffectEstimator as TEE
from GaussianEPClient import EPClient as client
from ApproxInfer import SG_MCMC_Infer,likelihood_p_k
from data_simulated_case1 import SyntheticDataGenerator as sy #data_simulated_case2_3....data_simulated_case_ext
from GaussianEPServer import EPServer
from sklearn.linear_model import LogisticRegression, LinearRegression

#####################################################################################
#########################Simulation start here#######################################
#####################################################################################
#Code for Case1, 2, 3
dim = 5 #Dimensions (Number of variables)
n = 5 #Number of Clients
all_round_rmse_fed = []
all_round_rmse_indi = []
all_round_rmse_center = []
all_round_ate_fed = []
all_round_ate_indi = []
all_round_ate_center = []
all_true_ate = []
use_fixed_data = []
para_all = []

for i in range(n):
    generator = sy()
    all_information = generator.generate_data()
    parap_s,data_all = all_information[0],all_information[1]#[1]
    para_all.append(parap_s)
    use_fixed_data.append(data_all)

"we do all of Centralized information here######################"
center_para = np.mean(np.vstack(para_all),axis = 0)
combined_array = np.vstack(use_fixed_data)

"Use logistic regression to get parameter"
log_reg_cen = LogisticRegression()
log_reg_cen.fit(combined_array[:, 4:], combined_array[:, 0])
coe_center = log_reg_cen.coef_
"RMSE compute"
rmse_center = np.sqrt(np.mean((coe_center - center_para) ** 2))
all_round_rmse_center.append(rmse_center)

"ATE compute"
TEG_center = TEE(X=combined_array[:, 4:], W=combined_array[:, 0], Y=combined_array[:, 1],
               coeff=coe_center[0])
TEG_center.calculate_propensity_scores()
treat_center = TEG_center.estimate_att()
"fed ATE computing here########################################"
all_round_ate_center.append(treat_center)
"we do all of Centralized information here######################"

round_in_data = []
#########################################################################################################


A = np.random.rand(dim, dim)
global_variance_full =  np.eye(dim)*(1/n)

local_list = {}
eta_global,Lambda_global =0,0
for i in range(n):
    local_list[f'local_para {i}'] =np.random.normal(0, 1, size=(dim,)),global_variance_full
    eta_global+=local_list[f'local_para {i}'][0]
    Lambda_global+=local_list[f'local_para {i}'][1]

delta_q = {}

#Communication start here
for t in range(0,20):
    print(f'This is the round {t} communication #####################################################')

    each_communi_rmse_fed = []
    each_communi_rmse_indi = []
    each_communi_ate_fed = []
    each_communi_ate_indi = []
    each_true_ate = [] ##true ATE is same for fed and individual


    cluster_for_client_indi_para = []

    delta_q_list_mean = []
    delta_q_list_cov = []
    localds = []
    round_in_data = []
    sgld = []

    #Estimating for each client in the system
    for i in range(n):
        log_reg = LogisticRegression()
        log_reg.fit(use_fixed_data[i][:, 4:], use_fixed_data[i][:, 0])
        coeffice = log_reg.coef_
        cluster_for_client_indi_para.append(coeffice)
        delta_q[f'delta_q{i}']= client(eta_global, Lambda_global, local_list[f'local_para {i}'][0],
                                       local_list[f'local_para {i}'][1],
                                       delta=0.2)
        sgld_see = SG_MCMC_Infer(p_k=likelihood_p_k,
                                 eta=delta_q[f'delta_q{i}'].cavity_distribution()[0],
                                 Lambda=delta_q[f'delta_q{i}'].cavity_distribution()[1],
                                 learning_rate=0.005,
                                 num_samples=700,
                                 theta=np.random.randn(dim),
                                 X= use_fixed_data[i][:, 4:],
                                 W =  use_fixed_data[i][:, 0],
                                 batch_size=int(len(use_fixed_data[i][:, 4:])*0.9))

        sample_set = sgld_see.sgld_sample()
        sample_set = sample_set[100:700]

        eta_est, Lambda_est = np.mean(sample_set, axis=0), 1*10**-20*np.eye(dim)# 5 * 10 ** -20 * np.eye(20)  # 5*10**20
        eta_add, Lambda_add = para_all[i] + delta_q[f'delta_q{i}'].cavity_distribution()[0], np.cov(
            sample_set.T) + delta_q[f'delta_q{i}'].cavity_distribution()[1]
        sgld.append([eta_est, Lambda_est])
        delta_q_list_mean.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                     0])  # update_local_distribution()[0])
        delta_q_list_cov.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                   1])  # update_local_distribution()[1])


        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0])
        loca_check = np.linalg.inv(temp_list[1]) @ temp_list[0]

        rmse_person = np.sqrt(np.mean((temp_list[0] - para_all[i]) ** 2))
        rmse_indi = np.sqrt(np.mean((coeffice - para_all[i]) ** 2))
        print(rmse_person, 'Local RMSE')
        print(rmse_indi, 'no fed RMSE')

        each_communi_rmse_fed.append(rmse_person)
        each_communi_rmse_indi.append(rmse_indi)

        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0] )
        #localds.append(temp_list[0])
        local_list[f'local_para {i}'] = tuple(temp_list)

        TEG_sgld = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
              coeff=temp_list[0])
        TEG_sgld.calculate_propensity_scores()
        treat_effG = TEG_sgld.estimate_att()
        "fed ATE computing here########################################"
        each_communi_ate_fed.append(treat_effG)

        print(treat_effG,f'The client{i+1} estimate ate')
        TEG_est = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
                       coeff=coeffice[0])#para_all[i]
        TEG_est.calculate_propensity_scores()
        "indivudial ATE computing here########################################"
        treat_est = TEG_est.estimate_att()
        each_communi_ate_indi.append(treat_est)
        each_true_ate.append(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]))



    Sever_state = EPServer(eta_global, Lambda_global,delta_q_list_mean,delta_q_list_cov)
    eta_global, Lambda_global = Sever_state.InfServer()
    stacked_arrays = np.stack(cluster_for_client_indi_para, axis=2)
    mean_array = np.mean(stacked_arrays, axis=2)

    TEG_mean = TEE(X=combined_array[:, 4:], W=combined_array[:, 0], Y=combined_array[:, 1],
                     coeff=mean_array[0])
    TEG_mean.calculate_propensity_scores()
    # calculate_propensity_scores(
    treat_mean = TEG_mean.estimate_att()
    mean_individual_rmse =np.sqrt(np.mean((mean_array - para_all[i]) ** 2))
    fed_Armse = np.mean(each_communi_rmse_fed)
    ind_Armse = np.mean(each_communi_rmse_indi)
    fed_ate = np.mean(each_communi_ate_fed)
    ind_ate = np.mean(each_communi_ate_indi)
    true_ate = np.mean(each_true_ate)
all_round_rmse_fed.append(fed_Armse)
all_round_rmse_indi.append(ind_Armse)
all_round_ate_fed.append(fed_ate)
all_round_ate_indi.append(treat_mean)
all_true_ate.append(true_ate)

print(all_round_rmse_fed, "rmse_fed")
print(all_round_rmse_indi,"rmse_indi")
print(all_round_rmse_center, "rmse_center")
print(all_round_ate_fed, "ate_fed")
print(all_round_ate_indi, "ate_indi")
print(all_round_ate_center, "ate_center")
print(all_true_ate, "true_ate")




########################Case4-5 code##############################
"""
n = 5
all_round_rmse_fed = []
all_round_rmse_indi = []

all_round_ate_fed = []
all_round_ate_indi = []


min_fed_set = []
min_indi_set = []
use_fixed_data = []
para_all = []
for i in range(n):
    generator = sy(flag=i)
    all_information = generator.generate_data()
    parap_s,data_all = all_information[0],all_information[1]#[100*i:500]
    para_all.append(parap_s)
    use_fixed_data.append(data_all)


round_in_data = []
A = np.random.rand(20, 20)
global_variance_full =  np.eye(20)*(1/n)#*(1/n)#10**-3
local_list = {}
eta_global,Lambda_global =0,0 #np.random.normal(0, 1, size=(5,)),global_variance_full #0,0
for i in range(n):
    local_list[f'local_para {i}'] =np.random.normal(0, 1, size=(20,)),global_variance_full #np.random.normal(0, 1, size=(5,)),global_variance_full#np.random.normal(loc=parap_s, scale=1),#np.random.normal(0, 2, size=(5,)), global_variance_full
    #np.random.normal(loc=parap_s, scale=np.std(parap_s)),glob_samp[1]#gaussian_q_k(5)
    eta_global+=local_list[f'local_para {i}'][0]
    Lambda_global+=local_list[f'local_para {i}'][1]

delta_q = {}

for t in range(0,40):#35:
    print(f'This is the round {t} communication #####################################################')
    each_communi_rmse_fed = []
    each_communi_rmse_indi = []
    each_communi_ate_fed = []
    each_communi_ate_indi = []
    min_fed_all = 0
    min_indi_all = 0
    each_true_ate = [] ##true ATE is same for fed and individual


    cluster_for_client_indi_para = []

    delta_q_list_mean = []
    delta_q_list_cov = []
    localds = []
    round_in_data = []
    sgld = []
    for i in range(n):
        log_reg = LogisticRegression()
        log_reg.fit(use_fixed_data[i][:, 4:], use_fixed_data[i][:, 0])
        coeffice = log_reg.coef_
        cluster_for_client_indi_para.append(coeffice)
        delta_q[f'delta_q{i}']= client(eta_global, Lambda_global, local_list[f'local_para {i}'][0],
                                       local_list[f'local_para {i}'][1],
                                       delta=0.2)
        sgld_see = SG_MCMC_Infer(p_k=likelihood_p_k,
                                 eta=delta_q[f'delta_q{i}'].cavity_distribution()[0],
                                 Lambda=delta_q[f'delta_q{i}'].cavity_distribution()[1],
                                 learning_rate=0.001,
                                 num_samples=600,
                                 theta=np.random.randn(20),#delta_q[f'delta_q{i}'].cavity_distribution()[0],
                                    #data_all[i][2],#np.random.randn(5),#np.random.normal(0, 1, 5),#data_all[i][2],#np.random.normal(0, 1, 5),#,#np.random.normal(0, 1, 5),#delta_q[f'delta_q{i}'].cavity_distribution()[0],##delta_q[f'delta_q{i}'].cavity_distribution()[0],#),
                                 X= use_fixed_data[i][:, 4:],
                                 W =  use_fixed_data[i][:, 0],
                                 batch_size=int(len(use_fixed_data[i][:, 4:])*0.8))
        sample_set = sgld_see.sgld_sample()
        sample_set = sample_set[100:600]

        eta_est, Lambda_est = np.mean(sample_set, axis=0), 1*10**-15*np.eye(20)
        sgld.append([eta_est, Lambda_est])
        delta_q_list_mean.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                     0])  # update_local_distribution()[0])
        delta_q_list_cov.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                   1])  # update_local_distribution()[1])


        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0])
        loca_check = np.linalg.inv(temp_list[1]) @ temp_list[0]

        rmse_person = np.sqrt(np.mean((temp_list[0] - para_all[i]) ** 2))
        rmse_indi = np.sqrt(np.mean((coeffice - para_all[i]) ** 2))
        print(rmse_person, f'This is the client {i} federated RMSE')
        print(rmse_indi,  f'This is the client {i} individual RMSE')

        each_communi_rmse_fed.append(rmse_person)
        each_communi_rmse_indi.append(rmse_indi)
        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0] )
        local_list[f'local_para {i}'] = tuple(temp_list)
        TEG_sgld = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
              coeff=temp_list[0])
        TEG_sgld.calculate_propensity_scores()
        treat_effG = TEG_sgld.estimate_att()
        "fed ATE computing here########################################"
        each_communi_ate_fed.append(treat_effG)
        TEG_est = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
                       coeff=coeffice[0])
        TEG_est.calculate_propensity_scores()
        "indivudial ATE computing here########################################"
        treat_est = TEG_est.estimate_att()
        each_communi_ate_indi.append(treat_est)
        "###########################compute true ATE here##############################"
        #print(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]), f'client{i} real ate for sure')
        each_true_ate.append(np.abs(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3])))
        " true_ate for this_client"
        true_ate_i_client = np.abs(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]))
        " true_ate for this_client"

        min_fed_ate = true_ate_i_client - treat_effG
        min_indi_ate = true_ate_i_client - treat_est
        min_fed_all += min_fed_ate ** 2
        min_indi_all += min_indi_ate ** 2

    Sever_state = EPServer(eta_global, Lambda_global,delta_q_list_mean,delta_q_list_cov)
    eta_global, Lambda_global = Sever_state.InfServer()
    fed_Armse = np.mean(each_communi_rmse_fed)
    ind_Armse = np.mean(each_communi_rmse_indi)
min_fed_set.append(np.sqrt(min_fed_all/n))
min_indi_set.append(np.sqrt(min_indi_all/n))
all_round_rmse_fed.append(fed_Armse)
all_round_rmse_indi.append(ind_Armse)

print(all_round_rmse_fed, "rmse_fed")
print(all_round_rmse_indi,"rmse_indi")
print(min_fed_set, 'ate rmse fed')
print(min_indi_set, 'ate rmse fed indi')
"""


########################Case6-7 code##############################
"""
n = 20 #Number of Clients

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
    generator = sy(flag=i)
    all_information = generator.generate_data()
    parap_s,data_all = all_information[0],all_information[1][i*10:300]#[1]
    #data_all = generator.simulate()[1]
    para_all.append(parap_s)
    use_fixed_data.append(data_all)
#generator.generate_parameters()

"we do all of Centralized information here######################"
center_para = np.mean(np.vstack(para_all))
combined_array = np.vstack(use_fixed_data)


round_in_data = []
#########################################################################################################

#np.random.normal(loc=parap_s, scale=0.01), 5 * glob_samp[1]
#print(eta_global)
A = np.random.rand(10, 10)
global_variance_full =  np.eye(10)*(1/n)#*(1/n)#10**-3


local_list = {}
eta_global,Lambda_global =0,0 #np.random.normal(0, 1, size=(5,)),global_variance_full #0,0
for i in range(n):
    local_list[f'local_para {i}'] =np.random.normal(0, 1, size=(10,)),global_variance_full #np.random.normal(0, 1, size=(5,)),global_variance_full#np.random.normal(loc=parap_s, scale=1),#np.random.normal(0, 2, size=(5,)), global_variance_full
    #np.random.normal(loc=parap_s, scale=np.std(parap_s)),glob_samp[1]#gaussian_q_k(5)
    eta_global+=local_list[f'local_para {i}'][0]
    Lambda_global+=local_list[f'local_para {i}'][1]

delta_q = {}

for t in range(0,40):#35:
    each_communi_rmse_fed = []
    each_communi_rmse_indi = []
    each_communi_ate_fed = []
    each_communi_ate_indi = []
    min_fed_all = 0
    min_indi_all = 0
    each_true_ate = [] ##true ATE is same for fed and individual


    cluster_for_client_indi_para = []

    delta_q_list_mean = []
    delta_q_list_cov = []
    localds = []
    round_in_data = []
    sgld = []
    #parallel computing for the clients, the num of lopps mean how many clients
    for i in range(n):
        input_X = scaler.fit_transform(use_fixed_data[i][:, 4:])
        log_reg = LogisticRegression()
        log_reg.fit(use_fixed_data[i][:, 4:], use_fixed_data[i][:, 0])
        coeffice = log_reg.coef_
        cluster_for_client_indi_para.append(coeffice)


        delta_q[f'delta_q{i}']= client(eta_global, Lambda_global, local_list[f'local_para {i}'][0],
                                       local_list[f'local_para {i}'][1],
                                       delta=0.2)
        sgld_see = SG_MCMC_Infer(p_k=likelihood_p_k,
                                 eta=delta_q[f'delta_q{i}'].cavity_distribution()[0],
                                 Lambda=delta_q[f'delta_q{i}'].cavity_distribution()[1],
                                 learning_rate=0.001,
                                 num_samples=600,
                                 theta=np.random.randn(10),
                                 X= use_fixed_data[i][:, 4:],
                                 W =  use_fixed_data[i][:, 0],
                                 batch_size=int(len(use_fixed_data[i][:, 4:])*0.8))

        sample_set = sgld_see.sgld_sample()
        sample_set = sample_set[100:600]

        eta_est, Lambda_est = np.mean(sample_set, axis=0), 1*10**-20*np.eye(10)

        eta_add, Lambda_add = para_all[i] + delta_q[f'delta_q{i}'].cavity_distribution()[0], np.cov(
            sample_set.T) + delta_q[f'delta_q{i}'].cavity_distribution()[1]
        sgld.append([eta_est, Lambda_est])

        delta_q_list_mean.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                     0])  # update_local_distribution()[0])
        delta_q_list_cov.append(delta_q[f'delta_q{i}'].get_localnew_parameters(eta_est, Lambda_est)[
                                   1])  # update_local_distribution()[1])


        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0])
        loca_check = np.linalg.inv(temp_list[1]) @ temp_list[0]

        rmse_person = np.sqrt(np.mean((temp_list[0] - para_all[i]) ** 2))
        rmse_indi = np.sqrt(np.mean((coeffice - para_all[i]) ** 2))
        print(rmse_person, 'Local RMSE')
        print(rmse_indi, 'no fed RMSE')

        each_communi_rmse_fed.append(rmse_person)
        each_communi_rmse_indi.append(rmse_indi)


        temp_list = list(local_list[f'local_para {i}'])
        temp_list[0], temp_list[1] = delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[2], delta_q[f'delta_q{i}'].get_localnew_parameters(sgld[i][0], sgld[i][1])[3]
        localds.append(np.linalg.inv(temp_list[1]) @ temp_list[0] )
        #localds.append(temp_list[0])
        local_list[f'local_para {i}'] = tuple(temp_list)

        TEG_sgld = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
              coeff=temp_list[0])
        TEG_sgld.calculate_propensity_scores()
    #calculate_propensity_scores(
        treat_effG = TEG_sgld.estimate_att()
        "fed ATE computing here########################################"
        each_communi_ate_fed.append(treat_effG)

        print(treat_effG,f'client{i} estimate ate')
        #print(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]), f'client{i} real ate')
        TEG_est = TEE(X=use_fixed_data[i][:, 4:], W=use_fixed_data[i][:, 0], Y=use_fixed_data[i][:, 1],
                       coeff=coeffice[0])
        TEG_est.calculate_propensity_scores()
        "indivudial ATE computing here########################################"
        treat_est = TEG_est.estimate_att()
        each_communi_ate_indi.append(treat_est)

        "###########################compute true ATE here##############################"
        print(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]), f'client{i} real ate for sure')
        each_true_ate.append(np.abs(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3])))

        " true_ate for this_client"
        true_ate_i_client = np.abs(np.mean(use_fixed_data[i][:, 2] - use_fixed_data[i][:, 3]))
        " true_ate for this_client"

        min_fed_ate = true_ate_i_client - treat_effG  # minus fed ate
        min_indi_ate = true_ate_i_client - treat_est  # minus indivudual ate
        min_fed_all += min_fed_ate ** 2
        min_indi_all += min_indi_ate ** 2
    Sever_state = EPServer(eta_global, Lambda_global,delta_q_list_mean,delta_q_list_cov)
    eta_global, Lambda_global = Sever_state.InfServer()
    stacked_arrays = np.stack(cluster_for_client_indi_para, axis=2)
    fed_Armse = np.mean(each_communi_rmse_fed)
    ind_Armse = np.mean(each_communi_rmse_indi)
min_fed_set.append(np.sqrt(min_fed_all/n))
min_indi_set.append(np.sqrt(min_indi_all/n))

all_round_rmse_fed.append(fed_Armse)
all_round_rmse_indi.append(ind_Armse)


print(all_round_rmse_fed, "rmse_fed")
print(all_round_rmse_indi,"rmse_indi")


print(min_fed_set)
print(min_indi_set)

"""













