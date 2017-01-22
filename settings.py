from string import Template

const_base = "data/const/degree_order_{}_{}.pkl"

#LRF settings
N = 500
minc = 100
maxc = 100
mu = 0.3
k = 5
#k = 10
#maxk = 20
maxk = 50
t1 = 2
t2 = 1

name_tmp = Template("LRF_${N}_${k}_${maxk}_${minc}_${maxc}_${mu}")
lrf_data_label = name_tmp.substitute(N=N, k=k, maxk=maxk, minc=minc, maxc=maxc,
                           mu=mu)

data_label = lrf_data_label

#data_label = "gn_1000_4"
#data_label = "polblogs"
exp_name = "{}_degree_order_asc".format(data_label)
print("*********************", exp_name)

#General settings
#densities =[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
densities = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
trials = 5
K = 5
lr_adam = 0.01
lr_sgd = 0.01
threshold = 10e-9
threads = 8
used_models = ["abs_adam","update_rule"]
#used_models = ["abs_adam", "update_rule"]
#used_models = ["abs_adam"]
max_iters = 1000

mlambda = 2
