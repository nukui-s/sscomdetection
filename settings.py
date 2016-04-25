from string import Template

#LRF settings
N = 500
minc = 100
maxc = 100
mu = 0.3
k = 5
maxk = 50
t1 = 2
t2 = 1

name_tmp = Template("LRF_${N}_${k}_${maxk}_${minc}_${maxc}_${mu}")
data_label = name_tmp.substitute(N=N, k=k, maxk=maxk, minc=minc, maxc=maxc,
                           mu=mu)


#General settings
densities =[0,1,2,3,4,5]
trials = 5
K = 5
lr_adam = 0.01
lr_sgd = 0.01
threshold = 10e-9
threads = 4
used_models = ["abs_sgd","clip_sgd","clip_adam","abs_adam","update_rule"]
#used_models = ["abs_adam","update_rule"]
max_iters = 1000

