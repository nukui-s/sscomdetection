import os
import time
from string import Template
from conv2pkl import conv2pkl
from visualize import plot_degree_dist
import pandas as pd
from make_constraint import make_constraints
import seaborn
import matplotlib.pyplot as plt
from string import Template
from settings import *


#cmd =Template("""./benchmark -k $k -N $N -mu $mu -minc $minc -maxc $maxc -maxk $maxk""")

cmd =Template("""./binary_networks/benchmark -k $k -N $N -mu $mu -t1 $t1 -t2 $t2 -minc $minc -maxc $maxc -maxk $maxk""")

query = cmd.substitute(N=N, k=k, mu=mu, t1=t1, t2=t2, maxc=maxc, minc=minc, maxk=maxk)
print(query)
ret = os.system(query)
if ret != 0:
    raise SystemError
    quit(1)

#os.system(cmd.substitute(N=N, k=k, mu=mu, maxc=maxc, minc=minc, maxk=maxk))
conv2pkl(data_label)
os.system("mv ./binary_networks/data/*.pkl data/")

path = os.path.join("data/",data_label + "_edge.pkl")
elist = pd.read_pickle(path)
print(len(elist))
#plot_degree_dist(elist)

print(data_label)
make_constraints(data_label)

