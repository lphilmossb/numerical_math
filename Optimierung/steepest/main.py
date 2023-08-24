from numpy import *
from steepest import GradientDescent
import matplotlib.pyplot as plt
from functions import *


graddes = GradientDescent(q, nabla_q, N=10000, M=100)

x0 = ones(3,dtype=float)

# print(graddes.run(x0))
print(graddes.run(x0,2))
