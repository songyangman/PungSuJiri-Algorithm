# test_please
from sko.GA import GA

func = lambda x: x[0] ** 2
ga = GA(func=func, n_dim=1)
ga.run(10)
ga.run(20)