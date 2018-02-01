import sobol_seq
import matplotlib.pyplot as plot


sob = sobol_seq.i4_sobol_generate(2,128)

plot.scatter(sob[:,0], sob[:,1])

plot.show()