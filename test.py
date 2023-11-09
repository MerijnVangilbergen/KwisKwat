import numpy as np

N = 20
p = np.column_stack([range(N),range(N)])
print(p)
jj=19
print(p[[jj-1,jj,(jj+1)%N]])