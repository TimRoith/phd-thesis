import numpy as np

#%%
num_pts = int(2)
a = np.ones((num_pts,))
b = np.ones((num_pts,))

res = np.sum(a)/(np.sum(1/b))
res2 = 1/num_pts * (np.sum(a*b))
print('Sum: ' + str(res))
print('Sum2 :' + str(res2))