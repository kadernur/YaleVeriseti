# -*- coding: utf-8 -*-
"""
Created on Wed Jan 26 10:51:59 2022

@author: kader
"""


import numpy as numpy
USE_SVD = True

# M = numpy.matrix([[3,2,1,4,5,8,4,5,4],[7,8,9,1,5,0,6,3,8],[1,5,4,4,2,3,1,7,9],[2,5,7,4,3,8,7,3,1],[8,4,3,1,7,5,2,2,1],[2,5,8,4,3,6,8,2,1],[7,3,5,4,6,8,4,0,3]])
# print(M)

# d,N = M.shape
# print(M.shape)

# k=7
# assert k<N
# m_estimate = []
# var_total = 0
# for row in range(N):
#   if row%500==0:print('finished %s out of %s' % (row,N))
#   M_Mi = numpy.array(M-M[:,row])
#   #print(M_Mi)
#   vec = (M_Mi**2).sum(0)
#   #print(vec)
#   nbrs = numpy.argsort(vec)[1:k+1]
#   print(nbrs)
#   x = numpy.matrix(M[:,nbrs] - M[:,row])
#   print(x)
#   sig2 = (numpy.linalg.svd(x,compute_uv=0))**2
#   print(sig2)
#   sig2 /= sig2.sum()
#   print(sig2)
#   S = sig2.cumsum()
#   print(S)
#   v=0,9
#   m = S.searchsorted(v)
#   print(m)
  
 
# W = numpy.zeros((N,N))
# #print(W)


 
# for row in range(N):
#      M_Mi = numpy.array(M-M[:,row])
#      print(M_Mi)
 
#      vec = (M_Mi**2).sum(0)
#      nbrs = numpy.argsort(vec)[1:k+1]
#      print(nbrs)
#      M_Mi = numpy.matrix(M_Mi[:,nbrs])
#      print(M_Mi)
#      # Q = M_Mi.T * M_Mi

x=numpy.matrix([[7,2],[5,4],[1,2],[1,3]])
print(x)
sig2 = (numpy.linalg.svd(x,compute_uv=0))
print(sig2)
sig2 = (numpy.linalg.svd(x,compute_uv=0))**2
print(sig2)
S = sig2.cumsum()
print(S)
v=0.9
m = S.searchsorted(v)
print(m)



# from sklearn.manifold import LocallyLinearEmbedding
# for i in range(10):
#     for j in range(20):
      
#         print(str((i+1)*3) + str((j+1)*5))
#         embedding = LocallyLinearEmbedding(n_neighbors = (i+1)*3, n_components=(j+1)*5)
#         trainKlpp = embedding.fit_transform(X_train)
#         testKlpp = embedding.transform(X_test)
    





