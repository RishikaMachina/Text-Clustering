class file_read():
    def __init__(self):
        pass
   
    def read_file(self):
        with open("train.dat") as file:
            line = file.readlines()

        rows = len(line)
        cols = 0 
        nonZeroCount = 0 
        n = 0

        for i in range(rows):
            line_splitted = line[i].split()
            nonZeroCount += len(line_splitted)/2
            for j in range(0, len(line_splitted), 2): 
                columval = int(line_splitted[j]) - 1
                if columval + 1 > cols:
                    cols = columval + 1
        indptr = np.zeros(int(rows) + 1, dtype = np.float)
        indices = np.zeros(int(nonZeroCount), dtype = np.int)
        data = np.zeros(int(nonZeroCount), dtype = np.float)

        for i in range(rows):
            p = line[i].split()
            for j in range(0, len(p), 2): 
                indices[n] = int(p[j]) - 1
                data[n] = float(p[j+1])
                n += 1
            indptr[i+1] = n 

        return csr_matrix((data, indices, indptr), shape=(rows, cols), dtype = np.float)
    
    def norm(self, mat):
        rows = mat.shape[0]
        val, p =  mat.data, mat.indptr
        for i in range(rows):
            row_Sums = 0.0    
            for j in range(p[i], p[i+1]):
                row_Sums += val[j]**2
            if row_Sums == 0.0:
                continue 
            row_Sums = float(1.0 / np.sqrt(row_Sums))
            for j in range(p[i], p[i+1]):
                val[j] *= row_Sums
    
class modeling(file_read):
    def __init__(self):
        pass
    
    def Cluster(mat, centroids):
        valueList = list()
        matrixsimilarity = mat.dot(centroids.T)

        for i in range(matrixsimilarity.shape[0]):
            row = matrixsimilarity.getrow(i).toarray()[0].ravel()
            indicestop = row.argsort()[-1]
            valueList.append(indicestop + 1)
        return valueList

    def bisectingKMeans(self, mat, z):
        d = mat
        final_clusters = []   
        present_clusters = []      
        for i in range(mat.shape[0]):
            present_clusters.append(i)

        while len(final_clusters) < z - 1:
            error_first_cluster = 0
            error_second_cluster = 0
            onelist, twolist, one_cluster, two_cluster, one, two = Kmean(mat, d, present_clusters)
            for row in one_cluster:
                error_first_cluster += (euclidean(row.toarray(),one.toarray()))**2
            for row in two_cluster:
                error_second_cluster += (euclidean(row.toarray(),two.toarray()))**2    
            if error_first_cluster < error_second_cluster:
                final_clusters.append(onelist)
                present_clusters = twolist
                d = two_cluster
            else:
                final_clusters.append(twolist)
                present_clusters = onelist
                d = one_cluster
        final_clusters.append(present_clusters)
        return final_clusters

    
from collections import Counter, defaultdict
import random
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import euclidean
import scipy.sparse
from sklearn.utils import shuffle
from scipy.sparse import find, csr_matrix

prog = modeling()
mat = prog.read_file()
prog.norm(mat)
cal = prog.bisectingKMeans(mat,7)

pred = [0]* mat.shape[0]
for i in range(len(cal)):
    for j in range(len(cal[i])):
        pred[cal[i][j]] = i + 1
        
d = {}
for i in range(len(cal)):
    for j in range(len(cal[i])):
        d[cal[i][j]] = i

final = []
for key in sorted(d.keys()):
    final.append(d[key]) 
    
    
ui = []
ui.append('ItemID,ClusterID\n')
for i in range(len(final)):
    tr = ""
    tr += str(str((i+1)) +','+ str(final[i]))
    ui.append(tr)

file = open('final.dat',"w")
file.write("\n".join(map(lambda x: x, ui)))
file.close()
