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
    
