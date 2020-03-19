import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

with open('format.dat') as fn:
    documents = fn.readlines()
