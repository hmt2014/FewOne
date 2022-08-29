import pickle
from sklearn.manifold import TSNE
import torch
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm


if __name__ == '__main__':
    type_ = "multi"
    X, _, label = pickle.load(open("./features/acd/DistilBert/"+type_+"_DistilBERT_maml.p", "rb"))


    X = np.array(X)
    n_samples, n_features = X.shape
    print(X.shape)

    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    #print(mean)

    norm_X = X - mean

    cov_matrix = np.cov(norm_X, rowvar=False)
    #print(cov_matrix)
    print("avg cov", abs(cov_matrix).mean())

