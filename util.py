import json
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans

def get_objectlatent_matrix(A, k, latentspace):
    latentMatrix = None
    if(latentspace == 'svd'):
        U, S, Vh = np.linalg.svd(A, full_matrices=False)
        return U
    elif latentspace == 'nmf':
        A = np.asarray(A)
        A[A<0] = 0
        nmf_model = NMF(init="nndsvd", n_components=k)
        return nmf_model.fit_transform(A)
    elif latentspace == 'lda':
        A = np.asarray(A)
        A[A<0] = 0
        lda = LatentDirichletAllocation(n_components=k)
        lda.fit(A)
        return lda.transform(A)
    elif latentspace == 'kmeans':
        A = np.asarray(A)
        A[A<0] = 0
        return KMeans(n_clusters=k, n_init="auto").fit_transform(A)


def get_latent_semantics(k, vectorspace, latentspace):
    json_file = open('descriptors/'+vectorspace+'_desc_a2.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    data = json.loads(loaded_model_json)
    int_ids = []
    for i in data.keys():
        int_ids.append(int(i))
    int_ids.sort()
    A = []
    for i in int_ids:
        A.append(data[str(i)]['feature-descriptor'])
    U = get_objectlatent_matrix(A, k, latentspace)
    res = {}
    j = 0
    for i in int_ids:
        sdictk = {}
        sdictk['feature-descriptor'] = list(U[j])
        sdictk['label'] = data[str(i)]['label']
        res[str(i)] =  sdictk
        j += 1
    return res