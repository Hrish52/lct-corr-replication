import numpy as np
import matplotlib.pyplot as plt

def heatmap(mask_vec, p, title="Significant edges"):
    iu, ju = np.triu_indices(p, 1)
    M = np.zeros((p,p), dtype=int)
    M[iu, ju] = mask_vec.astype(int)
    M = M + M.T
    plt.figure()
    plt.imshow(M, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    return M