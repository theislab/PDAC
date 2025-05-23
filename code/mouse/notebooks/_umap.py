import pandas as pd
import scanpy as sc
import warnings
import scarches as sca
warnings.filterwarnings("ignore")


from sklearn_ann.kneighbors.annoy import AnnoyTransformer
adata = sc.read_h5ad('/mnt/storage/Daniele/atlases/mouse/06_mouse_inhouse_integrated_scanvi.h5ad')
sc.pp.neighbors(adata, use_rep = 'X_scANVI', transformer=AnnoyTransformer(adata.n_obs // 1000))
sc.tl.umap(adata, min_dist=0.2)
sc.tl.leiden(adata, resolution=0.5, flavor='igraph', key_added='leiden_0.5')

adata.write_h5ad('/mnt/storage/Daniele/atlases/mouse/07_mouse_inhouse_integrated_scanvi_umap.h5ad')