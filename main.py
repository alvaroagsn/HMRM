import os
import sys
import warnings

warnings.filterwarnings("ignore")

from hmrm.configuration.data import OutputDirectory, InputDirectory
from hmrm.base.hmrmBaseline import HmrmBaseline
from hmrm.base.hmrm_new import HmrmBaselineNew

if __name__ == "__main__":
    base_dir = "C:/Users/alvar/OneDrive/Documentos/GitHub/HMRM/hmrm/data"
    file = os.path.join(base_dir, "pois_local_Illinois_cat_placeid.csv")

    weight = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    K = [7]
    embedding_size = [10, 20, 30, 40, 50, 100, 150, 200, 250, 300]

    # for w in weight:
    #     for k in K:
    #         for e in embedding_size:
    #             out = OutputDirectory(f"features-{w}-{k}-{e}.csv")
    #             hmrm = HmrmBaseline(file, w, k, e)
    #             df = hmrm.start()
    #             print("Weight: ", w, " K: ", k, " Embedding Size: ", e)
    #             df.to_csv(out.outfile, index=False)

    hmrm = HmrmBaseline(file, 0.1, 7, 50)
    df = hmrm.start()
    df.to_csv(OutputDirectory.outfile, index=False)

