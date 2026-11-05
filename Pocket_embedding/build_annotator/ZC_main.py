#from new_binding_pocket_analyse import process_all
#from binding_pocket_filter  import get_valids
#from new_build_dataset import get_binding_site_graphs_all
#from annotator import annotate_all

import pickle
import os
def recursive_dd():
    return defaultdict(recursive_dd)
#pdb_path = os.path.join("./11")
#print (pdb_path)
#process_all(pdb_path, "./1/1.p")

#d = pickle.load(open('./1/1.p', 'rb'))
#c = 10
#conc = .6
#ligs = get_valids(d, c, conc, min_size=4)
#pickle.dump(ligs, open("./2/2.p", "wb"))
#get_binding_site_graphs_all('./2/2.p', './out_dump', non_binding=False)

#annotate_all(parallel=False, graph_path="./out_dump", dump_path="./out_dump2", 
        #ablate=False, mode='fp')
