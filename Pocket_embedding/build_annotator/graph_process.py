"""
    Tools for processing pdb graphs.
"""
import torch
import itertools
from random import shuffle
import numpy as np
import networkx as nx
import torch.nn.functional as F
import math
import re
faces = ['W', 'S', 'H']
orientations = ['c', 't']
valid_edges = set(['B53'] + [orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations])

def remove_self_loops(G):
    selfs = list(nx.selfloop_edges(G))
    G.remove_edges_from(selfs)

def remove_non_standard_edges(G):
    remove = []
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in valid_edges:
            remove.append((n1, n2))
    G.remove_edges_from(remove)

def symmetric(label):

    #if not a base pair keep the same
    if label[0] not in ['c', 't']:
        return label

 
    special_cases = {
        "cWH": "cWH", "cHW": "cWH",
        "tWH": "tWH", "tHW": "tWH",
        "tWS": "tWS", "tSW": "tWS",
        "cSH": "cHS", "cHS": "cHS",
        "tHS": "tHS", "tSH": "tHS",
        "cWS": "cWS", "cSW": "cWS",
    }
    if label in special_cases:
         return special_cases[label] 
        #return special_cases.get[label] 

           
        

def get_max_idx(G):
    max_idx = -1
    for n in G.nodes():
        try:
            idx = int(n.split('.')[-1])
            max_idx = max(max_idx, idx)
        except (ValueError, IndexError):
            print(f"⚠️ 节点名 {n} 格式错误，无法提取编号")
    return max_idx
def replace_idx_regex(node_name, new_idx):
    return re.sub(r'\d+$', str(new_idx), node_name)
    
def fix_nan_features(G, target_keys,log_file="out_zc.txt"):
     
    max_index = get_max_idx(G)
    failed_nodes = set()
    for n, d in G.nodes(data=True):    #n:7sxp.A.6

        idx = int(n.split('.')[-1]) 
        

        for key in target_keys:
            value = d.get(key)
           # print (G.nodes[n].get(key))
            if value is None or (isinstance(value, float) and math.isnan(value)):
                # 查找前后邻居
                prev_val = next_val = None
                step = 1
                while idx - step > 0:     # idx>1
                      prev_node = replace_idx_regex(n, idx - step)
                      if prev_node in G.nodes:
                         prev = G.nodes[prev_node].get(key)
                         if isinstance(prev, float) and not math.isnan(prev):
                             prev_val = prev
                             break
                      step += 1 
                
 
                step = 1
                while idx + step <= max_index:
                      nxt_node = replace_idx_regex(n, idx + step) 
                      if nxt_node in G.nodes:
                         nxt = G.nodes[nxt_node].get(key)
                         if isinstance(nxt, float) and not math.isnan(nxt):
                            next_val = nxt
                            break
                      step += 1
                              
                if prev_val is not None and next_val is not None:
                      G.nodes[n][key] = (prev_val + next_val) / 2.0
                      #print (f"目标节点{n}的前邻居节点{prev_node}")
                      #print (f"目标节点{n}的后邻居节点{nxt_node}")
                elif prev_val is not None:
                      G.nodes[n][key] = prev_val
                     # print (f"目标节点{n}的前邻居节点{prev_node}")
                elif next_val is not None:
                      G.nodes[n][key] = next_val  
                     # print (f"目标节点{n}的后邻居节点{nxt_node}")
                else:
                      failed_nodes.add(
                        f"{n}\tidx={idx}\tkey={key}")
                     


def add_dbn_onehot(H):
    for n in H.nodes():
        dbn = H.nodes[n].get('dbn', None)       
        #print (n)      
        if dbn in ('(', ')'):
            H.nodes[n]['dbn_onehot'] = [0, 1]
        else:
            H.nodes[n]['dbn_onehot'] = [1, 0]

        


def add_sin_cos_features(H, angle_params):
    for param in angle_params:
        sin_param = f"sin_{param}"
        cos_param = f"cos_{param}"
        
        # 遍历图中所有节点
        for n in H.nodes():
            # 获取角度值，假设该节点有这个角度的属性
            angle_value = H.nodes[n].get(param, None)
            if angle_value is not None:
                # 将角度从度转换为弧度
                rad = np.deg2rad(angle_value)
                
                # 计算 sin 和 cos 值
                H.nodes[n][sin_param] = np.sin(rad)
                H.nodes[n][cos_param] = np.cos(rad)
            else:
                # 如果没有该角度的值，则给 sin 和 cos 设置为 NaN 或默认值
                H.nodes[n][sin_param] = np.nan
                H.nodes[n][cos_param] = np.nan
            if param in H.nodes[n]:
                del H.nodes[n][param]
angle_params = ['alpha','beta','gamma','delta','epsilon','zeta','epsilon_zeta', 'v0', 'v1', 'v2', 'v3', 'v4']
target_keys = ['sin_alpha','sin_beta','sin_gamma','sin_delta','sin_epsilon','sin_zeta','sin_epsilon_zeta', 'sin_v0', 'sin_v1', 'sin_v2', 'sin_v3', 'sin_v4','cos_alpha','cos_beta','cos_gamma','cos_delta','cos_epsilon','cos_zeta','cos_epsilon_zeta', 'cos_v0', 'cos_v1', 'cos_v2', 'cos_v3','cos_v4','ssZp','Dp','suiteness','dbn_onehot','nuc_onehot','bin_onehot']
def to_orig(G):
    """
        Convert new networkx graphs (from vlad 2019) to original format.
        Keep only B53 and 12 base pair interactions.
        Annotate nodes with position, chain, and nucleotide ID attributes.

        Arguments:
            G (networkx graph): networkx graph to convert to original.
        Returns:
            networkx graph
    """
    remove_self_loops(G)
    # remove_non_standard_edges(G)

    H = nx.Graph()


    for n1,n2, d in G.edges(data=True):   
        label = symmetric(d['LW'])  
        #print (valid_edges)
        if label in valid_edges:
            H.add_edge(n1, n2, label=label)      


    
    d_orig_1 = {n:d['nt_resnum'] for n,d in G.nodes(data=True)}    
    d_orig_2= {n:d['nt_code'] for n,d in G.nodes(data=True)}   
    d_orig_3 = {n:d['chain_name'] for n,d in G.nodes(data=True)} 
    d_orig_4 = {n:d['dbn'] for n,d in G.nodes(data=True)}    
    d_orig_5 = {n:d['alpha'] for n,d in G.nodes(data=True)}
    d_orig_6 = {n:d['beta'] for n,d in G.nodes(data=True)}
    d_orig_7 = {n:d['gamma'] for n,d in G.nodes(data=True)}
    d_orig_8 = {n:d['delta'] for n,d in G.nodes(data=True)}
    d_orig_9 = {n:d['epsilon'] for n,d in G.nodes(data=True)}
    d_orig_10 = {n:d['zeta'] for n,d in G.nodes(data=True)}
    d_orig_11 = {n:d['epsilon_zeta'] for n,d in G.nodes(data=True)}
    d_orig_13 = {n:d['ssZp'] for n,d in G.nodes(data=True)}
    d_orig_14 = {n:d['Dp'] for n,d in G.nodes(data=True)}
    d_orig_24 = {n:d['v0'] for n,d in G.nodes(data=True)}
    d_orig_25 = {n:d['v1'] for n,d in G.nodes(data=True)}
    d_orig_26 = {n:d['v2'] for n,d in G.nodes(data=True)}
    d_orig_27 = {n:d['v3'] for n,d in G.nodes(data=True)}
    d_orig_28 = {n:d['v4'] for n,d in G.nodes(data=True)}   
    d_orig_29 = {n:d['suiteness'] for n,d in G.nodes(data=True)}
    d_orig_31 = {n:d['bin'] for n,d in G.nodes(data=True)}

    #nx.set_node_attributes(H, {n:d_orig_1[n] for n in H.nodes()}, 'pdb_pos')    
   # nx.set_node_attributes(H, {n:d_orig_3[n] for n in H.nodes()}, 'chain')
    nx.set_node_attributes(H, {n:d_orig_5[n] for n in H.nodes()}, 'alpha')
    nx.set_node_attributes(H, {n:d_orig_6[n] for n in H.nodes()}, 'beta')
    nx.set_node_attributes(H, {n:d_orig_7[n] for n in H.nodes()}, 'gamma')
    nx.set_node_attributes(H, {n:d_orig_8[n] for n in H.nodes()}, 'delta')
    nx.set_node_attributes(H, {n:d_orig_9[n] for n in H.nodes()}, 'epsilon')
    nx.set_node_attributes(H, {n:d_orig_10[n] for n in H.nodes()}, 'zeta')
    nx.set_node_attributes(H, {n:d_orig_11[n] for n in H.nodes()}, 'epsilon_zeta')
    nx.set_node_attributes(H, {n:d_orig_24[n] for n in H.nodes()}, 'v0')
    nx.set_node_attributes(H, {n:d_orig_25[n] for n in H.nodes()}, 'v1')
    nx.set_node_attributes(H, {n:d_orig_26[n] for n in H.nodes()}, 'v2')
    nx.set_node_attributes(H, {n:d_orig_27[n] for n in H.nodes()}, 'v3')
    nx.set_node_attributes(H, {n:d_orig_28[n] for n in H.nodes()}, 'v4')
    nx.set_node_attributes(H, {n:d_orig_13[n] for n in H.nodes()}, 'ssZp')
    nx.set_node_attributes(H, {n:d_orig_14[n] for n in H.nodes()}, 'Dp')
    nx.set_node_attributes(H, {n:d_orig_29[n] for n in H.nodes()}, 'suiteness')    
    nx.set_node_attributes(H, {n:d_orig_4[n] for n in H.nodes()}, 'dbn')
    nx.set_node_attributes(H, {n:d_orig_2[n] for n in H.nodes()}, 'nt')
    nx.set_node_attributes(H, {n:d_orig_31[n] for n in H.nodes()}, 'bin')

    nuc_map = {n:i for i,n in enumerate(['A', 'U', 'C', 'G'])}
    bin_map = {n:i for i,n in enumerate(['22p', '32p', '23p', '33p', '22t', '32t', '23t', '33t', '22m', '32m', '23m', '33m','inc','trig' ])}
    onehot_nuc  = {node: F.one_hot(torch.tensor(nuc_map[label]),num_classes=4).float() for node, label in (nx.get_node_attributes(H, 'nt')).items()}  
    onehot_bin  = {node: F.one_hot(torch.tensor(bin_map[label]),num_classes=14).float() for node, label in (nx.get_node_attributes(H, 'bin')).items()}    
    add_sin_cos_features(H, angle_params)
    add_dbn_onehot(H) 
    nx.set_node_attributes(H, name='nuc_onehot', values=onehot_nuc) 
    nx.set_node_attributes(H, name='bin_onehot', values=onehot_bin)    
    fix_nan_features(H, target_keys)
 
 
    for n in H.nodes():
        del H.nodes[n]['bin']
        del H.nodes[n]['nt']
        del H.nodes[n]['dbn']
        
    


    return H

def kill_islands(G, min_size=4):
    """
        Kill components that are too small
    """
    remove = []
    for comp in nx.connected_components(G):
        if len(comp) < min_size:
            remove.extend(list(comp))
        pass
    G.remove_nodes_from(remove)


def find_node(graph, chain, pos):
    for n,d in graph.nodes(data=True):

        if (  str(d['nt_resnum']) == str(pos)):   
            return n

    return None

def has_NC(G):
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in ['cWW', 'B53']:
            return True
    return False

def bfs_expand(G, initial_nodes, depth=4):
    """
        Extend motif graph starting with motif_nodes.
        Returns list of nodes.
    """

    total_nodes = [list(initial_nodes)]
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))

def floaters(G):
    """
    Try to connect floating base pairs. (Single base pair not attached to backbone).
    Otherwise remove.
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u,v in G.edges():
        if deg_ok(G, u, v, 1):
            floaters.append((u,v))

    G.remove_edges_from(floaters)

    return G

def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.
    """
    is_backbone = lambda n,G: sum([G[n][nei]['label'] != 'B53' for nei in G.neighbors(n)]) == 0
    #degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        dangles = []
        for n in cur_G.nodes:
            # node_deg = degree(i, G, current_nodeset)
            # print(node_deg)
            # if node_deg == 2 and is_backbone(n, G):
            ##if cur_G.degree(n) == 1 and is_backbone(n, cur_G) or cur_G.degree(n) == 0:
            if   cur_G.degree(n) == 0:
                 dangles.append(n)
        if len(dangles) == 0:
            break
       
        cur_G.remove_nodes_from(dangles)
            #cur_G = cur_G.copy()
    return cur_G

def stack_trim(G):
    """
    Remove stacks from graph.
    """
    is_ww = lambda n,G: 'cWW' in [info['label'] for node,info in G[n].items()]
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        stacks = []
        for n in cur_G.nodes:
            if cur_G.degree(n) == 2 and is_ww(n, cur_G):
                #potential stack opening
                partner = None
                stacker = None
                for node, info in cur_G[n].items():
                    if info['label'] == 'B53':
                        stacker = node
                    elif info['label'] == 'cWW':
                        partner = node
                    else:
                        pass
                if cur_G.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_G[partner].items():
                    if info['label'] == 'B53':
                        stacker_2 = node
                    elif info['label'] == 'cWW':
                        partner_2 = node
                try:
                    if cur_G[stacker][stacker_2]['label'] == 'cWW':
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_G.remove_nodes_from(stacks)
            cur_G = cur_G.copy()
    return cur_G

def in_stem(G, u,v):
    non_bb = lambda G, n: len([info['label'] for node, info in G[n].items() if info['label'] != 'B53'])
    is_ww = lambda G, u, v: G[u][v]['label'] == 'cWW'
    if is_ww(G,u,v) and (non_bb(G,u) in (1, 2)) and (non_bb(G,v) in (1, 2)):
        return True
    return False

if __name__ == "__main__":
      y= {'LW': 'cHW'}
      label_y = symmetric(y['LW']) 
      print (label_y)
   # pass
