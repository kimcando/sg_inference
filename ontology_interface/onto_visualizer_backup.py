import json
import networkx as nx
from itertools import combinations
import numpy as np
import pdb
from matplotlib import pyplot as plt
class GTDictHandler:
    """
    For VG based scene graph generator
    """
    def __init__(self, json_path):
        # object_count, idx_to_label, predicate_to_idx, predicate_count, idx_to_predicate, label_to_idx
        self.vg_dict, self.vg_whole_key = self.load_ground_truth(json_path)
        self.idx_label = self.idx_to_dict('idx_to_label')
        self.idx_predicate = self.idx_to_dict('idx_to_predicate')

    def load_ground_truth(self, json_path):
        with open(json_path) as f:
            vg_dict = json.load(f)
        # only in first depth keys
        whole_key = self.check_key(vg_dict)
        return vg_dict, whole_key

    def check_key(self, data_dict):
        whole_key = dict()
        print('@ Checking keys')
        for idx, k in enumerate(data_dict.keys()):
            print(f' > {k}')
            whole_key[idx] = k
        return whole_key

    def idx_to_dict(self, key):
        new_obj = dict()
        print(f'@{key} dictrionary generating')
        for idx, (k, v) in enumerate(self.vg_dict[key].items()):
            new_obj[int(k)] = v
        print(f' > total length for {key} : {idx+1}')
        return new_obj

    @property
    def get_idx_label(self):
        return self.idx_label

    @property
    def get_idx_predicate(self):
        return self.idx_predicate

class GraphHandler:
    def __init__(self, gt_data_obj):
        self.G = self.new_graph()
        self.gt_data_obj = gt_data_obj
        self.idx_label = gt_data_obj.get_idx_label
        self.idx_predicate = gt_data_obj.get_idx_predicate

    def new_graph(self):
        G = nx.Graph()
        return G

    def print_graph_info(self, G=None):
        if G is None:
            print(f'number of nodes in this graph: {self.G.nodes()}')
            print(f'number of edges in this graph: {self.G.edges()}')
            print(f'edge label in this graph:{self.edge_label}')
        else:
            print(f'number of nodes in this graph: {G.nodes()}')
            print(f'number of edges in this graph: {G.edges()}')

    def generate_SG(self, triplet):
        obj = self.add_sub_obj(triplet)
        rel, self.edge_label = self.add_relation(triplet, obj)
        self.print_graph_info()

    def add_sub_obj(self, triplet, rank=20):
        """
        total_obj: dictionary of list
        """
        total_len = len(triplet['data_img']['triplet'][:rank])
        #  make subject/object dictionary
        obj = {i:[] for i in range(total_len)}
        for i in range(total_len):
            obj[i].append(triplet['data_img']['triplet'][i][0])
            obj[i].append(triplet['data_img']['triplet'][i][-1])

        #  add // can be done above but just for separting
        for k, v_list in obj.items():
            for v in v_list:
                # pdb.set_trace()
                self.G.add_node(self.idx_label[v])
        return obj

    def add_relation(self,triplet, obj, rank=20,show_edge_name=True):
        """
        This is done in only triplet case
        total_rel: dictionary of list
        """
        total_len = len(triplet['data_img']['triplet'][:rank])
        #  make relational dictionary
        rel = {i: [] for i in range(total_len)}
        edge_label = dict()
        for i in range(total_len):
            rel[i].append(triplet['data_img']['triplet'][i][1]) # position check requires

        #  add // can be done above but just for separting
        for k, v_list in obj.items():
            self.G.add_edge(self.idx_label[v_list[0]], self.idx_label[v_list[-1]])
            if show_edge_name:
                edge_label[(self.idx_label[v_list[0]], self.idx_label[v_list[-1]])] = self.idx_predicate[rel[k][0]]
        return rel, edge_label

class GraphDrawer:
    def __init__(self, G_obj):
        self.G_obj = G_obj
        self.G = G_obj.G
        # TO DO
        self.pos = nx.spring_layout(self.G)

    def draw_and_save(self, node_size=500, save=True, figure_name='test20',node_color='pink', alpha=0.9, linewidths=1, width=1, edge_font_size=10, node_font_size=10):
        # draw graph node-edge level
        # plt.figure()
        base_path = './result'
        nx.draw(self.G, pos= self.pos,
                node_size=node_size, node_color=node_color,
                alpha=alpha, linewidths=linewidths, width=width)
        # draw edge label
        nx.draw_networkx_edge_labels(self.G, pos=self.pos,
                                      edge_labels=self.G_obj.edge_label, font_size=edge_font_size)
        # draw node label
        nx.draw_networkx_labels(self.G, pos=self.pos, font_size=node_font_size)
        if save:
            plt.savefig(base_path+'/'+figure_name+'.png')

def make_dictionary(real_data_file, base_schema_file):
    with open(base_schema_file) as f:
        base_scheme = json.load(f)
    with open(real_data_file) as f:
        real_data = json.load(f)
    #TODO: id parsing
    # base_scheme['data_img']['id'] = real_data_file.split()
    base_scheme['data_img']['triplet'] = real_data['pred_triplets']
    base_scheme['data_bbox'] = real_data['pred_triplet_boxes']
    return base_scheme
if __name__=='__main__':
    # load ground truth VG dict
    gt_data_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data/VG_SGG_dicts.json'
    # real data file
    real_data_file = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data/test.json'
    pred_data_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data/pred_triplet_v1.json'

    gt_data_obj = GTDictHandler(gt_data_path)
    graph_obj = GraphHandler(gt_data_obj)

    # need to make new function to converse real_data_file to correct data field dictionary
    real_triple_obj = make_dictionary(real_data_file,pred_data_path )
    # with open(pred_data_path) as f:
    #     triplet = json.load(f)
    print('\n@loading triplet information')
    print(real_triple_obj,'\n')

    graph_obj.generate_SG(real_triple_obj)
    graph_drawer = GraphDrawer(graph_obj)
    graph_drawer.draw_and_save()

    # load output
    # function required




