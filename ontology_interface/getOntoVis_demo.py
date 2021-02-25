import json
import networkx as nx
from itertools import combinations
import numpy as np
import pdb
from matplotlib import pyplot as plt
import os
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

class GraphHandlerDemo_v2:
    def __init__(self, gt_data_obj, merge_dict):
        self.gt_data_obj = gt_data_obj
        self.idx_label = gt_data_obj.get_idx_label
        self.idx_predicate = gt_data_obj.get_idx_predicate

        self.G = self.new_graph()
        self.json_data = merge_dict

    def new_graph(self):
        G = nx.Graph()
        return G

    def get_pred_class(self, rank):
        for i in range(rank):
            self.json_data['triplet'][i]['predicate'] = self.idx_predicate[
                int(self.json_data['triplet'][i]['predicate'])]

    def get_obj_name(self):
        for i in range(len(self.json_data['bbox'])):
            self.json_data['bbox'][i]['obj_name'] = self.idx_label[int(self.json_data['bbox'][i]['obj_name'])]

    def get_dict(self):
        return self.json_data

    def get_name(self, rank, image_ids):

        for i in range(rank):
            # GT translation requires
            triple = dict()
            self.json_data['triplet'][i]['subject'] = self.idx_label[int(self.json_data['triplet'][i]['subject'][0])]+'_'+self.json_data['triplet'][i]['subject'][1]
            self.json_data['triplet'][i]['predicate'] = self.idx_predicate[int(self.json_data['triplet'][i]['predicate'])]
            self.json_data['triplet'][i]['object'] = self.idx_label[int(self.json_data['triplet'][i]['object'][0])]+'_'+self.json_data['triplet'][i]['object'][1]
        with open(f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_send.json', 'w') as f:
            json.dump(self.json_data, f)
            # new_data['triplet'].append(triple)

    def save_triplet(self, rank, image_ids):
        triple = {
            ''
        }
        for i in range(rank):
            # GT translation requires

            self.json_data['triplet'][i]['subject'] = self.idx_label[
                                                          int(self.json_data['triplet'][i]['subject'][0])] + '_' + \
                                                      self.json_data['triplet'][i]['subject'][1]
            self.json_data['triplet'][i]['predicate'] = self.idx_predicate[
                int(self.json_data['triplet'][i]['predicate'])]
            self.json_data['triplet'][i]['object'] = self.idx_label[
                                                         int(self.json_data['triplet'][i]['object'][0])] + '_' + \
                                                     self.json_data['triplet'][i]['object'][1]
        # with open(f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_send.json', 'w') as f:
        #     json.dump(self.json_data, f)
            # new_data['triplet'].append(triple)
        return

    def print_graph_info(self, G=None):
        if G is None:
            print(f'number of nodes in this graph: {self.G.nodes()}')
            print(f'number of edges in this graph: {self.G.edges()}')
            print(f'edge label in this graph:{self.edge_label}')
        else:
            print(f'number of nodes in this graph: {G.nodes()}')
            print(f'number of edges in this graph: {G.edges()}')

    def generate_SG(self, rank=1):

        obj = self.add_sub_obj(rank=rank)
        rel, self.edge_label = self.add_relation(obj, rank=rank)
        self.print_graph_info()

    def add_sub_obj(self, rank=1):

        #  make subject/object dictionary
        obj = {i:[] for i in range(rank)}
        # import pdb; pdb.set_trace()
        for i in range(rank):
            # import pdb; pdb.set_trace()
            # self.json_data['triplet'][i] = [str(class index), str(bbox_idx)]
            obj[i].append(self.json_data['triplet'][i]['subject'])
            obj[i].append(self.json_data['triplet'][i]['object'])
            print(i)
        #  add // can be done above but just for separting
        for k, v_list in obj.items():
            for v in v_list:
                # pdb.set_trace()
                self.G.add_node(v)
        return obj

    def add_relation(self,obj, rank=1,show_edge_name=True):
        """
        This is done in only triplet case
        total_rel: dictionary of list
        """
        #  make relational dictionary
        rel = {i: [] for i in range(rank)}
        edge_label = dict()
        for i in range(rank):
            rel[i].append(self.json_data['triplet'][i]['predicate']) # position check requires

        #  add edge
        for k, v_list in obj.items():
            self.G.add_edge(v_list[0], v_list[-1])
            if show_edge_name:
                edge_label[(v_list[0], v_list[-1])] = rel[k][0]
        return rel, edge_label


class GraphHandlerDemo:
    def __init__(self, data_path, gt_data_obj):
        self.gt_data_obj = gt_data_obj
        self.idx_label = gt_data_obj.get_idx_label
        self.idx_predicate = gt_data_obj.get_idx_predicate

        self.G = self.new_graph()
        with open(data_path) as f:
            self.json_data = json.load(f)

    def new_graph(self):
        G = nx.Graph()
        return G

    def get_name(self, rank, image_ids):

        for i in range(rank):
            # GT translation requires
            triple = dict()
            self.json_data['triplet'][i]['subject'] = self.idx_label[int(self.json_data['triplet'][i]['subject'][0])]+'_'+self.json_data['triplet'][i]['subject'][1]
            self.json_data['triplet'][i]['predicate'] = self.idx_predicate[int(self.json_data['triplet'][i]['predicate'])]
            self.json_data['triplet'][i]['object'] = self.idx_label[int(self.json_data['triplet'][i]['object'][0])]+'_'+self.json_data['triplet'][i]['object'][1]
        with open(f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_send.json', 'w') as f:
            json.dump(self.json_data, f)
            # new_data['triplet'].append(triple)

    def save_triplet(self, rank, image_ids):
        triple = {
            ''
        }
        for i in range(rank):
            # GT translation requires

            self.json_data['triplet'][i]['subject'] = self.idx_label[
                                                          int(self.json_data['triplet'][i]['subject'][0])] + '_' + \
                                                      self.json_data['triplet'][i]['subject'][1]
            self.json_data['triplet'][i]['predicate'] = self.idx_predicate[
                int(self.json_data['triplet'][i]['predicate'])]
            self.json_data['triplet'][i]['object'] = self.idx_label[
                                                         int(self.json_data['triplet'][i]['object'][0])] + '_' + \
                                                     self.json_data['triplet'][i]['object'][1]
        # with open(f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_send.json', 'w') as f:
        #     json.dump(self.json_data, f)
            # new_data['triplet'].append(triple)
        return

    def print_graph_info(self, G=None):
        if G is None:
            print(f'number of nodes in this graph: {self.G.nodes()}')
            print(f'number of edges in this graph: {self.G.edges()}')
            print(f'edge label in this graph:{self.edge_label}')
        else:
            print(f'number of nodes in this graph: {G.nodes()}')
            print(f'number of edges in this graph: {G.edges()}')

    def generate_SG(self, rank=1):

        obj = self.add_sub_obj(rank=rank)
        rel, self.edge_label = self.add_relation(obj, rank=rank)
        self.print_graph_info()

    def add_sub_obj(self, rank=1):

        #  make subject/object dictionary
        obj = {i:[] for i in range(rank)}
        # import pdb; pdb.set_trace()
        for i in range(rank):
            # import pdb; pdb.set_trace()
            # self.json_data['triplet'][i] = [str(class index), str(bbox_idx)]
            obj[i].append(self.json_data['triplet'][i]['subject'])
            obj[i].append(self.json_data['triplet'][i]['object'])
            print(i)
        #  add // can be done above but just for separting
        for k, v_list in obj.items():
            for v in v_list:
                # pdb.set_trace()
                self.G.add_node(v)
        return obj

    def add_relation(self,obj, rank=1,show_edge_name=True):
        """
        This is done in only triplet case
        total_rel: dictionary of list
        """
        #  make relational dictionary
        rel = {i: [] for i in range(rank)}
        edge_label = dict()
        for i in range(rank):
            rel[i].append(self.json_data['triplet'][i]['predicate']) # position check requires

        #  add edge
        for k, v_list in obj.items():
            self.G.add_edge(v_list[0], v_list[-1])
            if show_edge_name:
                edge_label[(v_list[0], v_list[-1])] = rel[k][0]
        return rel, edge_label


class GraphHandler:
    def __init__(self, data_path):
        self.G = self.new_graph()
        with open(data_path) as f:
            self.json_data = json.load(f)

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

    def generate_SG(self, rank=1):

        obj = self.add_sub_obj(rank=rank)
        rel, self.edge_label = self.add_relation(obj, rank=rank)
        self.print_graph_info()

    def add_sub_obj(self, rank=1):

        #  make subject/object dictionary
        obj = {i:[] for i in range(rank)}
        import pdb; pdb.set_trace()
        for i in range(rank):
            # import pdb; pdb.set_trace()
            # self.json_data['triplet'][i] = [str(class index), str(bbox_idx)]
            obj[i].append(self.json_data['triplet'][i]['subject'])
            obj[i].append(self.json_data['triplet'][i]['object'])
            print(i)
        #  add // can be done above but just for separting
        for k, v_list in obj.items():
            for v in v_list:
                # pdb.set_trace()
                self.G.add_node(v)
        return obj

    def add_relation(self,obj, rank=1,show_edge_name=True):
        """
        This is done in only triplet case
        total_rel: dictionary of list
        """
        #  make relational dictionary
        rel = {i: [] for i in range(rank)}
        edge_label = dict()
        for i in range(rank):
            rel[i].append(self.json_data['triplet'][i]['predicate']) # position check requires

        #  add edge
        for k, v_list in obj.items():
            self.G.add_edge(v_list[0], v_list[-1])
            if show_edge_name:
                edge_label[(v_list[0], v_list[-1])] = rel[k][0]
        return rel, edge_label

class GraphDrawer:
    def __init__(self, G_obj):
        self.G_obj = G_obj
        self.G = G_obj.G
        # TO DO
        self.pos = nx.spring_layout(self.G)

    def draw_and_save(self, node_size=500, live=False,save=True, sg_folder='sg_result',figure_name='test20',node_color='pink', alpha=0.9, linewidths=1, width=1, edge_font_size=10, node_font_size=10):
        # draw graph node-edge level
        # plt.figure()
        base_path = '/home/ncl/ADD_sy/inference/sg_inference/visualize/'
        if not os.path.exists(base_path + sg_folder):
            os.mkdir(base_path+sg_folder)
        nx.draw(self.G, pos= self.pos,
                node_size=node_size, node_color=node_color,
                alpha=alpha, linewidths=linewidths, width=width)
        # draw edge label
        nx.draw_networkx_edge_labels(self.G, pos=self.pos,
                                      edge_labels=self.G_obj.edge_label, font_size=edge_font_size)
        # draw node label
        nx.draw_networkx_labels(self.G, pos=self.pos, font_size=node_font_size)
        if live:
            plt.show()
        else:
            plt.savefig(os.path.join(base_path+sg_folder,figure_name+'.png'))
            # plt.show()
            # plt.savefig(base_path+'/'+figure_name+'.png')
        # clearining figure
        plt.clf()



class JsonTranslator(object):
    def __init__(self, gt_data_obj):
        self.gt_data_obj = gt_data_obj
        self.idx_label = gt_data_obj.get_idx_label
        self.idx_predicate = gt_data_obj.get_idx_predicate

    def make_json(self, data_file, img_name='test',rank=1, FileName=False):
        with open(data_file) as f:
            data = json.load(f)

        new_data = dict()
        for k, v in data.items():
            if type(v) == list:
                new_data[k] = []
            else:
                new_data[k] = "null"
        if FileName:
            new_data['FileName'] = FileName
        else:
            new_data['FileName'] = data['FileName']
        new_data['rank'] = rank

        # bbox, triplet append
        # bbox --> append all the detected boxes
        for i in range(len(data['bbox'])):
            bbox = dict()
            bbox['id'] = i
            bbox['x'] = data['bbox'][i][0]
            bbox['y'] = data['bbox'][i][1]
            bbox['x2'] = data['bbox'][i][2]
            bbox['y2'] = data['bbox'][i][3]
            #TODO
            # bbox['obj_name'] = self.idx_label[data['obj_name'][i][0]]
            # bbox['score'] = self.idx_label[data['score'][i][0]]
            new_data['bbox'].append(bbox)

        # triplet --> append top rank
        # triplet index 중 sub, obj가  obj bbox 가르키도록 수정돼야할듯
        for i in range(rank):
            # GT translation requires
            triple = dict()
            triple['subject'] = self.idx_label[data['triplet'][i][0]]
            triple['predicate'] = self.idx_predicate[data['triplet'][i][1]]
            triple['object'] = self.idx_label[data['triplet'][i][2]]
            new_data['triplet'].append(triple)


        with open('/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'+img_name+'.json', 'w') as outfile:
            json.dump(new_data, outfile)


class BboxDrawer(object):
    def __init__(self, path='/results/'):
        self.path = path

    # def compute_colors_for_labels(self, labels, pallette=):
    #     #TODO
    #     return colors
    #
    # def overlay_boxes(self, img, predictions):
    #     labels = predictions.get_field("labels")
    #     boxes = predictions.bbox
    #     colors = self.compute_colors_for_labels(labels)
    #
    #     for box, color in zip(boxes, colors):
    #         #TODO
    #     return image
def main():
    pass

if __name__=='__main__':
    # file in the result_data_path is generated after single inference
    gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
    # result_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/result_img_0.json'
    result_data_path = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/0_merged.json'
    gt_data_obj = GTDictHandler(gt_data_path)
    import pdb; pdb.set_trace()
    jsonMaker = JsonTranslator(gt_data_obj)

    # jsonMaker.make_json(result_data_path,img_name='0_test_image',
    #                     rank=20,FileName='0_test_image')

    # server will send generated_json_path
    generated_json_path = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/0_merged.json'
    graph_obj = GraphHandlerDemo(generated_json_path, gt_data_obj)
    graph_obj.get_name(rank=20)

    with open('sample_file.pickle', 'wb') as f:
        pickle.dump(merged_json, f,protocol=pickle.HIGHEST_PROTOCOL)

    # graph_obj.generate_SG(rank=20)
    # graph_drawer = GraphDrawer(graph_obj)
    # graph_drawer.draw_and_save(figure_name='test_sg')



