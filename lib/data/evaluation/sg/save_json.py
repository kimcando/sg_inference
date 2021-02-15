import json
import os
import numpy as np
# def save_value( k, gt_triplets,
#                 pred_triplets,
#                 gt_triplet_boxes,
#                 pred_triplet_boxes,
#                 iou_thresh, file_name='test'):
def save_value( pred_bbox_list,
                recall,
                pred_triplet_list,
                FileName="null", json_file_name='result_img_0'):
    """
    :param k: recall value
    :param gt_triplets: gt triplet
    :param pred_triplets: pred_triplet
    :param gt_triplet_boxes: gt_triplet_boxes
    :param pred_triplet_boxes: pred_triplet_boxes
    :param iou_thresh: iou threshold
    :return:
    """
    json_data = {
            "FileName": FileName,
            "bbox": [ ],
            "recall": recall,
            "triplet": []
        }

    json_data["bbox"] = pred_bbox_list.tolist()
    json_data["triplet"] = pred_triplet_list.tolist()

    base_dir_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data'
    with open(base_dir_path+'/'+json_file_name+'.json', 'w') as outfile:
        json.dump(json_data, outfile)

def save_scores(scores, name_dict,  json_file_name= 'result_score_img_0', FileName='null'):
    base_dir_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data'
    json_data = {
        "FileName": FileName,
        "sub_score": [],
        "sub_i":[],
        "pred_score": [],
        "obj_score" : [],
        "obj_i":[]
    }
    json_data["sub_score"] = scores[:,0].tolist()
    json_data["pred_score"] = scores[:, 1].tolist()
    json_data["obj_score"] = scores[:, 2].tolist()
    # json_data["sub_i"] = name_dict['sub'].tolist()
    # json_data["obj_i"] = name_dict['obj'].tolist()
    with open(base_dir_path+'/'+json_file_name+'.json', 'w') as outfile:
        json.dump(json_data, outfile)
    with open(base_dir_path+'/'+json_file_name+'_dict.json', 'w') as outfile:
        json.dump(name_dict, outfile)

def save_cls_scores(scores, json_file_name= 'cls_score_img_0', FileName='null' ):
    base_dir_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data'
    json_data = {
        "FileName": FileName,
        "cls_score":[]
    }
    json_data["cls_score"] = scores.tolist()

    with open(base_dir_path + '/' + json_file_name + '.json', 'w') as outfile:
        json.dump(json_data, outfile)

if __name__=='__main__':

    k = 20
    gt_triplets= np.array([[65,29, 135],
    [135 , 20 ,145],
    [136 ,29  , 65],
    [145 , 31 , 135],
    [146 , 31 , 135]])
    data = {}
    data['k'] = k
    data['gt_triplets'] = gt_triplets.tolist()
    print(gt_triplets.tolist())
    with open('./test.json', 'w') as test_file:
        json.dump(data, test_file)

