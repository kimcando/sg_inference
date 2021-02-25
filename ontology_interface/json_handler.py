import json
import os, sys
import torch
import pickle
from ontology_interface.getOntoVis_demo import GTDictHandler, GraphHandlerDemo_v2, GraphDrawer


def demo_merge_json(image_ids, top_k = 20):
    """
    make json file for GraphViz format
    :param image_ids:
    :param top_k:
    :return:
    """

    merged_json = dict()
    # files are saved in sg_eval.py
    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_triplets_sorted.pth')

    # import pdb;pdb.set_trace()
    # graph handler added
    gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
    gt_data_obj = GTDictHandler(gt_data_path)

    merged_json.update( url=image_ids[0],
                        bbox = [
                            {
                                "id": i,                                  #bbox id
                                "x" : bbox_info['pred_boxes'][i].tolist()[0],
                                "y": bbox_info['pred_boxes'][i].tolist()[1],
                                "x2": bbox_info['pred_boxes'][i].tolist()[2],
                                "y2": bbox_info['pred_boxes'][i].tolist()[3],
                                "obj_name" : bbox_info['pred_classes'].tolist()[i], #class name
                                "score" : bbox_info['obj_scores'].tolist()[i],
                            }
                            for i in range(len(bbox_info['pred_boxes']))
                        ],
                        triplet= [
                            {
                            'subject': bbox_info['pred_rel_inds'][k].tolist()[0], # class_idx, bbox
                            'predicate': triplet_info.tolist()[k][1], # predicate
                            'object': bbox_info['pred_rel_inds'][k].tolist()[-1] ,# class_idx, bbox
                            }
                            for k in range(top_k)
                        ]
                        )
    graph_obj = GraphHandlerDemo_v2(gt_data_obj, merged_json)
    graph_obj.get_pred_class(rank= top_k)
    graph_obj.get_obj_name()
    new_json = graph_obj.get_dict()
    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'
    # with open(os.path.join(save_dir, f'{image_ids[0]}_test.json'), 'w') as json_file:
    #     json.dump(merged_json,json_file)
    with open(os.path.join(save_dir, f'{image_ids[0]}_final.json'), 'w') as json_file:
        json.dump(new_json,json_file)


def demo_merge_json_org(image_ids, top_k = 20):
    """
    exits for internal meeting prototype
    not related to sending format
    :param image_ids:
    :param top_k:
    :return:
    """
    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_triplets_sorted.pth')

    class_names = bbox_info['pred_classes']
    # breakpoint()

    merged_json.update(FileName=image_ids[0],
                       bbox=[
                           {
                               "id": i,  # bbox id
                               "x": str(bbox_info['pred_boxes'][i][0]),
                               "y": str(bbox_info['pred_boxes'][i][1]),
                               "x2": str(bbox_info['pred_boxes'][i][2]),
                               "y2": str(bbox_info['pred_boxes'][i][3]),
                               "obj_name": str(bbox_info['pred_classes'][i]),  # class name
                               "score": str(bbox_info['obj_scores'][i]),
                           }
                           for i in range(len(bbox_info['pred_boxes']))
                       ],
                       recall=top_k,
                       triplet=[
                           {
                               'subject': [str(class_names[bbox_info['pred_rel_inds'][k][0]]),
                                           str(bbox_info['pred_rel_inds'][k][0])],  # class_idx, bbox
                               'predicate': str(triplet_info.tolist()[k][1]),  # predicate
                               'object': [str(class_names[bbox_info['pred_rel_inds'][k][-1]]),
                                          str(bbox_info['pred_rel_inds'][k][-1])],  # class_idx, bbox
                           }
                           for k in range(top_k)
                       ]
                       )

    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'

    with open(os.path.join(save_dir, f'{image_ids[0]}_merged.json'), 'w') as json_file:
        json.dump(merged_json,json_file)


if __name__=='__main__':
    demo_merge_json(image_ids=(10,))
