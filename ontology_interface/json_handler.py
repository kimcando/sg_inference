import json
import os, sys
import torch
import pickle
from ontology_interface.getOntoVis_demo import GTDictHandler, GraphHandlerDemo_v2, GraphDrawer

def demo_merge_json_dict(image_ids, top_k = 20):
    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    # bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/bbox/{image_ids[0]}_coco_results_bbox.pth')
    triplet_info = torch.load(
        f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')
    # TODO
    # merged_json.update( FileName=image_ids[0],
    #                     bbox_id = [bbox_info['bbox'][i]['bbox_id'] for i in range(len(bbox_info['bbox']))],
    #                     obj_name=[bbox_info['bbox'][i]['category_id'] for i in range(len(bbox_info['bbox']))],
    #                     bbox = [bbox_info['bbox'][i]['bbox'] for i in range(len(bbox_info['bbox']))],
    #                     recall = top_k,
    #                     triplet= triplet_info.tolist()[:top_k])
    class_names = bbox_info['pred_classes']
    merged_json.update(FileName=image_ids[0],
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
    with open('sample_file.pickle', 'wb') as f:
        pickle.dump(merged_json, f,protocol=pickle.HIGHEST_PROTOCOL)



def demo_merge_json(image_ids, top_k = 20):

    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')

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

    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    # bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/bbox/{image_ids[0]}_coco_results_bbox.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')
    #TODO
    # merged_json.update( FileName=image_ids[0],
    #                     bbox_id = [bbox_info['bbox'][i]['bbox_id'] for i in range(len(bbox_info['bbox']))],
    #                     obj_name=[bbox_info['bbox'][i]['category_id'] for i in range(len(bbox_info['bbox']))],
    #                     bbox = [bbox_info['bbox'][i]['bbox'] for i in range(len(bbox_info['bbox']))],
    #                     recall = top_k,
    #                     triplet= triplet_info.tolist()[:top_k])
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

    # merged_json.update( FileName=image_ids[0],
    #                     bbox = [
    #                         {
    #                             "id": i,                                  #bbox id
    #                             "x" : bbox_info['pred_boxes'][i][0],
    #                             "y": bbox_info['pred_boxes'][i][1],
    #                             "x2": bbox_info['pred_boxes'][i][2],
    #                             "y2": bbox_info['pred_boxes'][i][3],
    #                             "obj_name" : bbox_info['pred_classes'][i], #class name
    #                             "score" : bbox_info['obj_scores'][i],
    #                         }
    #                         for i in range(len(bbox_info['pred_boxes']))
    #                     ],
    #                     recall = top_k,
    #                     triplet= [
    #                         {
    #                         'subject': (class_names[bbox_info['pred_rel_inds'][k][0]], bbox_info['pred_rel_inds'][k][0]), # class_idx, bbox
    #                         'predicate': triplet_info.tolist()[k][1], # predicate
    #                         'object': (class_names[bbox_info['pred_rel_inds'][k][-1]],bbox_info['pred_rel_inds'][k][-1]) ,# class_idx, bbox
    #                         }
    #                         for k in range(top_k)
    #                     ]
    #                     )

    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'
    # with open(os.path.join(save_dir, f'{image_ids[0]}_test.json'), 'w') as json_file:
    #     json.dump(merged_json,json_file)
    with open(os.path.join(save_dir, f'{image_ids[0]}_merged.json'), 'w') as json_file:
        json.dump(merged_json,json_file)


def demo_merge_json_test(image_ids, top_k = 20):

    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/output/{image_ids[0]}_pred_entry.pth')
    # bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/bbox/{image_ids[0]}_coco_results_bbox.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')
    #TODO
    # merged_json.update( FileName=image_ids[0],
    #                     bbox_id = [bbox_info['bbox'][i]['bbox_id'] for i in range(len(bbox_info['bbox']))],
    #                     obj_name=[bbox_info['bbox'][i]['category_id'] for i in range(len(bbox_info['bbox']))],
    #                     bbox = [bbox_info['bbox'][i]['bbox'] for i in range(len(bbox_info['bbox']))],
    #                     recall = top_k,
    #                     triplet= triplet_info.tolist()[:top_k])
    class_names = bbox_info['pred_classes']
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@2')
    breakpoint()

    merged_json.update( FileName=image_ids[0],
                        bbox = [
                            {
                                "id": i,                                  #bbox id
                                "x" : str(bbox_info['pred_boxes'][i][0]),
                                "y": str(bbox_info['pred_boxes'][i][1]),
                                "x2": str(bbox_info['pred_boxes'][i][2]),
                                "y2": str(bbox_info['pred_boxes'][i][3]),
                                "obj_name" : str(bbox_info['pred_classes'][i]), #class name
                                "score" : str(bbox_info['obj_scores'][i]),
                            }
                            for i in range(len(bbox_info['pred_boxes']))
                        ],
                        recall = top_k,
                        triplet= [
                            {
                            'subject': [str(class_names[bbox_info['pred_rel_inds'][k][0]]), str(bbox_info['pred_rel_inds'][k][0])], # class_idx, bbox
                            'predicate': str(triplet_info.tolist()[k][1]), # predicate
                            'object': [str(class_names[bbox_info['pred_rel_inds'][k][-1]]),str(bbox_info['pred_rel_inds'][k][-1]) ] ,# class_idx, bbox
                            }
                            for k in range(top_k)
                        ]
                        )

    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'
    with open(os.path.join(save_dir, f'{image_ids[0]}_test.json'), 'w') as json_file:
        json.dump(merged_json,json_file)
    # with open(os.path.join(save_dir, f'{image_ids[0]}_merged.json'), 'w') as json_file:
    #     json.dump(merged_json,json_file)


def demo_merge_json_morning(image_ids, top_k = 20):
    bbox_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/bbox'
    triplet_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/triplet'
    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/bbox/{image_ids[0]}_coco_results_bbox.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')
    #TODO
    # merged_json.update( FileName=image_ids[0],
    #                     bbox_id = [bbox_info['bbox'][i]['bbox_id'] for i in range(len(bbox_info['bbox']))],
    #                     obj_name=[bbox_info['bbox'][i]['category_id'] for i in range(len(bbox_info['bbox']))],
    #                     bbox = [bbox_info['bbox'][i]['bbox'] for i in range(len(bbox_info['bbox']))],
    #                     recall = top_k,
    #                     triplet= triplet_info.tolist()[:top_k])
    import pdb; pdb.set_trace()
    merged_json.update( FileName=image_ids[0],
                        bbox = [
                            {
                                "id": bbox_info['bbox'][i]['bbox_id'], #test
                                "x" : bbox_info['bbox'][i]['bbox'][0],
                                "y": bbox_info['bbox'][i]['bbox'][1],
                                "x2": bbox_info['bbox'][i]['bbox'][2],
                                "y2": bbox_info['bbox'][i]['bbox'][3],
                                "obj_name" : bbox_info['bbox'][i]['category_id'],
                                "score" : bbox_info['bbox'][i]['score'],
                            }
                            for i in range(len(bbox_info['bbox']))
                        ],
                        recall = top_k,
                        triplet=triplet_info.tolist()[:top_k]
                        )
    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'
    with open(os.path.join(save_dir, f'{image_ids[0]}_test2.json'), 'w') as json_file:
        json.dump(merged_json,json_file)
    # with open(os.path.join(save_dir, f'{image_ids[0]}_merged.json'), 'w') as json_file:
    #     json.dump(merged_json,json_file)


def demo_merge_json_0222(image_ids, top_k = 20):
    bbox_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/bbox'
    triplet_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/triplet'
    merged_json = dict()

    # for final form
    bbox_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/bbox/{image_ids[0]}_coco_results_bbox.pth')
    triplet_info = torch.load(f'/home/ncl/ADD_sy/inference/sg_inference/results/triplet/{image_ids[0]}_pred_triplets_sorted.pth')
    #TODO
    merged_json.update( FileName=image_ids[0],
                        bbox = [bbox_info['bbox'][i]['bbox'] for i in range(len(bbox_info['bbox']))],
                        recall = top_k,
                        triplet= triplet_info.tolist()[:top_k])
    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send/'
    with open(os.path.join(save_dir, f'{image_ids[0]}_merged.json'), 'w') as json_file:
        json.dump(merged_json,json_file)

def merge_json(test_img, top_k=20):
    bbox_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/bbox'
    triplet_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/triplet'
    merged_json = dict()

    # for final form
    bbox_info = torch.load(os.path.join(bbox_dir, test_img))
    # / home / ncl / ADD_sy / inference / sg_inference / results / triplet / pred_triplets_sorted.pth
    triplet_info = torch.load(os.path.join(triplet_dir, test_img))
    # with open(bbox_dir+'/'+test_img,'r') as f:
    #     bbox_info = json.load(f)
    # with open(triplet_dir+'/'+test_img,'r') as f:
    #     triplet_info = json.load(f)
    # TODO
    merged_json.update(FileName=test_img,
                       bbox=bbox_info['bbox'].tolist(),
                       recall=top_k,
                       relationship=triplet_info.tolist()[:top_k])
    save_dir = '/home/ncl/ADD_sy/inference/sg_inference/results/to_send'
    json.dump(merged_json, os.path.join(save_dir, test_img))

if __name__=='__main__':
    demo_merge_json(image_ids=(10,))
    # with open('sample_file.pickle', 'wb') as f:
    #     pickle.dump(merged_json, f,protocol=pickle.HIGHEST_PROTOCOL)
    # demo_merge_json_dict(image_ids=(0,))

    # demo_merge_json_test(image_ids=(0,))

# {
#   "FileName": 'test.png',
#   "bbox": [
#     {
#       "id": 10,
#       "x": 100,
#       "y": 200,
#       "w": 50,
#       "h": 80,
#       "obj_name": 'class'
#     },
#     ...
#   ],
#   "recall": 20,
#   "relationship": [
#     {
#       "subject": 10,
#       "predicate": 20,
#       "object": 30
#     },
#     {
#       {
#       "subject": 110,
#       "predicate": 210,
#       "object": 310
#     },
#     },
#     ...
#   ]
# }