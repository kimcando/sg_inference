import json
import os, sys
import torch

def demo_merge_json(image_ids, top_k = 20):
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

    demo_merge_json(image_ids=(0,))

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