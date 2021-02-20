import cv2
import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
import numpy as np

def select_top_predictions(predictions, confidence_threshold=0.2):
    """
    Select only predictions which have a `score` > self.confidence_threshold,
    and returns the predictions in descending order of score
    Arguments:
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores`.
    Returns:
        prediction (BoxList): the detected objects. Additional information
            of the detection properties can be found in the fields of
            the BoxList via `prediction.fields()`
    """
    scores = predictions.get_field("scores")
    keep = torch.nonzero(scores > confidence_threshold).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)
    return predictions[idx]

def compute_colors_for_labels(labels, palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])):
    """
    Simple function that adds fixed colors depending on the class
    """
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_boxes(image, predictions):
    """
    Adds the predicted boxes on top of the image
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `labels`.
    """
    # import pdb; pdb.set_trace()
    labels = torch.rand(len(predictions))
    colors = compute_colors_for_labels(labels).tolist()

    for box, color in zip(predictions, colors):
        # box = box.to(torch.int64)
        top_left, bottom_right = [box['x'],box['y']] , [box['w'],box['h']]
        # top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        image = cv2.rectangle(
            image, tuple(top_left), tuple(bottom_right), tuple(color), 1
        )

    return image

def overlay_class_names(image, predictions):
    """
    Adds detected class names and scores in the positions defined by the
    top-left corner of the predicted bounding box
    Arguments:
        image (np.ndarray): an image as returned by OpenCV
        predictions (BoxList): the result of the computation by the model.
            It should contain the field `scores` and `labels`.
    """
    # labels = predictions.get_field("labels").tolist()
    # boxes = predictions.bbox
    template = "{}: {:.2f}" # score requires
    for box in predictions:
        x, y = box['x'], box['y']
        s = template.format(box['obj_name'], 0.8)
        cv2.putText(
            image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )
    # template = "{}: {:.2f}"
    # for box, score, label in zip(boxes, scores, labels):
    #     x, y = box[:2]
    #     s = template.format(label, score)
    #     cv2.putText(
    #         image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
    #     )

    return image

def visualize_detection(img_path, prediction,
                        visualize_folder="visualize",
                        raw_folder = "raw",
                        bbox_folder="bbox"):

    base_path = '/home/ncl/ADD/sg_inference/graph_rcnn/ontology/'
    if not os.path.exists(base_path+visualize_folder):
        os.mkdir(base_path+visualize_folder)
    if not os.path.exists(base_path+visualize_folder+'/'+raw_folder):
        os.mkdir(base_path+visualize_folder+'/'+raw_folder)
    if not os.path.exists(base_path+visualize_folder + '/' + bbox_folder):
        os.mkdir(base_path+visualize_folder + '/' + bbox_folder)

    img = cv2.imread(img_path)
    #glob
    # import pdb; pdb.set_trace()
    predictions = []
    predictions.append(prediction)
    for i, pred in enumerate(predictions):
        # result = np.array(img)
        # original image
        raw_path = os.path.join(base_path+visualize_folder + '/' + raw_folder)
        # import pdb; pdb.set_trace()
        cv2.imwrite(raw_path+"/raw_detection_{}.jpg".format(i),
                    img)

        # image with bounding box
        result = overlay_boxes(img, pred)
        result = overlay_class_names(result, pred)
        bbox_path = os.path.join(visualize_folder + '/' + bbox_folder)
        cv2.imwrite(bbox_path+"/bbox_detection_{}.jpg".format(i), result)

if __name__ == '__main__':
    import json
    with open('/home/ncl/ADD/sg_inference/graph_rcnn/ontology/data/json_img_0.json') as f:
        data_obj = json.load(f)

    # in single_img
    img = '/home/ncl/ADD/sg_inference/graph_rcnn/graph-rcnn.pytorch-master/visualize/raw/raw_detection_0.jpg'
    prediction = data_obj['bbox']
    visualize_detection(img, prediction)
