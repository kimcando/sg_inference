from .coco_eval import do_coco_evaluation


def coco_evaluation(
    dataset,
    predictions,
    output_folder,
    image_ids,
    box_only,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    return do_coco_evaluation(
        dataset=dataset,
        predictions=predictions,
        image_ids=image_ids,
        box_only=box_only,
        output_folder=output_folder,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )
