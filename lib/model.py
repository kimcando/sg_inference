import os
import datetime
import logging
import time
import numpy as np
import torch
import cv2
from .data.build import build_data_loader
from .scene_parser.parser import build_scene_parser
from .scene_parser.parser import build_scene_parser_optimizer
from .scene_parser.rcnn.utils.metric_logger import MetricLogger
from .scene_parser.rcnn.utils.timer import Timer, get_time_str
from .scene_parser.rcnn.utils.comm import synchronize, all_gather, is_main_process, get_world_size
from .scene_parser.rcnn.utils.visualize import select_top_predictions, overlay_boxes, overlay_class_names
from .data.evaluation import evaluate, evaluate_sg
from .utils.box import bbox_overlaps

from ontology_interface.json_handler import demo_merge_json, demo_merge_json_org
from ontology_interface.getOntoVis_demo import GTDictHandler, GraphHandlerDemo, GraphDrawer
import pdb
from glob import glob
from PIL import Image
import numpy as np
from torchvision.utils import save_image


class OntoSceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # TODO
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed)

        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start bulilding Ontology scene graph generation")

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        # print(self.scene_parser)
        # sys.exit(1)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes


    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions,live=False,visualize_folder="visualize",
                            raw_folder = "raw",
                            bbox_folder="bbox"):
        visualize_folder = "/home/ncl/ADD_sy/inference/sg_inference/visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
            if not os.path.exists(visualize_folder+'/'+raw_folder):
                os.mkdir(visualize_folder+'/'+raw_folder)
            if not os.path.exists(visualize_folder + '/' + bbox_folder):
                os.mkdir(visualize_folder + '/' + bbox_folder)
        # print(type(imgs)) # <class 'lib.scene_parser.rcnn.structures.image_list.ImageList'>
        # print('length of predictions',len(predictions))
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            # print('top predcition',len(top_prediction))
            # TODO
            img = imgs.permute(1, 2, 0).contiguous().cpu().numpy() # + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            # import pdb; pdb.set_trace()
            if live:
                # cv2.imwrite(
                #     os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                #     result)
                im = Image.fromarray(result.astype(np.uint8))
                im.save(f'{visualize_folder}/{raw_folder}/raw_detection_{img_ids[i]}.jpg')
                img = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/raw/raw_detection_{img_ids[0]}.jpg')
                cv2.imshow('raw_image', img)
                cv2.waitKey(12)
            else:
                # cv2.imwrite(os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                #             result)
                # breakpoint()
                im = Image.fromarray(result.astype(np.uint8))
                im.save(f'{visualize_folder}/{raw_folder}/raw_detection_{img_ids[i]}.jpg')


            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            if live:
                im = Image.fromarray(result.astype(np.uint8))
                im.save(f'{visualize_folder}/{bbox_folder}/bbox_detection_{img_ids[i]}.jpg')
                # cv2.imwrite(
                #     os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])),
                #     result)
                # BGR conversion requires TODO
                img2 = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/bbox/bbox_detection_{img_ids[0]}.jpg')
                cv2.imshow('bounding_box_image', img2)
                cv2.waitKey(12)
            else:
                # cv2.imwrite(os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])), result)
                im = Image.fromarray(result.astype(np.uint8))
                im.save(f'{visualize_folder}/{bbox_folder}/bbox_detection_{img_ids[i]}.jpg')
                # im.save(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i]))

    # def load_test_data(self, img_dir, test_save_img=False):
    #
    #     img_list = glob(f'{img_dir}/*')
    #     # for img_file in img_list:
    #     img_pil = Image.open(img_list[0])
    #     img_pil = np.array(img_pil).transpose(2, 0, 1)
    #     # img = ToTensor()(img_pil).unsqueeze(0) # ToTensor 하면 normalize 도 자동으로 됨
    #     img = torch.from_numpy(img_pil).unsqueeze(0).float()
    #     file_name = img_dir.split('/')[-1].split('.')[0]
    #     if test_save_img:
    #         save_image(img_dir)
    #     return img, file_name

    def test(self, data_loader, test_single=False,timer=None, visualize=False, live=False, output_folder="results/"):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []

        # imgs = imgs.to(self.device)

        # graph handler added
        gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
        gt_data_obj = GTDictHandler(gt_data_path)

        if test_single:
            img_dir = "/home/ncl/ADD_sy/inference/sg_inference/data/"
            imgs, file_name = self.load_test_data(img_dir)
            breakpoint()
            imgs = imgs.to(self.device)
            image_ids = (122,)
            output = self.scene_parser(imgs)

            if self.cfg.MODEL.RELATION_ON:
                output, output_pred = output
                output_pred = [o.to(cpu_device) for o in output_pred]

            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

            if visualize:
                self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs.squeeze(0), output, live=live)
            # CHECK
            # breakpoint()
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            # targets_dict.update(
            #     {img_id: target for img_id, target in zip(image_ids, targets)}
            # )
            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )

            predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
            if not is_main_process():
                return

            extra_args = dict(
                box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                iou_types=("bbox",),
                expected_results=self.cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            )
            eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                                        predictions=predictions,
                                        output_folder=output_folder,
                                        image_ids=image_ids,
                                        **extra_args)
            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            image_ids=image_ids,
                            **extra_args)
                if visualize:
                    # import pdb; pdb.set_trace()
                    # breakpoint()
                    demo_merge_json(image_ids)
                    # jsonMaker = JsonTranslator(gt_data_obj)
                    result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                    # jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image', recall=10,
                    #                     FileName=f'{image_ids[0]}_image')
                    # generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                    graph_obj = GraphHandlerDemo(result_data_path, gt_data_obj)
                    graph_obj.get_name(rank=20)
                    graph_obj.generate_SG(recall=20)
                    graph_drawer = GraphDrawer(graph_obj)
                    print('done')
                    graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                    # graph_drawer.draw_and_save(figure_name=f'{122}_sg')
                if live:
                    sg = cv2.imread(
                        f'/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/{image_ids[0]}_sg.png')
                    cv2.imshow('scene_graph', sg)
                    cv2.waitKey(10)

            print(f'{image_ids[0]} done')

        else:
            for i, data in enumerate(data_loader, 0):
                logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))

                imgs, _ = data
                imgs = imgs.to(self.device)
                idx = i+10000
                image_ids = (i+1000000,)

                with torch.no_grad():
                    if timer:
                        timer.tic()

                    output = self.scene_parser(imgs)

                    if self.cfg.MODEL.RELATION_ON:
                        output, output_pred = output
                        output_pred = [o.to(cpu_device) for o in output_pred]

                    torch.save(output,
                               os.path.join('/home/ncl/ADD_sy/inference/sg_inference/results/output',
                                            f'{image_ids[0]}_output.pth'))
                    torch.save(output_pred,
                               os.path.join('/home/ncl/ADD_sy/inference/sg_inference/results/output',
                                            f'{image_ids[0]}_output_pred.pth'))

                    if timer:
                        torch.cuda.synchronize()
                        timer.toc()
                    output = [o.to(cpu_device) for o in output]

                    # if visualize:
                    #     self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs.squeeze(0), output, live=live)
                    # CHECK
                    # breakpoint()
                    results_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output)}
                    )
                    # targets_dict.update(
                    #     {img_id: target for img_id, target in zip(image_ids, targets)}
                    # )
                    # breakpoint()
                    if self.cfg.MODEL.RELATION_ON:
                        results_pred_dict.update(
                            {img_id: result for img_id, result in zip(image_ids, output_pred)}
                        )

                    predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

                    # breakpoint()
                    if self.cfg.MODEL.RELATION_ON:
                        predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
                    if not is_main_process():
                        return

                    extra_args = dict(
                        box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                        iou_types=("bbox",),
                        expected_results=self.cfg.TEST.EXPECTED_RESULTS,
                        expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                    )
                    eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                                                predictions=predictions,
                                                output_folder=output_folder,
                                                image_ids=image_ids,
                                                **extra_args)
                    # breakpoint()
                    if self.cfg.MODEL.RELATION_ON:
                        evaluate_sg(dataset=self.data_loader_test.dataset,
                                    predictions=predictions,
                                    predictions_pred=predictions_pred,
                                    output_folder=output_folder,
                                    image_ids=image_ids,
                                    **extra_args)
                        demo_merge_json(image_ids)
                        if visualize:
                            # import pdb; pdb.set_trace()
                            demo_merge_json_org(image_ids)
                            result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                            # jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image', recall=10,
                            #                     FileName=f'{image_ids[0]}_image')
                            # generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                            graph_obj = GraphHandlerDemo(result_data_path, gt_data_obj)
                            graph_obj.get_name(rank=20, image_ids=image_ids)
                            graph_obj.generate_SG(rank=20)
                            graph_drawer = GraphDrawer(graph_obj)
                            print('done')
                            graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                            self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs.squeeze(0), output,
                                                     live=live)
                            # original
                            # jsonMaker = JsonTranslator(gt_data_obj)
                            # result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                            # jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image',
                            #                     FileName=f'{image_ids[0]}_image')
                            # generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                            # graph_obj = GraphHandler(generated_json_path)
                            # graph_obj.generate_SG(recall=5)
                            # graph_drawer = GraphDrawer(graph_obj)
                            # graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                        if live:
                            sg = cv2.imread(
                                f'/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/{image_ids[0]}_sg.png')
                            cv2.imshow('scene_graph', sg)
                            cv2.waitKey(10)

                    print(f'{image_ids[0]} done')


class SceneGraphGeneration:
    """
    Scene graph generation
    """
    def __init__(self, cfg, arguments, local_rank, distributed):
        """
        initialize scene graph generation model
        """
        self.cfg = cfg
        self.arguments = arguments.copy()
        self.device = torch.device("cuda")

        # build data loader
        self.data_loader_train = build_data_loader(cfg, split="train", is_distributed=distributed)
        #TODO
        self.data_loader_test = build_data_loader(cfg, split="test", is_distributed=distributed)

        cfg.DATASET.IND_TO_OBJECT = self.data_loader_train.dataset.ind_to_classes
        cfg.DATASET.IND_TO_PREDICATE = self.data_loader_train.dataset.ind_to_predicates

        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Train data size: {}".format(len(self.data_loader_train.dataset)))
        logger.info("Test data size: {}".format(len(self.data_loader_test.dataset)))
        # pdb.set_trace()

        if not os.path.exists("freq_prior.npy"):
            logger.info("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:,:,0] = bg_matrix

            prob_matrix[:,:,0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:,:,None]
            # prob_matrix /= float(fg_matrix.max())
            np.save("freq_prior.npy", prob_matrix)

        # build scene graph generation model
        self.scene_parser = build_scene_parser(cfg); self.scene_parser.to(self.device)
        # print(self.scene_parser)
        # sys.exit(1)
        self.sp_optimizer, self.sp_scheduler, self.sp_checkpointer, self.extra_checkpoint_data = \
            build_scene_parser_optimizer(cfg, self.scene_parser, local_rank=local_rank, distributed=distributed)

        self.arguments.update(self.extra_checkpoint_data)

    def _get_freq_prior(self, must_overlap=False):

        fg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
            ), dtype=np.int64)

        bg_matrix = np.zeros((
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
            self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
        ), dtype=np.int64)

        for ex_ind in range(len(self.data_loader_train.dataset)):
            gt_classes = self.data_loader_train.dataset.gt_classes[ex_ind].copy()
            gt_relations = self.data_loader_train.dataset.relationships[ex_ind].copy()
            gt_boxes = self.data_loader_train.dataset.gt_boxes[ex_ind].copy()

            # For the foreground, we'll just look at everything
            o1o2 = gt_classes[gt_relations[:, :2]]
            for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
                fg_matrix[o1, o2, gtr] += 1

            # For the background, get all of the things that overlap.
            o1o2_total = gt_classes[np.array(
                self._box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
            for (o1, o2) in o1o2_total:
                bg_matrix[o1, o2] += 1

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, len(self.data_loader_train.dataset)))

        return fg_matrix, bg_matrix

    def _box_filter(self, boxes, must_overlap=False):
        """ Only include boxes that overlap as possible relations.
        If no overlapping boxes, use all of them."""
        n_cands = boxes.shape[0]

        overlaps = bbox_overlaps(torch.from_numpy(boxes.astype(np.float)), torch.from_numpy(boxes.astype(np.float))).numpy() > 0
        np.fill_diagonal(overlaps, 0)

        all_possib = np.ones_like(overlaps, dtype=np.bool)
        np.fill_diagonal(all_possib, 0)

        if must_overlap:
            possible_boxes = np.column_stack(np.where(overlaps))

            if possible_boxes.size == 0:
                possible_boxes = np.column_stack(np.where(all_possib))
        else:
            possible_boxes = np.column_stack(np.where(all_possib))
        return possible_boxes

    def train(self):
        """
        main body for training scene graph generation model
        """
        start_iter = self.arguments["iteration"]
        logger = logging.getLogger("scene_graph_generation.trainer")
        logger.info("Start training")
        meters = MetricLogger(delimiter="  ")
        max_iter = len(self.data_loader_train)
        self.scene_parser.train()
        start_training_time = time.time()
        end = time.time()
        for i, data in enumerate(self.data_loader_train, start_iter):
            data_time = time.time() - end
            self.arguments["iteration"] = i
            self.sp_scheduler.step()
            imgs, targets, _ = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            loss_dict = self.scene_parser(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = loss_dict
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)

            self.sp_optimizer.zero_grad()
            losses.backward()
            self.sp_optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - i)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if i % 20 == 0 or i == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "model: {tag}",
                            "eta: {eta}",
                            "iter: {iter}/{max_iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        tag="scene_parser",
                        eta=eta_string,
                        iter=i, max_iter=max_iter,
                        meters=str(meters),
                        lr=self.sp_optimizer.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            if (i + 1) % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                self.sp_checkpointer.save("checkpoint_{:07d}".format(i), **self.arguments)
            if (i + 1) == max_iter:
                self.sp_checkpointer.save("checkpoint_final", **self.arguments)

    def _accumulate_predictions_from_multiple_gpus(self, predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return
        # merge the list of dicts
        predictions = {}
        for p in all_predictions:
            predictions.update(p)
        # convert a dict where the key is the index in a list
        image_ids = list(sorted(predictions.keys()))
        if len(image_ids) != image_ids[-1] + 1:
            logger = logging.getLogger("scene_graph_generation.inference")
            logger.warning(
                "Number of images that were gathered from multiple processes is not "
                "a contiguous set. Some images might be missing from the evaluation"
            )

        # convert to a list
        predictions = [predictions[i] for i in image_ids]
        return predictions

    def visualize_detection(self, dataset, img_ids, imgs, predictions,live=False,visualize_folder="visualize",
                            raw_folder = "raw",
                            bbox_folder="bbox"):
        visualize_folder = "/home/ncl/ADD_sy/inference/sg_inference/visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
            if not os.path.exists(visualize_folder+'/'+raw_folder):
                os.mkdir(visualize_folder+'/'+raw_folder)
            if not os.path.exists(visualize_folder + '/' + bbox_folder):
                os.mkdir(visualize_folder + '/' + bbox_folder)
        # print(type(imgs)) # <class 'lib.scene_parser.rcnn.structures.image_list.ImageList'>
        # print('length of predictions',len(predictions))
        for i, prediction in enumerate(predictions):
            top_prediction = select_top_predictions(prediction)
            # print('top predcition',len(top_prediction))
            img = imgs.permute(1, 2, 0).contiguous().cpu().numpy() # + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            # import pdb; pdb.set_trace()
            if live:
                cv2.imwrite(
                    os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                    result)
                img = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/raw/raw_detection_{img_ids[0]}.jpg')
                cv2.imshow('raw_image', img)
                cv2.waitKey(12)
            else:
                cv2.imwrite(os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                            result)
                # im = Image.fromarray(result)
                # im.save('/home/ncl/ADD_sy/inference/sg_inference/visualize/'+'test.png')

            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            if live:
                cv2.imwrite(
                    os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])),
                    result)
                img2 = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/bbox/bbox_detection_{img_ids[0]}.jpg')
                cv2.imshow('bounding_box_image', img2)
                cv2.waitKey(12)
            else:
                cv2.imwrite(os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])), result)
                # im = Image.fromarray(result)
                # im.save('/home/ncl/ADD_sy/inference/sg_inference/visualize/' + 'test2.png')

    def visualize_detection_2020(self, dataset, img_ids, imgs, predictions,live=False,visualize_folder="visualize",
                            raw_folder = "raw",
                            bbox_folder="bbox"):
        visualize_folder = "/home/ncl/ADD_sy/inference/sg_inference/visualize"
        if not os.path.exists(visualize_folder):
            os.mkdir(visualize_folder)
            if not os.path.exists(visualize_folder+'/'+raw_folder):
                os.mkdir(visualize_folder+'/'+raw_folder)
            if not os.path.exists(visualize_folder + '/' + bbox_folder):
                os.mkdir(visualize_folder + '/' + bbox_folder)
        # print(type(imgs)) # <class 'lib.scene_parser.rcnn.structures.image_list.ImageList'>
        # print('length of predictions',len(predictions))
        for i, prediction in enumerate(predictions):
            # import pdb; pdb.set_trace()
            top_prediction = select_top_predictions(prediction)
            # print('top predcition',len(top_prediction))
            img = imgs.tensors[i].permute(1, 2, 0).contiguous().cpu().numpy() + np.array(self.cfg.INPUT.PIXEL_MEAN).reshape(1, 1, 3)
            result = img.copy()
            # import pdb; pdb.set_trace()
            if live:
                cv2.imwrite(
                    os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                    result)
                img = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/raw/raw_detection_{img_ids[0]}.jpg')
                cv2.imshow('raw_image', img)
                cv2.waitKey(20)
            else:
                cv2.imwrite(os.path.join(visualize_folder + '/' + raw_folder, "raw_detection_{}.jpg".format(img_ids[i])),
                            result)

            result = overlay_boxes(result, top_prediction)
            result = overlay_class_names(result, top_prediction, dataset.ind_to_classes)
            if live:
                cv2.imwrite(
                    os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])),
                    result)
                img2 = cv2.imread(
                    f'/home/ncl/ADD_sy/inference/sg_inference/visualize/bbox/bbox_detection_{img_ids[0]}.jpg')
                cv2.imshow('bounding_box_image', img2)
                cv2.waitKey(20)
            else:
                cv2.imwrite(os.path.join(visualize_folder + '/' + bbox_folder, "bbox_detection_{}.jpg".format(img_ids[i])), result)

    def test_nono(self, data_loader, timer=None, visualize=False, live=False, output_folder="results/"):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []

        # imgs = imgs.to(self.device)

        # graph handler added
        gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
        gt_data_obj = GTDictHandler(gt_data_path)

        for i, data in enumerate(data_loader, 0):
            logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            logger.info("analyzing_model.py 229")
            # TODO
            imgs, _ = data
            imgs = imgs.to(self.device)
            image_ids = (122,)
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                if timer:
                    timer.tic()
                # import pdb; pdb.set_trace()
                # bbox +
                output = self.scene_parser(imgs)

                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]

                # output_pred
                # ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                # reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                # reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]

                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs.squeeze(0), output, live=live)
                # CHECK
                # breakpoint()
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
                # targets_dict.update(
                #     {img_id: target for img_id, target in zip(image_ids, targets)}
                # )
                # breakpoint()
                if self.cfg.MODEL.RELATION_ON:
                    results_pred_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output_pred)}
                    )

                predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

                # breakpoint()
                if self.cfg.MODEL.RELATION_ON:
                    predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
                if not is_main_process():
                    return

                extra_args = dict(
                    box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                    iou_types=("bbox",),
                    expected_results=self.cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                )
                eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                                            predictions=predictions,
                                            output_folder=output_folder,
                                            image_ids=image_ids,
                                            **extra_args)
                breakpoint()
                if self.cfg.MODEL.RELATION_ON:
                    evaluate_sg(dataset=self.data_loader_test.dataset,
                                predictions=predictions,
                                predictions_pred=predictions_pred,
                                output_folder=output_folder,
                                image_ids=image_ids,
                                **extra_args)
                    if visualize:
                        # import pdb; pdb.set_trace()
                        demo_merge_json_org(image_ids)
                        jsonMaker = JsonTranslator(gt_data_obj)
                        result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                        jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image',
                                            FileName=f'{image_ids[0]}_image')
                        generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                        graph_obj = GraphHandler(generated_json_path)
                        graph_obj.generate_SG(recall=20)
                        graph_drawer = GraphDrawer(graph_obj)
                        graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                    if live:
                        sg = cv2.imread(
                            f'/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/{image_ids[0]}_sg.png')
                        cv2.imshow('scene_graph', sg)
                        cv2.waitKey(10)

                print(f'{image_ids[0]} done')
    def singe_test(self, imgs, timer=None, visualize=False, live=False, output_folder="results/"):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []

        imgs = imgs.to(self.device)

        # graph handler added
        gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
        gt_data_obj = GTDictHandler(gt_data_path)

        # for i, data in enumerate(self.data_loader_test, 0):
        #     logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
        #     logger.info("analyzing_model.py 229")
        #     # TODO
        #     imgs, targets, image_ids = data
        imgs = imgs.to(self.device)
        image_ids = (122,)
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if timer:
                timer.tic()
            # import pdb; pdb.set_trace()
            # bbox +
            output = self.scene_parser(imgs)

            if self.cfg.MODEL.RELATION_ON:
                output, output_pred = output
                output_pred = [o.to(cpu_device) for o in output_pred]
            # output_pred
            # ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
            # reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
            # reg_recalls.append(reg_recall)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]

            if visualize:
                self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs.squeeze(0), output, live=live)
            # CHECK
            # breakpoint()
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            # targets_dict.update(
            #     {img_id: target for img_id, target in zip(image_ids, targets)}
            # )
            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )

            predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
            if not is_main_process():
                return

            extra_args = dict(
                box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                iou_types=("bbox",),
                expected_results=self.cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            )
            eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            output_folder=output_folder,
                            image_ids= image_ids,
                            **extra_args)
            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                evaluate_sg(dataset=self.data_loader_test.dataset,
                                predictions=predictions,
                                predictions_pred=predictions_pred,
                                output_folder=output_folder,
                                image_ids = image_ids,
                                **extra_args)
                if visualize:
                    # import pdb; pdb.set_trace()
                    demo_merge_json(image_ids)
                    jsonMaker = JsonTranslator(gt_data_obj)
                    result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                    jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image',FileName=f'{image_ids[0]}_image')
                    generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                    graph_obj = GraphHandler(generated_json_path)
                    graph_obj.generate_SG(recall=20)
                    graph_drawer = GraphDrawer(graph_obj)
                    graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                if live:
                    sg = cv2.imread(
                        f'/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/{image_ids[0]}_sg.png')
                    cv2.imshow('scene_graph', sg)
                    cv2.waitKey(10)


            print(f'{image_ids[0]} done')

    def test(self, timer=None, visualize=False, live=False, output_folder="results/"):
        """
        test_0220
        original
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []

        # graph handler added
        gt_data_path = '/home/ncl/ADD_sy/inference/sg_inference/ontology_interface/gt_data/VG_SGG_dicts.json'
        gt_data_obj = GTDictHandler(gt_data_path)

        for i, data in enumerate(self.data_loader_test, 0):
            logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            logger.info("analyzing_model.py 229")
            # TODO
            # import pdb; pdb.set_trace()
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            # if i % 10 == 1:
            #     logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            #     logger.info("analyzing_model.py 229")
            #     break
            with torch.no_grad():
                if timer:
                    timer.tic()
                # import pdb; pdb.set_trace()
                # parser.py 의 forward 결과값
                # detections 는 bbox, detection_pairs는 pair
                # output = result = (detections, detection_pairs)
                output = self.scene_parser(imgs)
                # breakpoint()
                # output check
                # output check
                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                torch.save(output,
                           os.path.join('/home/ncl/ADD_sy/inference/sg_inference/results/output',
                                        f'{image_ids[0]}_output.pth'))
                torch.save(output_pred,
                           os.path.join('/home/ncl/ADD_sy/inference/sg_inference/results/output',
                                        f'{image_ids[0]}_output_pred.pth'))
                # output_pred
                ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]

                # if visualize:
                #     self.visualize_detection_2020(self.data_loader_test.dataset, image_ids, imgs, output, live=live)
                # CHECK
                # breakpoint()
                results_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output)}
                )
                targets_dict.update(
                    {img_id: target for img_id, target in zip(image_ids, targets)}
                )
                # breakpoint()
                #"test"
                if self.cfg.MODEL.RELATION_ON:
                    results_pred_dict.update(
                        {img_id: result for img_id, result in zip(image_ids, output_pred)}
                    )
                if self.cfg.instance > 0 and i > self.cfg.instance:
                    break

                synchronize()
                total_time = total_timer.toc()
                total_time_str = get_time_str(total_time)
                num_devices = get_world_size()
                logger.info(
                    "Total run time: {} ({} s / img per device, on {} devices)".format(
                        total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
                    )
                )
                total_infer_time = get_time_str(inference_timer.total_time)
                logger.info(
                    "Model inference time: {} ({} s / img per device, on {} devices)".format(
                        total_infer_time,
                        inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                        num_devices,
                    )
                )
                predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

                # breakpoint()
                if self.cfg.MODEL.RELATION_ON:
                    predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
                if not is_main_process():
                    return

                # output_folder = "results"
                # if output_folder:
                #     if not os.path.exists(output_folder):
                #         os.mkdir(output_folder)
                #     torch.save(predictions, os.path.join(output_folder, str(image_ids[0])+"_predictions.pth"))
                #     if self.cfg.MODEL.RELATION_ON:
                #         torch.save(predictions_pred, os.path.join(output_folder, str(image_ids[0])+"_predictions_pred.pth"))
                # ? TEST.EXPECTED.RESULTS
                extra_args = dict(
                    box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
                    iou_types=("bbox",),
                    expected_results=self.cfg.TEST.EXPECTED_RESULTS,
                    expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                )
                eval_det_results = evaluate(dataset=self.data_loader_test.dataset,
                                predictions=predictions,
                                output_folder=output_folder,
                                image_ids= image_ids,
                                **extra_args)
                # breakpoint()
                if self.cfg.MODEL.RELATION_ON:
                    evaluate_sg(dataset=self.data_loader_test.dataset,
                                    predictions=predictions,
                                    predictions_pred=predictions_pred,
                                    output_folder=output_folder,
                                    image_ids = image_ids,
                                    **extra_args)
                    demo_merge_json(image_ids)
                    if visualize:
                        # import pdb; pdb.set_trace()
                        demo_merge_json_org(image_ids)
                        # jsonMaker = JsonTranslator(gt_data_obj)
                        result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                        # jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image', recall=10,
                        #                     FileName=f'{image_ids[0]}_image')
                        # generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                        graph_obj = GraphHandlerDemo(result_data_path, gt_data_obj)
                        graph_obj.get_name(rank=20, image_ids = image_ids)
                        graph_obj.generate_SG(rank=20)
                        graph_drawer = GraphDrawer(graph_obj)
                        print('done')
                        graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                        self.visualize_detection_2020(self.data_loader_test.dataset, image_ids, imgs, output, live=live)

                        # original
                        # demo_merge_json(image_ids)
                        # jsonMaker = JsonTranslator(gt_data_obj)
                        # result_data_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_merged.json'
                        # jsonMaker.make_json(result_data_path, img_name=f'{image_ids[0]}_image',FileName=f'{image_ids[0]}_image')
                        # generated_json_path = f'/home/ncl/ADD_sy/inference/sg_inference/results/to_send/{image_ids[0]}_image.json'
                        # graph_obj = GraphHandler(generated_json_path)
                        # graph_obj.generate_SG(rank=20)
                        # graph_drawer = GraphDrawer(graph_obj)
                        # graph_drawer.draw_and_save(figure_name=f'{image_ids[0]}_sg')
                    if live:
                        sg = cv2.imread(
                            f'/home/ncl/ADD_sy/inference/sg_inference/visualize/sg_result/{image_ids[0]}_sg.png')
                        cv2.imshow('scene_graph', sg)
                        cv2.waitKey(10)


                print(f'{image_ids[0]} done')

    def original_test(self, timer=None, visualize=False, live=False, output_folder="results"):
        """
        main body for testing scene graph generation model
        """
        logger = logging.getLogger("scene_graph_generation.inference")
        logger.info("Start evaluating")
        self.scene_parser.eval()
        targets_dict = {}
        results_dict = {}
        if self.cfg.MODEL.RELATION_ON:
            results_pred_dict = {}
        cpu_device = torch.device("cpu")
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        reg_recalls = []
        for i, data in enumerate(self.data_loader_test, 0):
            logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            logger.info("analyzing_model.py 229")
            imgs, targets, image_ids = data
            imgs = imgs.to(self.device); targets = [target.to(self.device) for target in targets]
            # if i % 10 == 1:
            #     logger.info("inference on batch {}/{}...".format(i, len(self.data_loader_test)))
            #     logger.info("analyzing_model.py 229")
            #     break
            with torch.no_grad():
                if timer:
                    timer.tic()
                # import pdb; pdb.set_trace()
                # bbox +
                output = self.scene_parser(imgs)

                if self.cfg.MODEL.RELATION_ON:
                    output, output_pred = output
                    output_pred = [o.to(cpu_device) for o in output_pred]
                # output_pred
                ious = bbox_overlaps(targets[0].bbox, output[0].bbox)
                reg_recall = (ious.max(1)[0] > 0.5).sum().item() / ious.shape[0]
                reg_recalls.append(reg_recall)
                if timer:
                    torch.cuda.synchronize()
                    timer.toc()
                output = [o.to(cpu_device) for o in output]

                if visualize:
                    self.visualize_detection(self.data_loader_test.dataset, image_ids, imgs, output, live=live)
            # CHECK
            # breakpoint()
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
            targets_dict.update(
                {img_id: target for img_id, target in zip(image_ids, targets)}
            )
            # breakpoint()
            if self.cfg.MODEL.RELATION_ON:
                results_pred_dict.update(
                    {img_id: result for img_id, result in zip(image_ids, output_pred)}
                )
            if self.cfg.instance > 0 and i > self.cfg.instance:
                break
            break
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        num_devices = get_world_size()
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(self.data_loader_test.dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(self.data_loader_test.dataset),
                num_devices,
            )
        )
        predictions = self._accumulate_predictions_from_multiple_gpus(results_dict)

        # breakpoint()
        if self.cfg.MODEL.RELATION_ON:
            predictions_pred = self._accumulate_predictions_from_multiple_gpus(results_pred_dict)
        if not is_main_process():
            return

        # output_folder = "results"
        # if output_folder:
        #     if not os.path.exists(output_folder):
        #         os.mkdir(output_folder)
        #     torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        #     if self.cfg.MODEL.RELATION_ON:
        #         torch.save(predictions_pred, os.path.join(output_folder, "predictions_pred.pth"))
        # ? TEST.EXPECTED.RESULTS
        extra_args = dict(
            box_only=False if self.cfg.MODEL.RETINANET_ON else self.cfg.MODEL.RPN_ONLY,
            iou_types=("bbox",),
            expected_results=self.cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=self.cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        evaluate(dataset=self.data_loader_test.dataset,
                        predictions=predictions,
                        output_folder=output_folder,
                        **extra_args)
        # breakpoint()
        if self.cfg.MODEL.RELATION_ON:
            evaluate_sg(dataset=self.data_loader_test.dataset,
                            predictions=predictions,
                            predictions_pred=predictions_pred,
                            output_folder=output_folder,
                            **extra_args)

        # merge_json()
        # generate_graph()
        # if live:
        #     cv2.imshow()
        # else:
        #     cv2.imwrite()

def build_model(cfg, args, arguments, local_rank, distributed):

    if args.raw_img:
        # image inference
        # image stroed at '/home/ncl/ADD_sy/inference/sg_inference/lib/data'
        return OntoSceneGraphGeneration(cfg, arguments, local_rank, distributed)
    else:
        # test mode with visual genome dataset
        return SceneGraphGeneration(cfg, arguments, local_rank, distributed)

# def build_model(cfg, arguments, local_rank, distributed):
#     return SceneGraphGeneration(cfg, arguments, local_rank, distributed)
