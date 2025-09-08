from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from sam2.sam2_video_predictor import SAM2VideoPredictor, SAM2VideoPredictorVOS
from tqdm import tqdm

from .misc import load_video_frames_from_images


class TemporalVideoPredictor(SAM2VideoPredictor):
    """The predictor class to handle user interactions and manage inference states."""

    def __init__(self, ignore_empty_mask_prompts=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_empty_mask_prompts = ignore_empty_mask_prompts

    def set_predictor_state(self, image_paths: List[str]) -> Dict:
        """
        target_size: Tuple[int, int]: the size of the query image. All context images will be resized to this size.
        """
        inference_state = self.init_state_from_frames_list(image_paths=image_paths)

        return inference_state

    def add_context_masks(
        self,
        inference_state: dict,
        masks_context: Dict[int, torch.Tensor],
        category_ids: Tuple[int, ...] = (1, 2, 3),
    ):
        """
        Prompts the video prediction models with context masks provided as arrays.

        Args:
            inference_state: Current inference state dictionary
            context_mask_arrays: A dictionary mapping from category id to mask tensor
            img_size: Target size for mask resizing (width, height)
            category_ids: Tuple of category IDs to process
            device: Torch device for processing

        Returns:
            None: Updates inference_state in-place
        """
        self.reset_state(inference_state)

        is_init_cond_frame = [True] * len(category_ids)
        for frame_idx, masks_frame in enumerate(masks_context):
            assert len(masks_frame) == len(category_ids), "Number of categories must match category IDs"

            for cat_id in category_ids:
                mask_prompt = masks_frame[cat_id]

                # This is very important,
                # when the dataset is partially labeled, the absense of a label does not mean the absense of the object
                # So, we ignore context with empty masks. The idea is that the annotators may have skipped this category even though it exists in the frame.
                # But depening on your dataset, you may want to remove this check.

                if self.ignore_empty_mask_prompts and not mask_prompt.sum() > 0:
                    print(f"Empty mask for category {category_id} in frame {frame_idx}")
                    continue

                self.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=cat_id,
                    mask=mask_prompt,
                    is_init_cond_frame=is_init_cond_frame[category_ids.index(cat_id)],
                )
                # print(f"Added mask for category {category_id} in frame {frame_idx}")

                # All future frames with mask input for the same object are treated as non-init conditioning frames (memory from other frames is used)
                is_init_cond_frame[category_ids.index(cat_id)] = False

    @torch.inference_mode()
    def add_new_mask(
        self,
        inference_state,
        frame_idx,
        obj_id,
        mask,
        is_init_cond_frame=True,
    ):
        """

        Add new mask to a frame.

        # the input points will be used to correct the already tracked masks.
        - `is_init_cond_frae` indicates whether this frame is the first frame that received a mask prompt (i.e., without using any memory from other frames).
        -  non -init conditioning frames will use memory and the current mask to make a prediction.

        """
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        assert mask.dim() == 2
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        # We consider all frames receiving mask as conditioning frames, i.e., they'll be carried over as memory when making predictions on other frames in the future.
        is_cond = True
        reverse = False
        storage_key = "cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=True,
        )
        # Add the output to the output dict (to be used as future memory)
        # obj_temp_output_dict[storage_key][frame_idx] = current_out
        obj_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(inference_state, consolidated_out["pred_masks_video_res"])
        return frame_idx, obj_ids, video_res_masks

    def propagate_video_segments(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate segmentation through video frames.

        Args:
            predictor: Video predictor model
            inference_state: Current inference state

        Returns:
            Dict mapping frame index to dict of object masks
        """
        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu() for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def propagate_segments_for_vos(
        self,
        inference_state,
        keyframe_indices: List[int],
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ) -> Dict[int, Dict[int, np.ndarray]]:
        """
        Propagate segmentation through video frames.

        Args:
            predictor: Video predictor model
            inference_state: Current inference state

        Returns:
            Dict mapping frame index to dict of object masks
        """
        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.propagate_in_video_for_vos(
            inference_state=inference_state,
            keyframe_indices=keyframe_indices,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        ):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu() for i, out_obj_id in enumerate(out_obj_ids)
            }
        return video_segments

    def forward_vos(
        self,
        image_paths: List[str],
        keyframe_indices: List[int],
        keyframe_masks: List[Dict[int, torch.Tensor]],
    ):
        inference_state = self.set_predictor_state(image_paths=image_paths)
        categories = []
        for mask_ann in keyframe_masks:
            categories.extend(list(mask_ann.keys()))
        categories = list(set(categories))

        # Add context (image + masks prompt - for all objects/categories)
        is_init_cond_frame = [True] * len(categories)
        for keyframe_idx, mask_ann in zip(keyframe_indices, keyframe_masks):
            # the first frame with mask input is treated as an init conditioning frame (no memory from other frames is used)
            for cat, mask_tensor in mask_ann.items():
                cat_mask = mask_tensor.squeeze(0).float().to(self.device)
                if cat_mask.sum() > 0:
                    self.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=keyframe_idx,
                        obj_id=cat,
                        mask=cat_mask,
                        is_init_cond_frame=is_init_cond_frame[categories.index(cat)],
                    )
                    # All future frames with mask input for the same object are treated as non-init conditioning frames (memory from other frames is used)
                    is_init_cond_frame[categories.index(cat)] = False

        # Propagate throughout the rest of the video
        video_segments = self.propagate_segments_for_vos(
            inference_state, keyframe_indices=keyframe_indices, start_frame_idx=0
        )

        return video_segments

    def forward_in_context(
        self,
        image_query: str,
        images_context: List[str],
        masks_context: Dict[int, torch.Tensor],
        category_ids: Tuple[int],
    ):
        inference_state = self.set_predictor_state(image_paths=images_context + [image_query])

        self.add_context_masks(
            inference_state=inference_state,
            masks_context=masks_context,
            category_ids=category_ids,
        )

        video_prediction = self.propagate_video_segments(inference_state, start_frame_idx=0)

        # remove all masks in all frames throughout the video, for next round of inference.
        self.reset_state(inference_state)

        del inference_state
        return video_prediction

    @torch.no_grad()
    def init_state_from_frames_list(
        self,
        image_paths: List[str],
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
        async_loading_frames=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames_from_images(
            image_paths=image_paths,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            async_loading_frames=async_loading_frames,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["image_paths"] = image_paths
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """Propagate the input points across frames to track in the entire video."""

        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]

                # We skip frames that already have outputs consolidated from prompting
                if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    # if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    print(f"Skipping frame {frame_idx} for object {obj_idx}")
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                else:
                    storage_key = "non_cond_frame_outputs"
                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks

    @torch.inference_mode()
    def propagate_in_video_for_vos(
        self,
        inference_state,
        start_frame_idx,
        keyframe_indices,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)

        # set start index, end index, and processing order
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)

        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            pred_masks_per_obj = [None] * batch_size
            for obj_idx in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]

                # Refine the predictions for the keyframe, based on predictions from recent frames
                # We skip frames that already have outputs consolidated from prompting
                if frame_idx not in keyframe_indices and frame_idx in obj_output_dict["cond_frame_outputs"]:
                    # if frame_idx in obj_output_dict["cond_frame_outputs"]:
                    print(f"Skipping frame {frame_idx} for object {obj_idx}")
                    storage_key = "cond_frame_outputs"
                    current_out = obj_output_dict[storage_key][frame_idx]
                    device = inference_state["device"]
                    pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                else:
                    if frame_idx in keyframe_indices:
                        storage_key = "cond_frame_outputs"  # We are going to carry over keyframe mask as a condtioning memory for future predictions
                    else:
                        storage_key = "non_cond_frame_outputs"

                    current_out, pred_masks = self._run_single_frame_inference(
                        inference_state=inference_state,
                        output_dict=obj_output_dict,
                        frame_idx=frame_idx,
                        batch_size=1,  # run on the slice of a single object
                        is_init_cond_frame=False,
                        point_inputs=None,
                        mask_inputs=None,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                    obj_output_dict[storage_key][frame_idx] = current_out

                inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {"reverse": reverse}
                pred_masks_per_obj[obj_idx] = pred_masks

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            if len(pred_masks_per_obj) > 1:
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
            else:
                all_pred_masks = pred_masks_per_obj[0]
            _, video_res_masks = self._get_orig_video_res_output(inference_state, all_pred_masks)
            yield frame_idx, obj_ids, video_res_masks


class SAM2VideoPredictorVOS(TemporalVideoPredictor, SAM2VideoPredictorVOS):
    """Optimized for the VOS setting"""
