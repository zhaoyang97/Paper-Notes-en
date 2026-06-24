---
title: >-
  ECCV2024 Object Detection Papers · 31 Notes
description: >-
  31 ECCV2024 papers in the Object Detection area, covering Object Detection, Few-/Zero-Shot Learning, Self-Supervised Learning, Layout & Composition, Object Tracking and more. Each note has TL;DR, motivation, method, experiments, highlights, and limitations — 5-minute reads of core ideas.
tags:
  - "ECCV2024"
  - "Object Detection"
  - "AI paper notes"
  - "paper summaries"
  - "Few-/Zero-Shot Learning"
  - "Self-Supervised Learning"
  - "Layout & Composition"
  - "Object Tracking"
item_list:
  - u: "adaptive_bounding_box_uncertainties_via_twostep_conformal_pr/"
    t: "Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction"
  - u: "adaptive_multi-task_learning_for_few-shot_object_detection/"
    t: "Adaptive Multi-task Learning for Few-Shot Object Detection"
  - u: "augdetr_improving_multi-scale_learning_for_detection_transformer/"
    t: "AugDETR: Improving Multi-scale Learning for Detection Transformer"
  - u: "bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro/"
    t: "BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos"
  - u: "bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de/"
    t: "Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection"
  - u: "can_ood_object_detectors_learn_from_foundation_models/"
    t: "Can OOD Object Detectors Learn from Foundation Models?"
  - u: "damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu/"
    t: "DAMSDet: Dynamic Adaptive Multispectral Detection Transformer"
  - u: "dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning/"
    t: "DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning"
  - u: "gra_detecting_oriented_objects_through_group-wise_rotating_and_attention/"
    t: "GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention"
  - u: "lami-detr_open-vocabulary_detection_with_language_model_instruction/"
    t: "LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction"
  - u: "mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection/"
    t: "MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection"
  - u: "nonverbal_interaction_detection/"
    t: "Nonverbal Interaction Detection"
  - u: "on_calibration_of_object_detectors_pitfalls_evaluation_and_baselines/"
    t: "On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines"
  - u: "openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection/"
    t: "OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection"
  - u: "plain-det_a_plain_multi-dataset_object_detector/"
    t: "Plain-Det: A Plain Multi-Dataset Object Detector"
  - u: "portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer/"
    t: "Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer"
  - u: "projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio/"
    t: "Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation"
  - u: "rectify_the_regression_bias_in_long-tailed_object_detection/"
    t: "Rectify the Regression Bias in Long-Tailed Object Detection"
  - u: "reground_improving_textual_and_spatial_grounding_at_no_cost/"
    t: "ReGround: Improving Textual and Spatial Grounding at No Cost"
  - u: "responsible_visual_editing/"
    t: "Responsible Visual Editing"
  - u: "self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection/"
    t: "Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection"
  - u: "shifted_autoencoders_for_point_annotation_restoration_in_object_counting/"
    t: "Shifted Autoencoders for Point Annotation Restoration in Object Counting"
  - u: "shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr/"
    t: "SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding"
  - u: "stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo/"
    t: "Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization"
  - u: "taptr_tracking_any_point_with_transformers_as_detection/"
    t: "TAPTR: Tracking Any Point with Transformers as Detection"
  - u: "tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap/"
    t: "Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography"
  - u: "towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat/"
    t: "Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching"
  - u: "visible_and_clear_finding_tiny_objects_in_difference_map/"
    t: "Visible and Clear: Finding Tiny Objects in Difference Map"
  - u: "walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc/"
    t: "WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs"
  - u: "weak-to-strong_compositional_learning_from_generative_models_for_language-based_/"
    t: "Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection"
item_total: 31
---

<!-- Auto-generated by src/gen_blog_index.py --lang en -->
# 🎯 Object Detection

**🎞️ ECCV2024** · **31** paper notes

📌 **Same area in other venues:** [📷 CVPR2026 (99)](../../CVPR2026/object_detection/index.md) · [🔬 ICLR2026 (31)](../../ICLR2026/object_detection/index.md) · [🧪 ICML2026 (6)](../../ICML2026/object_detection/index.md) · [🤖 AAAI2026 (29)](../../AAAI2026/object_detection/index.md) · [🧠 NeurIPS2025 (27)](../../NeurIPS2025/object_detection/index.md) · [📹 ICCV2025 (28)](../../ICCV2025/object_detection/index.md)

🔥 **Top topics:** Object Detection ×7 · Few-/Zero-Shot Learning ×2 · Self-Supervised Learning ×2 · Layout & Composition ×2 · Object Tracking ×2

**[Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](adaptive_bounding_box_uncertainties_via_twostep_conformal_pr.md)**

:   This paper proposes a two-step conformal prediction framework for uncertainty quantification in multi-object detection: the first step generates conformal prediction sets of class labels to handle classification errors, and the second step produces adaptive bounding box uncertainty intervals based on ensembles and quantile regression, providing practically useful tight prediction intervals while guaranteeing coverage.

**[Adaptive Multi-task Learning for Few-Shot Object Detection](adaptive_multi-task_learning_for_few-shot_object_detection.md)**

:   This paper proposes an adaptive multi-task learning method (MTL-FSOD) that dynamically adjusts the gradient scales of classification and localization tasks using a precision-driven gradient balancer to alleviate their conflict. It also introduces CLIP-based knowledge distillation and a classification refinement scheme to enhance individual task performance, achieving consistent improvements across multiple few-shot object detection benchmarks.

**[AugDETR: Improving Multi-scale Learning for Detection Transformer](augdetr_improving_multi-scale_learning_for_detection_transformer.md)**

:   This paper proposes AugDETR (Augmented DETR), which expands the receptive field of the deformable encoder and introduces global context features to enhance feature representations through a Hybrid Attention Encoder. It then adaptively utilizes information from multiple encoder layers using Encoder-Mixing Cross-Attention to accelerate convergence, yielding improvements of 1.2, 1.1, and 1.0 AP over DINO, AlignDETR, and DDQ on COCO, respectively.

**[BAM-DETR: Boundary-Aligned Moment Detection Transformer for Temporal Sentence Grounding in Videos](bam-detr_boundary-aligned_moment_detection_transformer_for_temporal_sentence_gro.md)**

:   Proposes the Boundary-Aligned Moment Detection Transformer (BAM-DETR), which models moments using an anchor-boundary triplet $(p, d_s, d_e)$ instead of the traditional center-length duplet $(c, l)$. Combined with a dual-pathway decoder and a quality-based ranking mechanism, it effectively addresses the issue of imprecise localization caused by center ambiguity.

**[Bridge Past and Future: Overcoming Information Asymmetry in Incremental Object Detection](bridge_past_and_future_overcoming_information_asymmetry_in_incremental_object_de.md)**

:   This paper proposes the Bridge Past and Future (BPF) method, which bridges past stages via pseudo-labels, excludes potential future objects using an attention mechanism, and incorporates dual-teacher distillation (Distillation with Future) to resolve the optimization goal inconsistency caused by cross-stage information asymmetry in incremental object detection.

**[Can OOD Object Detectors Learn from Foundation Models?](can_ood_object_detectors_learn_from_foundation_models.md)**

:   SyncOOD proposes an automated data curation method that leverages LLMs to imagine semantically novel OOD concepts and performs region-level editing on ID images via Stable Diffusion Inpainting to synthesize scene-level OOD samples. After refining bounding boxes with SAM and filtering via feature similarity, a lightweight MLP classifier is trained, substantially outperforming SOTA on multiple OOD detection benchmarks with a minimal amount of synthetic data.

**[DAMSDet: Dynamic Adaptive Multispectral Detection Transformer](damsdet_dynamic_adaptive_multispectral_detection_transformer_with_competitive_qu.md)**

:   DAMSDet proposes a dynamic adaptive infrared-visible object detection method based on the DETR architecture. By utilizing Modality Competitive Query Selection (dynamically selecting the dominant modality feature as the initial query for each object) and Multispectral Deformable Cross-Attention (adaptively sampling and aggregating bi-modal features across multiple semantic levels), it simultaneously addresses the dual challenges of complementary information fusion and modality misalignment, significantly outperforming the state-of-the-art (SOTA) on four public datasets.

**[DSPDet3D: 3D Small Object Detection with Dynamic Spatial Pruning](dspdet3d_3d_small_object_detection_with_dynamic_spatial_pruning.md)**

:   Proposed a Dynamic Spatial Pruning (DSP) strategy to progressively remove voxel features in areas where large objects have already been detected within the decoders of multi-scale 3D detectors. This allows the detector to process scenes at extremely high spatial resolutions, significantly improving small object detection accuracy (ScanNet small object mAP@0.25 boosted from 27.5% to 44.8%) while reducing GPU memory to 1/5 of the baseline method with the same resolution.

**[GRA: Detecting Oriented Objects Through Group-Wise Rotating and Attention](gra_detecting_oriented_objects_through_group-wise_rotating_and_attention.md)**

:   A lightweight Group-wise Rotating and Attention (GRA) module is proposed. By grouping and rotating convolution kernels and applying group-wise spatial attention, it outperforms the previous SOTA method ARC with nearly 50% fewer parameters, achieving new state-of-the-art performance on DOTA-v2.0.

**[LaMI-DETR: Open-Vocabulary Detection with Language Model Instruction](lami-detr_open-vocabulary_detection_with_language_model_instruction.md)**

:   LaMI-DETR is proposed to address two core challenges in open-vocabulary object detection—insufficient concept representation and base-category overfitting—by leveraging GPT to generate visual concept descriptions and T5 to mine inter-category visual similarity relationships. It outperforms previous state-of-the-art methods by 7.8 rare AP on OV-LVIS, achieving 43.4 AP_rare.

**[MutDet: Mutually Optimizing Pre-training for Remote Sensing Object Detection](mutdet_mutually_optimizing_pre-training_for_remote_sensing_object_detection.md)**

:   MutDet is proposed, a mutually optimizing pre-training framework for remote sensing oriented object detection. It systematically mitigates the feature discrepancy issue between object embeddings and detector features in detection pre-training via bidirectional cross-attention fusion of object embeddings and encoder features, a contrastive alignment loss, and an auxiliary Siamese head.

**[Nonverbal Interaction Detection](nonverbal_interaction_detection.md)**

:   This work presents the first systematic study of human nonverbal interaction (gestures, expressions, gaze, postures, touch), introducing a large-scale dataset NVI, a new task NVI-DET, and a dual multi-scale hypergraph-based detection model NVI-DEHR, which achieves state-of-the-art performance on both nonverbal interaction detection and HOI detection tasks.

**[On Calibration of Object Detectors: Pitfalls, Evaluation and Baselines](on_calibration_of_object_detectors_pitfalls_evaluation_and_baselines.md)**

:   This paper systematically reveals significant flaws in existing evaluation frameworks, evaluation metrics, and the use of Temperature Scaling in object detector calibration research. It proposes a principled joint evaluation framework along with post-hoc calibration methods tailored specifically for object detection (Platt Scaling and Isotonic Regression), demonstrating that correctly designed and evaluated post-hoc calibrators far outperform recent train-time calibration methods.

**[OpenKD: Opening Prompt Diversity for Zero- and Few-shot Keypoint Detection](openkd_opening_prompt_diversity_for_zero-_and_few-shot_keypoint_detection.md)**

:   This paper proposes the OpenKD model, which opens up prompt diversity across three dimensions: modality (vision + text), semantics (seen vs. unseen), and language (diverse text). By employing a multimodal prototype set, auxiliary keypoint-text interpolation, and LLM-based text parsing, OpenKD achieves generalized zero- and few-shot keypoint detection, obtaining SOTA performance on Animal Pose, AwA, CUB, and NABirds.

**[Plain-Det: A Plain Multi-Dataset Object Detector](plain-det_a_plain_multi-dataset_object_detector.md)**

:   Plain-Det proposes a simple and flexible multi-dataset object detection framework. By incorporating semantic space calibration, class-aware query compositor, and hardness-indicated dynamic sampling strategies, it achieves 51.9 mAP on COCO (matching the SOTA at that time) and can be flexibly scaled to new datasets while maintaining robust performance.

**[Portrait4D-v2: Pseudo Multi-View Data Creates Better 4D Head Synthesizer](portrait4d-v2_pseudo_multi-view_data_creates_better_4d_head_synthesizer.md)**

:   A new learning paradigm utilizing **pseudo multi-view videos** to train a feed-forward, one-shot 4D head synthesizer is proposed. It first learns a 3D head synthesizer from synthetic data to convert monocular videos into multi-view videos, and then trains the 4D synthesizer through **cross-view self-reenactment** using these pseudo multi-view videos. This avoids over-reliance on 3DMMs and significantly outperforms prior methods in reconstruction fidelity, geometric consistency, and motion control accuracy.

**[Projecting Points to Axes: Oriented Object Detection via Point-Axis Representation](projecting_points_to_axes_oriented_object_detection_via_point-axis_representatio.md)**

:   This paper proposes a Point-Axis representation method that decouples the position (point set) and orientation (axis encoding) of oriented objects. Facilitated by Max-Projection Loss and Cross-Axis Loss, this method achieves optimization without requiring extra annotations. Based on this, the Oriented DETR model is designed to resolve the loss discontinuity issue inherent in traditional oriented bounding box representations.

**[Rectify the Regression Bias in Long-Tailed Object Detection](rectify_the_regression_bias_in_long-tailed_object_detection.md)**

:   This work first reveals and systematically addresses the overlooked **regression bias** problem in long-tailed object detection. Due to insufficient samples, the parameters of class-specific regression heads for rare categories suffer from poor generalization. By incorporating an additional class-agnostic regression branch for trade-off, this method achieves state-of-the-art performance on datasets such as LVIS.

**[ReGround: Improving Textual and Spatial Grounding at No Cost](reground_improving_textual_and_spatial_grounding_at_no_cost.md)**

:   By changing the sequential connection of Gated Self-Attention (GSA) and Cross-Attention (CA) in GLIGEN to a parallel connection (network rewiring), the trade-off between textual and spatial grounding is significantly alleviated without introducing new parameters, fine-tuning, or computational overhead.

**[Responsible Visual Editing](responsible_visual_editing.md)**

:   Defines a new task of "Responsible Visual Editing" and proposes CoEditor, a cognitive editor that converts harmful images into responsible versions through a two-stage perceptual-behavioral cognitive process while minimizing modifications.

**[Self-supervised Feature Adaptation for 3D Industrial Anomaly Detection](self-supervised_feature_adaptation_for_3d_industrial_anomaly_detection.md)**

:   This paper proposes the LSFA (Local-to-global Self-supervised Feature Adaptation) framework. It performs task-oriented adaptation of pretrained features through two self-supervised strategies: Intramodal Feature Compactness (IFC) optimization and Cross-modal Local-to-global Consistency (CLC) alignment. LSFA achieves 97.1% I-AUROC on MVTec-3D AD, outperforming the state-of-the-art (SOTA) by +3.4%.

**[Shifted Autoencoders for Point Annotation Restoration in Object Counting](shifted_autoencoders_for_point_annotation_restoration_in_object_counting.md)**

:   Proposes **Shifted AutoEncoders (SAE)**, an MAE-inspired point annotation restoration method: by applying random shifts to point annotations and training a UNet to restore them, the model learns "general location knowledge" while ignoring individual annotation noise. The trained SAE is used to restore original annotations to make them more consistent, which consistently improves the performance of any counting model (density-map or localization-based), setting new records across 9 datasets.

**[SHINE: Saliency-aware HIerarchical NEgative Ranking for Compositional Temporal Grounding](shine_saliency-aware_hierarchical_negative_ranking_for_compositional_temporal_gr.md)**

:   To address the improper negative sample construction in existing compositional temporal grounding methods and the failure of DETR models to generate reasonable saliency responses for negative queries, this paper proposes leveraging an LLM (GPT-3.5 Turbo) to generate semantically feasible hierarchical hard negative samples, and designs a coarse-to-fine saliency ranking strategy to establish multi-granularity semantic relations between video clips and hierarchical negative queries, significantly improving compositional generalization performance.

**[Stepwise Multi-grained Boundary Detector for Point-Supervised Temporal Action Localization](stepwise_multi-grained_boundary_detector_for_point-supervised_temporal_action_lo.md)**

:   To address the semantic ambiguity of action boundaries caused by sparse annotations in point-supervised temporal action localization, this paper proposes a Stepwise Multi-grained Boundary Detector (SMBD). By employing a Background Anchor Generator (BAG) and a Dual Boundary Detector (DBD), SMBD provides fine-grained boundary supervision signals for training, achieving state-of-the-art performance on datasets such as THUMOS'14.

**[TAPTR: Tracking Any Point with Transformers as Detection](taptr_tracking_any_point_with_transformers_as_detection.md)**

:   TAPTR reformulates the Tracking Any Point (TAP) task as a DETR-like detection problem. It represents each tracking point as a point query containing both position and content, which is layer-wise optimized through a multi-layer Transformer decoder. Combined with a cost volume and sliding-window feature update strategy, it achieves SOTA performance on the TAP-Vid benchmark with faster inference speed.

**[Tensorial Template Matching for Fast Cross-Correlation with Rotations and Its Application for Tomography](tensorial_template_matching_for_fast_cross-correlation_with_rotations_and_its_ap.md)**

:   Proposes the Tensorial Template Matching (TTM) algorithm, which integrates the template information under all rotations into a symmetric tensor field to reduce the calculation to a fixed number of cross-correlations. This makes the computational complexity independent of rotational precision, achieving fast and accurate object detection and rotation estimation in 3D tomographic images.

**[Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching](towards_natural_language-guided_drones_geotext-1652_benchmark_with_spatial_relat.md)**

:   Constructed the first natural language-guided drone geolocalization benchmark GeoText-1652 (276K bbox-text pairs, 316K descriptions), and proposed a blending spatial matching method that achieves region-level spatial relation matching via grounding loss + spatial relation loss, achieving a text retrieval Recall@10 of 31.2%.

**[Visible and Clear: Finding Tiny Objects in Difference Map](visible_and_clear_finding_tiny_objects_in_difference_map.md)**

:   SR-TOD introduces the image self-reconstruction mechanism into object detection for the first time, discovering a strong correlation between reconstruction difference maps and tiny objects. It designs a Difference Map Guided Feature Enhancement (DGFE) module, achieving significant improvements on the self-built anti-UAV dataset DroneSwarms as well as VisDrone2019 and AI-TOD.

**[WALKER: Self-supervised Multiple Object Tracking by Walking on Temporal Appearance Graphs](walker_self-supervised_multiple_object_tracking_by_walking_on_temporal_appearanc.md)**

:   This paper proposes Walker, the first self-supervised multiple object tracker. By constructing a quasi-dense temporal object appearance graph, designing a multi-positive contrastive loss to optimize random walks on the graph for instance similarity learning, and introducing mutually-exclusive connectivity constraints and a motion-constrained bidirectional walk inference strategy, Walker achieves competitive self-supervised tracking performance on MOT17, DanceTrack, and BDD100K, outperforming prior self-supervised methods even with 400 times fewer annotations.

**[Weak-to-Strong Compositional Learning from Generative Models for Language-based Object Detection](weak-to-strong_compositional_learning_from_generative_models_for_language-based_.md)**

:   Proposes the WSCL framework: leveraging LLMs to generate diverse text descriptions, diffusion models to generate corresponding images, and a weak detector to decompose phrases and generate pseudo bounding boxes, constructing dense synthetic triplets (image, description, bbox). Together with compositional contrastive learning, it significantly improves language-guided object detection performance, achieving a +5.0 AP improvement for GLIP-T on OmniLabel.

**[YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information](yolov9_learning_what_you_want_to_learn_using_programmable_gradient_information.md)**

:   YOLOv9 proposes Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN) to address the information bottleneck problem in deep networks. It comprehensively outperforms existing real-time object detectors on MS COCO with fewer parameters and computation, surpassing methods pre-trained on large-scale datasets while training from scratch.
