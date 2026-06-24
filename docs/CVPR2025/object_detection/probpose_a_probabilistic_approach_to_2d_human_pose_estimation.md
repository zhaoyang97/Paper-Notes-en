---
title: >-
  [Paper Note] ProbPose: A Probabilistic Approach to 2D Human Pose Estimation
description: >-
  [CVPR 2025][Object Detection][Human Pose Estimation] ProbPose proposes replacing traditional heatmaps with calibrated probability maps for 2D human keypoint localization. It introduces presence probability to explicitly model whether keypoints are within the activation window. Through crop data augmentation and expected risk minimization of the OKS loss, it significantly improves the localization capability of out-of-image keypoints and the quality of the model's probability…
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Human Pose Estimation"
  - "Probability Map"
  - "Out-of-Image Keypoints"
  - "OKS Loss"
  - "Calibration Probability"
date: 2026-05-08
content_hash: e8a746ab8bf4a47b
---

# ProbPose: A Probabilistic Approach to 2D Human Pose Estimation

**Conference**: CVPR 2025  
**arXiv**: [2412.02254](https://arxiv.org/abs/2412.02254)  
**Code**: [https://MiraPurkrabek.github.io/ProbPose/](https://MiraPurkrabek.github.io/ProbPose/)  
**Area**: Object Detection / Human Pose Estimation  
**Keywords**: Human Pose Estimation, Probability Map, Out-of-Image Keypoints, OKS Loss, Calibration Probability

## TL;DR

ProbPose proposes replacing traditional heatmaps with calibrated probability maps for 2D human keypoint localization. It introduces presence probability to explicitly model whether keypoints are within the activation window. Through crop data augmentation and expected risk minimization of the OKS loss, it significantly improves the localization capability of out-of-image keypoints and the quality of the model's probability calibration.

## Background & Motivation

**Background**: Top-down heatmap-based methods (such as ViTPose) are the current mainstream paradigm in 2D human pose estimation, localizing keypoints by predicting dense heatmaps and selecting the argmax. Heatmaps use a Gaussian with a fixed sigma as the training target and MSE as the loss function.

**Limitations of Prior Work**: (1) All existing methods ignore out-of-image keypoints during training and evaluation—when a keypoint falls outside the activation window due to cropping or occlusion, the model is not penalized, leading it to incorrectly localize it onto other joints (e.g., misaligning the left leg to the right leg); (2) Heatmaps lack probabilistic calibration—training with a fixed Gaussian sigma and MSE forces the output to also be a fixed-shape Gaussian, failing to capture true localization uncertainty and, worse, unable to express "I don't know"; (3) Evaluation metrics (OKS/PCK) only assess in-image keypoints and do not penalize incorrect guesses.

**Key Challenge**: Traditional heatmaps conflate three semantic concepts—"localization", "quality assessment", and "presence"—into a single output (the peak heatmap value), encoding three different types of information inside a single scalar, which is neither precise nor flexible.

**Goal**: To design a system that fully describes the state of each keypoint—(1) whether it is within the activation window, (2) where it is, (3) how reliable the localization is, and (4) whether it is visible—while providing calibrated probabilities.

**Key Insight**: The authors observe that image boundaries are essentially a form of occlusion, and crop augmentation can be utilized to generate a large number of "out-of-window keypoint" training samples, thereby training the presence probability.

**Core Idea**: To replace heatmaps with probability maps that normalize to 1 and satisfy probability axioms, substitute MSE with expected risk minimization of the OKS loss, and incorporate an independent presence probability prediction head.

## Method

### Overall Architecture

The output of ProbPose consists of four parts: (1) a probability map—the calibrated probability of keypoints at each position within the activation window; (2) presence probability—the binary probability of whether keypoints are within the activation window; (3) quality estimation—the predicted OKS score; and (4) visibility prediction. During inference, the approach first determines whether the presence probability exceeds a threshold; if the keypoint is present, the localization is obtained from the probability map by maximizing the expected OKS.

### Key Designs

1. **Probability Map and OKS Expected Risk Loss**:

    - **Function**: To provide calibrated probability distributions of keypoint locations.
    - **Mechanism**: The probability map utilizes the Sparsemax activation function to ensure all values are in $[0,1]$ and sum to 1. Each pixel $p_L(x_i) = p(x_i | k_j \in AW, img)$ represents the posterior probability of the keypoint at that location. The loss function is formulated as expected risk minimization $R_{exp}(x_i) = (1 - OKS(x_i)) \cdot p_L(x_i)$, combined with Sobel gradient regularization $\mathcal{L}_{OKS}(x_i) = (1-\alpha) R_{exp}(x_i) + \alpha g(x_i)$. During inference, instead of applying argmax, the expected OKS of each pixel is calculated and the maximum value is selected, which is more robust to bimodal distributions.
    - **Design Motivation**: Traditional MSE + fixed Gaussian targets imply unreasonable shape assumptions—the true posterior distribution of human keypoints should reflect the body shape rather than annotation noise. The probability map does not assume any specific shape, and the calibrated probabilities support more flexible queries (e.g., "the smallest region containing 95% probability"). Gradient regularization prevents the probability map from over-fitting by forming sharp peaks too early.

2. **Presence Probability**:

    - **Function**: To explicitly predict whether a keypoint is within the activation window.
    - **Mechanism**: For each keypoint $k_j$, it predicts $p_p(k_j) = p(k_j \in AW | img)$ and is trained using binary cross-entropy loss. When the presence probability is lower than the threshold, the model does not output localization results; when it is higher than the threshold, the probability map is referenced. The "out-of-window keypoint" samples required for training are generated through crop data augmentation.
    - **Design Motivation**: Existing methods use the peak value of the heatmap for both localization confidence and presence determination, but these two semantics are not mathematically equivalent. By modeling presence independently, the presence classification error on CropCOCO is reduced by 30-45%.

3. **Crop Data Augmentation and Dual Heatmap Method**:

    - **Function**: To generate training samples and extend the localization range.
    - **Mechanism**: (1) Randomly crop training images to force some annotated keypoints out of the window, using these samples to train the presence probability and the empty outputs of the probability map; (2) The dual heatmap method introduces an additional, larger activation window (same resolution but larger field of view) outside the standard activation window. The expert heatmap handles precise in-window localization, while the large-window heatmap handles more distant, out-of-image keypoints. When the large window determines that a keypoint is within the small window, it is refined by the expert map.
    - **Design Motivation**: Crop augmentation acts like a Hide-and-Seek information-discarding strategy. It not only provides training data for presence probability but also improves localization accuracy near image boundaries (by ~+1% mAP). Dual heatmaps act as a trade-off between the field of view and accuracy.

### Loss & Training

The probability map utilizes a modified OKS loss (expected risk minimization + Sobel gradient regularization), and the presence probability uses binary cross-entropy loss. The probability maps and presence probabilities are post-hoc calibrated via temperature scaling on CropCOCO. All training is performed on COCO, and crop augmentation is enabled in specified experiments.

## Key Experimental Results

### Main Results

| Model | COCO mAP | CropCOCO mAP | CropCOCO Ex-mAP | OCHuman mAP |
|---|---|---|---|---|
| ViTPose-s | 75.9 | 72.7 | 66.5 | 60.3 |
| HRFormer-s | 75.2 | 70.9 | 64.3 | 60.3 |
| **ProbPose-s** | **76.6** | **81.7** | **73.9** | 60.4 |
| ProbPose-s-DH | 76.2 | 80.9 | 71.4 | **61.4** |

ProbPose achieves a massive improvement on CropCOCO (with mAP rising from 72.7 to 81.7, +9%) while also showing a minor improvement on standard COCO (75.9 $\rightarrow$ 76.6).

### Ablation Study

| Configuration | COCO mAP | CropCOCO mAP |
|---|---|---|
| ViTPose-s Baseline | 75.9 | 72.7 |
| + Crop Augmentation | ~76.5 | ~79 |
| + Probability Map | ~76 | ~80 |
| + Presence Probability (ProbPose) | 76.6 | 81.7 |

Each component brings gains, with crop augmentation yielding the most significant effect and presence probability providing additional improvements in Ex-mAP.

### Key Findings

- The presence probability reduces the error of presence determination on CropCOCO by 30% (and by 45% on balanced datasets) compared to using the peak value of heatmaps.
- The commonly used confidence threshold of 0.3 is close to optimal, but the optimal threshold varies significantly across different datasets (0.15-0.4).
- The dual heatmap method brings improvements on OCHuman (multi-person occlusion), where the expanded field of view helps distinguish occluded individuals.
- The calibration curve of the probability map is approximately diagonal, indicating that the calibrated probabilities truly reflect real localization uncertainty.
- COCO annotations themselves exhibit bias near boundaries—annotators tend to avoid placing keypoints exactly on the image edges.

## Highlights & Insights

1. **Conceptual Clarity**: Decomposes the three semantic concepts conflated by heatmaps (location, quality, and presence) into independent outputs, each with an explicit probabilistic interpretation.
2. **Redefining Out-of-Image Keypoints**: Views image boundaries as a special form of occlusion, providing an insightful perspective.
3. **Expected OKS Maximization**: More robust than simple argmax, especially under bimodal distributions where it is less likely to be misled by sharp, local peaks.
4. **High Practicality**: Calibrated probabilities are crucial for safety-critical applications (such as human-robot interaction), enabling the model to express "I am uncertain".

## Limitations & Future Work

- Experiments are only conducted at the ViT-s scale; whether the gains hold on larger models remains unverified.
- CropCOCO is constructed via synthetic cropping, which may differ from the distribution of out-of-image keypoints in real-world scenarios.
- The precision-field of view trade-off in the dual heatmap approach leads to a slight decline on COCO.
- Future work could extend probabilistic modeling to 3D pose estimation and multi-person scenarios.

## Related Work & Insights

- Relation to ViTPose: ProbPose is based on the ViTPose architecture, achieving improvements by modifying the output representation and loss functions.
- Relation to RLE: RLE employs regression methods to localize out-of-image points, whereas ProbPose retains the heatmap paradigm but incorporates presence probability.
- OKSLoss was originally proposed in prior work for keypoint prediction; this paper extends it to every pixel of the probability map.
- Insight: The limitations of evaluation metrics (neglecting out-of-image points) actually guide the models to optimize in incorrect directions.

## Rating

- **Novelty**: 8/10 — Probability maps, presence probability, and expected OKS maximization are all novel designs supported by theoretical formulation.
- **Experimental Thoroughness**: 8/10 — Evaluated across multiple datasets, with detailed ablation studies and the construction of a new benchmark.
- **Writing Quality**: 8/10 — Clear problem analysis and rigorous mathematical formulation of the probabilistic framework.
- **Value**: 7/10 — Provides clear, practical improvements for pose estimation, though its scope of impact within the field is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VOccl3D: A Video Benchmark Dataset for 3D Human Pose and Shape Estimation under Real Occlusions](../../ICCV2025/object_detection/voccl3d_a_video_benchmark_dataset_for_3d_human_pose_and_shape_estimation_under_r.md)
- [\[CVPR 2025\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching_human_videos.md)
- [\[ICCV 2025\] 3D-MOOD: Lifting 2D to 3D for Monocular Open-Set Object Detection](../../ICCV2025/object_detection/3dmood_lifting_2d_to_3d_for_monocular_openset_object_detecti.md)
- [\[CVPR 2025\] Boosting Domain Incremental Learning: Selecting the Optimal Parameters Is All You Need](boosting_domain_incremental_learning_selecting_the_optimal_parameters_is_all_you.md)
- [\[CVPR 2025\] Test-Time Backdoor Detection for Object Detection Models](test-time_backdoor_detection_for_object_detection_models.md)

</div>

<!-- RELATED:END -->
