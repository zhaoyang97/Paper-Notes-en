---
title: >-
  [Paper Note] SOS: Segment Object System for Open-World Instance Segmentation With Object Priors
description: >-
  [ECCV 2024][Segmentation][open-world instance segmentation] The SOS method is proposed to generate object-focused SAM prompt points using DINO self-attention maps as object priors, thereby producing high-quality pseudo annotations to train standard instance segmentation systems. It significantly outperforms prior SOTA methods under COCO/LVIS/ADE20k cross-category/cross-dataset settings, achieving precision improvements of up to 81.6%.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "open-world instance segmentation"
  - "pseudo annotation"
  - "SAM prompting"
  - "object prior"
  - "DINO self-attention"
date: 2026-05-08
content_hash: f8680b9b7091c9cd
---

# SOS: Segment Object System for Open-World Instance Segmentation With Object Priors

**Conference**: ECCV 2024  
**arXiv**: [2409.14627](https://arxiv.org/abs/2409.14627)  
**Code**: [https://github.com/chwilms/SOS](https://github.com/chwilms/SOS)  
**Area**: Segmentation  
**Keywords**: open-world instance segmentation, pseudo annotation, SAM prompting, object prior, DINO self-attention

## TL;DR

The SOS method is proposed to generate object-focused SAM prompt points using DINO self-attention maps as object priors, thereby producing high-quality pseudo annotations to train standard instance segmentation systems. It significantly outperforms prior SOTA methods under COCO/LVIS/ADE20k cross-category/cross-dataset settings, achieving precision improvements of up to 81.6%.

## Background & Motivation

**Background**: Open-World Instance Segmentation (OWIS) requires a system to segment arbitrary unknown object instances in images based on learning from only a small set of known categories. This task breaks the closed-world assumption of standard instance segmentation and holds significant value in real-world scenarios—for example, detecting surfboards or tennis rackets during testing even if they were unseen during training.

**Limitations of Prior Work**: Existing OWIS methods mainly follow two paths:

1. **Substituting classification with localization scores** (e.g., OLN, SWORD, OpenInst): Replacing the foreground/background binary classification with localization quality scores like IoU to avoid categorizing unknown objects as background. However, these methods generally suffer from low precision.

2. **Pseudo-annotation augmentation** (e.g., GGN, UDOS, LDET): Generating pseudo annotations for unlabeled objects to expand the training set. However, existing pseudo-annotation methods produce **highly noisy** annotations, often including large background regions, leading to low precision.

**Key Challenge**: There is a fundamental trade-off between improving recall (detecting more unknown objects) and maintaining precision (not misclassifying backgrounds as objects). Existing methods generally improve recall but suffer from very low precision. Furthermore, simply applying "segment anything" via SAM does not work well, as SAM segments both object and non-object regions (such as stuff regions like sky and ground).

**Key Insight**: The key lies in forcing SAM to focus only on objects rather than arbitrary regions. By systematically studying various "object priors" to generate object-focused prompt points, SAM can be guided to produce only high-quality object segmentations, fundamentally improving pseudo-annotation quality.

**Core Idea**: Using the self-attention maps of a self-supervised ViT (DINO) as object location priors to sample prompt points, shifting SAM from "Segment Anything" to "Segment Objects," and then training a standard instance segmentation system with the generated high-quality pseudo annotations.

## Method

### Overall Architecture

SOS consists of three cascaded modules, forming a pipeline of "prior localization → pseudo-annotation generation → instance segmentation training":
1. **OLM (Object Localization Module)**: Samples object-focused point prompts from object priors.
2. **PAC (Pseudo Annotation Creator)**: Uses the sampled point prompts to guide SAM in generating segmentations, followed by post-processing to obtain pseudo annotations.
3. **Instance Segmentation Training**: Merges pseudo annotations with ground-truth annotations to train a standard Mask R-CNN.

During inference, only the trained Mask R-CNN is run, requiring neither DINO nor SAM.

### Key Designs

#### 1. Object Localization Module (OLM)

- **Function**: Roughly localizes all objects from the input image and outputs a set of object-focused point coordinates as prompts for SAM.
- **Mechanism**:
    - The attention maps from 6 self-attention heads in the last layer of DINO (self-supervised ViT-S) are aggregated into a single scene layout map by taking the element-wise maximum.
    - This map is normalized into a probability mass function (minimum set to zero, sum normalized to one) to serve as the object prior $P(x,y)$.
    - $S=50$ point coordinates are randomly sampled from the object prior. After sampling each point, the prior values within a surrounding $N=20$ pixel radius are zeroed out and re-normalized to ensure that the sampled points are widely distributed across different objects.
- **Design Motivation**: The paper systematically compares 8 types of object priors (grid / superpixel / contour density / VOCUS2 saliency / DeepGaze / CAM / DINO / U-Net) and finds that the DINO self-attention map is the optimal choice. It precisely focuses on the discriminative regions of objects (such as a polar bear's face), unlike VOCUS2 or U-Net which also highlight background areas. DINO improves the F1 score by 7.5 points compared to the Grid baseline.

#### 2. Pseudo Annotation Creator (PAC)

- **Function**: Drives the pre-trained SAM with the point prompts output by OLM to generate candidate segmentations, which are filtered and merged with the ground-truth annotations.
- **Mechanism**:
    - For each point prompt, SAM generates 3 candidate segmentations (to handle ambiguity).
    - **Confidence Filtering**: Segmentations with a SAM confidence score lower than $\tau_{conf}=0.9$ are removed.
    - **NMS Deduplication**: Redundant segmentations are removed using an IoU threshold of $\tau_{NMS}=0.95$.
    - **Merging with Ground Truth**: If a pseudo annotation has an IoU greater than $\tau_{NMS}$ with any ground-truth annotation, it is suppressed (keeping only pseudo annotations that cover unknown objects).
    - **Quantity Constraint**: A maximum of $P=10$ pseudo annotations are kept per image (with an actual average of 7.8).
- **Design Motivation**: Driving SAM solely with point prompts still yields noisy segmentations (object parts, stuff regions, etc.), necessitating strict post-processing. A high threshold of $\tau_{conf}=0.9$ ensures that only segmentations with high SAM confidence are retained, and $\tau_{NMS}=0.95$ eliminates almost completely overlapping redundant segmentations. Compared to GGN, which only uses 3 pseudo annotations, SOS can handle up to 10 without introducing excessive noise, precisely due to the higher quality of its pseudo annotations.

#### 3. Instance Segmentation Training

- **Function**: Trains a class-agnostic Mask R-CNN (ResNet-50 + FPN) using the merged annotations (ground truth + high-quality pseudo annotations).
- **Design Motivation**: Selecting a standard instance segmentation system is intended to (a) provide maximum flexibility, as the system can be replaced with any instance segmentation network; and (b) verify that performance gains stem from pseudo-annotation quality rather than model architectural improvements. Class-agnostic training does not assume a fixed number of object categories, which is naturally suited for open-world scenarios.

### Loss & Training

- Uses the default training configuration of Mask R-CNN (in class-agnostic mode).
- DINO (ViT-S) is pre-trained via self-supervision on ImageNet without fine-tuning.
- SAM (ViT-H) is pre-trained on SA-1B without fine-tuning.
- Pseudo annotations are only required during training; during inference, only Mask R-CNN is run.
- No additional training tricks or custom losses are introduced.

## Key Experimental Results

### Main Results

**COCO (VOC) → COCO (non-VOC) Cross-Category Evaluation**:

| Method | AP | AR100 | F1 |
|------|-----|-------|-----|
| Mask R-CNN | 1.0 | 8.2 | 1.8 |
| SAM (Grid Prompt) | 3.6 | 48.1 | 6.7 |
| OLN | 4.2 | 28.4 | 7.3 |
| GGN | 4.9 | 28.3 | 8.4 |
| SWORD | 4.8 | 30.2 | 8.3 |
| **SOS (Ours)** | **8.9** | **39.3** | **14.5** |

vs GGN (second place): F1 **+6.1**, Precision **+81.6%** (4.9 $\rightarrow$ 8.9), Recall **+38.9%** (28.3 $\rightarrow$ 39.3).

**Cross-Dataset Evaluation**:

| Setting | SOS F1 | Second Place F1 | Gain |
|------|--------|----------|------|
| COCO → LVIS | **13.3** | GGN/LDET 10.5 | +2.8 |
| COCO → ADE20k | **17.0** | GGN 13.3 | +3.7 |
| COCO → UVO | 28.0 | LDET **28.5** | -0.5 |

Slightly lower than LDET on UVO, because UVO contains object categories outside of ImageNet, and DINO is trained on ImageNet, which might lead to missing these categories.

### Ablation Study

**Comparison of Object Priors** (on COCO cross-category setting):

| Object Prior | AP | AR100 | F1 | Description |
|---------|-----|-------|-----|------|
| Grid (SAM Default) | 3.8 | 36.5 | 6.9 | Contains a large amount of stuff |
| Dist (Training set distribution) | 3.4 | 27.4 | 6.0 | Ignores image content |
| Spx (Superpixels) | 5.6 | 34.8 | 9.6 | Simple and effective |
| Contour (Contour density) | 5.6 | 36.6 | 9.7 | |
| VOCUS2 (Saliency) | 6.1 | 37.7 | 10.5 | Highlights high-contrast background regions |
| CAM | 5.4 | 36.7 | 9.4 | Class Activation Map |
| DeepGaze | 5.4 | 35.9 | 9.4 | Eye-tracking data |
| U-Net | 7.3 | 37.3 | 12.2 | Learns pixel-level object locations |
| **DINO** | **8.9** | **38.1** | **14.4** | Self-attention scene layout |
| GT (Upper Bound) | 18.1 | 42.5 | 25.4 | Uses GT object centers |

**Component Ablation**:

| Configuration | AP | AR100 | F1 |
|------|-----|-------|-----|
| Mask R-CNN (No pseudo annotations) | 1.2 | 10.8 | 2.2 |
| + Grid pseudo annotations (No post-processing) | 3.4 | 35.2 | 6.2 |
| + DINO pseudo annotations (No post-processing) | 8.9 | 37.1 | 14.3 |
| + DINO + Post-processing (Full SOS) | **8.9** | **38.1** | **14.4** |

**Number of Pseudo Annotations**:

| Quantity | AP | AR100 | F1 |
|------|-----|-------|-----|
| 3 | 8.3 | 34.5 | 13.4 |
| 5 | 8.8 | 36.2 | 14.2 |
| **10** | **8.9** | **38.1** | **14.4** |
| 20 | 8.8 | 38.1 | 14.3 |

### Quantization of Pseudo-Annotation Quality

Evaluating the capability of pseudo annotations to cover non-VOC category objects on the COCO training set:

| Method | Precision | Recall | F1 |
|------|------|------|-----|
| GGN (3 annotations) | 7.3 | 12.1 | 9.1 |
| SOS (3 annotations) | **19.0** | 26.4 | **22.1** |
| SOS (10 annotations) | 15.5 | **41.7** | 22.6 |

The pseudo-annotation precision of SOS is **2.6 times** that of GGN, demonstrating the effectiveness of DINO in focusing on objects.

### Key Findings

- **The choice of object prior is core**: Compared to the Grid baseline, DINO doubles the F1 score (14.4 vs 6.9), primarily achieved through a substantial increase in precision (8.9 vs 3.8).
- **Precision improvement is the biggest highlight**: An 81.6% precision improvement compared to GGN, breaking the "high recall, low precision" bottleneck in the OWIS field.
- **The combination of SAM + DINO is complementary**: DINO knows where the objects are (localization), while SAM knows what the objects look like (segmentation). Their combination produces high-quality pseudo annotations.
- **10 pseudo annotations is optimal**: This is more than GGN's 3, but because of the high quality, it does not introduce excessive noise.

## Highlights & Insights

- **The Paradigm of "Object Prior + Foundation Model Prompting"**: Offers a general methodology on how to transform a general foundation model (SAM) into a task-specific tool. Designing a reasonable prompting strategy (instead of fine-tuning) maximizes the capabilities of the foundation model. This concept can be transferred to other downstream applications of SAM (such as weakly-supervised segmentation, zero-shot detection, etc.).
- **Systematic Study on Object Priors**: Comparing 8 types of object priors is itself a significant contribution, providing experimental guidance for any task that requires class-agnostic object localization.
- **High-Quality Pseudo Annotations > Quantity**: SOS using 10 high-quality pseudo annotations far outperforms GGN using 3 low-quality ones, illustrating the "quality first" principle of pseudo-labeling.
- **No Extra Inference Overhead**: DINO and SAM are only used to generate training pseudo annotations. Inference directly runs Mask R-CNN, introducing no extra deployment costs.

## Limitations & Future Work

1. **DINO relies on ImageNet**: DINO is pre-trained on ImageNet and may miss object classes outside of ImageNet (e.g., unique objects in UVO), leading to SOS performing slightly lower than LDET on COCO$\rightarrow$UVO. This can be mitigated by training DINO on more diverse unlabeled data.
2. **Pseudo annotations still contain noise**: Even the highest precision is only 19% (under the IoU=0.5 standard), showing that most pseudo annotations are still imperfect. Future work could introduce stronger filtering mechanisms or iterative refinement.
3. **Fixed prompting strategy**: OLM uses a fixed sampling strategy (50 points, with a exclusion zone of $N=20$ pixels) and has not explored adaptive strategies.
4. **Evaluated only on Mask R-CNN**: Although claiming compatibility with any instance segmentation system, the performance on other architectures (e.g., query-based methods) was not actually validated.
5. **Category information is unutilized**: SOS is completely class-agnostic and does not explore utilizing the semantic information of known classes to further enhance the detection of unknown classes.

## Related Work & Insights

- **vs GGN**: GGN generates pseudo annotations by learning pixel affinity, but the results are highly noisy (containing many background segments). SOS utilizes DINO+SAM to generate object-focused high-quality pseudo annotations, achieving a precision 2.6 times that of GGN.
- **vs OLN/SWORD**: These methods improve generalization by modifying the classification head without adding training data. SOS solves the problem from the data side by adding high-quality pseudo annotations, making these two approaches potentially complementary.
- **vs Direct Use of SAM**: The F1 score of vanilla SAM with Grid prompts is only 6.7 (precision 3.6), because it does not distinguish between objects and stuff. SOS improves SAM's precision from 3.6 to 8.9 through DINO priors, validating the importance of "prompt design" over "fine-tuning".

## Rating

- Novelty: ⭐⭐⭐⭐ The combined approach of object priors + SAM prompting is novel, and the systematic study of object priors has pioneering significance.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 evaluation settings, comparison of 8 types of priors, component ablations, and pseudo-annotation quality quantification, making it extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, with the study on object priors being exceptionally well-presented, backed up by thorough quantitative and qualitative analysis.
- Value: ⭐⭐⭐⭐ Provides a general paradigm for leveraging foundation models to solve open-world tasks, and the dramatic boost in precision resolves a key bottleneck in this domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] V-CLR: View-Consistent Learning for Open-World Instance Segmentation](../../CVPR2025/segmentation/v-clr_view-consistent_learning_for_open-world_instance_segmentation.md)
- [\[ECCV 2024\] ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)
- [\[ECCV 2024\] Unsupervised Moving Object Segmentation with Atmospheric Turbulence](unsupervised_moving_object_segmentation_with_atmospheric_turbulence.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ECCV 2024\] Frequency-Spatial Entanglement Learning for Camouflaged Object Detection](frequency-spatial_entanglement_learning_for_camouflaged_object_detection.md)

</div>

<!-- RELATED:END -->
