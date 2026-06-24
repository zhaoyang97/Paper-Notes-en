---
title: >-
  [Paper Note] ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting
description: >-
  [ECCV 2024][class-agnostic counting] Proposes ABC123, the first exemplar-free method capable of concurrently counting multiple classes of unknown objects in an image. By combining a ViT for multi-channel density map regression, Hungarian-matching-based training, and a SAM-based example discovery mechanism, this approach significantly outperforms exemplar-based methods on the self-created synthetic MCAC dataset while demonstrating strong generalization on the real-world FSC-14…
tags:
  - "ECCV 2024"
  - "class-agnostic counting"
  - "exemplar-free"
  - "multi-class"
  - "density map regression"
  - "Hungarian matching"
date: 2026-05-08
content_hash: 323fd337ef1f5144
---

# ABC Easy as 123: A Blind Counter for Exemplar-Free Multi-Class Class-Agnostic Counting

**Conference**: ECCV 2024  
**arXiv**: [2309.04820](https://arxiv.org/abs/2309.04820)  
**Code**: [https://ABC123.active.vision](https://ABC123.active.vision)  
**Area**: Object Counting / Class-Agnostic Counting  
**Keywords**: class-agnostic counting, exemplar-free, multi-class, density map regression, Hungarian matching  

## TL;DR
Proposes ABC123, the first exemplar-free method capable of concurrently counting multiple classes of unknown objects in an image. By combining a ViT for multi-channel density map regression, Hungarian-matching-based training, and a SAM-based example discovery mechanism, this approach significantly outperforms exemplar-based methods on the self-created synthetic MCAC dataset while demonstrating strong generalization on the real-world FSC-147 dataset.

## Background & Motivation
Class-agnostic counting aims to count objects of arbitrary categories without retraining for specific classes. Existing methods suffer from two major limitations: (1) most methods require users to provide exemplar images of the target categories to define "what to count", which increases manual intervention; (2) even recent zero-shot methods (such as RCC or the zero-shot mode of CounTR) can only handle scenarios with a single object class in the image. When multiple object classes coexist, the performance of these methods drops drastically or they fail to function. The root cause is the lack of a multi-class counting dataset for training and evaluation.

## Core Problem
How to automatically identify and separately count multiple co-existing unknown object classes in an image without any exemplar images? The key challenges lie in: (1) there is no external guidance for "what to count", requiring the model to autonomously determine how to group objects; (2) there is no pre-defined correspondence between the predicted multiple counts and the ground-truth annotations, necessitating a matching mechanism; (3) object categories do not overlap between training and testing, which demands strong generalization capability.

## Method
The core mechanism of ABC123 is "count blindly first, explain later"—it directly regresses the density maps and count values for each object category via a network without relying on any exemplars, and then identifies the corresponding physical objects for each count through an example discovery stage to help users interpret the results.

### Overall Architecture
Given an input image, a ViT-Small backbone extracts global patch-wise features. Then, $\hat{m}$ parallel convolutional upsampling heads (each consisting of 3 Conv-ReLU-Upsample layers) upsample the low-resolution features into $\hat{m}$ pixel-level density maps ($224 \times 224$). Integrating each density map yields a count value. During training, Hungarian matching pairs predictions with ground truths to compute losses. During inference, similar predictions are merged and zero counts are removed for output. An optional example discovery phase adopts SAM to localize and crop exemplar patches of the counted objects to help users understand "what was counted".

### Key Designs
1. **Multi-Head Density Map Regression**: Setting $\hat{m}=5$ parallel output heads (exceeding the maximum of 4 classes in MCAC), where each head independently regresses a density map. The redundant heads allow the network to discover "valid-but-unknown counts" (e.g., partitioning a single semantic class into sub-classes). No loss is applied to unmatched heads, allowing the network to freely explore multi-granular grouping methods.

2. **Hungarian Matching Training Strategy**: Since there is no pre-defined pairing between predictions and ground truths under the multi-class exemplar-free setting, the authors model this as a bipartite matching problem. The cost matrix is defined as the $L_2$ distance between normalized density maps (normalization eliminates the scales of counts, forcing matching to be based on spatial distributions rather than count magnitudes), which is solved for the optimal matching using the Jonker-Volgenant algorithm. This draws inspiration from matching concepts in Novel Class Discovery (NCD) and DETR.

3. **Example Discovery**: This reverses the traditional paradigm of "using exemplars to guide counting" to "counting blindly first, then finding exemplars". Specifically, it identifies points with high activations in one density map but low activations in other density maps. Multiple points with maximum feature-space distance are selected to increase diversity. These points are then fed as prompts into a pre-trained SAM for segmentation, and the segmented regions are cropped and displayed to the user.

4. **MCAC Synthetic Dataset**: The first multi-class class-agnostic counting dataset, rendered with Blender based on ShapeNetSem. It features 1–4 classes per image and 1–300 instances per class, with 287 classes for training, 37 for validation, and 19 for testing (disjoint classes). It provides instance labels, class labels, bounding boxes, center points, and occlusion percentages. Synthetic data avoids the mutual exclusion constraints of classes that are hard to satisfy in natural images.

### Loss & Training
- **Loss Function**: The sum of $L_1$ losses for all matched paired density maps: $\mathcal{L} = \sum_{i,j} \|d_i - \hat{d}_j\|_1 \cdot \mathcal{X}_{i,j}$, where unmatched heads do not contribute to the loss.
- **Backbone Initialization**: DINO self-supervised pre-trained ViT-Small (21M parameters) to minimize the risk of overfitting on small datasets.
- **Training Configuration**: Adam optimizer, batch-size = 2, learning rate of $3 \times 10^{-5}$ halved every 35 epochs, 100 epochs in total; takes less than 8 hours on two 1080Ti GPUs.
- **Input Resolution**: $224 \times 224$ (constrained by ViT-S, lower than the 384+ used in compared methods).
- **Density Map Generation**: Standard pseudo-density maps (applying Gaussian kernels at object centers) are used to ensure a fair comparison with baseline methods.

## Key Experimental Results

| Dataset | Metric | ABC123 | LOCA (3-shot) | CounTR (3-shot) | BMNet+ (3-shot) |
|--------|------|--------|---------------|-----------------|-----------------|
| MCAC Val | MAE | **8.96** | 10.45 | 15.07 | 15.83 |
| MCAC Val | RMSE | **15.93** | 20.81 | 26.26 | 27.07 |
| MCAC Test | MAE | **9.52** | 10.91 | 16.12 | 17.29 |
| MCAC Test | NAE | **0.28** | 0.37 | 0.67 | 0.75 |
| FSC-147 Val | MAE | **11.13** | 25.70 | 21.22 | 29.47 |
| FSC-147 Test | MAE | **11.75** | 29.93 | 21.09 | 30.74 |

Note: On FSC-147, ABC123 uses the sum method to merge subclass density maps and excludes images with counts $> 300$ (3% of Val, 1.1% of Test).

### Ablation Study
- **Impact of the Number of Heads**: Increasing from 4 to 100 heads yields a continuous drop in MAE (from 9.43 with 4 heads to 7.11 with 100 heads), but head utilization decreases from 100% to 39%, indicating that redundant heads are rarely matched during training. The authors select $\hat{m}=5$ as the balance point to avoid "unfair" quantitative advantages stemming from the matching phase.
- **Frozen Backbone vs. Full Fine-tuning**: ABC123❄ (frozen backbone) obtains an MAE of 14.64 vs. full fine-tuning ABC123 MAE of 8.96, showing that end-to-end fine-tuning of the ViT backbone significantly improves performance.
- **Single-Head vs. Multi-Head**: A single head ($\hat{m}=1$) behaves similarly to 5 heads on MCAC-M1, demonstrating that the matching mechanism itself does not introduce an unfair quantitative advantage.
- **Training on MCAC vs. MCAC-M1**: ABC123 performs similarly under both training schemes, suggesting robust handling of inter-class/intra-class variations without mistakenly merging different categories.

## Highlights & Insights
- **Paradigm Shift**: The idea of "counting blindly first and finding exemplars later" is highly ingenious—shifting the exemplar from the input end to the output end completely eliminates the dependency on human-provided exemplars while preserving interpretability (showing what was counted).
- **Matching-based Training Strategy**: Elegantly formulates the multi-class unlabeled pairing counting problem as a bipartite matching problem and tolerates valid-but-unknown counts, a design that is both practical and theoretically grounded.
- **Synthetic-to-Real Generalization**: Trained solely on the synthetic MCAC dataset and directly transferred to FSC-147, outperforming all compared methods, which verifies the effectiveness of synthetic data for counting tasks.
- **Lightweight & Efficient**: Uses ViT-Small with 21M parameters, trainable on two 1080Ti GPUs, and simultaneously infers counts for all categories in a single pass (instead of class-by-class sequential invocations).

## Limitations & Future Work
- **Semantic Gap between Synthetic and Real**: MCAC defines classes by "identical mesh + texture", whereas humans group by high-level genetics in real-world scenarios (e.g., chairs of different colors are still counted as the same class). This leads to "over-segmentation" on FSC-147 (splitting a single category into multiple sub-classes), requiring post-processing to merge them.
- **Limited Input Resolution**: An input resolution of $224 \times 224$ might lose fine-grained details in dense small-object and high-count scenarios, leading to the exclusion of images with counts $> 300$ during evaluation.
- **Fixed Maximum Number of Classes**: The number of heads $\hat{m}$ is hard-coded to 5, failing to adapt to scenarios with more co-existing classes.
- **Dependency on SAM for Example Discovery**: The example discovery phase requires inference on an external SAM model, increasing computational overhead.
- **Potential Directions**: (1) Use larger backbones or higher resolutions to adapt to high-density scenarios; (2) Replace pure blind counting with natural language guidance to reduce ambiguity; (3) Extend the matching scheme to open-world detection or segmentation.

## Related Work & Insights
- **vs. RCC (CVPR 2023)**: Both are exemplar-free, but RCC outputs only a single scalar count (suitable only for single-class settings) and cannot handle multi-class scenarios. ABC123 overcomes this fundamental limit via multi-head regression and matching.
- **vs. CounTR**: CounTR is primarily a few-shot method whose performance drops in the zero-shot setting, and it outputs only one count at a time. ABC123 handles multiple classes simultaneously under zero-shot conditions without requiring exemplars.
- **vs. LOCA**: LOCA achieves an MAE of 10.91 on 3-shot MCAC, whereas ABC123 achieves 9.52 under the zero-shot setting. LOCA also relies on exemplars to define categories, requiring multiple independent invocations for multi-class scenarios, which is inefficient. ABC123 returns counts for all classes in a single forward pass.

## Inspirations & Connections
- The Hungarian matching training strategy shares similarities with DETR (optimally matching predictions with an unordered set of ground truths); this idea can be transferred to other tasks handling "unordered prediction sets".
- The training paradigm of "predict first, align with matching, and omit penalties on redundant predictions" is highly valuable for open-world perception tasks.
- The paradigm of synthetic data combined with disjoint class training/testing validates its feasibility in counting tasks, and could potentially be applied to data construction for other low-level vision tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to propose a complete pipeline for multi-class exemplar-free counting with a creative paradigm shift, though the core technologies (ViT, density maps, Hungarian matching) are not individually novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Verified on both MCAC and FSC-147 datasets with ablation studies covering the number of heads, backbone, and training sets, but lacks evaluations on real-world multi-class scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, smooth method description, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Initiates a new track in multi-class exemplar-free counting with MCAC as a potentially long-lasting dataset; however, the practical deployment value is somewhat limited by the synthetic-to-real gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Rebalancing Using Estimated Class Distribution for Imbalanced Semi-Supervised Learning under Class Distribution Mismatch](rebalancing_using_estimated_class_distribution_for_imbalanced_semi-supervised_le.md)
- [\[ECCV 2024\] Rethinking Data Bias: Dataset Copyright Protection via Embedding Class-Wise Hidden Bias](rethinking_data_bias_dataset_copyright_protection_via_embedding_class-wise_hidde.md)
- [\[CVPR 2026\] Drainage: A Unifying Framework for Addressing Class Uncertainty](../../CVPR2026/others/drainage_a_unifying_framework_for_addressing_class_uncertainty.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](../../ICCV2025/others/intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ECCV 2024\] Improving Point-based Crowd Counting and Localization Based on Auxiliary Point Guidance](improving_point-based_crowd_counting_and_localization_based_on_auxiliary_point_g.md)

</div>

<!-- RELATED:END -->
