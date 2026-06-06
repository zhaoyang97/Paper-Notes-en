---
title: >-
  [Paper Note] NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization
description: >-
  [CVPR 2026][Medical Imaging][Neuron segmentation] NeurINO proposes to initialize a 3D neuron segmentation model by inflating DINOv3 pretrained 2D convolutional kernels into 3D operators…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Neuron segmentation"
  - "DINOv3"
  - "2D-to-3D transfer"
  - "topology-aware loss"
  - "data-efficient learning"
date: 2026-05-08
content_hash: ae9c0eb363e6f91d
---

# NeuroSeg Meets DINOv3: Transferring 2D Self-Supervised Visual Priors to 3D Neuron Segmentation via DINOv3 Initialization

**Conference**: CVPR 2026
**arXiv**: [2603.23104](https://arxiv.org/abs/2603.23104)  
**Code**: [https://github.com/yy0007/NeurINO](https://github.com/yy0007/NeurINO)  
**Area**: Medical Imaging / 3D Segmentation
**Keywords**: Neuron segmentation, DINOv3, 2D-to-3D transfer, topology-aware loss, data-efficient learning

## TL;DR

NeurINO proposes to initialize a 3D neuron segmentation model by inflating DINOv3 pretrained 2D convolutional kernels into 3D operators, while introducing a Topology-Aware Skeleton Loss (TASL) to explicitly supervise skeleton-level structural fidelity. The method achieves average improvements of 2.9% in ESA, 2.8% in DSA, and 3.8% in PDS across four neuroimaging datasets.

## Background & Motivation

1. **Background**: Accurate reconstruction of neuronal morphology from volumetric optical microscopy images is critical for brain connectomics, requiring the tracking of axons and dendrites over long-range projections across multiple brain regions. Deep learning methods such as 3D U-Net and V-Net have substantially improved segmentation quality.

2. **Limitations of Prior Work**:
    - **Data scarcity**: Acquiring and annotating 3D neuroimaging data is costly and laborious, limiting the performance of data-driven approaches;
    - **Lack of structural supervision**: Existing methods typically optimize voxel-level accuracy (Dice/CE loss) without explicit supervision of branching topology or connectivity integrity, leading to results that may perform well on voxel metrics yet be topologically unfaithful (broken or merged structures);
    - **Absence of 3D foundation models**: 2D visual foundation models (DINO, SAM, etc.) excel on natural images, but no analogous general-purpose foundation model exists for 3D biomedical volumetric data.

3. **Key Challenge**: 2D foundation models possess rich semantic priors but cannot be directly applied to 3D volumetric data, while the scarcity of 3D data precludes training strong feature representations from scratch.

4. **Goal**: To efficiently transfer the visual priors of 2D foundation models to 3D neuron segmentation while ensuring morphological topological fidelity.

5. **Key Insight**: Decompose 3D volumetric learning into intra-slice feature extraction (exploiting DINOv3 priors) and inter-slice aggregation (requiring additional learning), while employing a topology-aware loss to guide cross-slice structural learning.

6. **Core Idea**: Transfer DINOv3's 2D weights to a 3D encoder via an inflation strategy, allowing the model to focus solely on learning inter-slice correlations, and enforce morphological fidelity through a skeleton-level topological loss.

## Method

### Overall Architecture

A 3D neuroimaging patch is fed into a 3D encoder initialized by inflating DINOv3 ConvNeXt weights, which extracts multi-scale features. A symmetric MedNeXt-style decoder then recovers fine-grained neuronal morphology. Multi-level outputs are jointly supervised by Dice + CE + TASL with deep supervision.

### Key Designs

1. **Inflation-Based Adaptation**:

    - **Function**: Losslessly elevates DINOv3 pretrained 2D convolutional kernels into 3D convolutional kernels.
    - **Mechanism**: Given a 2D convolutional kernel $W_{2D} \in \mathbb{R}^{C_{out} \times C_{in} \times k_h \times k_w}$, it is inflated to $W_{3D} \in \mathbb{R}^{C_{out} \times C_{in} \times k_d \times k_h \times k_w}$. Two schemes are proposed: **center inflation** places the 2D kernel at the central slice of the 3D kernel ($W_{3D}[:,:,c,:,:] = W_{2D}$, all other slices set to zero); **average inflation** uniformly copies the 2D kernel to all depth slices and divides by $k_d$. Experiments verify that center inflation is superior, as it preserves the spatial alignment between the 2D receptive field and its 3D extension.
    - **Design Motivation**: Training 3D models from scratch under data-scarce conditions yields poor results, while freezing the pretrained encoder prevents adaptation to the neuronal data distribution. The inflation strategy allows the model to inherit DINOv3's intra-slice semantic priors (edges, textures, spatial patterns) and focus learning exclusively on inter-slice structural continuity.

2. **Topology-Aware Skeleton Loss (TASL)**:

    - **Function**: Explicitly penalizes morphological discontinuities and topological errors at the skeleton-graph level.
    - **Mechanism**: Binary segmentations of the prediction and ground truth are skeletonized via operator $\mathcal{S}(\cdot)$ and converted to graphs $G=(V,E)$. TASL consists of three complementary terms: (a) **node-level discrepancy** $L_{node}$—symmetric nearest-neighbor distance between predicted and ground-truth skeleton node sets, penalizing bifurcation misalignment and missing endpoints; (b) **edge-level discrepancy** $L_{edge}$—compares the proportional difference in edge counts between predicted and ground-truth graphs, capturing over- and under-connectivity errors; (c) **path-level discrepancy** $L_{path}$—compares the difference in mean connected-component size, emphasizing long-range structural continuity.
    - **Design Motivation**: Standard voxel-level losses (Dice, CE) are topology-agnostic and cannot penalize breaks or merges. TASL serves as a regularizer that guides segmentation learning toward topologically consistent volumetric representations.

3. **Deep Supervision**:

    - **Function**: Improves gradient flow and refines multi-scale features.
    - **Mechanism**: Dice + CE + TASL are jointly applied at each output branch. The total loss is defined as $\mathcal{L}_{total} = \sum_s \lambda_s (1 + \beta \mathcal{L}_{TASL}^s)(\mathcal{L}_{Dice}^s + \mathcal{L}_{CE}^s)$. TASL modulates the standard losses multiplicatively, amplifying the learning signal at locations with large topological errors.
    - **Design Motivation**: The skeletonization operation in TASL is non-differentiable; propagating its signal indirectly as a multiplicative modulation factor is more stable than direct backpropagation.

### Loss & Training

- TASL composite loss: $\mathcal{L}_{TASL} = \lambda_{node}L_{node} + \lambda_{edge}L_{edge} + \lambda_{path}L_{path}$
- Total loss is jointly optimized across multiple scales via deep supervision.
- Training for 110 epochs with AdamW optimizer, learning rate 0.001, and AMP mixed-precision training.
- Sliding window inference strategy for large volumetric data.
- Decoder normalization layers replaced with Batch Renormalization to mitigate tiling artifacts.
- All experiments conducted on two NVIDIA RTX 4060 Ti GPUs (16 GB each).

## Key Experimental Results

### Main Results

| Method | Drosophila F1/HD95 | Mouse F1/HD95 | NeuroFly F1/HD95 | CWMBS F1/HD95 |
|------|-------------------|---------------|------------------|---------------|
| nnUNet | 47.20/3.20 | 52.05/10.12 | 63.36/18.33 | 36.50/16.34 |
| MedNeXt | 47.74/3.15 | 50.61/13.77 | 62.50/19.23 | 33.46/18.37 |
| NeurINO-T | **50.06/3.07** | 52.50/9.50 | **65.23/16.38** | **36.77/16.10** |
| NeurINO-S | 50.19/**3.02** | **52.73/9.24** | 65.44/16.53 | 36.55/16.27 |

Reconstruction metrics (Drosophila + SmartTracing):

| Method | ESA↓ | DSA↓ | PDS↓ |
|------|------|------|------|
| nnUNet | 1.67 | 4.48 | 0.20 |
| MedNeXt | 1.68 | 4.44 | 0.20 |
| NeurINO-T | **1.62** | **4.29** | **0.20** |

### Ablation Study

| Configuration | F1(%) | SmartTracing ESA/DSA/PDS |
|------|-------|--------------------------|
| Average inflation | 49.64 | 1.71/4.34/0.21 |
| Center inflation (default) | **50.06** | **1.62/4.29/0.20** |
| w/o TASL | 50.59 | 1.68/4.41/0.21 |
| with TASL | 50.06 | **1.62/4.29/0.20** |
| Frozen encoder | 47.22 | 1.78/4.49/0.23 |
| Training from scratch | 48.56 | 1.76/4.37/0.23 |
| Fine-tuning (default) | **50.06** | **1.62/4.29/0.20** |

### Key Findings

- **Center inflation outperforms average inflation**: Center inflation preserves the spatial anchoring of the 2D receptive field, whereas average inflation dilutes depth-wise semantics.
- **TASL trades a marginal F1 drop for substantial reconstruction gains**: Removing TASL yields a slightly higher F1 (50.59 vs. 50.06) but causes significant degradation in reconstruction metrics, indicating that TASL guides the network to prioritize global connectivity.
- **Full fine-tuning substantially outperforms both freezing and training from scratch**: DINOv3 priors provide critical intra-slice semantics, but fine-tuning is necessary to adapt to the target distribution.
- NeurINO-T frequently surpasses NeurINO-S on reconstruction metrics, suggesting that a smaller encoder combined with the inflation strategy may generalize better.

## Highlights & Insights

- **2D-to-3D transfer paradigm**: Decomposing volumetric learning into "intra-slice 2D" and "inter-slice 3D" components and leveraging the inflation strategy to obtain intra-slice priors at no additional cost is a transferable idea applicable to other 3D medical tasks (e.g., organ segmentation, vessel tracing).
- **The multiplicative modulation design of TASL** elegantly circumvents the non-differentiability of skeletonization—TASL does not serve as a direct gradient source but instead modulates the weights of Dice/CE losses.
- **The trade-off between F1 and topological fidelity** is an important insight: voxel-level optimality does not imply structural optimality, a finding broadly relevant to the segmentation of tubular and network-like structures.
- All experiments are completed on only two RTX 4060 Ti GPUs, imposing minimal hardware requirements.

## Limitations & Future Work

- The skeletonization operation in TASL is non-differentiable and can only serve as an indirect regularizer, limiting gradient propagation of topological information.
- The inflation strategy is a static initialization method and cannot dynamically adapt to the anisotropic resolution of different imaging modalities.
- Only the ConvNeXt architecture is evaluated; inflation-based transfer for ViT-style architectures remains unexplored.
- Dataset scale is relatively small (up to 245 volumes); performance on large-scale data has yet to be validated.
- Comparisons with recent 3D general segmentation methods such as SAM3D and UniverSeg are absent.

## Related Work & Insights

- **vs. nnUNet**: nnUNet is an adaptive general-purpose segmentation framework trained from scratch; NeurINO significantly outperforms it in data-scarce settings by transferring 2D priors (F1 improvement of 2–3%).
- **vs. MedNeXt**: MedNeXt has a larger parameter count (62M vs. 39M) yet achieves lower performance, demonstrating that larger models do not necessarily hold an advantage on small datasets.
- **vs. Skeleton Recall Loss**: Previous skeleton-based losses focus solely on centerline coverage, whereas TASL models node, edge, and path consistency at the graph level, providing finer-grained topological supervision.
- The inflation strategy is inspired by I3D but is simplified and adapted for the segmentation task.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Inflation-based transfer from DINOv3 to 3D is novel in the context of neuron segmentation; the three-level topological loss design in TASL is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, two tracing algorithms, and detailed ablations are provided, though comparisons with a broader range of 3D pretrained methods are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The paper is clearly structured with well-motivated arguments and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Provides a practical, low-cost solution for resource-constrained 3D biomedical segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Addressing Data Scarcity in 3D Trauma Detection through Self-Supervised and Semi-Supervised Learning with Vertex Relative Position Encoding](addressing_data_scarcity_in_3d_trauma_detection_through_self-supervised_and_semi.md)
- [\[CVPR 2026\] Learning Generalizable 3D Medical Image Representations from Mask-Guided Self-Supervision](learning_generalizable_3d_medical_image_representations_from_mask-guided_self-su.md)
- [\[ICCV 2025\] An OpenMind for 3D Medical Vision Self-supervised Learning](../../ICCV2025/medical_imaging/an_openmind_for_3d_medical_vision_selfsupervised_learning.md)
- [\[AAAI 2026\] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](../../AAAI2026/medical_imaging/neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)
- [\[CVPR 2026\] From Adaptation to Generalization: Adaptive Visual Prompting for Medical Image Segmentation](apex_adaptive_visual_prompting.md)

</div>

<!-- RELATED:END -->
