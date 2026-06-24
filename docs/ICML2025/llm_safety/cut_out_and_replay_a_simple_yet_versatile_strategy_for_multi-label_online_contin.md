---
title: >-
  [Paper Note] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning
description: >-
  [ICML 2025][LLM Safety][Multi-label learning] Proposed CUTER (CUT-out-and-Experience-Replay), which converts multi-label online continual learning into multiple single-label sub-image classification tasks by cropping label-specific regions from images and storing them in a memory buffer for replay. This simultaneously addresses the three challenges of catastrophic forgetting, missing labels, and class imbalance.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Multi-label learning"
  - "online continual learning"
  - "experience replay"
  - "spectral clustering"
  - "object localization"
date: 2026-05-08
content_hash: e90305e3add5340b
---

# Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning

**Conference**: ICML 2025  
**arXiv**: [2505.19680](https://arxiv.org/abs/2505.19680)  
**Code**: [GitHub](https://github.com/wxr99/Cut-Replay)  
**Area**: LLM Safety  
**Keywords**: Multi-label learning, online continual learning, experience replay, spectral clustering, object localization

## TL;DR

Proposed CUTER (CUT-out-and-Experience-Replay), which converts multi-label online continual learning into multiple single-label sub-image classification tasks by cropping label-specific regions from images and storing them in a memory buffer for replay. This simultaneously addresses the three challenges of catastrophic forgetting, missing labels, and class imbalance.

## Background & Motivation

Multi-label online continual learning (MOCL) requires a model to continuously learn from an incoming stream of multi-label data, which faces three major challenges:

**Pervasive Missing Labels**: Samples for task $t$ are only labeled with the current label set $Y_t$. Even if the image contains objects from old or future classes, these unlabeled classes act as false negatives, exacerbating catastrophic forgetting.

**Uncontrollable Class Imbalance**: Classes typically exhibit a long-tailed distribution, and the co-occurrence of head and tail classes in the same sample further complicates the issue.

**Limitations of Prior Work**: Existing methods (PRS, OCDM, AGCN, KRT, etc.) all adopt image-level feature extraction, ignoring the identification and feature learning of label-specific regions. This leads to co-occurrence bias—where head and tail classes share the same feature vector, making pseudo-labeling and resampling unable to fundamentally resolve the issue.

**Key Insight**: If the regions corresponding to each label can be identified and cropped online, and the region-label pairs stored in the buffer for replay, label co-occurrence interference and missing label issues can be naturally avoided. Meanwhile, class imbalance can be easily solved by controlling the distribution of each class in the buffer.

## Method

### Overall Architecture

CUTER comprises three core steps:

1. **Zero-Shot Localization Capability Assessment of Pre-trained Models**: A label-free assessment protocol based on graph theory is proposed to select the pre-trained model most suitable for localization.
2. **Selective Replay (Cut-and-Replay)**: MaskCut is utilized to locate foreground object regions, establishing a one-to-one region-label correspondence, followed by cropping and storing these regions in the memory buffer.
3. **Localization-Aware Feature Regularization**: To prevent the degradation of localization capability during continual learning, a nuclear norm regularization is introduced to enhance the separability of feature maps.

### Key Designs

#### 1. Pre-trained Model Evaluation Based on Fiedler Value

Construct a weighted undirected graph $G=(V, E, A)$ using image patch features, where edge weights are defined as $A_{ij} = \exp(-\frac{\|\theta(x_i)-\theta(x_j)\|^2}{2\sigma^2})$. The second smallest eigenvalue of the graph Laplacian matrix $L=D-A$ (the Fiedler value $\lambda_2$) is then calculated.

**Key Theoretical Basis**: The Fiedler value and the Cheeger constant satisfy $\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2\Delta\lambda_2}$. A lower average Fiedler value implies weaker graph connectivity and stronger feature separability, making it more suitable for spectral clustering-based localization.

**Assessment Conclusion**: Multi-crop consistency training (e.g., DINO) significantly enhances natural localization capabilities, followed by contrastive learning (MoCo), while reconstruction-based pre-training (MAE) performs the worst. DINO v1 ViT-S/16 is ultimately selected as the backbone network.

#### 2. Label-Region Matching and Selective Replay

For each input $(x, y)$:

- **Foreground Object Extraction**: MaskCut (MCut) is used iteratively to generate $N$ binary masks $\{m_j\}_{j=1}^N$, from which bounding boxes are derived to crop $N$ candidate foreground regions $\{x_{obj}^j\}$.
- **Establishing Region-Label Correspondence**: The cropped regions are fed into the classification model to obtain predictions $p_{obj}^j$. Regions satisfying the following conditions are preserved:
  - Maximum prediction probability $p_{obj,(1)}^j > \tau$ (high confidence)
  - Second largest prediction probability $p_{obj,(2)}^j < 0.5$ (single-label correspondence)
- **Adaptive Threshold Balancing**: A dual-threshold scheme $\tau_1 < \tau_2$ is established. Classes with frequencies less than half of the most frequent class use a lower threshold $\tau_1$, while others use $\tau_2$.
- **Rebalanced Reservoir Sampling**: The sampling probability for new candidates is $1 - m/m_{max}$, where $m$ is the number of samples of the predicted label in the buffer, and $m_{max}$ is the quantity of the most frequent class. When the buffer is full, samples from the most frequent classes are randomly evicted.

This strategy converts multi-label image replay into multiple single-label sub-image replays, achieving better class balance than OCDM and PRS with lower computational overhead.

#### 3. Localization-Aware Feature Regularization

During continual learning, the localization capability of the pre-trained model gradually degenerates (evidenced by increasing Fiedler values and localization failure rates). To address this:

**Theoretical Basis (Theorem 2.3)**: Decomposing the adjacency matrix into an ideal block-diagonal matrix $A^*$ and a noise matrix $\epsilon$, i.e., $A = A^* + \epsilon$, yields $\lambda_2(L) \leq \|\epsilon\|_2 + \|\epsilon\|_\infty$.

**Nuclear Norm Regularization**: The nuclear norm $\|A\|_*$ of the adjacency matrix $A$ is directly constrained. Promoting a block-diagonal structure for $A$ via soft-thresholding of singular values indirectly reduces the Fiedler value, thereby enhancing separability.

### Loss & Training

The total loss function is:

$$L = L_{asl}(f, x, y) + \alpha \|A\|_*$$

where $L_{asl}$ is the Asymmetric Loss, which employs different focusing parameters $\gamma^+$ and $\gamma^-$ for positive and negative labels respectively:

$$L_{asl} = \frac{1}{|C_k|}\sum_{c=1}^{|C_k|} \begin{cases} (1-p_c)^{\gamma^+}\log(p_c), & y_c=1 \\ p_c^{\gamma^-}\log(1-p_c), & y_c=0 \end{cases}$$

Reasons for choosing the nuclear norm over sparse regularization or smoothing regularization: sparse regularization disrupts the inherent structure of ViT parameters, while smoothing regularization makes node features too similar, thereby hindering spectral clustering.

## Key Experimental Results

### Main Results

Experiments were conducted on three datasets: PASCAL VOC 2007 (5 tasks $\times$ 4 classes), MS-COCO (8 tasks $\times$ 10 classes), and NUS-WIDE (8 tasks), with a memory size of $1000 \times 224 \times 224 \times 3$.

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|-------|---------|------|
| VOC | Avg mAP | **82.07** | 76.24 (APPLE) | +5.83 |
| VOC | Last mAP | **67.89** | 58.27 (APPLE) | +9.62 |
| COCO | Avg mAP | **60.14** | 56.45 (AGCN) | +3.69 |
| COCO | Last mAP | **47.82** | 40.56 (OCDM) | +7.26 |
| NUSWIDE | Avg mAP | **51.14** | 49.16 (AGCN) | +1.98 |
| NUSWIDE | Last mAP | **42.92** | 40.89 (APPLE) | +2.03 |

### Ablation Study

| Configuration | VOC Avg mAP | COCO Avg mAP | Description |
|------|------------|-------------|------|
| Baseline (RS) | 75.05 | 48.12 | Baseline Replay |
| + Cut.Rep | 77.92 | 53.40 | Significant improvement with cropped replay alone |
| + Cut.Rep + Frozen Backbone | 78.62 | 59.01 | Frozen DINO backbone |
| CUTER (Updated Backbone) | 79.45 | 59.23 | Dynamic update outperforms freezing |
| CUTER + $R_l$ | **82.07** | **60.14** | Full method |

### Key Findings

1. **Cropped replay is the most critical component**: Cut.Rep improves Avg mAP from 75.05 to 77.92 on VOC, and from 48.12 to 53.40 on COCO, contributing the most.
2. **Strong orthogonal compatibility**: Cut.Rep can be seamlessly integrated as a plug-in with PRS (+3.43), OCDM (+3.17), KRT, and AGCN.
3. **Nuclear norm regularization is optimal**: Compared with sparse regularization ($R_{sp}$, 78.34) and smooth regularization ($R_{sm}$, 79.01), nuclear norm ($R_l$, 82.07) yields the best performance.
4. **DINO v1 possesses the strongest localization capability**: Under the ViT-S backbone, CUTER pre-trained with DINO v1 achieves 82.07 on VOC, outperforming Supervised (79.56), MoCo v3 (80.47), and MAE (77.31).

## Highlights & Insights

1. **Solving the problem at its root**: Unlike prior methods that "patch" issues via pseudo-labeling/distillation, this work targets the root of multi-label learning—label-specific feature learning—by decomposing the multi-label problem into multiple single-label sub-problems.
2. **Fiedler value as an unsupervised evaluation metric**: Leveraging spectral graph theory, this metric evaluates the localization potential of pre-trained models without GT boxes/masks, demonstrating generalizable value.
3. **Orthogonal design**: CUTER does not replace but complements existing methods, and can be freely combined with PRS, OCDM, KRT, AGCN, etc.
4. **Solid theoretical support**: A complete theoretical chain is established, extending from Cheeger constants to Fiedler values and onto the nuclear norm constraint of Theorem 2.3.

## Limitations & Future Work

1. **Dependency on ViT architecture**: The core of the method relies on patch-level feature graph construction, leading to significant performance degradation on CNNs (e.g., ResNet).
2. **Heavy computational overhead**: Multi-round MCut operations cannot be parallelized on GPUs, and nuclear norm gradient computation introduces additional overhead, limiting throughput.
3. **Sensitivity to thresholds**: Hyperparameters $\tau_1$, $\tau_2$, and $\alpha$ require careful tuning; an excessively large $\alpha$ collapses the adjacency matrix toward a zero matrix.
4. **Future directions**: Developing acceleration techniques, adaptive processing strategies, and adapting the method to CNN architectures.

## Related Work & Insights

- **OCDM (Liang & Li, 2022)**: Addresses class imbalance via optimized sampling. This work proposes a simpler and more effective rebalancing strategy on top of it.
- **AGCN++ (Du et al., 2023)**: Models label correlations via GCN to handle missing labels, but still relies on image-level features.
- **DINO (Caron et al., 2021)**: Self-supervised pre-training naturally possesses object localization capabilities, serving as the core premise of this work.
- **MaskCut (Wang et al., 2023)**: Unsupervised multi-object segmentation, which is introduced to the MOCL scenario in this work.
- **LIFT (Zhang & Wu, 2014)**: A classic work in offline multi-label learning showing that label-specific features outperform unified ones. This work generalizes this idea to online continual learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First to introduce label-specific region learning to MOCL, representing a highly novel perspective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets + 10 baselines + detailed ablation/sensitivity/visualization/backbone comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Rigorous theoretical derivation and clear structure, though some notations are heavy)
- Value: ⭐⭐⭐⭐⭐ (Strong orthogonality enables plug-and-play capability; both theoretical and empirical contributions are solid)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)

</div>

<!-- RELATED:END -->
