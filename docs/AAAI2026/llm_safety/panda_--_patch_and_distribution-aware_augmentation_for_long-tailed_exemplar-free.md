---
title: >-
  [Paper Note] PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning
description: >-
  [AAAI 2026][LLM Safety][Exemplar-Free Continual Learning] This paper proposes PANDA, a framework that achieves intra-task class balancing via CLIP-guided semantic patch grafting and alleviates inter-task distribution shi…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Exemplar-Free Continual Learning"
  - "Long-Tailed Distribution"
  - "CLIP-guided Augmentation"
  - "Dual-Level Imbalance"
  - "Distribution Smoothening"
date: 2026-05-08
content_hash: 22121dfb4a35892b
---

# PANDA: Patch and Distribution-Aware Augmentation for Long-Tailed Exemplar-Free Continual Learning

**Conference**: AAAI 2026
**arXiv**: [2511.09791](https://arxiv.org/abs/2511.09791)
**Code**: [GitLab](https://gitlab.com/viper-purdue/panda)
**Area**: Continual Learning / Long-Tailed Distribution / Data Augmentation
**Keywords**: Exemplar-Free Continual Learning, Long-Tailed Distribution, CLIP-guided Augmentation, Dual-Level Imbalance, Distribution Smoothening

## TL;DR

This paper proposes PANDA, a framework that achieves intra-task class balancing via CLIP-guided semantic patch grafting and alleviates inter-task distribution shift through a learnable distribution smoothening mechanism. PANDA operates as a plug-and-play module to improve pretrained model-based exemplar-free continual learning under long-tailed scenarios.

## Background & Motivation

### Problem Definition
Exemplar-Free Continual Learning (EFCL) prohibits storing data from previous tasks, making it highly susceptible to catastrophic forgetting. Recent advances leveraging the strong representational capacity of pretrained models (PTMs) have significantly improved EFCL, yet existing methods almost universally assume uniform class distributions across tasks.

### Real-World Challenge: Dual-Level Imbalance
The paper identifies a pervasive **Dual-Level Imbalance (DLI)** in real-world data streams:

**Dataset-level imbalance**: Certain classes dominate the overall dataset while others are scarce, governed by an exponential decay factor $\rho$.

**Task-level imbalance**: The class distribution within a single task may oppose or exaggerate the global trend, governed by $\rho^*$.

For example, camera traps in the wild may capture thousands of deer and rabbit images while predators are extremely rare; during migration seasons, deer images may temporarily surge. In medical imaging, pneumonia is common but pneumoconiosis may occasionally surpass it. The co-existence of intra-task imbalance and inter-task distribution drift has been largely unexplored.

### Limitations of Prior Work
- **Prompt-based methods** (L2P, CodaPrompt, DualPrompt, DAP): Optimizing only a small number of parameters makes it difficult to capture the diversity of tail classes under severe imbalance.
- **Representation-based methods** (SimpleCIL, RanPAC, MOS, etc.): Rely on prototype updates via nearest-mean classifiers, which become unreliable under long-tailed distribution shifts.
- **Existing augmentation methods** (CutMix, Mixup, Remix): Not designed for continual learning and ignore the constraint that the global distribution is unknown.

## Method

### Overall Architecture
PANDA is a **training-free debiasing augmentation framework** that integrates seamlessly into any PTM-based EFCL method. The core idea is to exploit the background diversity of head-class (high-frequency) images to enrich training samples for tail-class (low-frequency) categories. It comprises two complementary mechanisms:

1. **Intra-task Balancing**: Uses a frozen CLIP encoder to identify and graft semantically relevant patches between head- and tail-class images.
2. **Inter-task Smoothening**: Adaptively adjusts the distribution boundaries of the current task using distribution statistics from the previous task.

### Key Designs

#### 1. CLIP-Guided Semantic Patch Grafting
**Core Idea**: Extract the most class-semantically representative regions from tail-class images and graft them onto the background regions of head-class images, synthesizing new tail-class training samples.

**Pipeline**:
- Partition each image into $N \times N$ non-overlapping patches
- Compute embeddings for class labels (converted to pseudo-sentences "Image of a {label}") and each patch using frozen CLIP text and visual encoders, respectively
- Select the top-$N/2$ patches most semantically related to the class via cosine similarity $S_i = \frac{z_i \cdot z_t}{\|z_i\| \|z_t\|}$
- Apply a cosine similarity confidence threshold of 0.45 to prevent cross-class contamination
- Graft high-semantic patches from tail-class images onto head-class images using binary masks:

$$x' = (M^h)' \odot x^h + M^t \odot x^t, \quad y' = y^t$$

where $(M^h)'$ is the inverted head-class mask and $M^t$ is the tail-class semantic mask.

**Design Motivation**: Images generally consist of objects of interest and background; the background contributes little to classification. Grafting the semantic core of a tail-class image onto the background of a head-class image increases sample diversity for the tail class without introducing class confusion. Standard augmentations (flipping, cropping, color jitter, Gaussian blur) are further applied to the synthesized samples to prevent overfitting.

#### 2. Adaptive Distribution Smoothening
**Core Idea**: Maintain min/max sample count statistics from the previous task and blend them with the current task's statistics to mitigate inter-task distribution drift.

$$\text{adjusted}_m = \beta \cdot \text{prior}_m + (1 - \beta) \cdot \text{current}_m, \quad m \in \{\min, \max\}$$

The coefficient $\beta$ is dynamically adjusted based on the performance difference between consecutive tasks:
- Performance decreases on the current task → decrease $\beta$ to accelerate adaptation
- Performance increases → increase $\beta$ to reinforce stability
- Performance unchanged → $\beta$ remains constant

**Design Motivation**: Inter-task distribution drift in continual learning introduces classifier bias. Incorporating the distribution prior from the previous task smoothens the extreme distribution of the current task, reducing the overall gap in mean sample counts and enabling fairer learning on the frozen PTM.

### Loss & Training
PANDA itself is **training-free** (no additional trainable parameters introduced) and operates solely at the data preprocessing stage. The augmented data is fed into the original continual learning pipeline for training. The only hyperparameters are:
- Number of patches $N$ (determined by ViT patch size)
- Cosine similarity confidence threshold (0.45)
- Head-tail average sample gap threshold $q$

## Key Experimental Results

### Main Results (Single-Level Imbalance SLI, $\rho = 0.01$)

| Method | CIFAR100-LT Acc(%) | CIFAR100-LT For(%) | iNaturalist Acc(%) | iNaturalist For(%) |
|--------|--------------------|--------------------|--------------------|--------------------|
| L2P | 73.34 | 7.87 | 78.41 | 4.72 |
| L2P + PANDA | **81.32 (+7.98)** | **6.08 (-1.79)** | **85.47 (+7.06)** | **3.37 (-1.35)** |
| CodaPrompt | 76.52 | 7.55 | 83.85 | 4.58 |
| CodaPrompt + PANDA | **87.49 (+2.94)** | **4.61 (-2.94)** | **90.45 (+6.60)** | **3.30 (-1.28)** |
| RanPAC | 90.35 | 5.22 | 94.35 | 2.38 |
| RanPAC + PANDA | **91.91 (+1.56)** | **4.38 (-0.84)** | **95.70 (+1.35)** | **1.97 (-0.41)** |
| CoFiMA | 93.05 | 5.57 | 94.55 | 3.88 |
| CoFiMA + PANDA | **93.83 (+0.78)** | **4.91 (-0.66)** | 93.56 | **2.98 (-0.90)** |
| MOS | 91.60 | 4.69 | 95.49 | 2.77 |
| MOS + PANDA | **92.04 (+0.80)** | **4.48 (-0.21)** | **95.85 (+0.36)** | **2.63 (-0.14)** |

PANDA yields accuracy improvements or forgetting reductions across all 14 evaluated EFCL methods. Gains are most pronounced on prompt-based methods (L2P improves by ~7–8%), indicating that long-tailed scenarios impose the greatest burden on parameter-limited prompt approaches.

### Dual-Level Imbalance Experiments (DLI)

| Method | $\rho^*=0.05, *=2$ Acc | $\rho^*=0.05, *=3$ Acc | $\rho^*=0.05, *=4$ Acc |
|--------|------------------------|------------------------|------------------------|
| CoFiMA | 93.97 | 92.18 | 90.39 |
| CoFiMA + PANDA | **94.38 (+0.41)** | **93.25 (+1.07)** | **92.05 (+1.66)** |
| MOS | 93.54 | 92.10 | 91.69 |
| MOS + PANDA | 92.22 | **93.21 (+1.11)** | **92.82 (+1.13)** |

Under the DLI setting, where task-level imbalance compounds dataset-level imbalance, PANDA continues to improve performance through distribution smoothening.

### Ablation Study: Comparison with Other Augmentation Methods (RanPAC, CIFAR100-LT)

| Augmentation | SLI Acc(%) | SLI For(%) | DLI Acc(%) | DLI For(%) |
|--------------|-----------|-----------|-----------|-----------|
| No augmentation (baseline) | 84.39 | 5.82 | 85.07 | 5.97 |
| CutMix | 85.43 | 7.97 | 84.03 | 6.77 |
| Mixup | 81.33 | 8.03 | 77.29 | 7.06 |
| Remix | 86.50 | 7.55 | 86.51 | 5.73 |
| Con-CutMix | 87.27 | 6.48 | 84.19 | 6.01 |
| **PANDA (Ours)** | **90.31** | **5.03** | **90.08** | **4.52** |

PANDA substantially outperforms all conventional augmentation methods. CutMix and Mixup, being distribution-agnostic, even underperform the baseline; Con-CutMix approaches PANDA under SLI but collapses under DLI.

### Key Findings

1. **Semantic patch selection outperforms attention masks**: CLIP-guided semantic masks consistently outperform DinoV2 attention-affinity-based masks (83.39 vs. 80.87 accuracy on APART), as language-guided semantic alignment more precisely isolates representative regions.
2. **Manageable resource overhead**: PANDA typically increases GPU memory usage by no more than 600 MB and runtime by no more than 0.5 hours.
3. **Stability–plasticity trade-off**: On iNaturalist, CoFiMA + PANDA shows a slight accuracy drop accompanied by a substantial reduction in forgetting, reflecting the inherent trade-off when performance is near saturation.

## Highlights & Insights

- **Plug-and-play design**: PANDA is a pure data augmentation module with no additional trainable parameters, enabling seamless integration into any PTM-based EFCL method with high practical utility.
- **Formalization of dual-level imbalance**: The paper is the first to formally define the DLI setting, providing a more realistic evaluation framework for imbalance research in continual learning.
- **Novel application of CLIP**: CLIP's text–image alignment capability is repurposed to identify semantically representative regions within images, distinct from its conventional uses in zero-shot classification or retrieval.
- **Elegant simplicity of distribution smoothening**: Maintaining only the min/max statistics of the previous task effectively mitigates inter-task drift, achieving robust results with minimal implementation complexity.

## Limitations & Future Work

1. **Dependency on CLIP quality**: Patch selection quality relies entirely on CLIP's feature alignment; using different CLIP variants may introduce performance variability.
2. **Limited to image classification**: Applicability to more complex tasks such as object detection and semantic segmentation remains unexplored.
3. **Threshold sensitivity**: The cosine similarity threshold of 0.45 is determined empirically without theoretical justification and may require tuning for different datasets.
4. **Limited patch diversity for extremely rare tail classes**: When a tail class contains only 3 samples, the diversity of available semantic patches for grafting is inherently constrained.
5. **DLI setting affects only a single task**: $\rho^*$ is applied to one designated task; more complex scenarios where multiple tasks are simultaneously imbalanced are not considered.

## Related Work & Insights

- **CutMix/Mixup variants** show limited effectiveness in continual learning due to their disregard for class distribution information.
- **DAP** (AAAI 2024), despite its dual-anchor mechanism specifically designed for imbalance, still allows head-class gradients to dominate general prompts, leaving tail-class signals too weak.
- The PANDA framework could potentially be extended to continual learning in other modalities such as text and audio.
- The distribution smoothening mechanism may offer insights applicable to data heterogeneity problems in federated learning.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 4 | DLI formalization and CLIP-guided patch grafting are original contributions |
| Technical Depth | 3.5 | Method is elegant and effective but not technically complex |
| Experimental Thoroughness | 4.5 | 14 methods × 3 datasets × 2 imbalance settings with comprehensive ablations |
| Writing Quality | 4 | Clear motivation and readable experiments |
| Practicality | 5 | Plug-and-play, open-source, engineering-friendly |
| **Overall** | **4** | A pragmatic plug-and-play solution with thorough experiments, though limited in theoretical depth |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[AAAI 2026\] CATFormer: When Continual Learning Meets Spiking Transformers With Dynamic Thresholds](catformer_when_continual_learning_meets_spiking_transformers_with_dynamic_thresh.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/llm_safety/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)

</div>

<!-- RELATED:END -->
