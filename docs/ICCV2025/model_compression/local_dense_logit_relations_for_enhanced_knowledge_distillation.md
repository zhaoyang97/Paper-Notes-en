---
title: >-
  [Paper Note] Local Dense Logit Relations for Enhanced Knowledge Distillation
description: >-
  [ICCV 2025][Model Compression][Knowledge Distillation] This paper proposes Local Dense Relational Logit Distillation (LDRLD), which captures fine-grained inter-class relations by recursively decoupling and recombining logit knowledge, combined with an Adaptive Decay Weight (ADW) strategy that assigns higher weights to critical class pairs. LDRLD consistently outperforms existing logit distillation state-of-the-art methods on CIFAR-100, ImageNet-1K, and Tiny-ImageNet.
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Logit Distillation"
  - "Class-Pair Relations"
  - "Adaptive Decay Weights"
  - "Fine-Grained Knowledge Transfer"
date: 2026-05-08
content_hash: f7a49741720bc3bd
---

# Local Dense Logit Relations for Enhanced Knowledge Distillation

**Conference**: ICCV 2025
**arXiv**: [2507.15911](https://arxiv.org/abs/2507.15911)  
**Code**: None (promised by authors)  
**Area**: Model Compression
**Keywords**: Knowledge Distillation, Logit Distillation, Class-Pair Relations, Adaptive Decay Weights, Fine-Grained Knowledge Transfer

## TL;DR
This paper proposes Local Dense Relational Logit Distillation (LDRLD), which captures fine-grained inter-class relations by recursively decoupling and recombining logit knowledge, combined with an Adaptive Decay Weight (ADW) strategy that assigns higher weights to critical class pairs. LDRLD consistently outperforms existing logit distillation state-of-the-art methods on CIFAR-100, ImageNet-1K, and Tiny-ImageNet.

## Background & Motivation
Knowledge distillation (KD) transfers the "dark knowledge" of large teacher models to lightweight student models. Logit-based KD methods have attracted considerable attention due to their computational efficiency and generality. Classical KD computes KL divergence over global softmax outputs to transfer the teacher's output probability distribution.

However, existing logit distillation methods suffer from insufficient modeling of fine-grained inter-class relations:

**Global softmax weakens probability differences among low-probability classes**: Softmax concentrates on high-probability classes, compressing probability differences among low-probability classes and limiting the student's ability to capture fine-grained relations among them.

**Irrelevant classes introduce information redundancy**: When modeling the relation between "cat" and "dog," global softmax introduces probabilities of irrelevant classes such as "car," which interferes with and weakens the discriminability of the target class pair.

**Uniform weights ignore the varying importance of class pairs**: Existing methods assign equal weights to all class pairs, whereas semantically similar pairs (e.g., "cat" vs. "dog") should receive more attention than semantically distant pairs (e.g., "cat" vs. "airplane").

The paper's starting point is to decompose logit knowledge from a global distribution into local class-pair relations, enhance the discriminability of each class pair via local softmax, and focus on critical class pairs using a rank-based adaptive decay strategy. Intuitively, the probability difference between "cat" and "dog" under global KD is $\Delta P_{KD} = |p_1 - p_2| = |\frac{e^{Z_1} - e^{Z_2}}{\sum_{i=1}^C e^{Z_i}}|$, whereas under LDRLD it becomes $\Delta P_{LDRLD} = |\frac{e^{Z_1} - e^{Z_2}}{\sum_{i=1}^2 e^{Z_i}}|$, which is clearly larger.

## Method

### Overall Architecture
LDRLD comprises three loss terms: (1) weighted local dense relational distillation $\mathcal{L}^w$—transferring fine-grained knowledge via KL divergence over class pairs with ADW weights; (2) Residual Non-Target Knowledge distillation $\mathcal{L}_{RNTK}$—ensuring that the residual knowledge after recursive decoupling is not discarded; and (3) Local Logit Knowledge Integrity $\mathcal{L}_{LLKI}$—performing complete local KL distillation over the top-$d$ classes. The total loss is $\mathcal{L}_{LDRLD} = \mathcal{L}_{Task} + \alpha \mathcal{L}_{Local} + \beta \mathcal{L}_{RNTK}$.

### Key Designs
1. **Recursive Decoupling and Recombination (Core of LDRLD)**

    - Function: Recursively extracts the top-$d$ classes from the student logit output and forms all pairwise class combinations.
    - Mechanism:
        - Step 1: Extract the class with the largest student logit $\mathbf{Z}_1^s$ and remove it from the logit vector using a mask $\mathbf{M}_{\pi(1)}$ (setting the corresponding position to $-\infty$).
        - Step 2: Recursively extract the 2nd, 3rd, ..., $d$-th largest classes, combining each with all previously extracted classes to form class pairs via $\phi: \mathbb{R} \uplus \mathbb{R} \to \mathbb{R}^{1 \times 2}$.
        - This yields $\frac{d(d-1)}{2}$ class pairs in total.
        - Step 3: Apply independent softmax normalization to each class pair and compute the KL divergence.
    - Design Motivation: Local softmax normalization amplifies probability differences within each class pair, enhancing discriminability. The $O(d^2)$ number of class pairs provides dense relational information.
    - Base Loss: $\mathcal{L} = \sum_{i=1}^{d-1}\sum_{j=i+1}^{d} [p_i^t \log(p_i^t/p_i^s) + p_j^t \log(p_j^t/p_j^s)]$

2. **Adaptive Decay Weight (ADW) Strategy**

    - Function: Dynamically assigns different distillation weights to different class pairs.
    - Mechanism: Two components are included:
        - **Inverse Rank Weighting (IRW)**: $\Gamma_{IRW}(R', R) = \frac{1}{|R - R'| + \epsilon}$, assigning higher weights to class pairs with smaller rank gaps ($\epsilon = 1.50$).
        - **Exponential Rank Decay (ERD)**: $\Phi_{ERD}(R', R) = \delta \times \exp(-\lambda(R + R'))$, exponentially decaying the weight for class pairs with larger rank sums ($\delta = 2.0, \lambda = 0.05$).
        - Combined weight: $\Omega_{ADW}(R', R) = \Gamma_{IRW} \times \Phi_{ERD}$
    - Design Motivation:
        - IRW addresses the issue that "class pair [Z₁, Z₂] is harder to discriminate than [Z₁, Z₄]."
        - ERD addresses the issue that "[Z₁, Z₂] and [Z₁₂, Z₁₃] share the same rank gap of 1, yet the former is more important."
    - Weighted Loss: $\mathcal{L}^w = \sum_{i=1}^{d-1}\sum_{j=i+1}^{d} \Omega_{ADW}(i,j)[p_i^t \log(p_i^t/p_i^s) + p_j^t \log(p_j^t/p_j^s)]$

3. **Residual Non-Target Knowledge Distillation (RNTK)**

    - Function: Performs holistic distillation over the $C - d$ classes not selected during recursive decoupling.
    - Mechanism: $\mathcal{L}_{RNTK} = \mathcal{KL}(\bar{\mathcal{H}}^t, \bar{\mathcal{H}}^s) = \sum_{i=d+1}^C \bar{p}_i^t \log(\bar{p}_i^t / \bar{p}_i^s)$
    - Design Motivation: Ensures knowledge completeness, preventing the dark knowledge of tail classes from being discarded when attending only to the top-$d$ classes.

### Loss & Training
The overall optimization objective is:
$$\mathcal{L}_{LDRLD} = \mathcal{L}_{Task} + \alpha \mathcal{L}_{Local} + \beta \mathcal{L}_{RNTK}$$

where $\mathcal{L}_{Local} = \mathcal{L}^w + \mathcal{L}_{LLKI}$, and $\alpha$, $\beta$ are balancing coefficients.

## Key Experimental Results

### Main Results
CIFAR-100 homogeneous architecture distillation:

| Teacher→Student | KD | DKD | NKD | WTTM | **LDRLD** | Δ vs KD |
|---|---|---|---|---|---|---|
| ResNet56→ResNet20 | 70.66 | 71.97 | 71.18 | 71.92 | **72.20** | +1.54 |
| ResNet110→ResNet20 | 70.67 | 70.91 | 71.26 | 71.67 | **71.98** | +1.31 |
| ResNet32×4→ResNet8×4 | 73.33 | 76.32 | 76.35 | 76.06 | **77.20** | +3.87 |
| WRN-40-2→WRN-16-2 | 74.92 | 76.24 | 75.43 | 76.37 | **76.35** | +1.43 |
| VGG13→VGG8 | 72.98 | 74.68 | 74.86 | 74.44 | **75.06** | +2.08 |

ImageNet-1K:

| Teacher→Student | KD | DKD | WTTM | **LDRLD** | Δ vs KD |
|---|---|---|---|---|---|
| ResNet34→ResNet18 Top-1 | 70.66 | 71.70 | 72.19 | **71.88** | +1.22 |
| ResNet50→MobileNetV1 Top-1 | 70.49 | 72.05 | 73.09 | **73.12** | +2.63 |

### Ablation Study
CIFAR-100 heterogeneous architecture distillation (ResNet32×4→ShuffleNetV1):

| Method | Top-1 Acc | Note |
|---|---|---|
| KD | 74.07 | Baseline |
| DKD | 76.45 | Decoupled target/non-target |
| NKD | 76.31 | Normalized non-target |
| FCFD | 78.12 | Feature distillation |
| CAT-KD | 78.26 | Class activation transfer |
| **LDRLD** | **76.46** | ADW + LDRLD + RNTK |

### Key Findings
- **Comprehensive superiority among logit-based methods**: LDRLD outperforms DKD, NKD, WTTM, and other methods across all homogeneous architecture pairs and the majority of heterogeneous pairs.
- **Largest gains under large capacity gaps**: ResNet32×4→ResNet8×4 yields a +3.87% improvement, indicating that fine-grained relational knowledge is particularly valuable when the capacity gap is large.
- **Large-scale validation on ImageNet-1K**: A +2.63% gain on ResNet50→MobileNetV1 demonstrates that the method generalizes beyond small datasets.
- **Critical role of the ADW strategy**: The combination of IRW and ERD ensures focus on critical class pairs while avoiding excessive attention to low-ranked pairs.
- **Strong competitiveness within logit-based methods**, though a gap remains compared to some feature-based methods (e.g., FCFD, CAT-KD).

## Highlights & Insights
- **Local softmax enhances discriminability**: Decomposing the global probability distribution into pairwise local probabilities naturally amplifies probability differences among similar classes—a concise yet powerful observation.
- **Two-level weight design in ADW**: IRW operates on rank difference (inter-class distance) while ERD operates on rank sum (absolute position), capturing complementary aspects of class-pair importance.
- **Information density of recursive decoupling**: $d$ classes yield $\frac{d(d-1)}{2}$ class pairs, providing far richer relational information than a single KL divergence term.
- **Knowledge completeness guarantee**: The RNTK loss ensures that dark knowledge from unselected tail classes is not forgotten, reflecting the thoroughness of the design.

## Limitations & Future Work
- The recursive depth $d$ requires tuning; excessively large values introduce unnecessary low-ranked class pairs.
- Computational complexity is $O(d^2)$, which may incur additional overhead when $d$ is large.
- Hyperparameters in ADW ($\epsilon = 1.50, \delta = 2.0, \lambda = 0.05$) are manually set; adaptive learning of these values could yield further improvements.
- Validation is limited to classification tasks; extension to detection, segmentation, and other tasks remains unexplored.
- A performance gap persists compared to feature-based methods (e.g., FCFD at 78.12%), suggesting an inherent ceiling on the information extractable from the logit layer alone.

## Related Work & Insights
- DKD decouples logit knowledge into target and non-target components; LDRLD further recursively decomposes non-target knowledge into dense class-pair relations, achieving a deeper level of decomposition.
- The weight decay concept in ADW can be adapted to attention mechanism design.
- The discriminability enhancement via local softmax may inspire negative sample weighting strategies in contrastive learning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of recursive decoupling and local softmax is novel, and the ADW strategy is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers CIFAR-100/ImageNet-1K/Tiny-ImageNet, homogeneous/heterogeneous architectures, and extensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Figures and tables are clear and intuitive; method description is rigorous though formula-heavy.
- Value: ⭐⭐⭐⭐ Provides a new fine-grained modeling perspective for logit distillation with consistent empirical gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Knowledge Distillation with Refined Logits](knowledge_distillation_with_refined_logits.md)
- [\[ACL 2025\] Sparse Logit Sampling: Accelerating Knowledge Distillation in LLMs](../../ACL2025/model_compression/sparse_logit_sampling_accelerating_knowledge_distillation_in_llms.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICCV 2025\] EA-KD: Entropy-based Adaptive Knowledge Distillation](ea-kd_entropy-based_adaptive_knowledge_distillation.md)
- [\[ICML 2025\] Efficient Logit-based Knowledge Distillation of Deep Spiking Neural Networks for Full-Range Timestep Deployment](../../ICML2025/model_compression/efficient_logit-based_knowledge_distillation_of_deep_spiking_neural_networks_for.md)

</div>

<!-- RELATED:END -->
