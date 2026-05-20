---
title: >-
  [Paper Note] Boosting Adversarial Transferability via Ensemble Non-Attention
description: >-
  [AAAI 2026][adversarial transferability] This paper proposes NAMEA (Non-Attention Meta Ensemble Attack), which for the first time exploits the non-attention areas of ensemble models to integrate transferable information…
tags:
  - "AAAI 2026"
  - "adversarial transferability"
  - "ensemble attack"
  - "non-attention areas"
  - "meta-learning"
  - "cross-architecture attack"
date: 2026-05-08
content_hash: 75176e82ad938095
---

# Boosting Adversarial Transferability via Ensemble Non-Attention

**Conference**: AAAI 2026
**arXiv**: [2511.08937](https://arxiv.org/abs/2511.08937)  
**Code**: None  
**Area**: Other
**Keywords**: adversarial transferability, ensemble attack, non-attention areas, meta-learning, cross-architecture attack

## TL;DR

This paper proposes NAMEA (Non-Attention Meta Ensemble Attack), which for the first time exploits the non-attention areas of ensemble models to integrate transferable information from both CNNs and ViTs, and combines meta-learning gradient optimization to achieve an average improvement of 15.0% and 9.6% over the state-of-the-art methods AdaEA and SMER, respectively, on cross-architecture adversarial transferability.

## Background & Motivation

### Challenges in Adversarial Transferability

Deep neural networks are highly vulnerable to adversarial examples. Adversarial examples generated from surrogate models can transfer to unknown target models, enabling black-box attacks. Ensemble attacks improve transferability by aggregating predictions, losses, or logits from multiple surrogate models. However, **cross-heterogeneous-architecture transfer** (e.g., CNN→ViT or mixed ensemble→diverse targets) remains severely limited.

### Core Difficulty — Heterogeneous Gradient Direction Discrepancy

CNN and ViT exhibit large discrepancies in gradient update directions, making it difficult for existing methods to balance "reducing gradient variance across ensemble models" and "fully exploiting information from individual models":
- **AdaEA** ensures stable update directions via a discrepancy reduction filter, but sacrifices model diversity.
- **SMER** independently optimizes each surrogate model to preserve diversity, but the update directions may be insufficiently accurate.

### Key Observation — Complementarity of Attention Regions

The authors identify a critical phenomenon: **homogeneous models share large overlapping attention regions, whereas heterogeneous models focus on substantially different regions**. Specifically:
- Masking the attention regions of ResNet-18 causes CNN classification accuracy to drop by up to 30%, while ViT accuracy drops by less than 10%.
- The masked images actually produce higher attention overlap across heterogeneous models.

This indicates that **the non-attention regions of CNNs are likely the focus regions of ViTs, and vice versa** — non-attention areas contain transferable information across architectures.

## Method

### Overall Architecture

NAMEA models iterative ensemble attacks as a stochastic gradient descent optimization process with $T$ outer loops and $K$ inner loops. Each outer loop consists of three steps:
1. **Attention meta-train**: update gradients based on the attention regions of surrogate models.
2. **Non-attention meta-test**: update gradients based on the non-attention regions of surrogate models.
3. **Final update**: fuse the gradients from both steps to obtain the final update.

### Key Designs

#### 1. **Non-Attention Extraction Module (NAE)**

**Function**: Extract the attention regions of each surrogate model via Grad-CAM, then derive the corresponding non-attention regions.

**Mechanism**: For surrogate model $f_n$ and input $x$, compute the attention map $\mathbf{H}_n(x)$ via Grad-CAM:

$$\alpha_l^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial f_n(x)[y]}{\partial A_l^c[i,j]}$$

$$\mathbf{H}_n^l(x) = \text{ReLU}\left(\sum_c \alpha_l^c \cdot A_l^c\right)$$

An attention mask $\mathbb{M}_k$ is then generated (positions with values $\geq$ threshold $\eta$ are set to 1), and the non-attention mask is $\bar{\mathbb{M}}_k = \mathbf{1} - \mathbb{M}_k$.

**Design Motivation**: Non-attention information is extracted by replacing attention regions with random Gaussian noise: $x_{te}^k = \bar{\mathbb{M}}_k \odot x_{te}^k + \mathbb{M}_k \odot \xi, \quad \xi \sim \mathcal{N}(0,1)$. Filling with random noise yields approximately 2.7% improvement over filling with zeros or ones, as random noise more strongly disrupts model attention.

#### 2. **Meta-Gradient Optimization**

**Function**: Fuse gradients from attention and non-attention regions via a meta-learning framework, balancing update stability and model diversity.

**Mechanism**:

**① Attention meta-train**: Initialize $x_{tr}^0 = x_{adv}^t$ and iterate for $K$ steps:

$$g_{tr}^{k+1} = \nabla_{x_{tr}^k} \mathcal{L}(x_{tr}^k, y)$$

$$x_{tr}^{k+1} = \text{Clip}_\epsilon^x(x_{tr}^k + \alpha \cdot \text{sign}(g_{tr}^{k+1}))$$

**② Non-attention meta-test**: Initialize $x_{te}^0 = x_{adv}^t$, apply the NAE module to mask attention regions, then compute gradients:

$$g_{te}^{k+1} = \nabla_{x_{te}^k} \mathcal{L}(x_{te}^k, y)$$

**③ Final fusion**:

$$g^{t+1} = g_{tr}^K + g_{te}^K \odot \bar{\mathbb{M}}_K$$

The meta-test gradient is masked by the non-attention mask before fusion, ensuring that transferable gradient information from attention regions is not perturbed.

#### 3. **Gradient Scaling Optimization Module (GSO)**

**Function**: Separately optimize meta-test gradients tailored to the distinct properties of CNNs and ViTs.

- **CNN layer-wise gradient scaling**: Intermediate-layer features are more transferable; a scaling factor $\lambda(l) = \lambda_1 + \lambda_2 \cdot (L/l)$ is used to enhance shallow-layer gradient contributions.
- **ViT channel-wise gradient scaling**: ViT backpropagation produces smaller gradients that affect transferability; channels with gradient magnitudes below the mean are shrunk: $g_{te}[c] = g_{te}[c] \cdot \tanh(|{(g_{te}[c] - \phi)}/{\sigma}|)$.

### Loss & Training

NAMEA is a plug-and-play method compatible with various base attacks such as I-FGSM, MI-FGSM, and DI-MI-FGSM. Hyperparameter settings: outer loop $T=10$, inner loop $K=16$, step size $\alpha=0.8/255$, perturbation budget $\epsilon=8/255$, attention threshold $\eta=0.6$. Every $N$ consecutive inner loops guarantee that each surrogate model is selected at least once.

## Key Experimental Results

### Main Results

**ImageNet Cross-Architecture Transferability (ASR%, DI-MI-FGSM base attack)**

| Method | ViT Avg↑ | CNN Avg↑ | Overall Avg↑ |
|--------|----------|----------|--------------|
| Ens | 50.0 | 63.5 | 56.8 |
| SVRE | 55.8 | 70.4 | 63.1 |
| AdaEA | 54.8 | 63.9 | 59.4 |
| CWA | 61.4 | 70.3 | 65.9 |
| SMER | 72.1 | 78.5 | 75.3 |
| CSA | 60.3 | 69.0 | 64.7 |
| **NAMEA** | **77.5** | **83.7** | **80.6** |

Under DI-MI-FGSM, NAMEA achieves an average ASR of 80.6%, surpassing the strongest baseline SMER by 5.3%.

**Robustness Against Defended Models and Defense Methods (DI-MI-FGSM)**

| Method | Defended Models Avg↑ | Defense Methods Avg↑ |
|--------|----------------------|----------------------|
| AdaEA | 56.6 | 40.6 |
| SMER | 69.1 | 56.8 |
| **NAMEA** | **74.2** | **63.3** |

Even against the strong diffusion-based defense DiffPure, NAMEA outperforms baselines by 10% (50.3% vs. 39.9%).

**Real-World API Attack (DI-MI-FGSM)**

| API | SMER | **NAMEA** | Gain |
|-----|------|-----------|------|
| Google | 52 | **55** | +3 |
| Alibaba | 48 | **53** | +5 |
| Baidu | 61 | **64** | +3 |

### Ablation Study

| Configuration | Avg ASR↑ | Notes |
|---------------|----------|-------|
| NAMEA (full) | ~46.7 (I-FGSM) | Baseline |
| Remove meta-test step (-Mtest) | -9.2% | Non-attention gradient is critical |
| Remove meta-train step (-Mtrain) | -7.4% | Attention gradient also important |
| Remove GSO module | -2.0% | GSO contributes positively |
| Fill with zeros | -2.7% | Random noise outperforms fixed values |
| Fill with ones | -2.7% | Same as above |
| Extract attention regions (not non-attention) | Significant drop | Validates the key role of non-attention |
| Extract random patches | Moderate drop | Random positions inferior to semantics-driven non-attention |
| Threshold η=0.6 | Optimal | ASR is sensitive to η for both CNN and ViT |

### Key Findings

1. **The value of non-attention regions has been underestimated**: This work is the first to demonstrate that ensemble non-attention regions contain complementary cross-architecture transferable information.
2. **Meta-learning is essential**: Removing either meta-train or meta-test leads to significant performance degradation; their fusion outperforms any single strategy.
3. **Distinct from input diversity**: Controlled comparisons (NAMEA vs. NAMEA_RT) confirm that performance gains stem from semantic information in non-attention regions rather than simple input diversification.
4. **Insensitive to surrogate model selection strategy**: Same-architecture, cross-architecture, and randomly selected meta-test models yield comparable results, as random selection in inner loops already ensures sufficient exploration of non-attention regions.

## Highlights & Insights

1. **Observation-driven method design**: Starting from the empirical observation of "attention region discrepancy between heterogeneous models," the paper proposes the hypothesis that "non-attention regions contain complementary transferable information" and validates it from both theoretical and experimental perspectives.
2. **Plug-and-play practicality**: NAMEA seamlessly integrates with various base attacks (I-FGSM, MI-FGSM, DI-MI-FGSM, etc.), demonstrating strong generalizability.
3. **A new perspective via attention decoupling**: Unlike prior work focused on exploiting or disrupting attention regions, this paper is the first to systematically explore the value of non-attention regions.
4. **Real-world validation**: Beyond standard benchmarks, the attack is validated on commercial APIs including Google, Alibaba Cloud, and Baidu Cloud.

## Limitations & Future Work

1. **Computational overhead**: NAMEA maintains two adversarial examples per inner loop (meta-train and meta-test), resulting in approximately twice the computation of baselines.
2. **Limitations of Grad-CAM**: Attention extraction relies on the quality of Grad-CAM, which may be insufficiently accurate for certain architectures (e.g., very deep networks).
3. **Threshold sensitivity**: Experiments show that ASR is sensitive to the threshold $\eta$, and the optimal value may vary with different model combinations.
4. **Defense adaptability**: If defenders are aware of this attack strategy, they may design defenses that harden non-attention regions.
5. **Limited to image classification**: Effectiveness on more complex visual tasks such as object detection and semantic segmentation has not been verified.

## Related Work & Insights

- **Ensemble attacks**: AdaEA (adaptive fusion) and SMER (reinforcement learning-based reweighting) are the most closely related baselines; NAMEA surpasses them from a novel perspective of gradient decoupling and fusion.
- **Attention mechanisms in adversarial attacks**: Methods such as ATA and AoA focus on disrupting attention regions; this paper takes the opposite approach by focusing on non-attention regions.
- **Meta-learning in adversarial attacks**: MGAA employs meta-learning to simulate white-box/black-box attacks; NAMEA applies meta-learning to fuse attention and non-attention gradients.
- **Implications for defense research**: The transferability of non-attention regions suggests that current defenses may over-focus on protecting attention regions, necessitating more comprehensive robustness designs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (The concept of "ensemble non-attention" is novel; this is the first work to systematically demonstrate the adversarial transferability value of non-attention regions.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (17 target models + 6 defended models + 9 defense methods + 3 APIs + multiple rounds of ablation.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; the motivation–observation–method logical chain is complete.)
- Value: ⭐⭐⭐⭐ (Opens a new direction for cross-architecture adversarial transferability with strong practical utility.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)
- [\[CVPR 2026\] MyoVision: A Mobile Research Tool and NEATBoost-Attention Ensemble Framework for Real Time Chicken Breast Myopathy Detection](../../CVPR2026/others/myovision_a_mobile_research_tool_and_neatboost_attention_ensemble_framework.md)
- [\[AAAI 2026\] Parameterized Approximation Algorithms for TSP on Non-Metric Graphs](parameterized_approximation_algorithms_for_tsp_on_non-metric_graphs.md)
- [\[ICCV 2025\] NAPPure: Adversarial Purification for Robust Image Classification under Non-Additive Perturbations](../../ICCV2025/others/nappure_adversarial_purification_for_robust_image_classification_under_non-addit.md)
- [\[NeurIPS 2025\] Revisiting Agnostic Boosting](../../NeurIPS2025/others/revisiting_agnostic_boosting.md)

</div>

<!-- RELATED:END -->
