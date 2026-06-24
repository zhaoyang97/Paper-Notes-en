---
title: >-
  [Paper Note] MTL-UE: Learning to Learn Nothing for Multi-Task Learning
description: >-
  [ICML 2025][Self-Supervised Learning][Unlearnable Examples] MTL-UE is the first unlearnable example generation framework tailored for Multi-Task Learning (MTL). Utilizing an encoder-decoder architecture to inject task-specific class prior embeddings, it reduces the intra-class variance of shortcut features. Coupled with intra- and inter-task embedding cosine regularization, it increases inter-class distances and reduces redundancy. On CelebA (40 tasks)…
tags:
  - "ICML 2025"
  - "Self-Supervised Learning"
  - "Unlearnable Examples"
  - "Multi-Task Learning"
  - "Data Poisoning"
  - "Privacy Protection"
  - "Class Embedding Regularization"
date: 2026-05-08
content_hash: afe7da74a4439327
---

# MTL-UE: Learning to Learn Nothing for Multi-Task Learning

**Conference**: ICML 2025  
**arXiv**: [2505.05279](https://arxiv.org/abs/2505.05279)  
**Code**: None  
**Area**: Self-Supervised  
**Keywords**: Unlearnable Examples, Multi-Task Learning, Data Poisoning, Privacy Protection, Class Embedding Regularization

## TL;DR

MTL-UE is the first unlearnable example generation framework tailored for Multi-Task Learning (MTL). Utilizing an encoder-decoder architecture to inject task-specific class prior embeddings, it reduces the intra-class variance of shortcut features. Coupled with intra- and inter-task embedding cosine regularization, it increases inter-class distances and reduces redundancy. On CelebA (40 tasks), it degrades MTL model accuracy from 91% to 59%, demonstrating consistent effectiveness across 4 datasets, 3 base UE methods, 5 backbones, and 5 MTL strategies.

## Background & Motivation

**Background**: Unlearnable Examples (UEs) protect data from unauthorized training by adding imperceptible perturbations ($\|\delta\|_\infty \leq 8/255$) to training data, thereby forcing models to learn shortcut features instead of genuine features. Existing methods fall into two categories: surrogate-dependent methods (EM, TAP, SEP) that require optimization of perturbations on a sample-by-sample basis using a surrogate model, and surrogate-free methods (LSP, AR) that utilize predefined class-specific patterns.

**Limitations of Prior Work**: All existing UE methods only target Single-Task Learning (STL), whereas in practice, datasets are increasingly oriented toward Multi-Task Learning (MTL) — such as the 40 facial attribute classification tasks in CelebA, or semantic segmentation + depth estimation + surface normal estimation in NYUv2. Applying UEs to MTL data faces unique challenges: (1) multiple tasks' shortcut features must be encoded simultaneously within a single perturbation; (2) an increasing number of tasks leads to an exponential explosion of label combinations ($\prod_k C_k$ combinations), which causes surrogate-free patch-based methods to fail as patch sizes shrink continuously.

**Key Challenge**: Surrogate-dependent UEs perform sample-by-sample independent optimization $\rightarrow$ uncontrollable intra-class variance $\rightarrow$ inconsistent shortcut features $\rightarrow$ models struggle to learn the shortcuts. For surrogate-free patch-based methods, increasing tasks $\rightarrow$ smaller patches $\rightarrow$ degraded representation capacity $\rightarrow$ failed attacks. Both approaches systematically fail under MTL settings.

**Goal**: (1) To design a unified framework that fuses shortcut features across multiple tasks into a single perturbation; (2) to effectively reduce intra-class variance to enhance attack performance; and (3) to remain effective against both MTL and STL models.

**Key Insight**: Key observation — intra-class variance is the decisive factor for the effectiveness of UEs. The authors find that patch-based AR has the lowest intra-class standard deviation (20.59) and the strongest attack performance, whereas EM/TAP/SEP exhibit high intra-class standard deviations of 82–103, which is close to clean data, rendering their attacks ineffective.

**Core Idea**: Substituting pixel-level perturbation search with a generator injected with learnable task-class embeddings. This reduces the search space, allowing the shortcut features of samples from the same class to align naturally.

## Method

### Overall Architecture

Given a multi-task dataset $\mathcal{T}=\{(\mathbf{x}_i, \{y_i^k\}_{k=1}^K)\}_{i=1}^N$, MTL-UE generates perturbations $\delta_i$ using an encoder-decoder network and learnable class embeddings, transforming clean data into unlearnable data $\mathcal{P}=\{(\mathbf{x}_i+\delta_i, \{y_i^k\}_{k=1}^K)\}$. The perturbation generation process is as follows: the encoder $E$ maps the input to a latent variable $\mathbf{z}$; based on the sample's $K$ task labels, it retrieves the corresponding embeddings $\{e_{y^k}^k\}_{k=1}^K$; then $[\mathbf{z}, e_{y^1}^1, \ldots, e_{y^K}^K]$ are concatenated and fed into the decoder $D$ to generate the perturbation, which is finally clipped to $[-\epsilon, \epsilon]$.

### Key Designs

1. **Encoder-Decoder Perturbation Generator + Class Embedding Injection**

    - **Function**: Generate input-conditioned perturbations that carry task-label prior information.
    - **Mechanism**: The encoder $E(\mathbf{x}; \phi_E)$ extracts the latent representation $\mathbf{z}$ of the input. Learnable embeddings $e_c^k \in \mathbb{R}^{d_e}$ are maintained for each class $c$ of each task $k$. Based on the sample's label combination, the corresponding embeddings are retrieved, concatenated, and fed into the decoder: $\delta = \text{Clip}(D([\mathbf{z}, e_{y^1}^1, \ldots, e_{y^K}^K]; \phi_D), -\epsilon, \epsilon)$. The total number of embeddings is $\sum_k C_k$ (linear growth) instead of $\prod_k C_k$ (exponential growth), avoiding combinatorial explosion.
    - **Design Motivation**: Shifting the search space from pixel-level $\|\delta\|_\infty \leq \epsilon$ down to the decoder's output space ensures that samples of the same class are guided by the same embedding, naturally reducing intra-class variance. Training the generator on the entire dataset allows it to capture global patterns, outperforming sample-by-sample optimization. The concatenation strategy naturally merges multi-task information into a unified perturbation.

2. **Intra-task Embedding Regularization (Intra-task ER)**

    - **Function**: Maximize the directional disparity between different class embeddings within the same task.
    - **Mechanism**: Minimize the cosine similarity between all pairs of class embeddings within the same task: $\mathcal{L}_{Intra} = \frac{2}{\sum_k C_k(C_k-1)} \sum_k \sum_{m<n} \cos(e_m^k, e_n^k)$.
    - **Design Motivation**: Larger inter-class distances among shortcut features make it easier for models to learn the shortcuts. However, directly increasing the $L_2$ distance may be neutralized by the decoder's rescaling. Cosine similarity measures directional disparity, making it invariant to scaling.

3. **Inter-task Embedding Regularization (Inter-task ER)**

    - **Function**: Promote geometric independence among embeddings of different tasks.
    - **Mechanism**: Minimize the absolute value of the cosine similarity between embeddings of different tasks: $\mathcal{L}_{Inter} = \frac{1}{\sum_{k<l} C_k C_l} \sum_{k<l} \sum_m \sum_n |\cos(e_m^k, e_n^l)|$.
    - **Design Motivation**: The absolute value ensures that both positive and negative correlations are penalized, aiming to orthogonalize the embedding spaces of different tasks. This reduces redundancy (each embedding carries unique information), decreases coupling (allowing the decoder to process each task independently), and enhances interpretability.

### Dense Prediction Extension

For dense prediction datasets such as NYUv2, class embeddings are replaced with an embedding module $\mathcal{E}^k(\mathbf{y}^k; \phi_{\mathcal{E}^k})$ to map dense labels to embeddings. Since the shortcut feature redundancy in dense prediction is low, embedding regularization is not applied.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_b(F'_{MTL}, \mathbf{x}+\delta, \mathbf{y}) + \lambda_1 \mathcal{L}_{Intra} + \lambda_2 \mathcal{L}_{Inter}$$

Where $\mathcal{L}_b$ is the loss of the base UE method. If the base method requires training a surrogate model, the optimization is performed alternately. The framework is plug-and-play and can be integrated with any surrogate-dependent UE method, such as EM, TAP, or SEP.

## Key Experimental Results

### CelebA (40 Binary Classification Tasks, ResNet-18)

| Method | MTL Avg Acc↓ | STL Avg Acc↓ |
|------|-------------|-------------|
| Clean (No Attack) | 91.11 | 90.35 |
| AR (Patch) | 73.12 | 84.41 |
| EM | 75.66 | 89.91 |
| TAP | 85.24 | 87.00 |
| SEP | 84.25 | 89.91 |
| **MTL-UE + EM** | **74.38** | **74.26** |
| **MTL-UE + TAP** | **59.51** | **68.65** |
| **MTL-UE + SEP** | **63.77** | **72.51** |

### Cross-Dataset Verification

| Dataset | Task Type | Clean | MTL-UE Performance |
|--------|---------|-------|-----------|
| ChestX-ray14 | 14 diseases (AUC-ROC) | 0.7577 | 0.4813 |
| UTKFace | Age/Race/Gender | 78.97 | 25.84 (MTL) |
| NYUv2 | Segmentation/Depth/Normal | - | All tasks degrade |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Intra-ER | Degraded attack | Reduced directional differences between classes |
| w/o Inter-ER | Degraded attack | Increased embedding redundancy |
| w/o Generator (Direct EM) | 75.66→74.38 | Search space reduction is key |
| Across 5 backbones | Consistently effective | High generalization capability |
| Across 5 MTL strategies | Consistently effective | Strategy-agnostic |

### Key Findings

- **MTL-UE+TAP degrades the MTL accuracy on CelebA from 91.11% to 59.51%**, significantly outperforming all baseline methods.
- **Intra-class standard deviation analysis (Tab. 1)**: AR (Patch) maximum is 20.59 $\rightarrow$ MTL-UE is even lower; EM/TAP/SEP maximums are 82–103 $\rightarrow$ close to clean data (91.97), which explains their ineffective attacks.
- STL models are harder to attack (since MTL shared representations make it easier to learn shortcuts), but MTL-UE achieves effective attacks on STL for the first time (degrading STL Avg from 90%+ to 68–74%).
- Supporting partial protection: selectively rendering specified tasks unlearnable while keeping others normally learnable.

## Highlights & Insights

- **Core insight of search space reduction**: Restricting the search space from pixel-level $\epsilon$-balls down to the decoder's output space simultaneously addresses both the intra-class variance and multi-task fusion challenges. Injecting embeddings as "class signals" elegantly circumvents the label combinatorial explosion — requiring only $\sum_k C_k$ instead of $\prod_k C_k$.
- **A complete research trajectory from baseline analysis to method design**: Designing a systematic baseline benchmark first to reveal the essence of the problem (intra-class variance), then engineering a targeted solution, and finally verifying it via large-scale experiments — providing a comprehensive solution for MTL data protection.

## Limitations & Future Work

- Requiring surrogate MTL model training, where attackers must assume knowledge of the target model's architecture.
- The robustness under defenses like adversarial training is not fully evaluated.
- The impact of embedding dimension $d_e$ on performance requires a more systematic analysis.
- The encoder-decoder capacity might limit the effectiveness on high-resolution images.
- The paper might be more appropriately classified under AI Safety/Data Protection.

## Related Work & Insights

- **vs. EM (Huang et al., 2021)**: EM optimizes sample-by-sample, yielding uncontrollable intra-class variance, whereas MTL-UE drastically reduces it via the generator + embedding injection.
- **vs. AR Patch (Sandoval-Segura et al., 2022)**: AR Patch is effective with fewer tasks but fails as patch sizes shrink with more tasks; MTL-UE's embedding injection is unconstrained by the number of tasks.
- **vs. TAP (Fowl et al., 2021)**: TAP exhibits high intra-class variance (Tab. 1 maximum of 103), leading to weak attack performance (85.24%), whereas MTL-UE + TAP improves it to 59.51%.
- **Insights**: Generative approaches (using a network to generate perturbations rather than directly optimizing pixel values) can be generalized to more controllable data perturbation scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The first MTL-UE framework. The design of search space reduction and embedding regularization is ingenious, with a pioneering problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ A large-scale experimental matrix spanning 4 datasets $\times$ 3 methods $\times$ 5 backbones $\times$ 5 strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from baseline analysis to motivation to method, though the mathematical notation is slightly heavy.
- Value: ⭐⭐⭐⭐ MTL data protection represents a genuine demand in the context of large-scale model data scraping.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](../../NeurIPS2025/self_supervised/mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] Soft Task-Aware Routing of Experts for Equivariant Representation Learning](../../NeurIPS2025/self_supervised/soft_task-aware_routing_of_experts_for_equivariant_representation_learning.md)
- [\[CVPR 2025\] Task-Agnostic Guided Feature Expansion for Class-Incremental Learning](../../CVPR2025/self_supervised/task-agnostic_guided_feature_expansion_for_class-incremental_learning.md)
- [\[ICML 2025\] Update Your Transformer to the Latest Release: Re-Basin of Task Vectors](update_your_transformer_to_the_latest_release_re-basin_of_task_vectors.md)
- [\[NeurIPS 2025\] STaRFormer: Semi-Supervised Task-Informed Representation Learning via Dynamic Attention-Based Regional Masking](../../NeurIPS2025/self_supervised/starformer_semi-supervised_task-informed_representation_learning_via_dynamic_att.md)

</div>

<!-- RELATED:END -->
