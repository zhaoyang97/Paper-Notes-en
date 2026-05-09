---
title: >-
  [Paper Note] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning
description: >-
  [ICLR 2026][Model Compression][Continual Learning] Inspired by the mushroom body circuitry of *Drosophila*, FlyPrompt decomposes General Continual Learning (GCL) into two sub-problems—expert routing and expert capacity—and addresses them respectively with a Random Expanded Analytic Router (REAR) and Temporal-Ensemble Experts (TE2), achieving improvements of 11.23% / 12.43% / 7.62% on CIFAR-100 / ImageNet-R / CUB-200.
tags:
  - ICLR 2026
  - Model Compression
  - Continual Learning
  - Parameter-Efficient Fine-Tuning
  - Drosophila Neural System
  - Random-Expanded Routing
  - Temporal Ensemble
date: 2026-05-08
content_hash: ea871cf8943234c4
---

# FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2602.01976](https://arxiv.org/abs/2602.01976)
**Code**: [GitHub](https://github.com/FlyGCL)
**Area**: Model Compression / LLM Efficiency
**Keywords**: Continual Learning, Parameter-Efficient Fine-Tuning, Drosophila Neural System, Random-Expanded Routing, Temporal Ensemble

## TL;DR

Inspired by the mushroom body circuitry of *Drosophila*, FlyPrompt decomposes General Continual Learning (GCL) into two sub-problems—expert routing and expert capacity—and addresses them respectively with a Random Expanded Analytic Router (REAR) and Temporal-Ensemble Experts (TE2), achieving improvements of 11.23% / 12.43% / 7.62% on CIFAR-100 / ImageNet-R / CUB-200.

## Background & Motivation

**General Continual Learning (GCL)** requires intelligent systems to learn continuously from non-stationary, single-pass data streams without clear task boundaries. Compared to conventional continual learning, GCL poses more severe challenges: (1) rapid adaptation, (2) robust knowledge retention, and (3) efficient resource utilization under limited supervision and task ambiguity.

Recent parameter-efficient fine-tuning (PET) methods built upon pre-trained models (PTMs) have shown strong performance in continual learning; approaches such as L2P, DualPrompt, and CODA-P introduce trainable prompt experts to adapt PTMs. However, these methods face two fundamental challenges:

**Expert Routing Problem**: How to dynamically route inputs to appropriate experts without task labels or iterative training? Existing routers perform poorly under GCL's ambiguous task boundaries. Experiments confirm that methods such as DualPrompt and MVP exhibit low routing accuracy even after training.

**Expert Capacity Problem**: How to ensure expressive representation within each expert under sparse and imbalanced supervision? Even with an oracle router (perfect expert selection), existing methods still yield unsatisfactory accuracy, indicating that the representational quality of individual experts is also problematic.

**Biological Inspiration**: Despite possessing fewer than 100,000 neurons, *Drosophila* exhibits robust memory consolidation and context-aware behavior. Its mushroom body encodes sensory inputs via sparse random projections: projection neurons (PNs) connect randomly to Kenyon cells (KCs) with approximately 40-fold dimensional expansion, and distinct KC subregions display plasticity at different timescales (γ short-term / α′β′ intermediate / αβ long-term memory).

## Method

### Overall Architecture

FlyPrompt decomposes GCL into two sub-problems and solves them independently:

1. **REAR (Random Expanded Analytic Router)**: Simulates the sparse expansion circuitry of *Drosophila* to achieve fast, gradient-free instance-level expert selection.
2. **TE2 (Task-wise Experts with Temporal Ensemble)**: Employs exponential moving averages (EMA) at multiple timescales to capture knowledge, mimicking the compartmentalized consolidation mechanism of the mushroom body.

### Key Designs

**1. Random Expanded Analytic Router (REAR)**

The core idea of REAR is to leverage fixed random projections and closed-form analytic updates for expert assignment, requiring no gradient computation.

Given features $h = f(x)$ from a pre-trained backbone encoder, the method first applies random expansion: $\phi(x) = \sigma(hR)$, where $R$ is a fixed Gaussian random matrix of dimension $d \times M$ ($M > d$) and $\sigma$ denotes ReLU. This simulates the approximately 40-fold sparse expansion from PNs to KCs in *Drosophila*.

During online training, each expert $E_t$ is associated with task $t$. Two statistics are accumulated per batch:

- Gram matrix: $G \mathrel{+}= \Phi_i^\top \Phi_i$ (second-order feature correlation)
- Prototype matrix: $Q \mathrel{+}= \Phi_i^\top C_t$ (expert-level feature aggregation)

The router matrix is obtained via ridge regression in closed form: $\hat{U}^\top = (G + \lambda I)^{-1} Q$

The router matrix is computed only once at evaluation time. At inference, the expert is selected by $\arg\max\, \phi(x)\hat{U}^\top$.

**Theoretical Guarantee of REAR** (Theorem 1): The population excess risk decomposes into an approximation error (reducible by increasing $M$), an estimation variance (reducible by increasing $N$ or $\lambda$), and a regularization bias. With sufficiently large random expansion dimensionality and appropriate regularization, the misrouting probability can be made arbitrarily small.

Key distinction from methods such as RanPAC: REAR uses random projections solely for expert routing; each expert's prompt and classification head remain trainable. RanPAC, by contrast, directly uses ridge regression as the final classifier.

**2. Temporal-Ensemble Experts (TE2)**

Inspired by KC subtypes in *Drosophila*, each expert $E_t$ maintains a bank of $n$ EMA heads with decay rates $\{\alpha_j\}$.

During training, only the online head and prompt are updated. The loss function applies cross-entropy together with a non-parametric logit mask $m$: classes not present in the current batch are set to $-\infty$, suppressing predictions for unseen labels. After each update, the EMA heads are synchronized:

$$W_t^{(j)} \leftarrow \alpha_j \cdot W_t^{(j)} + (1 - \alpha_j) \cdot W$$

At inference, all $n+1$ heads (online + EMA) are ensembled; after computing softmax for each head, an element-wise maximum is taken:

$$\hat{z}(x) = \max_j \operatorname{softmax}(z^{(j)} + m)$$

**New Task Initialization**: The prompt for a new expert is initialized as the mean of all previously learned prompts, accelerating convergence under GCL's limited-data regime.

**Theoretical Guarantee of TE2** (Theorem 2): The parameter error of EMA heads satisfies a variance–bias decomposition. The geometric EMA bank always contains a head that closely approximates the optimal bias–variance trade-off. In practice, two EMA heads ($\alpha = 0.9$ and $0.99$, corresponding to windows of 10 and 100) suffice.

### Loss & Training

- Standard cross-entropy loss is used to train the online head and prompt.
- A non-parametric logit mask suppresses predictions for classes absent from the current batch, mitigating both cross-task and intra-task class imbalance.
- The REAR router matrix is computed once at evaluation via a closed-form solution over accumulated statistics.
- No replay buffer or distillation loss is required.
- Prompts are warm-started using the mean of historical prompts.

## Key Experimental Results

### Main Results

**Table 1: GCL Benchmark Performance (Sup-21K Pre-training)**

| Method | CIFAR-100 $A_\text{auc}$ | CIFAR-100 $A_\text{last}$ | ImageNet-R $A_\text{auc}$ | ImageNet-R $A_\text{last}$ | CUB-200 $A_\text{auc}$ | CUB-200 $A_\text{last}$ |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| L2P | 76.23 | 79.11 | 44.40 | 42.03 | 64.30 | 61.42 |
| DualPrompt | 76.04 | 76.62 | 46.13 | 40.80 | 65.03 | 62.43 |
| CODA-P | 79.13 | 80.91 | 51.87 | 48.09 | 66.01 | 62.90 |
| MVP | 67.74 | 63.22 | 39.50 | 32.63 | 54.69 | 50.07 |
| MISA | 80.35 | 80.75 | 51.52 | 45.08 | 65.40 | 60.20 |
| **FlyPrompt** | **83.24** | **86.76** | **56.58** | **55.27** | **70.64** | **73.40** |

FlyPrompt outperforms all baselines across the board, with particularly substantial gains in $A_\text{last}$ (CIFAR-100: +5.85%, ImageNet-R: +7.18%, CUB-200: +10.50%).

**Table 2: Generalization across Pre-trained Models (iBOT-21K)**

| Method | CIFAR-100 $A_\text{auc}$ | ImageNet-R $A_\text{auc}$ | CUB-200 $A_\text{auc}$ |
|--------|:---:|:---:|:---:|
| CODA-P | 62.13 | 45.50 | 17.72 |
| MISA | 65.30 | 40.94 | 18.62 |
| **FlyPrompt** | **75.58** | **57.75** | **28.86** |

FlyPrompt maintains a large margin even when evaluated on self-supervised pre-trained models.

### Ablation Study

Ablation experiments validate the contribution of each core component:

- Removing REAR (replacing with alternative routing strategies): routing accuracy drops significantly.
- Removing TE2 (using a single head): $A_\text{last}$ degrades noticeably.
- Removing the logit mask: performance deteriorates in class-imbalanced scenarios.
- Number of EMA heads: two EMA heads ($\alpha = 0.9$, $0.99$) achieve the best performance.

### Key Findings

1. **Routing accuracy is a bottleneck**: Existing methods achieve far below ideal routing accuracy in the GCL setting; REAR substantially improves routing precision via fixed random projections and closed-form solutions.
2. **Expert capacity matters equally**: Even with an oracle router, existing methods remain suboptimal; TE2 effectively enhances the robustness of individual experts through multi-timescale EMA heads.
3. **Cross-PTM generalization**: FlyPrompt is effective across diverse pre-trained models (Sup-21K, iBOT-21K, DINO-1K, MoCo v3-1K, etc.).
4. **Routing via forward pass only**: REAR requires no gradient updates, making it well-suited to the single-pass online constraint of GCL.
5. **CKA analysis** confirms that different experts do specialize in distinct feature subspaces.

## Highlights & Insights

- GCL is cleanly decomposed into two orthogonal sub-problems—expert routing and expert capacity—yielding a more structured analytical framework than direct end-to-end design.
- The biological analogy to the *Drosophila* mushroom body is apt: sparse random expansion corresponds to REAR, and multi-timescale plasticity corresponds to TE2.
- REAR's closed-form solution is computed only at evaluation time; training requires only the accumulation of sufficient statistics, making the computational overhead negligible.
- Theoretical guarantees are provided for both REAR and TE2, grounding the contributions beyond purely empirical improvements.
- The method remains effective under extreme settings (e.g., DINO-1K pre-training on CUB-200).

## Limitations & Future Work

1. Validation is primarily on visual classification tasks; extension to NLP and multimodal settings is needed.
2. The number of experts scales linearly with the number of tasks, potentially causing linear parameter growth at large task counts.
3. While the Si-Blurry benchmark is the standard GCL protocol, it may not fully reflect certain real-world deployment scenarios.
4. The random projection dimension $M$ must be specified in advance and may require tuning for problems of different scales.
5. The current framework assumes tasks arrive in sessions; performance in purely streaming scenarios requires further investigation.

## Related Work & Insights

FlyPrompt demonstrates the potential of neuroscience-inspired AI design (NeuroAI) in continual learning. REAR's random expansion with closed-form routing is generalizable to expert routing in Mixture-of-Experts (MoE) architectures; TE2's multi-timescale EMA is also applicable to online learning, federated learning, and other non-stationary data settings.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (biological inspiration + problem decomposition + theoretical grounding)
- Technical Depth: ⭐⭐⭐⭐⭐ (complete theoretical analysis + closed-form routing + multi-timescale ensemble)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets × 6+ PTMs, detailed ablations)
- Practicality: ⭐⭐⭐⭐ (gradient-free router is deployment-friendly)
- Writing Quality: ⭐⭐⭐⭐⭐ (rigorous problem analysis, apt biological analogy)

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] LD-MoLE: Learnable Dynamic Routing for Mixture of LoRA Experts](ld-mole_learnable_dynamic_routing_for_mixture_of_lora_experts.md)
- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](ider_idempotent_experience_replay_for_reliable_continual_learning.md)
- [\[ICLR 2026\] Null-Space Filtering for Data-Free Continual Model Merging: Preserving Stability, Promoting Plasticity](null-space_filtering_for_data-free_continual_model_merging_preserving_stability_.md)
- [\[ICLR 2026\] Knowledge Fusion of Large Language Models Via Modular Skillpacks](knowledge_fusion_of_large_language_models_via_modular_skillpacks.md)
- [\[ICLR 2026\] ABBA-Adapters: Efficient and Expressive Fine-Tuning of Foundation Models](abba-adapters_efficient_and_expressive_fine-tuning_of_foundation_models.md)

<!-- RELATED:END -->
