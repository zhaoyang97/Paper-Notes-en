---
title: >-
  [Paper Note] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models
description: >-
  [ICML 2026][Model Compression][Embedding condensation] This paper systematically observes the universal phenomenon of "embedding condensation…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Embedding condensation"
  - "dispersion loss"
  - "small model generalization"
  - "knowledge distillation"
  - "GPT2 / Qwen3"
date: 2026-05-08
content_hash: e76390a1a130ad11
---

# Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.00217](https://arxiv.org/abs/2602.00217)  
**Code**: https://github.com/KrishnaswamyLab/LM-Dispersion  
**Area**: Model Compression / Representation Learning / Small Model Training  
**Keywords**: Embedding condensation, dispersion loss, small model generalization, knowledge distillation, GPT2 / Qwen3

## TL;DR
This paper systematically observes the universal phenomenon of "embedding condensation," where token embeddings in small language models collapse into a narrow cone as depth increases—a phenomenon absent in large models. The authors design an angular dispersion loss $\mathcal{L}_{\text{disp}}$ to force embeddings to spread out, achieving an average improvement of 3.3% across 10 benchmarks for Qwen3 / GPT2 without adding any parameters.

## Background & Motivation
**Background**: While LLM capabilities scale with size, the soaring costs of training and deployment necessitate "replicating the key properties of large models in smaller ones." Existing compression strategies like distillation, quantization, and pruning primarily focus on mimicking the output distribution of large models.

**Limitations of Prior Work**: From the perspective of representation geometry, the authors find that token embeddings of small models (GPT2-small, Qwen3-0.6B) in deep layers align almost entirely in the same direction, with pairwise cosine similarity approaching 1. In contrast, large models (GPT2-xl, Qwen3-32B) maintain dispersed embeddings. While theoretical work by Geshkovski (2025) proved that Transformer embeddings collapse to a point as the number of layers approaches infinity, the relationship between this phenomenon and performance has not been systematically verified.

**Key Challenge**: Embedding condensation implies that the "representation directions" available to the model are increasingly restricted, effectively locking its expressive power geometrically. Even if distillation learns logit distributions from a large teacher, it fails to inherit the geometric properties because distillation objectives constrain outputs rather than intermediate embeddings.

**Goal**: (1) Quantify the measurement of embedding condensation and confirm that "resistance to collapse" is a universal rule for large models; (2) Verify that distillation cannot alleviate this; (3) Design an auxiliary loss acting directly on the geometry to actively disperse small model embeddings.

**Key Insight**: Since large models "automatically" maintain dispersion, dispersion itself may be a bottleneck for performance. Rather than stacking parameters to allow the model to disperse "naturally," it is more effective to explicitly add an objective function to enforce dispersion.

**Core Idea**: Utilize an angular-based dispersion loss $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-\arccos(\cos\text{sim}(z_i, z_j)) / \pi\tau)$ to push all token embeddings toward a uniform distribution on the unit hypersphere with zero additional parameters.

## Method

### Overall Architecture
The method consists of two stages: (1) Diagnosis stage—quantifying embedding condensation; (2) Intervention stage—incorporating the dispersion loss into training. Diagnosis uses Spearman $\rho$ and Kendall $\tau$ to measure the monotonic increase in average layer-wise cosine similarity. Intervention treats $\mathcal{L}_{\text{disp}}$ as a regularization term added to the original training objective: $\mathcal{L} = \mathcal{L}_{\text{train}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$.

### Key Designs

1.  **Angular Dispersion Loss (Core Dispersion Loss)**:
    - **Function**: Pushes all token embeddings apart on the unit hypersphere.
    - **Mechanism**: For every token pair $(z_i, z_j)$ in each layer, the cosine similarity is first mapped to an angular distance $D(z_i, z_j) = \arccos(\cos\text{sim}(z_i, z_j)) / \pi \in [0, 1]$. These are aggregated via log-sum-exp: $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-D(z_i, z_j)/\tau)$. When distances are small (same direction), the $\exp$ term is large, resulting in a high loss that pushes them apart; when distances are large (near orthogonal), the loss is negligible. The sum of losses across all layers has a per-batch complexity of $\mathcal{O}(N^2 F)$, which can be reduced via sub-sampling of tokens.
    - **Design Motivation**: (1) Using $\arccos$ instead of raw cosine ensures numerical stability and avoids saturation at the boundaries; (2) Log-sum-exp is numerically more robust than the mean, and the additive constant does not affect gradients; (3) Explicitly excluding diagonal terms prevents gradients from exploding due to self-similarity; (4) Angular distance is chosen over Euclidean distance because condensation is fundamentally a directional issue rather than a magnitude issue.

2.  **Three Alternative Formulations (for Ablation)**:
    - **Function**: Validates whether different implementations of the "dispersion" goal are effective, isolating the advantages of "angular dispersion."
    - **Mechanism**: (a) Decorrelation—minimizing off-diagonal elements of the embedding covariance matrix to indirectly reduce coupling between feature dimensions; (b) $\ell_2$-repel—directly increasing Euclidean distance between tokens, requiring norm regularization $\lambda_{\text{norm}} \|\mathcal{Z}\|_2^2$ to prevent cheating via norm expansion; (c) Orthogonalization—a hinge loss $\max(0, 1/2 - D(z_i, z_j))^2$ that only penalizes distances $< 1/2$ (acute pairs).
    - **Design Motivation**: Dispersion is an abstract requirement; comparing these four implementations demonstrates that "uniform dispersion in angular space" is more direct and effective than "decorrelation in feature dimensions" or "repulsion in Euclidean space."

3.  **Application Strategy Covering Mid-training + Full Pre-training**:
    - **Function**: Integrates dispersion loss into two actual training workflows, proving it can both retrofit existing models and be used for training from scratch.
    - **Mechanism**: Mid-training—Existing GPT2 / Qwen3 models are continued on Wikitext-103 for 200M tokens; Full pre-training—Qwen3 is trained from scratch on 156B tokens of C4 using 640 GPUs. In both scenarios, $\lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$ is added to the cross-entropy loss, with the dispersion of multi-layer embeddings calculated during every forward pass.
    - **Design Motivation**: Mid-training serves as a low-cost proof-of-concept and hyperparameter search; full pre-training verifies that this signal can shape better geometry from the start, fundamentally altering model capacity.

### Loss & Training
The training objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. Temperature $\tau$ and weight $\lambda_{\text{disp}}$ are the primary hyperparameters. Mid-training reports mean and variance across 3 seeds, while full pre-training uses a single seed with sufficient token volume for stability.

## Key Experimental Results

### Main Results
GPT2 mid-training (Average over 10 benchmarks):

| Configuration | Training Cost | Avg Score↑ | Rank↓ | Significance |
| :--- | :--- | :--- | :--- | :--- |
| GPT2 Original (No mid-training) | — | 34.35 | 6.1 | p<0.0001 |
| + $\mathcal{L}_{\text{CE}}$ only | 1.122 A100h | 34.95 | 6.2 | p<0.01 |
| + Noisy embedding | 1.122 | 35.15 | 4.3 | p<0.01 |
| + Active forgetting | 1.127 | 35.36 | 3.2 | n.s. |
| **+ Dispersion loss** | 1.13 (1.01×) | **35.52+** | **Best** | — |

Qwen3 full pre-training (156B tokens from scratch): Adding dispersion loss yields an average improvement of +1.17 points (3.3% relative gain), with stable improvements across all benchmarks.

### Ablation Study
Comparison of four dispersion variants:

| Variant | Avg Score | Notes |
| :--- | :--- | :--- |
| Decorrelation | 35.1 | Indirect, affected by feature dim |
| $\ell_2$-repel | 35.0 | Requires norm regularization for stability |
| Orthogonalization | 35.2 | Only penalizes acute angles |
| **Dispersion (Canonical)** | **35.5+** | Uniform angular dispersion, optimal |

Scale Control (Confounder-controlled): Four GPT2-like models were trained from scratch with only MLP dimensions varied. Larger MLP $\to$ lower condensation, verifying that "large model resistance to collapse" is not a pseudo-correlation caused by other factors.

### Key Findings
- **Distillation fails to fix collapse**: The embedding geometry of the distilled Qwen2.5 series is nearly identical to models trained from scratch, as KD loss only constrains output logits without regulating intermediate representations.
- **Collapse exists at initialization but is mitigated by training**: Checkpoints of Olmo-3-7B show high condensation at initialization which decreases during training, suggesting SGD itself resists collapse, while dispersion loss accelerates and strengthens this process.
- **Small models benefit more**: The gain is most pronounced for Qwen3-0.6B, while Qwen3-32B shows almost no benefit, consistent with the hypothesis that "large models are already dispersed."
- **Mid-training is effective**: No full retraining is required; adding dispersion loss for 200M tokens to an existing checkpoint yields significant gains at extremely low cost.
- **Cost is < 1% of training time**: 1.13 vs 1.122 A100h, as $N^2$ pairs can be sub-sampled.

## Highlights & Insights
- **"Small model bottlenecks are geometric, not parametric"**: Attributing performance gaps to representation geometry rather than capacity is a highly imaginative premise, suggesting that large model performance can be approached without adding parameters.
- **Angles over Euclidean distance**: Mapping cosine to uniform angular distance via $\arccos$ avoids saturation issues, serving as a critical engineering trick for stability.
- **Closed-loop of Theory $\to$ Empirical $\to$ Intervention $\to$ Verification**: Starting from Geshkovski's (2025) theoretical collapse theorem, providing large-scale empirical evidence, designing an intervention, and confirming it with confounder-controlled experiments creates a highly complete chain of argument.
- **Zero parameter overhead**: Unlike pruning, quantization, or LoRA, which modify architecture, dispersion loss is a plug-and-play addition to any mainstream LM training pipeline.

## Limitations & Future Work
- **Not extended to alignment/reasoning tasks**: Experiments were limited to zero/few-shot general NLU; reasoning, math, and code tasks requiring complex representation geometry were not tested.
- **Negligible gains for large models**: No improvement seen for 32B models; it is unconfirmed if the dispersion hypothesis holds for them, or if their bottlenecks lie elsewhere.
- **Cost of token sub-sampling**: $N^2$ complexity necessitates sub-sampling for large models, yet the impact of sub-sampling on convergence and final performance was not thoroughly ablated.
- **Lack of comparison with anti-collapse techniques**: Comparisons with self-supervised methods like SimSiam stop-gradient or BarlowTwins are missing.
- **Unexplained $\lambda_{\text{disp}}$ scheduling**: Intuitively, stronger dispersion might be needed early in training, yet a fixed weight performed well without deep analysis.

## Related Work & Insights
- **vs. Wang & He 2025 (Dispersion in Diffusion)**: While that work used dispersion in generative models, this paper adapts it to LMs with an angular form and explicit diagonal exclusion.
- **vs. Noisy embedding / Active forgetting**: These tricks attempt to increase diversity indirectly without geometric explanation; dispersion is direct, interpretable, and more effective.
- **vs. Distillation-based compression**: Distillation transfers output behavior, while dispersion transfers the more fundamental property of representation geometry; the two can be stacked.
- **vs. Cai 2021, Bis 2021 (Isotropy research)**: While both focus on isotropy/anisotropy, this paper provides the first clear training objective to make it controllable.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Translating a theoretical collapse theorem into a trainable auxiliary loss with four variants is a clean and credible "known phenomenon + clean intervention" type of novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ High-standard validation across mid-training and full pre-training (156B tokens / 640 GPUs), 10 benchmarks, 3 seeds, and extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical flow is excellent, and diagrams are clear, presenting abstract geometric phenomena in a narrative style.
- **Value**: ⭐⭐⭐⭐ Immediately beneficial for the small model training community with near-zero cost; however, the lack of gain for large models and limited task variety ceilings its long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](../../ICLR2026/model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](../../NeurIPS2025/model_compression/reordering_patches_improves_vision_models.md)

</div>

<!-- RELATED:END -->
