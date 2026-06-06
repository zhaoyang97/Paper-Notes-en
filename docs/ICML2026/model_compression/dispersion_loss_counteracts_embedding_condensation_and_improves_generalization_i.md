---
title: >-
  [Paper Note] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models
description: >-
  [ICML 2026][Model Compression][Embedding Condensation] This work systematically observes the prevalent phenomenon of "token embeddings in small language models condensing into a narrow cone with depth" (embedding condens…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Embedding Condensation"
  - "Dispersion Loss"
  - "Small Model Generalization"
  - "Knowledge Distillation"
  - "GPT2 / Qwen3"
date: 2026-05-08
content_hash: 13aecb4eaeb65c3b
---

# Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.00217](https://arxiv.org/abs/2602.00217)  
**Code**: https://github.com/KrishnaswamyLab/LM-Dispersion  
**Area**: Model Compression / Representation Learning / Small Model Training  
**Keywords**: Embedding Condensation, Dispersion Loss, Small Model Generalization, Knowledge Distillation, GPT2 / Qwen3

## TL;DR
This work systematically observes the prevalent phenomenon of "token embeddings in small language models condensing into a narrow cone with depth" (embedding condensation)—a phenomenon not seen in large models—and proposes an angular dispersion loss $\mathcal{L}_{\text{disp}}$ that directly encourages embedding dispersion. Without introducing extra parameters, this loss yields an average improvement of 3.3% on 10 benchmarks for Qwen3 / GPT2.

## Background & Motivation
**Background**: The capabilities of LLMs scale with size, but training and deployment costs are soaring, creating an urgent need to "replicate key properties of large models with small models." Existing compression approaches—distillation, quantization, pruning—mainly focus on mimicking the output distribution of large models.

**Limitations of Prior Work**: From a representation geometry perspective, the authors find that token embeddings in small models (GPT2-small, Qwen3-0.6B) become nearly aligned in deeper layers, with pairwise cosine similarity approaching 1; large models (GPT2-xl, Qwen3-32B) maintain dispersed embeddings. Geshkovski 2025 theoretically shows that Transformer embeddings collapse to a point as depth increases, but no one has systematically verified the empirical relationship to performance.

**Key Challenge**: Embedding condensation means the model's effective "representation directions" are reduced, geometrically constraining expressiveness. Even if distillation teaches the logit distribution from a large teacher, it cannot inherit the geometric properties—since distillation only constrains outputs, not intermediate embeddings.

**Goal**: (1) Quantitatively measure embedding condensation and confirm that "large models resist condensation" is a universal pattern; (2) verify that distillation does not alleviate condensation; (3) design an auxiliary loss that directly acts on geometry to encourage dispersion in small models.

**Key Insight**: Since large models "automatically" maintain dispersion, dispersion itself may be a bottleneck for performance. Rather than increasing parameters to achieve "natural" dispersion, it is preferable to explicitly add an objective to enforce it.

**Core Idea**: An angular dispersion loss $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-\arccos(\cos\text{sim}(z_i, z_j)) / \pi\tau)$ pushes all token embeddings toward a uniform distribution on the unit hypersphere, with zero additional parameters.

## Method

### Overall Architecture
The method consists of two stages: (1) Diagnosis—quantifying embedding condensation; (2) Intervention—adding the dispersion loss during training. Diagnosis uses Spearman $\rho$ and Kendall $\tau$ to measure the monotonic increase of layer-wise mean cosine similarity. Intervention adds $\mathcal{L}_{\text{disp}}$ as a regularizer to the original training objective: $\mathcal{L} = \mathcal{L}_{\text{train}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$.

### Key Designs

1. **Angular Dispersion Loss (core dispersion loss)**:

    - **Function**: Pushes all token embeddings apart on the unit hypersphere.
    - **Mechanism**: For each token pair $(z_i, z_j)$ in each layer, cosine similarity is mapped to angular distance $D(z_i, z_j) = \arccos(\cos\text{sim}(z_i, z_j)) / \pi \in [0, 1]$; then aggregated via log-sum-exp: $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-D(z_i, z_j)/\tau)$. When distance is small (aligned), the $\exp$ term is large and the loss is high, pushing them apart; when distance is large (nearly orthogonal), the term is near zero. Losses are summed across layers; per batch complexity is $\mathcal{O}(N^2 F)$, which can be reduced by subsampling tokens.
    - **Design Motivation**: (1) Using $\arccos$ instead of raw cosine ensures numerical stability, avoiding saturation at the ends; (2) log-sum-exp is more robust than mean, and the additive constant does not affect gradients; (3) explicitly excluding diagonal terms prevents exploding gradients from self-similarity; (4) angular rather than Euclidean distance is chosen because condensation is fundamentally about direction, not magnitude.

2. **Three Alternative Formulations (for ablation)**:

    - **Function**: To test whether different implementations of "dispersion" are effective, isolating the advantages of angular dispersion over other variants.
    - **Mechanism**: (a) Decorrelation—minimizes off-diagonal elements of the embedding covariance matrix, indirectly reducing coupling between feature dimensions; (b) $\ell_2$-repel—directly increases Euclidean distance between tokens, but requires norm regularization $\lambda_{\text{norm}} \|\mathcal{Z}\|_2^2$ to prevent cheating by norm inflation; (c) Orthogonalization—uses a hinge loss $\max(0, 1/2 - D(z_i, z_j))^2$, penalizing only pairs with distance < 1/2 (acute angles), allowing obtuse pairs to grow freely.
    - **Design Motivation**: Dispersion is an abstract goal; by comparing four implementations, the authors show that "uniform angular dispersion" is more direct and effective than "decorrelation in feature space" or "repulsion in Euclidean space," reinforcing the rationale for the main method.

3. **Application Strategies for Mid-training + Full Pre-training**:

    - **Function**: Embeds the dispersion loss into two practical training regimes, demonstrating applicability for both retrofitting existing models and training from scratch.
    - **Mechanism**: Mid-training—continue training existing GPT2 / Qwen3 on wikitext-103 for 200M tokens, runnable on a single A100; Full pre-training—train Qwen3 from scratch on C4 for 156B tokens using 640 GPUs. In both scenarios, $\lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$ is added to cross-entropy, and the loss computes dispersion across multiple layers per forward pass.
    - **Design Motivation**: Mid-training serves as a low-cost proof-of-concept and hyperparameter search; full pre-training tests whether the signal can shape better geometry from scratch, fundamentally altering model capacity.

### Loss & Training
The training objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. Temperature $\tau$ and weight $\lambda_{\text{disp}}$ are the main hyperparameters, with scan results provided in the appendix. Mid-training uses 3 seeds for mean and variance reporting; full pre-training uses a single seed but with sufficient token volume for stability.

## Key Experimental Results

### Main Results
GPT2 mid-training (average over 10 benchmarks):

| Configuration | Training Cost | Avg. Score↑ | Rank↓ | Significance |
|---------------|--------------|-------------|-------|--------------|
| GPT2 original (no mid-training) | — | 34.35 | 6.1 | p<0.0001 |
| + $\mathcal{L}_{\text{CE}}$ only | 1.122 A100h | 34.95 | 6.2 | p<0.01 |
| + noisy embedding | 1.122 | 35.15 | 4.3 | p<0.01 |
| + active forgetting | 1.127 | 35.36 | 3.2 | n.s. |
| **+ Dispersion loss** | 1.13 (1.01×) | **35.52+** | **Best** | — |

Qwen3 full pre-training (156B tokens from scratch): Adding dispersion loss yields an average improvement of +1.17 points (3.3% relative gain), with consistent gains across all benchmarks.

### Ablation Study
Comparison of four dispersion variants:

| Variant | Avg. Score | Notes |
|---------|------------|-------|
| Decorrelation | 35.1 | Indirect, affected by feature dimension |
| $\ell_2$-repel | 35.0 | Requires norm regularization for stability |
| Orthogonalization | 35.2 | Penalizes only acute angles |
| **Dispersion (canonical)** | **35.5+** | Uniform angular dispersion, optimal |

Scale control (confounder-controlled): Four GPT2-like models trained from scratch, only varying MLP dimension with other factors fixed; larger MLP → less condensation, confirming that "large models resist condensation" is not a spurious correlation from other factors.

### Key Findings
- **Distillation does not alleviate condensation**: After distillation, Qwen2.5 series embeddings are geometrically similar to those trained from scratch, since KD loss only constrains output logits, not intermediate representations—this is the most direct motivation for the paper.
- **Condensation exists at initialization but is mitigated by training**: Olmo-3-7B checkpoints show condensation metrics are initially positive and large, decreasing to negative with training, indicating SGD inherently resists condensation, with dispersion loss accelerating and reinforcing this effect.
- **Greater gains for small models**: Qwen3-0.6B shows the most improvement, Qwen3-32B almost none, consistent with the hypothesis that "large models are already dispersed."
- **Effective with mid-training**: No need for retraining; simply adding 200M tokens + dispersion to existing checkpoints yields significant gains at minimal cost.
- **<1% training time overhead**: 1.13 vs 1.122 A100h, as $N^2$ pairing can be subsampled.

## Highlights & Insights
- **"The bottleneck for small models is geometry, not parameters"**: Attributing performance gaps to representation geometry rather than capacity is the most imaginative proposition of the paper—implying that large-model performance can be approached without increasing parameters.
- **Angular rather than Euclidean distance**: Using $\arccos$ to map cosine to uniform angular distance avoids saturation at the ends, a crucial engineering stabilization trick.
- **Theory → Empirics → Intervention → Verification loop**: Starting from Geshkovski 2025's theoretical condensation theorem, providing large-scale empirical evidence, designing interventions, and confirming with confounder-controlled experiments, the argumentation chain is highly complete.
- **Zero parameter overhead**: Unlike pruning, quantization, or LoRA, which require architectural changes, dispersion loss is an auxiliary training term, plug-and-play with any mainstream LM training pipeline.

## Limitations & Future Work
- **Not extended to alignment/reasoning tasks**: Experiments are all on zero/few-shot general NLU, not on reasoning/math/code tasks that may require more complex representation geometry.
- **No gains for large models**: No improvement observed on 32B; whether the dispersion hypothesis holds for large models remains unconfirmed; their bottleneck may lie elsewhere.
- **Token subsampling cost**: For large models, $N^2$ still requires subsampling; the impact of subsampling on convergence and final performance is not thoroughly ablated.
- **No comparison with anti-collapse techniques (e.g., SimSiam stop-gradient, BarlowTwins)**: Many anti-collapse methods exist in self-supervised representation learning and should be compared horizontally.
- **No explanation for why $\lambda_{\text{disp}}$ does not require complex scheduling**: Intuitively, stronger dispersion may be needed early in training and less later, but the paper finds a fixed weight suffices—an interesting phenomenon not deeply analyzed.
- **Architectural sensitivity**: Effectiveness under RMSNorm (no affine LN) and different positional encodings is untested.

## Related Work & Insights
- **vs Wang & He 2025 (dispersion in diffusion)**: That work applies dispersion to generative models; this paper adapts it to language models, reformulates it in angular terms, and explicitly removes diagonal terms—a domain adaptation of the same idea.
- **vs noisy embedding / active forgetting**: These tricks also aim to increase representation diversity, but are indirect and lack geometric interpretation; dispersion is direct, interpretable, and more effective.
- **vs distillation-based compression**: Distillation only transfers output behavior, while dispersion transfers the more fundamental property of representation geometry; the two are complementary.
- **vs Cai 2021, Bis 2021 (isotropy studies)**: These works focus on isotropy/anisotropy in embedding space; this paper provides the first explicit training objective to control it.
- **Insights**: This approach can be extended to (1) patch token embeddings in vision encoders; (2) dispersion between different modality embeddings in multimodal alignment; (3) dispersion of MoE expert activation patterns to avoid expert collapse.

## Rating
- Novelty: ⭐⭐⭐⭐ Translates the theoretical condensation theorem into a trainable auxiliary loss, proposes four variants; a "known phenomenon + clean intervention" type of novelty—solid and credible, if not groundbreaking.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual validation with mid-training and full pre-training (156B tokens / 640 GPUs), 10 benchmarks, 3 seeds, confounder control, and 4 dispersion ablations—exceptionally rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ The argument chain "theory → empirical observation → counterexample (distillation ineffective) → intervention → verification" is very smooth, with clear figures and tables, making abstract geometric phenomena accessible.
- Value: ⭐⭐⭐⭐ Near-zero-cost, plug-and-play; directly benefits the small model training community. However, no gains for large models and untested on reasoning/code tasks, so long-term impact may be limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](../../ICLR2026/model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](../../NeurIPS2025/model_compression/reordering_patches_improves_vision_models.md)
- [\[ICLR 2026\] Embedding Compression via Spherical Coordinates](../../ICLR2026/model_compression/embedding_compression_via_spherical_coordinates.md)
- [\[CVPR 2026\] Beyond Loss Values: Robust Dynamic Pruning via Loss Trajectory Alignment](../../CVPR2026/model_compression/beyond_loss_values_robust_dynamic_pruning_via_loss_trajectory_alignment.md)
- [\[ICML 2026\] Resting Neurons, Active Insights: Robustify Activation Sparsity for Large Language Models](resting_neurons_active_insights_robustify_activation_sparsity_for_large_language.md)

</div>

<!-- RELATED:END -->
