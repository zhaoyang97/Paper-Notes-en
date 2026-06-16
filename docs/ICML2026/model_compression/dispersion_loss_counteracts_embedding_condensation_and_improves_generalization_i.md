---
title: >-
  [Paper Note] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] This paper systematically observes the universal phenomenon that "token embeddings in small language models collapse into a narrow cone as depth increases" (embedding condensation)—a behavior large models resist. It introduces an angular dispersion loss $\mathcal{L}_{\text{disp}}$ to explicitly force embedding dispersi
tags:
  - ICML 2026
  - Model Compression
  - Knowledge Distillation
  - GPT2 / Qwen3
date: 2026-05-08
content_hash: cc9b58b561f1c6b2
---
# Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.00217](https://arxiv.org/abs/2602.00217)  
**Code**: https://github.com/KrishnaswamyLab/LM-Dispersion  
**Area**: Model Compression / Representation Learning / Small Model Training  
**Keywords**: Embedding Collapse, Dispersion Loss, Small Model Generalization, Knowledge Distillation, GPT2 / Qwen3

## TL;DR
This paper systematically observes the universal phenomenon that "token embeddings in small language models collapse into a narrow cone as depth increases" (embedding condensation)—a behavior large models resist. It introduces an angular dispersion loss $\mathcal{L}_{\text{disp}}$ to explicitly force embedding dispersion, achieving an average 3.3% improvement for Qwen3 / GPT2 across 10 benchmarks without adding parameters.

## Background & Motivation
**Background**: As LLM capabilities scale with size, training and deployment costs skyrocket, creating an urgent need to "reproduce key large-model properties in small models." Existing compression routes—distillation, quantization, and pruning—primarily focus on mimicking the output distribution of large models.

**Limitations of Prior Work**: From a representation geometry perspective, the authors found that in small models (GPT2-small, Qwen3-0.6B), token embeddings in deep layers almost entirely align in the same direction, with pairwise cosine similarity approaching 1. Large models (GPT2-xl, Qwen3-32B) maintain embedding dispersion. While theoretical work by Geshkovski (2025) proved Transformers collapse to a point as layer count approaches infinity, the relationship between this phenomenon and performance has not been systematically verified.

**Key Challenge**: Embedding condensation implies that a model's available "representation directions" are severely limited, geometrically locking its expressive capacity. Even if knowledge distillation (KD) learns logit distributions from a teacher, it fails to inherit the large model's geometric properties because distillation targets only constrain the final output, not intermediate embeddings.

**Goal**: (1) Quantitatively measure embedding condensation and confirm "large-model resistance to collapse" as a universal law; (2) verify that distillation cannot alleviate this; (3) design an auxiliary loss acting directly on geometry to enforce embedding dispersion in small models.

**Key Insight**: Since large models "automatically" maintain dispersion, dispersion itself may be a bottleneck condition for performance. Rather than stacking parameters to allow "natural" dispersion, one can explicitly add an objective function to force it.

**Core Idea**: Use an angular-based dispersion loss $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-\arccos(\cos\text{sim}(z_i, z_j)) / \pi\tau)$ to push all token embeddings toward a uniform distribution on the unit hypersphere with zero additional parameters.

## Method

### Overall Architecture
The methodology consists of two stages: diagnosing the "geometric disease" and prescribing a direct intervention. In the diagnostic stage, embedding condensation is quantified using Spearman $\rho$ and Kendall $\tau$ to measure whether layer-wise average cosine similarity climbs monotonically with depth. In the intervention stage, an angular dispersion loss $\mathcal{L}_{\text{disp}}$ is attached as a regularizer to the training objective: $\mathcal{L} = \mathcal{L}_{\text{train}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. This requires zero extra parameters and can retrofit existing checkpoints or shape representations from scratch during pre-training.

### Key Designs

**1. Quantifying Condensation: Measuring the "Disease" via Rank Correlation**

To intervene effectively, the "disease" must be accurately measured. The authors process input sequences and collect token embeddings $z_i^{(l)}$ for each layer $l$. They compute all $N^2$ pairwise cosine similarities and summarize the alignment at each layer using the mean $\mu^{(l)} = \frac{1}{N^2}\sum_{i,j}\cos\text{sim}(z_i^{(l)}, z_j^{(l)})$. Finally, they compute the rank correlation (Spearman $\rho$ and Kendall $\tau$) between the sequence of means $\{\mu^{(l)}\}_{l=1}^L$ and the layer indices $\{l\}_{l=1}^L$. A $\rho/\tau$ close to $+1$ indicates monotonic collapse with depth; values near $0$ indicate no trend; negative values indicate dispersion. Using rank correlation instead of absolute mean values provides a robust metric unaffected by scale or non-linear distortions, allowing for clean cross-model comparisons.

**2. Angular Dispersion Loss: Pushing Embeddings Toward Hyperspherical Uniformity**

The loss targets the issue where small model embeddings align in a single direction. For each token pair $(z_i, z_j)$ at a given layer, cosine similarity is mapped to an angular distance $D(z_i, z_j) = \arccos(\cos\text{sim}(z_i, z_j)) / \pi \in [0, 1]$. These are aggregated via log-sum-exp: $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-D(z_i, z_j)/\tau)$. As embeddings align, $D$ decreases, the exponential term increases, and the loss pushes them apart. Several engineering details ensure stability: using $\arccos$ avoids gradient saturation at $\pm 1$; the log-sum-exp is more robust than a simple mean; and diagonal terms ($i=j$) are excluded to prevent gradient explosion.

**3. Three Alternative Formulas: Isolating the Benefit of Angular Dispersion**

The authors compared the main loss against three variants: Decorrelation (minimizing covariance matrix off-diagonals), $\ell_2$-repel (directly increasing Euclidean distance, requiring a norm regularizer $\lambda_{\text{norm}} \|\mathcal{Z}\|_2^2$), and Orthogonalization (a hinge loss $\max(0, 1/2 - D(z_i, z_j))^2$ that only penalizes acute angles). This comparison demonstrates that uniform dispersion in angular space is more direct and effective than feature-dimension decorrelation or Euclidean repulsion.

**4. Mid-training and Full Pre-training Workflows**

To prove the method benefits both existing and new models, it was verified in two scenarios. In Mid-training, existing GPT2 / Qwen3 models were trained on 200M tokens of Wikitext-103—a low-cost proof-of-concept. In Full Pre-training, Qwen3 was trained from scratch on 156B tokens of C4 using 640 GPUs to verify if geometric signals from the start could fundamentally expand available capacity. In both cases, the dispersion term is simply added to the cross-entropy objective.

### Loss & Training
The final training objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. The temperature $\tau$ and weight $\lambda_{\text{disp}}$ are the primary hyperparameters. Mid-training results report mean and variance over 3 seeds, while full pre-training uses a single stable seed over a large token count.

## Key Experimental Results

### Main Results
GPT2 mid-training (Average of 10 benchmarks):

| Configuration | Training Cost | Avg Score↑ | Rank↓ | Significance |
| :--- | :--- | :--- | :--- | :--- |
| GPT2 Original (No mid-training) | — | 34.35 | 6.1 | p<0.0001 |
| + $\mathcal{L}_{\text{CE}}$ only | 1.122 A100h | 34.95 | 6.2 | p<0.01 |
| + Noisy embedding | 1.122 | 35.15 | 4.3 | p<0.01 |
| + Active forgetting | 1.127 | 35.36 | 3.2 | n.s. |
| **+ Dispersion loss** | 1.13 (1.01×) | **35.52+** | **Best** | — |

Qwen3 full pre-training (156B tokens from scratch): Adding dispersion loss yielded an average improvement of +1.17 points (3.3% relative gain), with consistent gains across all benchmarks.

### Ablation Study
Comparison of four dispersion variants:

| Variant | Avg Score | Notes |
| :--- | :--- | :--- |
| Decorrelation | 35.1 | Indirect, sensitive to feature dim |
| $\ell_2$-repel | 35.0 | Requires norm regularization |
| Orthogonalization | 35.2 | Only penalizes acute angles |
| **Dispersion (Canonical)** | **35.5+** | Angular uniform dispersion, optimal |

Confounder-controlled size experiments: Four GPT2-like models were trained from scratch by only varying MLP dimensions. Results confirmed that larger MLPs lead to lower condensation, validating that large-model resistance to collapse is not a spurious correlation.

### Key Findings
- **Distillation fails to fix collapse**: Qwen2.5 distilled models show nearly the same geometric collapse as those trained from scratch because KD loss does not regularize intermediate representations.
- **Collapse exists at init, but SGD helps**: Olmo-3-7B checkpoints show high condensation at initialization that decreases during training, suggesting SGD naturally resists collapse. Dispersion loss accelerates and strengthens this process.
- **Greater gains for small models**: Qwen3-0.6B saw the most significant improvement, while Qwen3-32B saw almost no gain, supporting the hypothesis that large models are already dispersed.
- **Effective via Mid-training**: Significant gains are achieved by adding dispersion loss to existing checkpoints with only 200M additional tokens.
- **Minimal overhead**: Costs < 1% of training time due to token sub-sampling for dispersion calculations.

## Highlights & Insights
- **"Geometry, not parameters, is the bottleneck"**: Attributing the performance gap to representation geometry rather than parameter count suggests small models can approach large-model performance limits without increasing size.
- **Angular vs. Euclidean distance**: Realizing the problem is directional collapse, using $\arccos$ to map cosines to uniform angular distances is a critical engineering trick to avoid gradient saturation.
- **Methodological consistency**: The logical progression from theoretical theorems to empirical evidence, intervention design, and confounder-controlled verification makes the argument highly robust.
- **Zero parameter increase**: Unlike pruning or LoRA, dispersion loss is a training-only addition compatible with any standard LM pipeline.

## Limitations & Future Work
- **Alignment / Reasoning tasks neglected**: Experiments focused on zero/few-shot NLU, leaving reasoning, math, and code tasks (which may require more complex geometry) unexplored.
- **Large models see no gain**: No improvement on 32B models suggests the dispersion hypothesis may not be their primary bottleneck.
- **Token sub-sampling costs**: The impact of sub-sampling $N^2$ pairs on convergence and final performance was not exhaustively ablated.
- **No comparison with existing anti-collapse tech**: Traditional self-supervised methods like BarlowTwins or SimSiam stop-gradients were not benchmarked.
- **Unexplained weight stability**: It remains unclear why a fixed $\lambda_{\text{disp}}$ works effectively without a complex scheduling strategy.

## Related Work & Insights
- **vs. Wang & He 2025 (Dispersion in Diffusion)**: While dispersion has been used in generative models, this work adapts it for LM representations using an angular form and explicit diagonal exclusion.
- **vs. Noisy Embedding / Active Forgetting**: These tricks indirectly increase representation diversity, whereas dispersion loss is direct and geometrically interpretable.
- **vs. Distillation-based Compression**: Distillation transfers behavior; dispersion transfers geometric properties. They are complementary.
- **Insights**: This approach could be extended to (1) vision encoder patch embeddings; (2) multimodal alignment; and (3) MoE expert activation patterns to prevent expert collapse.

## Rating
- Novelty: ⭐⭐⭐⭐ Translates theoretical collapse into a practical auxiliary loss with robust variants.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across mid-training and massive full pre-training scales.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow and high-quality visualizations.
- Value: ⭐⭐⭐⭐ High utility for small model training; limited by lack of gain in larger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](../../NeurIPS2025/model_compression/reordering_patches_improves_vision_models.md)
- [\[ICLR 2026\] FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](../../ICLR2026/model_compression/futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](../../ICLR2026/model_compression/scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)

</div>

<!-- RELATED:END -->
