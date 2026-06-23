---
title: >-
  [Paper Note] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models
description: >-
  [ICML 2026][Model Compression][Knowledge Distillation] This paper systematically observes the universal phenomenon of "embedding condensation," where token embeddings in small language models collapse into a narrow cone as depth increases—unlike in large models. The authors design an angular dispersion loss $\mathcal{L}_{\text{disp}}$ to explicitly force embeddings to spre
tags:
  - ICML 2026
  - Model Compression
  - Knowledge Distillation
  - GPT2 / Qwen3
date: 2026-05-08
content_hash: 55b9291b70b89d18
---
# Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.00217](https://arxiv.org/abs/2602.00217)  
**Code**: https://github.com/KrishnaswamyLab/LM-Dispersion  
**Area**: Model Compression / Representation Learning / Small Model Training  
**Keywords**: Embedding Collapse, Dispersion Loss, Small Model Generalization, Knowledge Distillation, GPT2 / Qwen3

## TL;DR
This paper systematically observes the universal phenomenon of "embedding condensation," where token embeddings in small language models collapse into a narrow cone as depth increases—unlike in large models. The authors design an angular dispersion loss $\mathcal{L}_{\text{disp}}$ to explicitly force embeddings to spread out. Without adding any parameters, this approach achieves an average improvement of 3.3% for Qwen3 / GPT2 across 10 benchmarks.

## Background & Motivation
**Background**: While LLM capabilities scale with size, training and deployment costs have skyrocketed, creating an urgent need to "replicate the key properties of large models in small models." Existing compression routes, such as distillation, quantization, and pruning, primarily focus on mimicking the output distribution of large models.

**Limitations of Prior Work**: From a representation geometry perspective, the authors found that token embeddings in small models (e.g., GPT2-small, Qwen3-0.6B) align almost entirely in the same direction in deeper layers, with pairwise cosine similarity approaching 1. In contrast, large models (e.g., GPT2-xl, Qwen3-32B) maintain dispersed embeddings. Theoretical work (Geshkovski 2025) proved that Transformer embeddings collapse to a single point as the number of layers goes to infinity, but this has not been systematically verified empirically in relation to performance.

**Key Challenge**: Embedding condensation implies that the model has fewer "representational directions" available, effectively locking its expressive capacity geometrically. Even if distillation learns logit distributions from a large teacher, it cannot inherit the geometric properties of the large model because distillation targets only constrain the output, not the intermediate embeddings.

**Goal**: (1) Quantitatively measure the embedding condensation phenomenon and confirm that "resistance to collapse in large models" is a universal law; (2) Verify that distillation does not mitigate this; (3) Design an auxiliary loss acting directly on the geometry to help small models actively disperse their embeddings.

**Key Insight**: Since large models "automatically" maintain dispersion, dispersion itself might be a bottleneck condition for performance. Rather than stacking parameters to allow the model to disperse "naturally," it is better to explicitly add an objective function to force dispersion.

**Core Idea**: Use an angular-based dispersion loss $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-\arccos(\cos\text{sim}(z_i, z_j)) / \pi\tau)$ to push all token embeddings toward a uniform distribution on the unit hypersphere with zero extra parameters.

## Method

### Overall Architecture
The methodology revolves around first "diagnosing the disease" and then "prescribing a geometric treatment." In the diagnosis phase, embedding condensation is quantified using Spearman $\rho$ and Kendall $\tau$ to measure whether layer-wise average cosine similarity monotonically increases with depth. In the intervention phase, an angular dispersion loss $\mathcal{L}_{\text{disp}}$ is attached to the original training objective as a regularizer: $\mathcal{L} = \mathcal{L}_{\text{train}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. This requires zero extra parameters and can be used to either retrofit existing checkpoints or shape the model during pre-training from scratch.

### Key Designs

**1. Quantitative Diagnosis of Embedding Condensation: Measuring the "Severity" via Hierarchical Cosine Similarity Rank Correlation**

To treat the problem, it must first be measured accurately. The approach involves feeding input sequences into the model, collecting embeddings $z_i^{(l)}$ for all tokens at each layer $l$, and calculating the pairwise cosine similarity for all $N^2$ pairs. The alignment at each layer is summarized by the mean $\mu^{(l)} = \frac{1}{N^2}\sum_{i,j}\cos\text{sim}(z_i^{(l)}, z_j^{(l)})$. Finally, the rank correlation—Spearman $\rho$ and Kendall $\tau$—between the sequence $\{\mu^{(l)}\}_{l=1}^L$ and the layer indices $\{l\}_{l=1}^L$ is computed. Values of $\rho/\tau$ close to $+1$ indicate that similarity increases monotonically with depth (signaling severe condensation); values near $0$ indicate no systematic trend, and negative values suggest dispersion. Rank correlation is used instead of "mean cosine of the last few layers" because it captures monotonic trends regardless of absolute scale or non-linear distortion, providing a cleaner, cross-model metric. This metric allowed the authors to quantify the law that "larger models collapse less," providing a target for intervention.

**2. Angular Dispersion Loss: Pushing Embeddings Toward a Uniform Distribution on the Hypersphere**

The target is the issue where deep-layer embeddings in small models align in a single direction. For every token pair $(z_i, z_j)$ at each layer, the cosine similarity is mapped to an angular distance $D(z_i, z_j) = \arccos(\cos\text{sim}(z_i, z_j)) / \pi \in [0, 1]$. These are aggregated via log-sum-exp: $\mathcal{L}_{\text{disp}} = \log \sum_{i \neq j} \exp(-D(z_i, z_j)/\tau)$. The more co-directional the tokens are (smaller $D$), the larger the $\exp$ term, forcing the loss to push them apart. When nearly orthogonal, the $\exp$ term approaches zero, exerting no force. Summing over all layers yields a per-batch complexity of $\mathcal{O}(N^2 F)$, which can be mitigated via sub-sampling. Key stability details include: using $\arccos$ to avoid gradient saturation at $\pm 1$; using log-sum-exp for robustness; and excluding diagonal terms $i=j$ to prevent gradient explosions from self-similarity. Angular distance is preferred over Euclidean distance because condensation is fundamentally a directional problem.

**3. Three Alternative Formulations: Isolating Why "Uniform Angular Dispersion" is Superior**

To justify the choice of canonical dispersion, the authors compared it against three alternatives. *Decorrelation* minimizes the off-diagonal elements of the embedding covariance matrix to decouple feature dimensions. *$\ell_2$-repel* directly increases the Euclidean distance between tokens, but requires a norm regularizer $\lambda_{\text{norm}} \|\mathcal{Z}\|_2^2$ to prevent the model from simply inflating norms. *Orthogonalization* uses a hinge-style loss $\max(0, 1/2 - D(z_i, z_j))^2$, penalizing only acute angle pairs while ignoring obtuse ones. This comparison demonstrates that uniform dispersion in angular space is more direct and effective than feature-dimension decorrelation or Euclidean repulsion.

**4. Coverage of Mid-training and Full Pre-training Workflows**

To prove the generalizability of this "remedy," the authors verified it in two scenarios. *Mid-training* uses existing GPT2 / Qwen3 checkpoints and continues training on 200M tokens of wikitext-103—a low-cost setup for proof-of-concept and hyperparameter sweeping. *Full pre-training* involves training Qwen3 on 156B tokens of C4 using 640 GPUs to verify if the geometric signal can fundamentally shape a better representation structure and increase available capacity from the start. Both scenarios simply add $\lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$ to the cross-entropy loss, involving minimal pipeline modifications.

### Loss & Training
The final training objective is $\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda_{\text{disp}} \cdot \mathcal{L}_{\text{disp}}$. The temperature $\tau$ and weight $\lambda_{\text{disp}}$ are the only primary hyperparameters. Mid-training results report mean and variance across 3 seeds, while full pre-training uses a single seed due to the stabilizing effect of the large token volume.

## Key Experimental Results

### Main Results
GPT2 mid-training (average of 10 benchmarks):

| Configuration | Training Cost | Avg. Score↑ | Rank↓ | Significance |
|---------------|---------------|-------------|-------|--------------|
| GPT2 Original (no mid-training) | — | 34.35 | 6.1 | p<0.0001 |
| + $\mathcal{L}_{\text{CE}}$ only | 1.122 A100h | 34.95 | 6.2 | p<0.01 |
| + noisy embedding | 1.122 | 35.15 | 4.3 | p<0.01 |
| + active forgetting | 1.127 | 35.36 | 3.2 | n.s. |
| **+ Dispersion loss** | 1.13 (1.01×) | **35.52+** | **Best** | — |

Qwen3 full pre-training (156B tokens from scratch): Adding dispersion loss yielded an average improvement of +1.17 points (3.3% relative gain), with stable improvements across all benchmarks.

### Ablation Study
Comparison of four dispersion variants:

| Variant | Avg. Score | Notes |
|---------|------------|-------|
| Decorrelation | 35.1 | Indirect, affected by feature dim |
| $\ell_2$-repel | 35.0 | Requires norm regularization for stability |
| Orthogonalization | 35.2 | Only penalizes acute angles |
| **Dispersion (canonical)** | **35.5+** | Uniform angular dispersion, optimal |

Confounder-controlled scale study: Training four GPT2-like models from scratch while only varying MLP dimension showed that larger MLP → less condensation, confirming the link between model size and collapse is not a spurious correlation.

### Key Findings
- **Distillation does not solve collapse**: Distilled Qwen2.5 models show nearly identical embedding geometry to scratch-trained models, as KD loss only constrains output behavior, not internal representation topology.
- **Collapse exists at initialization but is mitigated by training**: Indicators for Olmo-3-7B show high condensation at initialization that decreases during training, suggesting SGD naturally resists collapse; dispersion loss accelerates and strengthens this process.
- **Small models benefit more**: Qwen3-0.6B showed the most significant gains, while Qwen3-32B saw almost no benefit, consistent with the hypothesis that large models are already sufficiently dispersed.
- **Mid-training is effective**: Significant gains can be achieved by applying dispersion loss to existing checkpoints for only 200M tokens, allowing for low-cost deployment.
- **Cost < 1% of training time**: 1.13 vs 1.122 A100h, as the $O(N^2)$ pairing can be sub-sampled.

## Highlights & Insights
- **"The bottleneck of small models is geometry, not parameters"**: Attributing performance gaps to representation geometry rather than raw capacity is an inspired proposition, suggesting performance can approach large-model upper bounds without adding parameters.
- **Angular vs. Euclidean Distance**: Using $\arccos$ to map cosine similarity to a uniform angular distance avoids gradient saturation at the extremes, a critical engineering trick for stability.
- **Theory → Empirical → Intervention → Validation Loop**: Starting from Geshkovski’s 2025 theorem, providing large-scale empirical evidence, designing an intervention, and verifying it via confounder control makes for a highly robust argument chain.
- **Zero Parameter Increment**: Unlike pruning, quantization, or LoRA, dispersion loss is just a training-time addition that is plug-and-play with any standard LM training pipeline.

## Limitations & Future Work
- **Not extended to alignment or reasoning tasks**: Experiments centered on zero/few-shot NLU; complex tasks like reasoning, math, or coding requiring specific representation geometries were not tested.
- **No gain for large models**: No improvement seen at 32B; whether the dispersion hypothesis holds for massive scales remains unconfirmed—perhaps their bottlenecks lie elsewhere.
- **Cost of token sub-sampling**: $N^2$ scaling still requires sub-sampling for large batches; the impact of sub-sampling on convergence and final performance was not exhaustively ablated.
- **Comparison with other anti-collapse techniques**: A direct comparison with SSL methods like SimSiam stop-gradient or BarlowTwins is lacking.
- **Weight scheduling**: It is intuitive that dispersion should be stronger early in training and weaker later, but the paper uses a fixed weight; this phenomenon warrants deeper analysis.

## Related Work & Insights
- **vs. Wang & He 2025 (Dispersion in Diffusion)**: While that work used dispersion in generative models, this paper adapts it for LMs with an angular form and explicit diagonal exclusion.
- **vs. Noisy Embedding / Active Forgetting**: These tricks attempt to increase diversity indirectly without geometric explanation; dispersion is direct, interpretable, and more effective.
- **vs. Distillation-based Compression**: Distillation transfers behavior; dispersion transfers a more fundamental property: representational geometry. The two could be combined.
- **Insights**: This logic could be extended to (1) vision encoder patch embeddings; (2) multimodal alignment; (3) MoE expert activation patterns to avoid expert collapse.

## Rating
- Novelty: ⭐⭐⭐⭐ Translates theoretical collapse into a trainable auxiliary loss with four variants; "known phenomenon + clean intervention" style novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Strong dual validation (mid-training + full pre-training on 156B tokens), 10 benchmarks, and extensive confounder controls/ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical progression is excellent, and the transformation of an abstract geometric phenomenon into a clear narrative is well-executed.
- Value: ⭐⭐⭐⭐ High practical value for the small-model training community given the zero-cost plug-and-play nature, though the lack of gain for large models limits long-term impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">
</div>

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](entropy-aware_on-policy_distillation_of_language_models.md)
- [\[NeurIPS 2025\] REOrdering Patches Improves Vision Models](../../NeurIPS2025/model_compression/reordering_patches_improves_vision_models.md)
- [\[ICLR 2026\] FutureMind: Equipping Small Language Models with Strategic Thinking-Pattern Priors via Adaptive Knowledge Distillation](../../ICLR2026/model_compression/futuremind_equipping_small_language_models_with_strategic_thinking-pattern_prior.md)
- [\[ICML 2026\] Making Models Unmergeable via Scaling-Sensitive Loss Landscape](making_models_unmergeable_via_scaling-sensitive_loss_landscape.md)

</div>

<!-- RELATED:END -->
