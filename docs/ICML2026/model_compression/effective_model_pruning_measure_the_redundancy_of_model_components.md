---
title: >-
  [Paper Note] Effective Model Pruning: Measure the Redundancy of Model Components
description: >-
  [ICML 2026][Model Compression][Model pruning] This paper borrows the concept of "effective sample size" from particle filtering to directly map any scoring vector to an adaptive retention count $N_{\text{eff}} = \lfloor…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Model pruning"
  - "effective sample size"
  - "inverse Simpson index"
  - "adaptive sparsity"
  - "universal threshold"
date: 2026-05-08
content_hash: 4f5b431a7c7c843f
---

# Effective Model Pruning: Measure the Redundancy of Model Components

**Conference**: ICML 2026  
**arXiv**: [2509.25606](https://arxiv.org/abs/2509.25606)  
**Code**: https://github.com/noMushroomw/Effective-model-pruning  
**Area**: Model Compression  
**Keywords**: Model pruning, effective sample size, inverse Simpson index, adaptive sparsity, universal threshold  

## TL;DR
This paper borrows the concept of "effective sample size" from particle filtering to directly map any scoring vector to an adaptive retention count $N_{\text{eff}} = \lfloor 1/\sum_i \omega_i^2 \rfloor$ as a pruning threshold. This avoids manual sparsity settings and provides a theoretical upper bound for the loss change before and after pruning.

## Background & Motivation

**Background**: Neural network pruning has formed a rich lineage of methods, categorizable by three dimensions: "what to prune" (unstructured weights / structured channels / attention heads), "when to prune" (pre-training / during training / post-training), and "what to score by" (magnitude, sensitivity, data-driven metrics). However, the vast majority of methods still require a human to decide how many components to keep after obtaining a scoring vector $s$.

**Limitations of Prior Work**: Sparsity selection is extremely sensitive—being too aggressive leads to immediate model performance collapse, while being too conservative wastes efficiency gains. Current practices either rely on high-cost iterative pruning (like Lottery Ticket rewriting), manual per-layer budget settings, or treating sparsity as a hyperparameter requiring meticulous tuning (SparseGPT/Wanda etc., require a pre-specified global sparsity rate). At the scale of Large Language Models (LLMs), this tuning cost becomes unbearable.

**Key Challenge**: Pruning's "scoring" and "quantifying" (deciding the amount) are tightly coupled in discussion, but they are actually two independent problems. Existing methods continuously develop new scoring metrics but almost default to "how much to prune" being a user-defined decision; meanwhile, the score distribution itself already carries information about "how many elements are truly significant," which has not been utilized.

**Goal**: Design a **scoring-agnostic and architecture-agnostic** universal threshold rule to decouple "how many components to keep" from hyperparameters, allowing it to be determined directly by the score distribution itself, while providing a provable upper bound on loss changes.

**Key Insight**: The authors noticed a similar problem in particle filtering—given a set of weighted particles, how to judge "how many particles are statistically effective." The answer is the effective sample size $N_{\text{eff}} = 1/\sum_i \omega_i^2$, known in ecology as the Inverse Simpson Diversity Index, directly linked to Rényi entropy. If the scoring vector is normalized into a probability distribution, this quantity naturally reflects "score concentration": higher concentration implies dominance by a few components, allowing for more pruning; higher uniformity implies each component contributes equally, allowing for almost no pruning.

**Core Idea**: Normalize any scoring vector $s$ into $\omega_i = |s_i|/\|s\|_1$ and retain exactly the top $N_{\text{eff}} = \lfloor 1/\sum_i \omega_i^2 \rfloor$ components while pruning the rest—a unified, parameter-free pruning threshold universal across architectures and criteria.

## Method

### Overall Architecture
EMP (Effective Model Pruning) is a three-step universal rule that takes any pre-trained network and a scoring vector $s \in \mathbb{R}^N$ as input and outputs a binary mask $M \in \{0,1\}^N$. The entire pipeline depends solely on the shape of the score distribution: (1) Normalize the absolute values of the scores according to the $\ell_1$ norm to obtain the probability vector $\omega$; (2) Calculate the effective sample size $N_{\text{eff}}$ and truncate it to $[1, N]$; (3) Set indices for the top-$N_{\text{eff}}$ values in $|s|$ to 1 and the rest to 0. The algorithm complexity is $O(N \log N)$, primarily from a single sort. An optional deployment knob $\beta \in [0.5, 2]$ is also introduced to change the actual retention count to $\beta N_{\text{eff}}$, used only for fine-tuning when hardware strictly requires specific sparsity rates.

### Key Designs

1. **$N_{\text{eff}}$ as a Universal Threshold**:

    - Function: Maps any non-negative scoring distribution $\omega \in \Delta$ (standard $(N-1)$ simplex) to an integer retention count.
    - Mechanism: Defined as $N_{\text{eff}} \triangleq \lfloor 1/\sum_i \omega_i^2 \rfloor$, which geometrically corresponds to the inverse of the squared distance between $\omega$ and the simplex centroid $\zeta_{[N]}$. When $\omega$ is perfectly uniform, $N_{\text{eff}} = N$ (nothing can be pruned); when $\omega$ degrades to a single point, $N_{\text{eff}} = 1$ (only the maximum value is kept). The authors prove $A_\nu = \tilde{\Delta} \cap (B_\nu - B_{\nu+1})$, slicing the entire $\tilde{\Delta}$ into several spherical shells, each corresponding to an $N_{\text{eff}}$ value.
    - Design Motivation: This quantity possesses three properties simultaneously—it depends only on the score distribution, adapts to the dimension $N$, and is invariant to coordinate permutations. Particle filtering and ecology long ago proved it as the best proxy for "distribution concentration." Bringing it to pruning means sharper distributions allow for more aggressive pruning without manually specifying sparsity rates.

2. **Tight Lower Bound for Effective Mass $s_{\text{eff}}$**:

    - Function: Provides a provable lower bound for the retained normalized mass $s_{\text{eff}} = \sum_{i=1}^{N_{\text{eff}}} \omega_{(i)}$ relative to $N_{\text{eff}}$, thereby controlling "how heavy the pruned part actually is."
    - Mechanism: Solves for the infimum of $\varphi_\nu(\omega) = \sum_{i=1}^{\nu} \omega_i$ on $A_\nu$ within $\tilde{\Delta}$. Direct relaxation to $\tilde{\Delta}$ yields only the trivial bound $s_{\text{eff}} \geq N_{\text{eff}}/N$. By constructing point $p_\nu = \zeta_{[N]} + \frac{r_{\nu+1}}{r_1}(\zeta_{[1]} - \zeta_{[N]})$, the authors prove it is the minimum point of $\varphi_\nu$ on $A_\nu$, yielding the tight bound $1 - s_{\text{eff}} \leq \frac{N-N_{\text{eff}}}{N}\left(1 - \sqrt{\frac{N-N_{\text{eff}}-1}{(N_{\text{eff}}+1)(N-1)}}\right)$, which asymptotically approximates $\frac{N-N_{\text{eff}}}{N}\left(1 - \sqrt{\frac{N-N_{\text{eff}}}{N N_{\text{eff}}}}\right)$.
    - Design Motivation: The primary concern in pruning is "how important the discarded part is," which is precisely $1 - s_{\text{eff}}$. The tight bound allows deriving the theoretical upper limit of pruning cost directly from the score distribution shape without running experiments.

3. **Upper Bound Propagation of Loss Change $\epsilon$**:

    - Function: When the scoring criterion is parameter magnitude, translates the $s_{\text{eff}}$ lower bound into an upper bound for the loss difference $\epsilon = |L(\theta^*) - L(\theta^k)|$ between the dense and pruned models.
    - Mechanism: Starting from the lemma by Zhang et al. 2023, $\rho \leq 1 - 2\epsilon N / (\|\theta^* - \theta^k\|_2^2 \mathrm{Tr}(H) + 2\epsilon N)$, back-solving yields $\epsilon \leq \frac{1-\rho}{2N\rho}\mathrm{Tr}(H)\|\theta^* - \theta^{N_{\text{eff}}}\|_2^2$. Using $\|\theta^* - \theta^{N_{\text{eff}}}\|^2 \leq \|\theta^*\|_1^2 (1-s_{\text{eff}})^2 (N - N_{\text{eff}})$, the right side is rewritten as an analytical expression involving only $\rho$ and $N$, resulting in the asymptotic upper bound $\epsilon \lesssim \|\theta^*\|_1^2 \mathrm{Tr}(H) \frac{(1-\rho)^4}{2\rho} \left(1 - \sqrt{\frac{1-\rho}{N\rho}}\right)^2$.
    - Design Motivation: This completes the causal chain from "distribution geometry → pruning cost." Experiments show that when $N = 1000$ and $\rho > 0.2$, this bound is near zero, indicating that as long as the $N_{\text{eff}}$ threshold falls within a reasonable range, the loss increment is theoretically suppressed. This chain strictly holds for magnitude criteria and can be generalized to other differentiable criteria.

### Loss & Training
EMP is a **pure post-training** pruning rule that does not modify training objectives or require post-pruning fine-tuning. In experiments, the authors intentionally omit any fine-tuning to isolate the effect of the EMP threshold itself. The optional $\beta$ coefficient is only for hardware deployment: when the target hardware requires a sparsity rate lower than $N_{\text{eff}}/N$, the retention count is scaled to $\beta N_{\text{eff}}$, though $\beta = 1$ remains the "watershed."

## Key Experimental Results

### Main Results
The authors tested the combination of EMP and magnitude pruning across five major architectures: FC, CNN, Transformer, KAN, and LLM. No fine-tuning was performed in any experiment.

| Dataset | Model | Sparsity (%) | Dense Loss | EMP Loss | $\epsilon$ |
|--------|------|-----------|------------|----------|------------|
| CIFAR10 | FC12 | 42.89 | 1.5123 | 1.4454 | 0.0669 |
| CIFAR10 | AlexNet | 62.22 | 0.4664 | 0.4286 | 0.0378 |
| CIFAR10 | VGG16 | 59.47 | 0.4234 | 0.3184 | 0.1050 |
| CIFAR100 | ResNet18 | 56.20 | 0.8740 | 0.9287 | 0.0547 |
| CIFAR100 | ResNet50 | 54.74 | 0.8586 | 0.8387 | 0.0199 |
| TinyImageNet | ResNet50 | 48.10 | 2.0213 | 1.9853 | 0.0360 |

Across all architectures, $\epsilon \leq 0.105$, consistent with theoretical upper bounds. LLM side tests showed average performance on 7 zero-shot tasks for LLaMA and LLaMA-2:

| Method | Avg. Sparsity (%) | Avg. $\Delta$PPL | Avg. $\Delta$Acc (%) |
|------|---------------|------------------|---------------------|
| Wanda (Fixed) | 50.00 | +0.799 | -1.40 |
| Magnitude (Fixed) | 50.00 | +2.982 | -2.60 |
| EMP-Wanda | 40.47 | +0.678 | -1.37 |
| EMP-Magnitude | 36.63 | +0.752 | -0.93 |

EMP-Magnitude pulled naive magnitude pruning from a "2.6 point drop" back to "only a 0.93 point drop," at the cost of reducing sparsity from 50% to 36.63%.

### Ablation Study
The robustness of $N_{\text{eff}}$ as a threshold was verified by scanning $\beta \in \{0.5, 0.75, 1, 1.25, 1.5, 2\}$.

| $\beta$ Setting | Behavior | Description |
|-------------|------|------|
| $\beta < 1$ | Sharp performance drop | Pruning more than $N_{\text{eff}}$ begins to affect truly important components |
| $\beta = 1$ | Performance inflection point | Across all architectures and criteria, this sits exactly at the boundary of "lossless → performance drop" |
| $\beta > 1$ | Plateauing performance | Retaining more components yields no gain, just less pruning |
| GPT-2 Head Pruning (Taylor) | $N_{\text{eff}} = 141.4$, PPL +1.0% | Attention head importance is almost uniformly distributed |
| GPT-2 Head Pruning (Weight) | $N_{\text{eff}} = 134.0$, PPL +6.5% | Pruning only 10 heads; weight norm criterion is more aggressive |

### Key Findings
- $\beta = 1$ **consistently** marks the inflection point where "further pruning leads to performance drops" across FC, CNN, Transformer, and LLMs, indicating that $N_{\text{eff}}$ captures an architecture-agnostic intrinsic sparsity.
- Different criteria for the same model yield different $N_{\text{eff}}$ values (e.g., 141.4 for Taylor vs. 134.0 for Weight on GPT-2), which can serve as a metric to **evaluate the quality of scoring criteria**—better criteria result in more concentrated distributions, smaller $N_{\text{eff}}$, and more prunability.
- On LLMs, the real reason naive magnitude pruning collapses at 50% sparsity is not that the scores themselves are poor, but that "fixed global budget" is too crude; using the $N_{\text{eff}}$ adaptive threshold allows the magnitude criterion to match Wanda's performance.
- Applying EMP to the RGB pixel level, calculating $N_{\text{eff}}$ locally for $4\times4$ patches achieved PSNR 38.3 dB / SSIM 0.991 at 32.3% sparsity, proving the criterion applies to features as well as parameters.

## Highlights & Insights
- Completely decouples "how much to keep" from the hyperparameter pool. EMP introduces no knobs requiring tuning ($\beta$ is only for deployment adaptation), which directly eliminates one dimension of grid search in large-scale LLM pruning experiments.
- Back-defines pruning cost using the "geometrical shape of the score distribution." $N_{\text{eff}}$ essentially treats the inverse of the distribution's distance-to-centroid 2-norm as the "effective dimension." This "distribution as budget" idea can be directly transferred to mixture-of-experts (MoE) activation, attention sparsification, and rank selection in low-rank decomposition.
- Provides a **new scale for evaluating scoring criteria**. Previously, judging a pruning criterion required comparing performance drops at a fixed sparsity rate; EMP allows direct comparison of $N_{\text{eff}}$ values given by criteria, where sharper distribution indicates better redundancy identification.
- Naturally compatible with gated attention. EMP can be viewed as a deterministic hard gating—passing pre-softmax scores through a top-$N_{\text{eff}}$ truncation acts as a parameter-free hard gate, potentially alleviating the attention sink phenomenon.

## Limitations & Future Work
- The derivation of the $\epsilon$ upper bound holds strictly only for magnitude criteria; for Wanda, Taylor, etc., it is only experimentally validated and needs theoretical extension to general differentiable scores.
- $N_{\text{eff}}$ is a global threshold (or per-layer global), lacking coordination for importance across layers, which might lead to suboptimal allocation between shallow and deep layers.
- Skipping fine-tuning completely still results in visible performance drops in LLMs at high sparsity (>50%), which may require combination with SparseGPT-style local reconstruction to squeeze further gains.
- The authors acknowledge that hybrid schemes combining learned gating and using EMP as an initialization for adaptive feature selection during training have not been systematically verified and are explicit future directions.

## Related Work & Insights
- **vs. Lottery Ticket / iterative magnitude pruning**: They rely on multiple rounds of retraining to find subnets. EMP provides a one-time threshold without retraining, though LTH can theoretically find subnets with higher sparsity; EMP is more suitable for rapid deployment without retraining budgets.
- **vs. SparseGPT / Wanda**: Both require pre-specifying sparsity as a hyperparameter. EMP derives this hyperparameter from the metric distribution; experiments show EMP-Wanda achieves better PPL at lower sparsity, indicating "adaptive sparsity" can be stacked with "good scoring."
- **vs. OBD / OBS**: Classic second-order methods require Hessian estimation for locally optimal pruning. EMP requires only first-order or zero-order scores for a global threshold, trading "optimality" for "controllable error."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bringing the $N_{\text{eff}}$ concept from particle filtering/ecology into pruning and providing a geometric lower bound is a true interdisciplinary transfer rather than a simple combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five major architectures and four scoring criteria, but lacks direct comparison with the latest LLM pruning (e.g., ShortGPT, LLM-Pruner).
- Writing Quality: ⭐⭐⭐⭐ Mathematical derivations are clear, and geometric intuitions (simplex + spherical shells) are very persuasive, though notation is slightly dense for beginners.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the pain point of "how to choose sparsity" in pruning practice, and the rule is simple enough to implement in 5 lines of code, offering high deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Continual Model Routing in Evolving Model Hubs](continual_model_routing_in_evolving_model_hubs.md)
- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICLR 2026\] AdaRank: Adaptive Rank Pruning for Enhanced Model Merging](../../ICLR2026/model_compression/adarank_adaptive_rank_pruning_for_enhanced_model_merging.md)
- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](../../NeurIPS2025/model_compression/towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)

</div>

<!-- RELATED:END -->
