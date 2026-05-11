---
title: >-
  [Paper Note] Preference Optimization by Estimating the Ratio of the Data Distribution
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] This paper reinterprets DPO as a likelihood ratio (ratio matching) estimation problem and proposes BPO (Bregman Preference Optimization) under a Bregman divergence framework. BPO define…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "DPO"
  - "Bregman divergence"
  - "likelihood ratio estimation"
  - "preference optimization"
  - "alignment"
date: 2026-05-08
content_hash: 9572cd9de5d99304
---

# Preference Optimization by Estimating the Ratio of the Data Distribution

**Conference**: NeurIPS 2025
**arXiv**: [2505.19601](https://arxiv.org/abs/2505.19601)
**Code**: [GitHub](https://github.com/aailab-kaist/BPO)
**Area**: Alignment / RLHF
**Keywords**: DPO, Bregman divergence, likelihood ratio estimation, preference optimization, alignment

## TL;DR
This paper reinterprets DPO as a likelihood ratio (ratio matching) estimation problem and proposes BPO (Bregman Preference Optimization) under a Bregman divergence framework. BPO defines a generalized family of loss functions that subsumes DPO as a special case, and introduces the SBA (Scaled Basu's Power Divergence) instantiation, achieving a state-of-the-art 55.9% AlpacaEval2 length-controlled win rate on Llama-3-8B.

## Background & Motivation

**Background**: DPO is the most widely adopted direct preference optimization method, simplifying RLHF into logistic regression over preference data. Subsequent work (f-DPO, f-PO) has extended DPO's loss function, but each approach has notable drawbacks.

**Limitations of Prior Work**:
- **f-DPO**: Extends the loss function but **sacrifices optimality guarantees**—minimizing f-DPO does not necessarily converge to the optimal policy as defined by DPO.
- **f-PO**: Preserves optimality but **requires training an additional reward model plus Monte Carlo estimation of the partition function**, imposing substantial computational overhead.
- No existing method simultaneously satisfies: (O) optimality guarantee, (S) simplicity (no additional training overhead), and (G) generality (support for diverse objective functions).

**Key Challenge**: When extending the DPO loss, optimality and simplicity appear to be mutually exclusive—f-PO preserves optimality but is not simple, while f-DPO is simple but does not preserve optimality.

**Goal**: To develop a unified preference optimization framework that simultaneously maintains optimality guarantees, incurs no additional computational overhead, and supports multiple loss function instantiations.

**Key Insight**: DPO is reinterpreted through the lens of likelihood ratio estimation—the optimal policy can be uniquely characterized by its **likelihood ratio** without requiring a reward model or partition function. The problem is thus reformulated as ratio matching via Bregman divergence.

**Core Idea**: DPO fundamentally matches the model ratio $R_\theta$ to the data ratio $R_{\text{data}}$. Choosing different Bregman divergences $h$ yields different loss functions, all of which preserve optimality and require no additional overhead.

## Method

### Overall Architecture
Preference optimization is reformulated as a matching problem between two ratios. $R_{\text{data}} = \frac{p_{\text{data}}(\mathbf{y}_w \prec \mathbf{y}_l | \mathbf{x})}{p_{\text{data}}(\mathbf{y}_w \succ \mathbf{y}_l | \mathbf{x})}$ denotes the data preference ratio, and $R_\theta = \left[\frac{\pi_\theta(\mathbf{y}_l|\mathbf{x})\pi_{\text{ref}}(\mathbf{y}_w|\mathbf{x})}{\pi_\theta(\mathbf{y}_w|\mathbf{x})\pi_{\text{ref}}(\mathbf{y}_l|\mathbf{x})}\right]^\beta$ denotes the model ratio. Minimizing $D_h(R_{\text{data}} || R_\theta)$ drives $\pi_\theta$ to converge to the optimal policy. The key technical contribution is deriving a tractable equivalent objective that does not require direct access to $R_{\text{data}}$, analogous to implicit score matching.

### Key Designs

1. **Proposition 1: Likelihood Ratio Representation of the Optimal Policy**:

   - Function: Proves that the optimal policy can be characterized solely via the reference model and the preference data distribution, without a reward model or partition function.
   - Core formula: $\frac{\pi_{\theta^*}(\mathbf{y}_w|\mathbf{x})}{\pi_{\theta^*}(\mathbf{y}_l|\mathbf{x})} = \frac{\pi_{\text{ref}}(\mathbf{y}_w|\mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l|\mathbf{x})} \times \left(\frac{p_{\text{data}}(\mathbf{y}_w \succ \mathbf{y}_l|\mathbf{x})}{p_{\text{data}}(\mathbf{y}_w \prec \mathbf{y}_l|\mathbf{x})}\right)^{1/\beta}$
   - Design Motivation: The likelihood ratio (concrete score) is sufficient to uniquely determine the distribution; matching it is therefore sufficient to recover the target policy.

2. **BPO Objective (Theorem 2 & 3)**:

   - Function: Constructs a tractable generalized loss function.
   - Core formula: $\mathcal{L}^h_{\text{BPO}}(R_\theta; p_{\text{data}}) = \mathbb{E}_{p_{\text{data}}}[h'(R_\theta)R_\theta - h(R_\theta) - h'(R_\theta^{-1})]$
   - Theorem 2 proves that for any strictly convex $h$, the minimizer is $\pi_{\theta^*}$ (optimality guarantee); Theorem 3 proves that $\mathcal{L}^h_{\text{BPO}}$ differs from the intractable $D_h(R_{\text{data}} || R_\theta)$ by only a constant (tractability guarantee).
   - Setting $h(R) = \frac{R\log R - (1+R)\log(1+R)}{2}$ recovers standard DPO.

3. **Gradient Analysis (Proposition 4)**:

   - Function: Analyzes differences in learning dynamics across choices of $h$.
   - Core finding: $\nabla_\theta \mathcal{L} = \mathbb{E}[G_h(R_\theta) \nabla_\theta R_\theta]$—all BPO instantiations share the same gradient direction (determined by $\nabla_\theta R_\theta$), differing only in gradient magnitude $G_h(R_\theta)$. The choice of $h$ controls the weighting assigned to samples with different confidence levels.
   - Design Motivation: Explains why different choices of $h$ all converge to the optimal policy yet exhibit distinct training behavior—the key lies in sample reweighting.

4. **SBA (Scaled Basu's Power Divergence)**:

   - Function: Proposes a new BPO instantiation that addresses the gradient scale issue of the BA divergence.
   - Mechanism: $G_{\text{SBA}_\lambda}(R_\theta) = (R_\theta^\lambda + R_\theta^{-\lambda-1})/s$, with $s=4$ chosen so that the gradient scale at initialization matches that of DPO. The hyperparameter $\lambda$ controls sensitivity to high- and low-confidence samples.
   - Design Motivation: The BA divergence amplifies gradient magnitude by a factor of $(\lambda+1)$, requiring hyperparameter re-tuning. SBA eliminates this issue.

### Loss & Training
- BPO serves as a drop-in replacement for DPO, requiring only minimal code changes to the loss computation.
- BPO is orthogonally composable with other DPO variants: substituting the model ratio $R_\theta^{f\text{-DPO}}$ from f-DPO into the BPO framework yields combined instantiations.

## Key Experimental Results

### Main Results
Dialogue generation (Pythia-2.8B, Anthropic-HH):

| Method | Win Rate vs Preferred ↑ | Win Rate vs SFT ↑ | Entropy ↑ |
|--------|------------------------|-------------------|-----------|
| DPO | 48.5% | 71.5% | 2.801 |
| f-DPO (χ²) | 53.5% | 72.0% | 2.369 ↓ |
| f-PO (JS) | 54.5% | 76.0% | 2.531 ↓ |
| **BPO-SBA** | **57.0%** | **77.0%** | **3.010** ↑ |

Llama-3-8B-Instruct on AlpacaEval2:

| Method | LC Win Rate |
|--------|------------|
| DPO | 51.3% |
| SimPO | 53.7% |
| **BPO-SBA** | **55.9%** |

### Ablation Study

| BPO Instance | Win Rate vs Pref | Entropy | Notes |
|-------------|-----------------|---------|-------|
| LR (= DPO) | 48.5% | 2.801 | baseline |
| KLIEP | 48.5% | 2.901 | improved diversity, similar generation quality |
| LSIF | 50.5% | 2.908 | improvement on both metrics |
| BA | 51.0% | 2.803 | requires lr re-tuning |
| **SBA** | **57.0%** | **3.010** | best on both quality and diversity |

### Key Findings
- **Core advantage of BPO**: Competing extensions (f-DPO, f-PO) exhibit a trade-off between win rate and diversity, whereas BPO-SBA improves both simultaneously.
- **Gradient scale is critical**: BA divergence is theoretically equivalent to SBA, but performs substantially worse in practice due to gradient scale issues, highlighting the extreme sensitivity of preference optimization to hyperparameters.
- **Effect of $\lambda$**: Larger $\lambda$ increases attention to high-confidence samples (where $R_\theta$ deviates far from 1), making it more suitable for settings with higher-quality preference data.

## Highlights & Insights
- **The likelihood ratio perspective on DPO** is highly elegant: DPO is neither "learning a reward" nor "distribution matching," but rather "matching preference ratios." This perspective directly eliminates the dependence on reward models and partition functions, making extensions natural.
- **The Bregman divergence framework unifies all extensions of DPO**: DPO corresponds to logistic regression, while KLIEP, LSIF, and BA correspond to different choices of $h$. This provides practitioners with a clear "menu" for selecting loss functions.
- **The gradient scale normalization in SBA** is a practical engineering contribution: a simple rescaling makes different values of $\lambda$ trainable under the same hyperparameter configuration.

## Limitations & Future Work
- **Optimal $\lambda$ requires tuning**: Although the framework is unified, selecting the best $h$ (or $\lambda$) still relies on empirical search, and no automatic selection mechanism is provided.
- **Theoretical analysis assumes infinite model capacity**: How finite model capacity affects the choice of different $h$ instances remains unanalyzed.
- **Limited experimental scale**: Experiments are primarily conducted on Pythia-2.8B and Llama-3-8B; performance on larger models remains unknown.
- **Future directions**: (1) adaptive $h$ selection strategies; (2) theoretical comparison of BPO instances under finite capacity; (3) integration with online DPO / RLHF.

## Related Work & Insights
- **vs. DPO**: BPO subsumes DPO as a special case ($h$ = logistic regression) while offering additional loss function choices.
- **vs. f-DPO**: f-DPO extends the loss function but sacrifices optimality; BPO preserves optimality.
- **vs. f-PO**: f-PO preserves optimality but requires an additional reward model and partition function estimation; BPO incurs no such overhead.
- **vs. engineering variants (SimPO, ORPO, etc.)**: Complementary relationship—BPO can be combined with the model ratio definitions used in these methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The likelihood ratio estimation perspective is highly original; the Bregman divergence framework is unifying and principled.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers dialogue, summarization, and AlpacaEval2; compared against multiple baselines with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; Table 1 provides a concise summary; minimal code changes required.
- Value: ⭐⭐⭐⭐⭐ Provides a unified theoretical framework for preference optimization together with a practical state-of-the-art method.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[NeurIPS 2025\] On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)
- [\[NeurIPS 2025\] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization](mitigating_hallucination_through_theory-consistent_symmetric_multimodal_preferen.md)
- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[NeurIPS 2025\] LongVPO: From Anchored Cues to Self-Reasoning for Long-Form Video Preference Optimization](longvpo_from_anchored_cues_to_selfreasoning_for_longform_vid.md)

</div>

<!-- RELATED:END -->
