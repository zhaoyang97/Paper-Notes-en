---
title: >-
  [Paper Note] On Extending Direct Preference Optimization to Accommodate Ties
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] This paper replaces the Bradley-Terry preference model in DPO with the Rao-Kupper and Davidson extensions, enabling preference optimization to explicitly model "tie" data. This avoids discarding ambiguous preference pairs and yields improved regularization and performance on translation and mathematical reasoning tasks.
tags:
  - NeurIPS 2025
  - LLM Alignment
  - DPO
  - preference optimization
  - ties
  - Bradley-Terry
  - Rao-Kupper
  - Davidson model
date: 2026-05-08
content_hash: 0a7aa3d33a1a0d11
---

# On Extending Direct Preference Optimization to Accommodate Ties

**Conference**: NeurIPS 2025
**arXiv**: [2409.17431](https://arxiv.org/abs/2409.17431)
**Code**: None
**Area**: LLM Alignment / Preference Optimization
**Keywords**: DPO, preference optimization, ties, Bradley-Terry, Rao-Kupper, Davidson model

## TL;DR
This paper replaces the Bradley-Terry preference model in DPO with the Rao-Kupper and Davidson extensions, enabling preference optimization to explicitly model "tie" data. This avoids discarding ambiguous preference pairs and yields improved regularization and performance on translation and mathematical reasoning tasks.

## Background & Motivation

**Background**: DPO is built on the Bradley-Terry model, which requires each training pair to have a clear winner: $y_w \succ y_l$. In practice (e.g., Llama 3, Qwen2), preference pairs that annotators find difficult to distinguish—so-called "tie" pairs—are frequently discarded.

**Limitations of Prior Work**: Discarding tie data is wasteful—such data is expensive to collect and carries genuine information (the fact that two responses are of comparable quality is itself a preference signal). Naively treating tie data as random wins/losses in DPO degrades performance.

**Key Challenge**: The Bradley-Terry model admits only two outcomes ($y_i$ wins or $y_j$ wins), leaving no probability mass for ties. When $\lambda_i \neq \lambda_j$, the model always favors the stronger candidate.

**Goal**: Enable DPO to correctly utilize tie data without discarding it or incurring performance degradation.

**Key Insight**: Classical paired comparison theory already offers solutions—Rao-Kupper (1967) and Davidson (1970) proposed Bradley-Terry extensions that accommodate ties, and these can be directly embedded into the DPO framework.

**Core Idea**: Replace DPO's Bradley-Terry model with tie-aware preference models from classical statistics, so that the optimization objective increases the reward margin for win/loss pairs and drives the reward margin toward zero for tie pairs.

## Method

### Overall Architecture
The input is a labeled preference dataset in which each pair $(x, y_w, y_l)$ carries a flag $t \in \{0, 1\}$, where $t=0$ denotes a clear preference and $t=1$ denotes a tie. The algorithm outputs a trained policy $\pi_\theta$. The loss function consists of two components: maximizing the preference log-likelihood for win/loss pairs and maximizing the tie log-likelihood for tie pairs.

### Key Designs

1. **Rao-Kupper Model (DPO-RK)**:

    - **Function**: Introduces a perception threshold $\alpha_{RK}$; differences in reward smaller than the threshold are treated as ties.
    - **Mechanism**: The win probability is $p^{RK}(y_w \succ y_l) = \sigma(d_\theta - \alpha_{RK})$, equivalent to shifting DPO's sigmoid rightward by $\alpha_{RK}$. The tie probability is given by the product of sigmoids at both ends. For win/loss pairs, the gradient pushes $d_\theta$ upward; for tie pairs, the gradient drives $d_\theta$ toward zero.
    - **Design Motivation**: Rao-Kupper is motivated by "perceptual resolution"—when the difference between two items is too small, annotators cannot distinguish them, which closely mirrors real-world preference annotation.

2. **Davidson Model (DPO-D)**:

    - **Function**: Allocates tie probability via the geometric mean of the two item strengths.
    - **Mechanism**: $p^D(y_w \succ y_l) = \frac{1}{1 + e^{-d_\theta} + 2\nu_D e^{-d_\theta/2}}$. The tie probability is proportional to the geometric mean of the two strengths. This satisfies the Luce choice axiom $p(y_i \succ y_j)/p(y_j \succ y_i) = \lambda_i/\lambda_j$, a property the Rao-Kupper model does not possess.
    - **Design Motivation**: Davidson derives the model axiomatically to ensure logical consistency of comparisons.

3. **Regularization Effect of Ties (Theoretical Explanation)**:

    - **Function**: Uses the ideal DPO policy theory to explain theoretically why tie data enhances regularization.
    - **Mechanism**: For a tie pair with true preference probability $\gamma = 0.5$, the ideal policy satisfies $\pi^*(y_w|x)/\pi^*(y_l|x) = \pi_{ref}(y_w|x)/\pi_{ref}(y_l|x)$, meaning the policy should not deviate from the reference model. This constitutes a regularization constraint on $\pi_\theta$—50% of the data requires the policy to preserve reference model behavior.
    - **Design Motivation**: This provides a theoretical explanation for why even vanilla DPO exhibits lower KL divergence when tie data is included.

### Loss & Training
$$\mathcal{L}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{t=0}[\log p_\theta(y_w \succ y_l)] - \mathbb{E}_{t=1}[\log p_\theta(y_w \sim y_l)]$$

Hyperparameter selection: $\nu_{RK} = 3$, $\nu_D = 1$, corresponding to a prior of "50% tie probability between equally matched candidates." Experiments show that results are insensitive to these parameters.

## Key Experimental Results

### Main Results
Three tasks: WMT21 ZH-EN translation, IWSLT17 FR-EN translation, and TL;DR summarization.

| Method | WMT21 BLEURT | IWSLT17 BLEURT | TL;DR Win-Rate |
|--------|-------------|----------------|----------------|
| DPO(CP) | Best baseline | Best baseline | Best baseline |
| DPO(CP+TP) | Notable drop | Notable drop | Drop |
| DPO-RK(CP+TP) | ≈DPO(CP), lower KL | ≈DPO(CP), lower KL | ≈DPO(CP), lower KL |
| DPO-D(CP+TP) | ≈DPO(CP), lower KL | ≈DPO(CP), lower KL | ≈DPO(CP), lower KL |

Results on mathematical reasoning (GSM8K/MATH):

| Method | GSM8K Pass@1 | MATH Pass@1 |
|--------|-------------|-------------|
| DPO | 81.4 | 44.1 |
| DPO-RK | **82.7** | **45.3** |
| DPO-D | 82.1 | 44.8 |

### Ablation Study

| Configuration | Effect | Notes |
|--------------|--------|-------|
| CP only (vanilla DPO) | High performance, high KL | Standard setting |
| CP+TP (DPO) | Performance drop, low KL | Ties harm DPO |
| CP+TP (DPO-RK/D) | High performance, low KL | Correctly models ties |
| Increasing tie ratio | Further KL reduction | Regularization scales with tie proportion |

### Key Findings
- Adding tie data to DPO does provide regularization (reducing KL), but at the cost of performance degradation; DPO-RK/D preserves performance while retaining the regularization benefit.
- The regularization effect scales with the proportion of tie data—more pronounced in translation tasks (50% ties) than in TL;DR (12.5% ties).
- The Rao-Kupper and Davidson variants perform comparably and are insensitive to their respective hyperparameters.
- On mathematical reasoning, constructing tie data from reward model score differences yields a 1.3 percentage point improvement in GSM8K accuracy with DPO-RK.

## Highlights & Insights
- **Elegant theoretical connection**: The paper seamlessly bridges classical statistical models from the 1960s–70s (Rao-Kupper 1967, Davidson 1970) with modern RLHF. DPO-RK requires only adding an offset $\alpha_{RK}$ inside the sigmoid—a minimal modification with significant effect.
- **Theoretical account of regularization via ideal policy**: The regularization effect of ties is rigorously explained through the ideal DPO policy theory of Chen et al. (2024), grounding an empirical observation in formal derivation.
- **High practical value**: For any DPO-based training pipeline, one need only label previously discarded "ambiguous" preference pairs as ties and switch to the DPO-RK loss to obtain dual benefits—improved data utilization and regularization—at virtually no additional cost.

## Limitations & Future Work
- Experiments are conducted at a relatively small scale (translation and TL;DR); validation on large LLMs (>7B parameters) is absent.
- Only two classical models are considered; whether superior tie probability allocation schemes exist remains an open question.
- The definition of ties relies on heuristic thresholds (e.g., BLEURT score differences); better tie detection methods warrant further investigation.
- No comparison is made against other approaches for handling ambiguous preferences (e.g., soft labels, label smoothing).

## Related Work & Insights
- **vs. DPO**: DPO-RK/D strictly generalize DPO (recovering DPO when $\alpha_{RK}=0$ or $\nu_D=0$), at the cost of a single additional hyperparameter.
- **vs. IPO**: IPO also addresses preference uncertainty, but by replacing the sigmoid with a squared loss rather than explicitly modeling ties. Experiments show that IPO exhibits analogous problems when tie data is introduced.
- **vs. KTO**: KTO can leverage unpaired data but still requires binary labels; the proposed method retains the paired framework while introducing a third label category.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of classical statistical models with modern DPO is highly elegant, though not technically complex.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three tasks plus mathematical reasoning, supported by theoretical analysis, but lacking large-model experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clear and well-structured, with motivation, theory, and experiments forming a coherent narrative.
- Value: ⭐⭐⭐⭐ A highly practical improvement that can be adopted at nearly zero cost.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] KL Penalty Control via Perturbation for Direct Preference Optimization](kl_penalty_control_via_perturbation_for_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[NeurIPS 2025\] DP²O-SR: Direct Perceptual Preference Optimization for Real-World Image Super-Resolution](dp2o-sr_direct_perceptual_preference_optimization_for_real-world_image_super-res.md)
- [\[NeurIPS 2025\] Robust LLM Alignment via Distributionally Robust Direct Preference Optimization](robust_llm_alignment_via_distributionally_robust_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)

</div>

<!-- RELATED:END -->
