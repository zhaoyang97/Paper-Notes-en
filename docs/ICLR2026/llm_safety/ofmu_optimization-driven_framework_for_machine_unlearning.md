---
title: >-
  [Paper Note] OFMU: Optimization-Driven Framework for Machine Unlearning
description: >-
  [ICLR 2026][LLM Safety][Machine Unlearning] This work formulates machine unlearning as a bilevel optimization problem: the inner level maximizes the forgetting loss with gradient decorrelation to prevent damage to the retain set, while the outer level minimizes the retain loss with a penalty term enforcing stationary points of the inner objective. On the TOFU benchmark, OFMU simultaneously achieves high forgetting quality and high model utility, outperforming GA/GradDiff/NPO/RMU in terms of forget-retain balance.
tags:
  - ICLR 2026
  - LLM Safety
  - Machine Unlearning
  - Bilevel Optimization
  - Gradient Decorrelation
  - Forget-Retain Trade-off
  - LLM Privacy
date: 2026-05-08
content_hash: fd4f3c17454444e3
---

# OFMU: Optimization-Driven Framework for Machine Unlearning

**Conference**: ICLR 2026
**arXiv**: [2509.22483](https://arxiv.org/abs/2509.22483)
**Code**: None
**Area**: AI Safety / Machine Unlearning
**Keywords**: Machine Unlearning, Bilevel Optimization, Gradient Decorrelation, Forget-Retain Trade-off, LLM Privacy

## TL;DR
This work formulates machine unlearning as a bilevel optimization problem: the inner level maximizes the forgetting loss with gradient decorrelation to prevent damage to the retain set, while the outer level minimizes the retain loss with a penalty term enforcing stationary points of the inner objective. On the TOFU benchmark, OFMU simultaneously achieves high forgetting quality and high model utility, outperforming GA/GradDiff/NPO/RMU in terms of forget-retain balance.

## Background & Motivation

**Background**: LLMs require on-demand forgetting of specific knowledge (GDPR compliance, copyright, outdated information), yet retraining from scratch is impractical. Existing approaches fall into input-level (refusal strategies), data-level (auxiliary data construction), and model-level (parameter modification) categories.

**Limitations of Prior Work**:
   - Input-level methods are fragile and can be bypassed by adversarial prompts.
   - Model-level methods rely on static weight balancing between forgetting and retention objectives, lacking dynamic adaptability.
   - GradAscent/GradDiff are destructive on hard-to-forget samples—sample difficulty and utility loss are strongly coupled.

**Key Challenge**: When forgetting gradients and retain gradients are correlated, improving forgetting inevitably degrades retention.

**Core Idea**: Bilevel optimization + gradient decorrelation = forgetting without harming retention.

## Method

### Overall Architecture
A bilevel optimization scheme: the inner level performs gradient ascent to maximize the forgetting loss (with decorrelation to protect the retain set), while the outer level performs gradient descent to minimize the retain loss with a penalty term that enforces inner-level convergence.

### Key Designs

1. **Bilevel Optimization Formulation**:

    - Inner objective: $\Phi(\theta) = \mathcal{L}_f(\theta) - \beta \cdot \text{Sim}(\nabla\mathcal{L}_f, \nabla\mathcal{L}_r)$
    - Outer objective: $F(\theta) = \mathcal{L}_r(\theta) + \rho\|\nabla\Phi(\theta)\|^2$
    - The gradient decorrelation term $\text{Sim}$ uses cosine similarity to enforce orthogonality between the forgetting and retain gradient directions.
    - The penalty term $\rho\|\nabla\Phi\|^2$ ensures the inner level reaches a stationary point.

2. **Two-Loop Algorithm**:

    - Inner loop ($T$ steps of gradient ascent): $\theta'^{(t+1)} = \theta'^{(t)} + \eta_{\text{in}}\nabla\Phi(\theta'^{(t)})$
    - Outer loop (retain + penalty): $\theta^{(k+1)} = \theta^{(k)} - \eta_{\text{out}}(\nabla\mathcal{L}_r + 2\rho\nabla^2\Phi\cdot\nabla\Phi)$
    - Theoretical convergence guarantees: $O(1/K)+O(K/T^2)$ for convex settings; $O(1/K)+O(1/T)+O(\sigma^2)$ for non-convex settings.

### Loss & Training
- Inner loop runs $T=5$–$10$ steps without requiring full convergence.
- The penalty parameter $\rho_k$ is increased progressively; outer-level gradients are computed via Hessian-vector products.

## Key Experimental Results

### Main Results: TOFU Benchmark (LLaMA-2-7B)

| Method | FQ (forget01) | MU | FTR | Note |
|--------|-------------|----|----|------|
| Retrain | 1.00 | 0.63 | 0.68 | Ideal upper bound |
| GradAscent | 1.88e-4 | 0.55 | 0.36 | Weak forgetting + poor retention |
| GradDiff | 3.02e-3 | 0.57 | 0.41 | Marginally better |
| NPO | 0.40 | 0.58 | 0.65 | Moderate |
| RMU | 0.40 | 0.62 | 0.64 | Moderate |
| **OFMU** | **0.42** | **0.63** | **0.68** | **Approaches Retrain** |

### Ablation Study

| Configuration | Key Findings |
|------|--------|
| Remove gradient decorrelation | Forgetting improves but retention degrades severely |
| Remove bilevel structure (linear weighting) | $\lambda$ trade-off is unstable and difficult to tune |
| Full OFMU | Best overall balance |

### Key Findings
- **OFMU approaches the Retrain upper bound**: MU=0.63 and FTR=0.68, matching Retrain exactly.
- **GA/GradDiff collapse on forget05/10**: FQ drops to e-119–e-239, indicating complete failure at large-scale forgetting.
- **Gradient decorrelation decouples the coupling problem in hard-to-forget samples.**

## Highlights & Insights
- **Bilevel optimization perspective redefines the unlearning problem**: Rather than simple multi-objective linear weighting, unlearning is modeled as an outer optimization subject to inner gradient stationarity constraints—a formulation more faithful to the nature of unlearning.
- **Elegant design of gradient decorrelation**: Cosine similarity penalties enforce orthogonality between forgetting and retain gradients, geometrically eliminating conflicts—conceptually similar to the null-space projection in NSPO but applied to machine unlearning rather than safety alignment.

## Limitations & Future Work
- Hessian-vector product computation introduces non-trivial overhead.
- Models larger than 70B and multimodal settings have not been evaluated.
- Continual unlearning scenarios (sequential unlearning requests) remain unexplored.

## Related Work & Insights
- **vs. GradAscent/GradDiff**: Simple gradient ascent collapses under large-scale forgetting; OFMU remains stable via the bilevel structure.
- **vs. NPO/RMU**: These methods rely on heuristic weight balancing, whereas OFMU employs a rigorous bilevel optimization framework with stronger theoretical guarantees.
- **vs. NSPO (same venue)**: Both adopt gradient orthogonality/decorrelation strategies, but NSPO targets safety alignment while OFMU targets machine unlearning.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of bilevel optimization and gradient decorrelation is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers TOFU + CIFAR across multiple settings, but lacks large-scale LLM experiments.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded optimization framework for machine unlearning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation](../../NeurIPS2025/llm_safety/a_reliable_cryptographic_framework_for_empirical_machine_unl.md)
- [\[ICLR 2026\] Erase or Hide? Suppressing Spurious Unlearning Neurons for Robust Unlearning](erase_or_hide_suppressing_spurious_unlearning_neurons_for_robust_unlearning.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[NeurIPS 2025\] SIMU: Selective Influence Machine Unlearning](../../NeurIPS2025/llm_safety/simu_selective_influence_machine_unlearning.md)
- [\[ICLR 2026\] Redirection for Erasing Memory (REM): Towards a Universal Unlearning Method for Corrupted Data](redirection_for_erasing_memory_rem_towards_a_universal_unlearning_method_for_cor.md)

<!-- RELATED:END -->
