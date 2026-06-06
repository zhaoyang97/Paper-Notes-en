---
title: >-
  [Paper Note] Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study
description: >-
  [ICLR 2026][LLM Alignment][Safety Alignment] Through four systematic experiments (parallel projection, orthogonal projection, subspace overlap, and activation space analysis) conducted across five open-source LLMs…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Safety Alignment"
  - "Subspace"
  - "Fine-Tuning Attack"
  - "Linear Separability"
  - "Weight Space"
date: 2026-05-08
content_hash: ce128a555c9bd4c0
---

# Safety Subspaces are Not Linearly Distinct: A Fine-Tuning Case Study

**Conference**: ICLR 2026
**arXiv**: [2505.14185](https://arxiv.org/abs/2505.14185)  
**Code**: [GitHub](https://github.com/CERT-Lab/safety-subspaces)  
**Area**: LLM Safety / Alignment
**Keywords**: Safety Alignment, Subspace, Fine-Tuning Attack, Linear Separability, Weight Space

## TL;DR

Through four systematic experiments (parallel projection, orthogonal projection, subspace overlap, and activation space analysis) conducted across five open-source LLMs, this paper establishes a key finding: safety alignment behavior is highly entangled with general learning in both weight space and activation space, and no linearly separable independent safety subspace exists. Consequently, defense strategies based on subspace projection/filtering face fundamental limitations.

## Background & Motivation

**Background**: LLMs aligned via safety training (e.g., RLHF) can refuse harmful prompts, yet this safety is fragile — even fine-tuning on benign data can degrade safety behavior. A small number of malicious samples mixed into the training set can subvert alignment, revealing an attack surface deeper than prompt injection: weight-level alignment degradation.

**Limitations of Prior Work**: A line of research (e.g., SafeLoRA, LDIFS) has attempted to defend against fine-tuning attacks by exploiting "safety subspaces" — the core assumption being that safety alignment information concentrates in specific linear directions in weight space and can be extracted via SVD and protected during subsequent fine-tuning. This assumption, however, has never been rigorously tested.

**Key Challenge**: If safety information truly resides in an independent linear subspace, harmful updates could be orthogonalized away from safety directions via simple projection, preserving safety while retaining task performance. However, if safety and general learning are highly entangled — i.e., the same directions simultaneously amplify both safe and harmful behavior — projection-based defenses cannot selectively suppress harmfulness without sacrificing utility.

**Goal**: To systematically examine the foundational assumption of whether LLM safety behavior concentrates in specific linear subspaces.

**Key Insight**: Rather than proposing a new defense, the authors conduct rigorous empirical investigation. Two candidate safety subspaces are constructed — from the alignment update $\Delta_A$ (aligned − base) and the safety fine-tuning update $\Delta_S$ (safety-tuned − base) — and their specificity is then tested through projection experiments.

**Core Idea**: Four carefully designed experiments demonstrate that safety-related weight updates and activation patterns are not linearly separable from general learning, exposing fundamental limitations of subspace-based defense strategies.

## Method

### Overall Architecture

The experimental design proceeds across four levels: (1) Do useful and harmful updates differ in their expressiveness within candidate safety subspaces? (2) Can orthogonalizing mixed updates against candidate safety subspaces selectively remove harmfulness? (3) What is the geometric relationship (subspace overlap) among useful, harmful, and safety updates? (4) Are representations of harmful and benign prompts separable in activation space?

### Key Designs

1. **Parallel Projection Experiment (Experiment 1: Do Subspaces Encode Safety?)**

    - **Function**: Tests behavioral outcomes when useful/harmful fine-tuning updates are projected onto the top directions of candidate safety subspaces.
    - **Mechanism**: Models are fine-tuned separately on MetaMathQA (useful data) and a harmful subset of BeaverTails, yielding $\Delta_T^{\text{Useful}}$ and $\Delta_T^{\text{Harmful}}$. SVD is applied to $\Delta_{A/S}$; the top-$k$ directions form a projection matrix $P_k$, and task updates are projected as $\tilde{\Delta}_T^j = P_k \Delta_T^j$. The projected model is then evaluated on utility (GSM8k accuracy) and harmfulness (AdvBench harmful score).
    - **Key Findings**: **Energy is uniformly distributed within the subspace** (the energy retention ratios for useful and harmful updates are nearly identical), yet **behavioral impact is asymmetric** — top-$k$ directions simultaneously amplify both utility and harmfulness, outperforming random projection. This indicates these directions are "high-impact learning directions" rather than "safety directions."

2. **Orthogonal Projection Experiment (Experiment 2: Can the Harmful Subspace Be Removed?)**

    - **Function**: In a mixed fine-tuning setting (80% useful + 20% harmful), tests whether removing candidate safety subspace directions can selectively suppress harmfulness.
    - **Mechanism**: The orthogonal projection $\tilde{\Delta}_T = P_k^{\perp} \Delta_T$ removes update components aligned with candidate safety directions; utility and harmfulness are then evaluated.
    - **Key Findings**: Utility and harmfulness decline in tandem. After removing top-$k$ directions, utility degrades faster than with random projection, while harmfulness declines at a comparable rate. That is, **no selective suppression effect exists — safety gains are always accompanied by proportional task performance loss.**

3. **Mode Subspace Overlap (MSO) Analysis (Experiment 3: Geometric Relationships Among Updates)**

    - **Function**: Directly compares subspace overlap among useful, harmful, and safety updates.
    - **Mechanism**: SVD is applied to each of the three update types; top-$k$ directions corresponding to energy retention ratio $\eta$ are selected, and pairwise MSO is computed as: $\mathrm{MSO}(\mathbf{V}, \mathbf{W}; \eta) = \frac{\|S\|_F^2}{\min(k_V, k_W)}$, with range $[0, 1]$ where 0 indicates orthogonality and 1 indicates identity.
    - **Key Findings**: The harmful–safety update pair never achieves the highest MSO, and is sometimes the lowest. This refutes the hypothesis that safety subspaces share a special geometric relationship with harmful behavior.

4. **Activation Space Analysis (Experiment 4: Are Representations Separable?)**

    - **Function**: Examines whether harmful and benign prompts occupy distinct regions in the model's internal activation space.
    - **Mechanism**: Intermediate-layer activations are analyzed for harmful vs. benign prompts.
    - **Key Findings**: Activations for harmful and benign prompts occupy overlapping regions; no safety-specific linear direction exists in activation space either.

### Loss & Training

Standard fine-tuning is employed throughout: useful data uses a 20K subset of MetaMathQA; harmful data uses a 4K unsafe subset of BeaverTails; mixed data consists of 20% harmful + 80% useful. Safety fine-tuning uses BeaverTails entries with `is_safe=True` (a different split from the harmful fine-tuning data to avoid methodological circularity). Harmfulness is evaluated by GPT-4o-mini scoring AdvBench outputs on a 1–5 scale.

## Key Experimental Results

### Parallel Projection Experiment (Qwen-2.5 1.5B)

| Method | SVD 0.01 | 0.25 | 0.50 | 0.75 | 0.99 | Full FT |
|--------|----------|------|------|------|------|---------|
| Top-K (Utility↑) | 0.50 | 0.53 | 0.55 | 0.57 | 0.58 | 0.61 |
| Random (Utility↑) | 0.49 | 0.50 | 0.53 | 0.53 | 0.56 | 0.61 |
| Top-K (Harm↓) | 1.62 | 1.80 | 1.92 | 1.90 | 1.97 | 2.09 |
| Random (Harm↓) | 1.56 | 1.65 | 1.74 | 1.83 | 1.95 | 2.09 |

### Orthogonal Projection Experiment (Mixed Fine-Tuning, Qwen-2.5 1.5B)

| Method | SVD 0.01 | 0.25 | 0.50 | 0.75 | 0.99 | Full FT |
|--------|----------|------|------|------|------|---------|
| Top-K (Utility↑) | 0.50 | 0.53 | 0.55 | 0.57 | 0.58 | 0.60 |
| Top-K (Harm↓) | 1.58 | 1.65 | 1.80 | 1.91 | 1.92 | 2.16 |

### Ablation Study

| Configuration | Description |
|---------------|-------------|
| Alignment subspace $\Delta_A$ | Amplifies both useful and harmful behavior; no safety specificity |
| Safety subspace $\Delta_S$ | Similarly amplifies both behaviors; no selectivity |
| Random-K control | $k$ singular vectors selected randomly; behavioral impact weaker than Top-K |
| Random control | SVD of random matrix; weakest behavioral impact |

### Key Findings

- **Core negative conclusion**: Across five LLMs (Llama 3.2 1B, Llama 2 7B, Qwen-2.5 1B/3B/7B), no linearly separable safety subspace is consistently observed.
- Top-$k$ alignment directions simultaneously amplify utility and harmfulness — they are "high-impact learning directions," not "safety directions."
- Orthogonal projection cannot selectively remove harmfulness: removing top-$k$ directions degrades utility faster than harmfulness.
- MSO analysis: harmful–safety update overlap is no greater than useful–safety overlap, refuting the hypothesis of a special geometric relationship between safety subspaces and harmful behavior.

## Highlights & Insights

- The **progressive experimental design** is rigorous — projection effects, orthogonal removal, geometric overlap, and activation space are mutually corroborating across four angles, lending strong credibility to the negative conclusion.
- The finding that "top-$k$ directions amplify all behaviors" provides an important insight: the principal directions identified by alignment/safety training are not safety-specific but are general high-sensitivity learning directions.
- The control experiments (Random-K, Random) ensure that the conclusions stem from the content of the subspace rather than from the projection operation itself.

## Limitations & Future Work

- The paper only refutes the separability of **linear** subspaces; nonlinear methods (e.g., manifold learning, kernel methods) are not ruled out.
- Harmfulness evaluation relies on GPT-4o-mini scoring, which introduces evaluator bias and instability.
- Safety fine-tuning data (`is_safe=True` from BeaverTails) and harmful fine-tuning data (unsafe BeaverTails) originate from different splits of the same dataset, potentially introducing methodological dependency.
- Whether different geometric structures might emerge in larger-scale models (e.g., 70B, 405B) remains unexplored.
- The paper primarily presents negative results without proposing alternative defense strategies.

## Related Work & Insights

- **vs. SafeLoRA**: SafeLoRA assumes safety information concentrates in specific directions of LoRA updates and protects them via projection. This paper's conclusions directly challenge the foundation of that assumption.
- **vs. LDIFS**: LDIFS defends against fine-tuning attacks by identifying "safety directions" and constraining updates accordingly. This paper demonstrates that such "safety directions" are simultaneously "utility directions," so constraining them incurs proportional task performance loss.
- **vs. Refusal Direction research**: Some works show that refusing behavior can be eliminated by ablating specific directions. The present findings are consistent with but go further — ablating these directions simultaneously removes useful behavior, as both share the same high-impact directions.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Rigorously and systematically tests a widely adopted but previously unvalidated assumption.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five models, four experimental angles, multiple control conditions, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear problem motivation, rigorous experimental logic, and precise formulation of conclusions.
- **Value**: ⭐⭐⭐⭐⭐ Raises fundamental challenges to the subspace-based defense paradigm in LLM safety, with far-reaching implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Why Supervised Fine-Tuning Fails to Learn: A Systematic Study of Incomplete Learning in Large Language Models](../../ACL2026/llm_alignment/why_supervised_fine-tuning_fails_to_learn_a_systematic_study_of_incomplete_learn.md)
- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models](a2d_any-order_any-step_safety_alignment_for_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
