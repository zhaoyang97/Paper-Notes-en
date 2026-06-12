---
title: >-
  [Paper Note] Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem
description: >-
  [ACL 2026][LLM Safety][LLM Unlearning] LLM unlearning is explicitly modeled as an asymmetric two-task problem where "retention is primary and forgetting is secondary." The proposed SAGO applies **element-wise sign alignm…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "LLM Unlearning"
  - "gradient conflict"
  - "multi-task learning"
  - "PCGrad"
  - "sign alignment"
date: 2026-05-08
content_hash: 6ea6a6ea7a41da73
---

# Modeling LLM Unlearning as an Asymmetric Two-Task Learning Problem

**Conference**: ACL 2026  
**arXiv**: [2604.14808](https://arxiv.org/abs/2604.14808)  
**Code**: https://github.com/sustech-nlp/SAGO  
**Area**: LLM Safety / Machine Unlearning  
**Keywords**: LLM Unlearning, gradient conflict, multi-task learning, PCGrad, sign alignment

## TL;DR
LLM unlearning is explicitly modeled as an asymmetric two-task problem where "retention is primary and forgetting is secondary." The proposed SAGO applies **element-wise sign alignment gating** to retain/forget gradients, bringing retention performance close to the original model on WMDP and RWKU benchmarks with almost no loss in forgetting effectiveness.

## Background & Motivation

**Background**: The mainstream approach for LLM unlearning is to perform gradient ascent (GA) on a forget set while conducting gradient descent (GD) on a retain set—the GradDiff paradigm. Subsequent works like NPO and SimNPO reframed the unbounded GA into bounded forms using sigmoid functions to mitigate catastrophic collapse.

**Limitations of Prior Work**: Most combinations of forget-objective and retain-objective treat the two tasks as "symmetric" multi-task learning by performing weighted loss summation. This results in systematic conflicts between forget and retain gradients, leading to continuous degradation in retention: for instance, after running SimNPO+GD on WMDP Bio, MMLU drops to 26.7 (original 59.8), meaning only 44.6% of retention is recovered.

**Key Challenge**: The unlearning task is inherently **asymmetric**—retention is the "primary task" (do-no-harm), while forgetting is merely the "subsidiary task." Current methods rely on loss balancing and fail to distinguish priorities from a gradient geometry perspective, leaving the issue of the retain gradient being "pulled away" by the forget gradient unaddressed.

**Goal**: Model unlearning explicitly as asymmetric two-task learning, treating the retain gradient as the "anchor direction" and only allowing forget signals to be injected into dimensions that do not harm retention.

**Key Insight**: The authors noted that (1) PCGrad in multi-task learning projects conflicting gradients onto the orthogonal complement, yet it has not been systematically applied in unlearning; (2) at the parameter level, forget and retain gradient signs sometimes align and sometimes conflict—dimensions with consistent signs are "task-specific" and safe for forgetting, while opposite signs indicate general knowledge that must be protected by the retain signal.

**Core Idea**: Use "element-wise sign gating" instead of "loss weighting." In dimensions where retain/forget signs match, the forget signal is passed; where signs differ, only the retain signal is kept. This mathematically ensures that the final update direction maintains a non-negative angle with the retain gradient and does not oppose it at any coordinate.

## Method

### Overall Architecture
SAGO is a modular two-stage iterative framework (Algorithm 1): at each step, a mini-batch is sampled from both the forget set $\mathcal{D}_f$ and retain set $\mathcal{D}_r$ to compute gradients $g_f^t$ and $g_r^t$. A plug-and-play `CombineGradients` module merges them into the final update direction $g_{\text{final}}^t$, and parameters are updated as $\theta^t = \theta^{t-1} - \eta\, g_{\text{final}}^t$. The framework is decoupled from the forget objective, supporting GA+GD, NPO+GD, or SimNPO+GD. Two implementations of `CombineGradients` are provided: module-level PCGrad and SAGO.

### Key Designs

1.  **Module-level PCGrad (per-module projection)**:
    - **Function**: When the angle between forget and retain gradients in a specific module exceeds 90°, the forget gradient is projected onto the normal plane of the retain gradient to remove conflicting components.
    - **Mechanism**: For each module $j$, it independently calculates $\tilde{g}_f^j = g_f^j - \frac{g_f^j \cdot g_r^j}{\|g_r^j\|^2} g_r^j$, then synthesizes $g_{\text{final}}^j = \alpha g_r^j + \gamma \tilde{g}_f^j$. Module-wise rather than global flattening avoids the entire network being adjusted due to local conflicts.
    - **Design Motivation**: Original PCGrad (used in methods like GRU) projects onto flattened vectors, which is too coarse; module-wise projection empirically yields higher retention (Table 2 shows Global PCGrad MMLU 51.0 vs. module-wise PCGrad 53.0).

2.  **SAGO: Element-wise Sign Alignment Gating**:
    - **Function**: Uses indicator functions to partition the two gradients into two parts with disjoint supports based on sign consistency before synthesis.
    - **Mechanism**: For each parameter dimension, it evaluates the sign of $g_f \odot g_r$. Consistent signs (task-specific dimensions) retain the forget signal $\tilde{g}_f = g_f \odot \mathbb{I}(g_f \odot g_r \ge 0)$; opposite signs (general knowledge dimensions) retain the retain signal $\tilde{g}_r = g_r \odot \mathbb{I}(g_f \odot g_r < 0)$. The final update is $g_{\text{final}} = \alpha \tilde{g}_r + \gamma \tilde{g}_f$. The two paths are naturally orthogonal ($\tilde{g}_f^\top \tilde{g}_r = 0$) due to disjoint supports, and no coordinate opposes $g_r$.
    - **Design Motivation**: PCGrad only guarantees the overall angle is $\ge 90^\circ$, but certain dimensions might still "flip" the final direction against the retain gradient if $|\tilde{g}_f^i| > |g_r^i|$. SAGO prevents this at the element level, making retention more geometrically stable.

3.  **Theory: Comparison of Cosine Similarity**:
    - **Function**: Establishes lower bounds for "retention alignment" for PCGrad and SAGO from a cosine similarity perspective.
    - **Mechanism**: For PCGrad, orthogonal projection yields $\cos\theta_P = (1 + \|\tilde{g}_f\|^2 / \|g_r\|^2)^{-1/2} \ge 0$. For SAGO, disjoint supports result in a $\cos\theta_S$ numerator $= \sum_{i\in C}(g_r^i)^2 + \sum_{i\in S} g_f^i g_r^i$, where $g_f^i g_r^i > 0$ on $S$. Structurally, SAGO provides more positive contribution than PCGrad, thus strictly aligning more tightly with $g_r$ under equal weighting $\alpha=\gamma=1$.
    - **Design Motivation**: Transforms "retention-first" from an intuition into a provable geometric property, justifying the engineering choice of element-wise gating.

### Loss & Training
The forget objective $\mathcal{L}_f$ can be GA, NPO, or SimNPO, while the retain objective is fixed as standard cross-entropy GD. Weights are defaulted to $\alpha=\gamma=1.0$, with sweeping in $[0.1, 1.0]$ when necessary to align the forget strength of baselines (forget metrics are aligned before comparing retention to avoid misleading improvements). WMDP is run for 100 steps, and RWKU for 2 epochs.

## Key Experimental Results

### Main Results
Evaluations on WMDP (Bio + Cyber, Zephyr-7B-beta) and RWKU (bulk unlearning of 50 personas, LLaMA3-8B-Instruct) using three groups of base objective × {naive, +PCGrad, +SAGO}:

| Configuration | WMDP Bio Forget↓ | WMDP Bio MMLU↑ | WMDP Cyber MMLU↑ | RWKU Neighbor All↑ |
|:---|:---:|:---:|:---:|:---:|
| Target Model | 64.4 | 59.8 | 59.8 | 85.1 |
| SimNPO + GD (naive) | 26.1 | 26.7 | 51.4 | 44.9 |
| SimNPO + GD + PCGrad | 28.7 | 56.4 | 57.9 | 53.5 |
| SimNPO + GD + **SAGO** | 28.2 | **57.4** | **58.3** | **56.9** |
| NPO + GD (naive) | 30.5 | 38.2 | 46.8 | 13.1 |
| NPO + GD + **SAGO** | 30.0 | **56.0** | **58.5** | **41.7** |
| GA + GD (naive) | 24.7 | 25.1 | 55.6 | 15.5 |
| GA + GD + **SAGO** | 26.0 | **54.1** | **59.7** | **32.1** |

SAGO increases the MMLU of SimNPO+GD from 26.7 to 57.4, recovering 96.0% of the original model's retention, while forgetting barely degrades. On RWKU, Neighbor retention increases by +3.4 to +13.3 points over PCGrad.

### Ablation Study
Comparison of SAGO with existing conflict mitigation methods (Global PCGrad from GRU, BLUR, module-wise PCGrad), and gradient geometry analysis:

| Configuration (WMDP Bio) | Forget↓ | MMLU↑ | Description |
|:---|:---:|:---:|:---|
| NPO + GD + GRU | 26.2 | 42.1 | Flattened global PCGrad, coarse grain |
| RMU + BLUR | 27.6 | 53.5 | Bi-level optimization |
| GA + GD + Global PCGrad | 24.8 | 51.0 | Flattened global projection |
| GA + GD + module-wise PCGrad | 24.5 | 53.0 | Module granularity, +2.0 MMLU |
| GA + GD + **SAGO** | 26.0 | **54.1** | Element granularity, further +1.1 MMLU |

Gradient Geometry (Mean over 100 steps, WMDP Cyber):

| Method | Forget-Retain cos | Comb-Forget cos | Comb-Retain cos |
|:---|:---:|:---:|:---:|
| GradDiff | −0.40 | **0.50** | 0.42 |
| PCGrad | −0.52 | 0.29 | 0.52 |
| **SAGO** | **−0.35** | 0.17 | **0.57** |

### Key Findings
- **Finer granularity is better**: Moving from global PCGrad to module-wise PCGrad and finally to element-wise SAGO leads to a monotonic increase in retention, validating that gradient conflicts should be resolved locally at the finest granularity.
- **SAGO reduces intrinsic Forget-Retain conflict**: The cosine similarity improved from −0.52 (PCGrad) to −0.35, suggesting this sign alignment implicitly reshapes the parameter space, making the two original tasks naturally less antagonistic.
- **"Purified" forget signal**: SAGO's Comb-Forget cosine is only 0.17 (compared to PCGrad's 0.29 and GradDiff's 0.50), yet forget metrics remain stable—indicating that removed forget components were "impurities" conflicting with retention.
- **Pushing the Pareto frontier**: Figures 3/4 show that SAGO significantly elevates retention at the same forget levels, even strictly dominating all previous points on RWKU.

## Highlights & Insights
- **Explicit modeling of task asymmetry**: Previous unlearning research treated it as symmetric multi-tasking; this work shifts the methodology from loss balancing to gradient geometry by asserting "retention is the primary task."
- **Evolution from "projection" to "sign-based gating"**: While PCGrad removes conflicting components, it can still cause single-dimension reversals. SAGO prevents these at the element level, proving geometrically superior—a technique transferable to any scenario where a primary direction must remain undisturbed.
- **Lightweight and plug-and-play**: The algorithm requires only one sign comparison and one indicator mask per step, adding almost zero computational overhead while being compatible with any forget objective.
- **Consistency between theory and phenomena**: The proof that SAGO's cosine alignment is strictly tighter ($\cos\theta_S > \cos\theta_P$ under equal weights) is validated by real measurements (Comb-Retain cos 0.57 vs. 0.52).

## Limitations & Future Work
- **Ours**: (1) Validation is limited to text benchmarks (WMDP/RWKU), excluding multimodal or code models; (2) requires storing both forget and retain gradients, doubling VRAM usage compared to single-objective training.
- **Self-identified**: (1) Strict cosine alignment guarantees only hold for equal weights ($\alpha=\gamma=1$), yet hyperparameter sweeping in practice might violate this; (2) element-level sign comparison is a hard threshold susceptible to gradient noise (e.g., in low-rank LoRA training); (3) SAGO's "absolute trust in the retain direction" could be detrimental if the retain set is biased.
- **Future Improvements**: Replacing hard indicators with soft gating (e.g., sigmoid of $g_f \cdot g_r$) or introducing EMA for retain gradient estimation could improve robustness; applying SAGO to RLHF or continual learning is a promising direction.

## Related Work & Insights
- **vs. GRU (ICML'25)**: GRU uses PCGrad on globally flattened vectors; Ours demonstrates that module-wise yields +2 MMLU and element-wise (SAGO) adds another +1.1 MMLU, emphasizing granularity.
- **vs. BLUR (Reisizadeh et al. 2025)**: BLUR uses bi-level optimization with **forgetting as the outer loop**; Ours prioritizes retention and modifies only gradient synthesis without changing the optimization structure, making it simpler and more effective for retention (54.1 vs 53.5 on Bio).
- **vs. NPO / SimNPO**: These methods only modify the forget objective to be bounded; SAGO is orthogonal and can be layered on top to gain +30 MMLU in some cases.
- **vs. MTL methods (CAGrad / Nash-MTL)**: These aim for "task fairness," which protects the progress of the forget task—harmful in the asymmetric unlearning context.

## Rating
- Novelty: ⭐⭐⭐⭐ Clean perspective shift (asymmetric tasks + retention anchor); element-wise sign gating is a natural but effective extension of PCGrad.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 3 bases × 2 benchmarks × 3 conflict-mitigation baselines, plus gradient geometry and Pareto analysis.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from motivation to theory and experiments; clear formulas and figures, though a computational overhead table is missing.
- Value: ⭐⭐⭐⭐ Plug-and-play with minimal overhead and significant retention gains; directly applicable to LLM safety and privacy scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[AAAI 2026\] ALTER: Asymmetric LoRA for Token-Entropy-Guided Unlearning of LLMs](../../AAAI2026/llm_safety/alter_asymmetric_lora_for_token-entropy-guided_unlearning_of.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_safety/llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] Maximizing Local Entropy Where It Matters: Prefix-Aware Localized LLM Unlearning](maximizing_local_entropy_where_it_matters_prefix-aware_localized_llm_unlearning.md)
- [\[ACL 2026\] Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning](unlearners_can_lie_evaluating_and_improving_honesty_in_llm_unlearning.md)

</div>

<!-- RELATED:END -->
