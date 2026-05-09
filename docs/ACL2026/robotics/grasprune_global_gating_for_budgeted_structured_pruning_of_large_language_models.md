---
title: >-
  [Paper Note] GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models
description: >-
  [ACL 2026][Robotics][Structured Pruning] GRASPrune proposes a globally budget-constrained structured pruning framework that enforces hard mask budget constraints at every training step via Projected Straight-Through Estimator (Projected STE), jointly pruning FFN channels and KV head groups, achieving 12.18 PPL at 50% parameter retention on LLaMA-2-7B with only 6 minutes of single A100 training.
tags:
  - ACL 2026
  - Robotics
  - Structured Pruning
  - Global Budget
  - Gating Learning
  - KV Head Pruning
  - Projected STE
content_hash: 0d9a6141308f4b17
---

# GRASPrune: Global Gating for Budgeted Structured Pruning of Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.19398](https://arxiv.org/abs/2604.19398)
**Code**: [GitHub](https://github.com/ZiY-Wang/GRASPrune)
**Area**: Robotics & Embodied AI
**Keywords**: Structured Pruning, Global Budget, Gating Learning, KV Head Pruning, Projected STE

## TL;DR
GRASPrune proposes a globally budget-constrained structured pruning framework that enforces hard mask budget constraints at every training step via Projected Straight-Through Estimator (Projected STE), jointly pruning FFN channels and KV head groups, achieving 12.18 PPL at 50% parameter retention on LLaMA-2-7B with only 6 minutes of single A100 training.

## Method

### Key Designs

1. **Global Budget Joint Pruning**: FFN channels and KV head groups compete under a single budget with heterogeneous unit costs. FFN channel $c_i=1$, KV head group $c_i=\alpha$ where $\alpha = \frac{(2G+2)d_h}{3}$.

2. **Projected STE**: Projects continuous gate probabilities $\mathbf{p}$ into budget-feasible hard masks every step via greedy ranking by $p_i$ (not $p_i/c_i$). Forward uses hard mask $m_i$, backward uses soft probability $p_i$ via STE.

3. **Budget-Preserving Scale Calibration**: Post-pruning scalar multipliers $\gamma_i$ for retained units, folded into sliced weights for zero inference overhead.

## Key Experimental Results

| Retention | Method | Wiki PPL↓ |
|-----------|--------|----------|
| 50% | LLM-Pruner | ~18 |
| 50% | **GRASPrune** | **12.18** |

## Highlights & Insights
- "Learning under constraints" vs "learning then constraining" — a deep insight addressing a widely overlooked issue in structured pruning
- Extremely low training cost (6 minutes single GPU) makes the method highly practical

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[ICLR 2026\] JULI: Jailbreak Large Language Models by Self-Introspection](../../ICLR2026/robotics/juli_jailbreak_large_language_models_by_self-introspection.md)
- [\[ICLR 2026\] Sysformer: Safeguarding Frozen Large Language Models with Adaptive System Prompts](../../ICLR2026/robotics/sysformer_safeguarding_frozen_large_language_models_with_adaptive_system_prompts.md)
- [\[NeurIPS 2025\] Uncovering Strategic Egoism Behaviors in Large Language Models](../../NeurIPS2025/robotics/uncovering_strategic_egoism_behaviors_in_large_language_models.md)

</div>

<!-- RELATED:END -->
