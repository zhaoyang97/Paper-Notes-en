---
title: >-
  [Paper Note] DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models
description: >-
  [ICLR 2026][Prompt optimization] This work integrates distributionally robust optimization (DRO) into a Bayesian optimization framework for zero-shot instruction optimization, enabling optimized instructions to maintain reliable performance under distribution shift and adversarial evaluation conditions.
tags:
  - ICLR 2026
  - Prompt optimization
  - distributionally robust optimization
  - Bayesian optimization
  - zero-shot instruction learning
  - black-box LLM
date: 2026-05-08
content_hash: 65f4f2fb67a94862
---

# DRO-InstructZero: Distributionally Robust Prompt Optimization for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2510.15260](https://arxiv.org/abs/2510.15260)
**Code**: Not released
**Area**: Code Intelligence
**Keywords**: Prompt optimization, distributionally robust optimization, Bayesian optimization, zero-shot instruction learning, black-box LLM

## TL;DR

This work integrates distributionally robust optimization (DRO) into a Bayesian optimization framework for zero-shot instruction optimization, enabling optimized instructions to maintain reliable performance under distribution shift and adversarial evaluation conditions.

## Background & Motivation

Large language models are highly sensitive to prompt phrasing — even minor paraphrases of strong instructions can cause accuracy drops, and instructions effective in one evaluation setting often fail to transfer to slightly different domains. Existing automatic prompt search methods, including InstructZero, suffer from the following core issues:

1. **Distribution dependence**: Optimizing expected performance relies on a fixed validation distribution, ignoring the possibility of distribution shift.
2. **Fragility**: Optimized instructions tend to overfit the training distribution, with performance degrading sharply under adversarial conditions or domain mismatch.
3. **Practical demand**: Input distributions inevitably shift during real-world deployment, necessitating robust instruction optimization strategies.

While InstructZero demonstrates that Bayesian optimization (BO) is an effective framework for instruction optimization, its EI/UCB acquisition functions optimize only the expected score — an inherently average-case objective.

## Method

### Overall Architecture

DRO-InstructZero builds on the InstructZero pipeline: an open-source LLM (Vicuna) generates candidate instructions → a black-box LLM (ChatGPT) evaluates them → Bayesian optimization iteratively refines soft prompts. The core innovation replaces the acquisition function from an expected-case objective to a distributionally robust objective.

### Key Designs

**Standard objective from InstructZero:**

$$\max_{v \in \mathcal{V}} \; \mathbb{E}_{(X,Y) \sim \mathcal{D}_t} \left[ h(f([v;X]), Y) \right]$$

**Extended distributionally robust objective:**

$$\max_{v \in V} \; \inf_{Q \in \mathcal{U}(D^t)} \; \mathbb{E}_{(X,Y) \sim Q} \left[ h(f([v;X]), Y) \right]$$

where $\mathcal{U}(D^t)$ is an ambiguity set centered at a reference distribution $w_{\text{ref}}$, measured by $f$-divergence (KL divergence), with radius $\epsilon$.

**Continuous relaxation**: The DRO objective over the discrete instruction space is reformulated as a low-dimensional continuous black-box function:

$$H(p) \triangleq \inf_{Q \in \mathcal{U}(D^t)} \; \mathbb{E}_{(X,Y) \sim Q} \left[ h(f([g([Ap; \text{exemplars}]); X]), Y) \right]$$

**Robust acquisition rule**: For each candidate soft prompt $p_m$, an optimistic upper bound is first computed:

$$\text{ucb}_m = \left[ \mu^t(p_m) + \beta(m) \sigma^t(p_m) \right]_t$$

Worst-case weights are then solved within the ambiguity set:

$$w_m^* = \arg\min_{w': \|w' - w_{\text{ref}}\|_{\mathcal{M}} \leq \epsilon(m)} \langle \text{ucb}_m, w' \rangle$$

The prompt maximizing the robust acquisition value is selected:

$$p_{m+1} = \arg\max_p \langle \text{ucb}_m, w_m^* \rangle$$

**Instruction-coupled kernel (DRO extension)**:

$$\mathbf{K}_{ij}^t = l(p_i, p_j)^\top L^{-1} S L^{-1} l(p_j, p_i)$$

where $S$ is further weighted by the adversarial distribution $w^*$, jointly accounting for semantic similarity and distributional robustness.

### Loss & Training

- CMA-ES evolutionary strategy is used to search for optimal soft prompts.
- The mini-batch variant explores 25 soft prompts per round.
- The reference distribution $w_{\text{ref}}$ is updated via exponential moving average.
- Exploration coefficient: $\beta(t) = 2.0 \cdot \sqrt{2.0 \cdot \log(t+1)}$.
- Ambiguity radius: $\epsilon = 0.1$ (fixed constant).
- Adversarial weights are solved via the cvxpy convex optimization solver.
- Joint optimization is performed across 2 tasks per round.

## Key Experimental Results

### Main Results

**Overall performance on 32 BIG-Bench tasks:**

| Metric | InstructZero | DRO-InstructZero | Gain |
|--------|-------------|-----------------|------|
| Average accuracy | 0.719 | 0.756 | +3.6 pts |
| Median per-task gain | — | — | +5.5 pts |
| Win/Tie/Loss | — | 18/8/6 | — |

**Detailed results on representative tasks:**

| Task | InstructZero | DRO-InstructZero | Gain |
|------|-------------|-----------------|------|
| Translation (EN-DE/ES/FR) | 0.867 | 0.980 | +11.3 pts |
| Auto-Debugging (shift) | baseline | — | +25 pts |
| Formality Rewriting (shift) | 61.3±0.7% | 85–90% | +25–30 pts |
| Cause-and-Effect (in-dist.) | ≥96% | ≥96% | No degradation |
| Unscrambling | 0.67 | 0.80 | +13 pts |
| Second Letter | 0.62 | 0.74 | +12 pts |
| Taxonomy | 0.82 | 0.92 | +10 pts |
| Sentiment | 0.93 | 0.99 | +6 pts |

**Saturated tasks remain at 100%:** Sum, Periodic, Passivation, Num2Verbal, Letters List, First Letter, Diff

### Ablation Study

| Method | In-Dist. (ID) | Shift | Notes |
|--------|-------------|-------|-------|
| InstructZero-EI | Strong | 61.3±0.7% | Sharp degradation under shift |
| InstructZero-UCB | Moderate | Moderate | Standard BO alternative |
| DRO w/o BO | Weak | Moderate | Demonstrates necessity of BO search |
| DRO-InstructZero | Strong | 85–90% | Best overall |

Key ablation findings:
- DRO outperforms EI/UCB acquisition functions by 15–25 absolute percentage points under distribution shift.
- "DRO w/o BO" underperforms the full method, confirming that latent-space Bayesian search is critical for efficiency and scalability.
- The combination of DRO and BO is essential: DRO provides the robustness objective while BO enables efficient search.

### Key Findings

1. **Robustness gains stem from principled design**: The improvements are not a byproduct of simple regularization but arise from replacing the average-case acquisition function with a distributionally robust counterpart.
2. **Regression on a minority of tasks**: Antonyms −11 pts, Object Counting −10 pts, CS-algorithm −8 pts — occurring when worst-case weighting emphasizes patterns that diverge from the exact lexical rules used by the evaluator.
3. **Query efficiency is preserved**: The same query budget as InstructZero is used, with no additional API overhead.

## Highlights & Insights

1. **Theoretical elegance**: Combining DRO with BO for prompt learning is a natural formalization; intuitively, worst-case optimization is precisely what is required for practical deployment.
2. **Plug-and-play design**: Modifying only the acquisition function allows any BO-based prompt optimization method to be upgraded to a robust variant.
3. **Distribution shift is a genuine pain point in prompt optimization**: The paper accurately identifies a critical issue overlooked by existing methods.
4. **Large gains on translation tasks** (+11.3 pts) suggest that robust optimization is especially valuable in cross-lingual settings.

## Limitations & Future Work

1. **Adversarial reweighting increases computational complexity**: Each iteration incurs additional overhead, particularly from the cvxpy convex optimization solver.
2. **Fixed divergence measure and ambiguity radius**: The current use of fixed KL divergence and $\epsilon = 0.1$ may not be appropriate for all forms of distributional uncertainty.
3. **Experiments constrained by API costs**: Only ChatGPT is used as the black-box LLM; experiments are not extended to stronger models such as GPT-4.
4. **Regression on a minority of lexical/classification tasks** requires further analysis — ablations on mixed acquisition functions are deferred to the appendix.
5. **Multimodal and reasoning-intensive settings remain unexplored.**

## Related Work & Insights

- **InstructZero** (Chen et al., 2024) is the direct foundation; this paper's contribution lies in adding a DRO layer on top of it.
- **DRBO** (Kirschner et al., 2020) provides the theoretical basis for distributionally robust Bayesian optimization.
- **APE / Auto-prompt line**: Works such as Zhou et al., 2022 automate prompt design via alternative approaches.
- Inspiration: The DRO paradigm is extensible to other search-based LLM optimization settings, such as chain-of-thought optimization and RLHF reward modeling.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of DRO and BO for prompt optimization is the first of its kind in this area, with elegant formalization.
- **Technical Depth**: ⭐⭐⭐⭐ — Solid theoretical grounding (information theory + convex optimization + Gaussian processes) with well-designed algorithms.
- **Experimental Thoroughness**: ⭐⭐⭐ — Broad coverage across 32 tasks, but lacks stronger baselines (e.g., GPT-4) and more detailed ablations.
- **Practicality**: ⭐⭐⭐⭐ — Plug-and-play design facilitates easy integration into existing pipelines.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated problem motivation.

**Overall**: ⭐⭐⭐⭐ (4/5) — A well-directed and elegantly formalized work that effectively bridges robust optimization theory with LLM prompt engineering, though experimental depth and model coverage leave room for improvement.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Training Large Language Models To Reason In Parallel With Global Forking Tokens](training_large_language_models_to_reason_in_parallel_with_global_forking_tokens.md)
- [\[ICLR 2026\] ShieldedCode: Learning Robust Representations for Virtual Machine Protected Code](shieldedcode_learning_robust_representations_for_virtual_machine_protected_code.md)
- [\[ICLR 2026\] A Problem-Oriented Perspective and Anchor Verification for Code Optimization](a_problem-oriented_perspective_and_anchor_verification_for_code_optimization.md)
- [\[ICLR 2026\] Sharing State Between Prompts and Programs](sharing_state_between_prompts_and_programs.md)
- [\[ICLR 2026\] ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory](reasoningbank_scaling_agent_self-evolving_with_reasoning_memory.md)

<!-- RELATED:END -->
