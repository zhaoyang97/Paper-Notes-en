---
title: >-
  [Paper Note] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization
description: >-
  [ACL 2026][search agent] CW-GRPO reframes process supervision as "advantage redistribution": a LLM judge evaluates the retrieval utility and reasoning correctness of each search turn, computes a contribution score to rescale outcome-based advantages, and achieves turn-level credit assignment without introducing an unstable value function. The approach outperforms standard GRPO by 5.0% on Qwen3-8B.
tags:
  - ACL 2026
  - search agent
  - GRPO
  - contribution weighting
  - process supervision
  - credit assignment
date: 2026-05-08
content_hash: c238afb4c802f871
---

# Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization

**Conference**: ACL 2026
**arXiv**: [2604.14267](https://arxiv.org/abs/2604.14267)
**Code**: [GitHub](https://github.com/zsxmwjz/CW-GRPO)
**Area**: Information Retrieval
**Keywords**: search agent, GRPO, contribution weighting, process supervision, credit assignment

## TL;DR
CW-GRPO reframes process supervision as "advantage redistribution": a LLM judge evaluates the retrieval utility and reasoning correctness of each search turn, computes a contribution score to rescale outcome-based advantages, and achieves turn-level credit assignment without introducing an unstable value function. The approach outperforms standard GRPO by 5.0% on Qwen3-8B.

## Background & Motivation

**Background**: Search agents (e.g., Search-R1, R1-Searcher) enhance LLM factual reliability through iterative retrieval of external evidence. Training paradigms fall into two categories: process supervision (turn-level rewards + PPO) and outcome supervision (final-answer reward + GRPO).

**Limitations of Prior Work**: Process supervision requires learning a value function for turn-level reward estimation, but the diversity of intermediate states leads to unstable estimates and brittle training. Outcome supervision (GRPO) is stable but yields sparse reward signals—assigning equal credit to all search turns in a successful trajectory without distinguishing pivotal from redundant searches.

**Key Challenge**: Process supervision is fine-grained but unstable; outcome supervision is stable but coarse-grained—a balance between the two is needed.

**Goal**: Achieve turn-level credit assignment while preserving the training stability of GRPO.

**Key Insight**: Rather than directly optimizing process rewards, the method uses process signals to modulate (rescale) outcome advantages—treating process supervision as an advantage redistribution problem.

**Core Idea**: A LLM judge evaluates the retrieval utility $u$ and reasoning correctness $v$ of each turn → a conjunctive contribution score $p = u \cdot v$ → outcome advantages are redistributed to high-contribution turns via temperature-controlled softmax.

## Method

### Overall Architecture
For each question, $G$ trajectories are sampled and outcome advantages $A_i^O$ are computed via within-group relative comparison. For each turn in successful trajectories, a LLM judge evaluates retrieval utility and reasoning correctness, computes a conjunctive contribution score, and redistributes advantages via softmax. Failed trajectories retain uniform distribution. The policy is optimized with a clipped surrogate objective.

### Key Designs

1. **Conjunctive Contribution Signal**:

    - Function: Identify search turns that causally contribute to task success.
    - Mechanism: Two orthogonal binary signals are evaluated per turn—retrieval utility $u_i^t$ (whether new, task-relevant evidence is retrieved) and reasoning correctness $v_i^t$ (whether the reasoning chain correctly interprets the current context). The contribution score is their logical conjunction $p_i^t = u_i^t \cdot v_i^t$; a turn is credited only when it simultaneously satisfies "good retrieval" and "correct use of information."
    - Design Motivation: Useful retrieval with incorrect reasoning wastes good evidence; correct reasoning with useless retrieval is idle effort; only their conjunction constitutes genuine progress.

2. **Asymmetric Treatment of Successful and Failed Trajectories**:

    - Function: Avoid introducing noisy supervision when credit attribution is ambiguous.
    - Mechanism: For successful trajectories, a temperature-controlled softmax emphasizes high-contribution turns: $c_i^t = \exp(\alpha p_i^t) / \sum \exp(\alpha p_i^{t'})$. For failed trajectories, uniform distribution is applied: $c_i^t = 1/(T_i-1)$. Credit in successful trajectories can be reliably attributed (good turns lead to success), whereas attribution in failed trajectories is ambiguous (failure may stem from corpus gaps rather than agent decision errors).
    - Design Motivation: Failure attribution is far more difficult than success attribution—errors may originate from external factors rather than agent behavior. Uniform distribution preserves the stability of outcome supervision.

3. **Advantage-Preserving Redistribution**:

    - Function: Redistribute credit while preserving the total learning signal at the trajectory level.
    - Mechanism: The redistributed advantage is $A_i^t = A_i^O \cdot c_i^t \cdot (T_i-1)$, designed to guarantee $\frac{1}{T_i-1}\sum A_i^t = A_i^O$, i.e., the mean advantage within a trajectory remains unchanged. High-contribution turns receive amplified signals while low-contribution turns are suppressed, with the total preserved.
    - Design Motivation: Maintaining the same gradient magnitude as standard GRPO avoids training instability introduced by process signals.

### Loss & Training
Clipped surrogate objective: $\mathcal{L}(\theta) = -\mathbb{E}[\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)]$. The LLM judge achieves a 95% agreement rate with human experts (validated on annotations of 97 search turns).

## Key Experimental Results

### Main Results

| Model | Method | Performance Gain | Notes |
|-------|--------|-----------------|-------|
| Qwen3-8B | CW-GRPO vs. GRPO | +5.0% | Multiple knowledge-intensive benchmarks |
| Qwen3-1.7B | CW-GRPO vs. GRPO | +6.3% | Greater gains for smaller models |
| — | CW-GRPO vs. process supervision baselines | Consistently superior | Avoids value function instability |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Retrieval utility only | Below conjunctive | Single signal insufficient |
| Reasoning correctness only | Below conjunctive | Single signal insufficient |
| Contribution redistribution applied to failed trajectories | Worse than uniform | Validates necessity of asymmetric design |
| Varying temperature $\alpha$ | Optimal at moderate values | Too high over-concentrates; too low degenerates to GRPO |

### Key Findings
- In successful trajectories, contribution is highly concentrated in a small number of pivotal turns—a structural characteristic of search agent tasks.
- Smaller models (1.7B) benefit more from CW-GRPO (+6.3%), likely because fine-grained credit assignment is more critical for improving search efficiency in capacity-limited models.
- The 95% agreement rate between the LLM judge and human annotations validates the feasibility of LLM-based process evaluation.
- The difficulty of credit attribution in failed trajectories is a structural challenge—many failures do not arise from incorrect agent decisions.

## Highlights & Insights
- **Reframing process supervision as advantage redistribution** is an elegant conceptual shift—no value function is trained, no process rewards are directly optimized; instead, process signals modulate outcome advantages.
- The conjunctive contribution signal ($u \cdot v$) reflects the core nature of search tasks: good retrieval must be accompanied by correct interpretation, and neither alone suffices.
- The philosophy behind asymmetric treatment is profound—"we know success results from doing something right, but we do not necessarily know what went wrong in failure."

## Limitations & Future Work
- The LLM judge's own evaluations may be biased, particularly in assessing reasoning correctness.
- Validation is limited to knowledge-intensive QA tasks; applicability to other agentic tasks such as code generation remains to be explored.
- The temperature $\alpha$ is a hyperparameter requiring task-specific tuning.
- Binary contribution signals (0/1) may be overly coarse; continuous-valued assessment could offer finer granularity.

## Related Work & Insights
- **vs. Search-R1**: Search-R1 applies standard GRPO for outcome supervision; CW-GRPO adds turn-level credit assignment.
- **vs. PPO-based process supervision**: PPO requires learning a value function and suffers from training instability; CW-GRPO eliminates the value function entirely.
- **vs. PRM methods**: PRMs require human annotations at the turn level; CW-GRPO replaces these with a LLM judge.

## Rating
- Novelty: ⭐⭐⭐⭐ The reframing of process supervision as advantage redistribution is novel, and the conjunctive contribution signal is well-motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two model scales, multiple benchmarks, and judge calibration validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation chain is clear, method derivation is coherent, and the formulation is elegant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ACL 2026\] ChAIRO: Contextual Hierarchical Analogical Induction and Reasoning Optimization for LLMs](chairo_contextual_hierarchical_analogical_induction_and_reasoning_optimization_f.md)
- [\[ACL 2026\] CarO: Chain-of-Analogy Reasoning Optimization for Robust Content Moderation](caro_chain-of-analogy_reasoning_optimization_for_robust_content_moderation.md)
- [\[ACL 2026\] Enhancing Multilingual RAG Systems with Debiased Language Preference-Guided Query Fusion](enhancing_multilingual_rag_systems_with_debiased_language_preference-guided_quer.md)

</div>

<!-- RELATED:END -->
