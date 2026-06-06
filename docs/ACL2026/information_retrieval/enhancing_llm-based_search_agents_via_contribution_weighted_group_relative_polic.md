---
title: >-
  [Paper Note] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization
description: >-
  [ACL 2026][Information Retrieval & RAG][Search Agents] CW-GRPO redefines process supervision as "advantage redistribution": using an LLM judge to evaluate the retrieval utility and reasoning correctness of each search ro…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Search Agents"
  - "GRPO"
  - "Contribution Weighting"
  - "Process Supervision"
  - "Credit Assignment"
date: 2026-05-08
content_hash: b898182ab1b6dcb9
---

# Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization

**Conference**: ACL 2026  
**arXiv**: [2604.14267](https://arxiv.org/abs/2604.14267)  
**Code**: [GitHub](https://github.com/zsxmwjz/CW-GRPO)  
**Area**: Information Retrieval  
**Keywords**: Search Agents, GRPO, Contribution Weighting, Process Supervision, Credit Assignment

## TL;DR
CW-GRPO redefines process supervision as "advantage redistribution": using an LLM judge to evaluate the retrieval utility and reasoning correctness of each search round, calculating contribution scores to scale outcome-based advantages. This achieves round-level credit assignment without introducing unstable value functions, outperforming standard GRPO by 5.0% on Qwen3-8B.

## Background & Motivation

**Background**: Search agents (such as Search-R1, R1-Searcher) enhance the factual reliability of LLMs by iteratively retrieving external evidence. Training methods are divided into process supervision (round-level rewards + PPO) and outcome supervision (final answer rewards + GRPO).

**Limitations of Prior Work**: Process supervision requires learning a value function for round-level reward estimation, but diverse intermediate states lead to unstable estimation and fragile training. Outcome supervision (GRPO) features stable training but sparse reward signals—it assigns the same credit to all search rounds of a successful trajectory, failing to distinguish between critical and redundant searches.

**Key Challenge**: Process supervision is fine-grained but unstable, while outcome supervision is stable but coarse-grained—a balance must be found between the two.

**Goal**: To achieve round-level credit assignment while maintaining the training stability of GRPO.

**Key Insight**: Instead of directly optimizing process rewards, use process signals to modulate (rescale) outcome advantages—treating process supervision as an advantage redistribution problem.

**Core Idea**: An LLM judge evaluates the retrieval utility $u$ and reasoning correctness $v$ of each round → joint contribution score $p = u \cdot v$ → redistribute outcome advantages to high-contribution rounds through temperature softmax.

## Method

### Overall Architecture
For each problem, $G$ trajectories are sampled, and the outcome advantage $A_i^O$ is calculated (intra-group relative comparison). For each round of a successful trajectory, an LLM judge evaluates retrieval utility and reasoning correctness to calculate a joint contribution score, and the advantage is redistributed via softmax. Failed trajectories maintain uniform distribution. The policy is optimized using a clipped surrogate objective.

### Key Designs

1. **Conjunctive Contribution**:

    - **Function**: Identifies search rounds that truly make a causal contribution to task success.
    - **Mechanism**: Each round evaluates two orthogonal binary signals—retrieval utility $u_i^t$ (retrieving new, task-relevant evidence) and reasoning correctness $v_i^t$ (the reasoning chain correctly interprets the current context). The contribution score is the logical AND $p_i^t = u_i^t \cdot v_i^t$, where a contribution is only recognized if both "good information was retrieved" and "information was used correctly" are met.
    - **Design Motivation**: Useful retrieval but incorrect reasoning = wasting good evidence; correct reasoning but useless retrieval = idling; only the combination of both constitutes genuine progress.

2. **Asymmetric Treatment of Success/Failure Trajectories**:

    - **Function**: Avoids introducing noisy supervision when attribution is ambiguous.
    - **Mechanism**: Successful trajectories use temperature-controlled softmax to emphasize high-contribution rounds: $c_i^t = \exp(\alpha p_i^t) / \sum \exp(\alpha p_i^{t'})$. Failed trajectories are distributed uniformly $c_i^t = 1/(T_i-1)$. Contributions of successful trajectories can be reliably attributed (good rounds lead to success), but attribution for failed trajectories is ambiguous (it might be due to insufficient corpus coverage rather than agent decision errors).
    - **Design Motivation**: The difficulty of failure attribution is much higher than that of success attribution—errors may stem from external factors rather than agent behavior. Uniform distribution maintains the stability of outcome supervision.

3. **Advantage-Preserving Redistribution**:

    - **Function**: Redistributes credit while maintaining the total volume of trajectory-level learning signals.
    - **Mechanism**: The redistributed advantage $A_i^t = A_i^O \cdot c_i^t \cdot (T_i-1)$ is designed to ensure $\frac{1}{T_i-1}\sum A_i^t = A_i^O$, meaning the mean advantage within the trajectory remains unchanged. This implies that signals from high-contribution rounds are amplified and signals from low-contribution rounds are suppressed, but the total amount remains constant.
    - **Design Motivation**: Maintains the same gradient magnitude as the original GRPO, avoiding training instability introduced by process signals.

### Loss & Training
Clipped surrogate objective: $\mathcal{L}(\theta) = -\mathbb{E}[\min(rA, \text{clip}(r, 1-\epsilon, 1+\epsilon)A)]$. The LLM judge achieves a 95% consensus rate with human experts (validated by annotation of 97 search rounds).

## Key Experimental Results

### Main Results

| Model | Method | Gain | Description |
|------|------|---------|------|
| Qwen3-8B | CW-GRPO vs GRPO | +5.0% | Multiple knowledge-intensive benchmarks |
| Qwen3-1.7B | CW-GRPO vs GRPO | +6.3% | Larger gains for small models |
| - | CW-GRPO vs Process Supervision Baseline | Consistently Superior | Avoids value function instability |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Retrieval Utility Only | Lower than Joint | Single signal is insufficient |
| Reasoning Correctness Only | Lower than Joint | Single signal is insufficient |
| Contribution Assignment for Failed Trajectories | Inferior to Uniform | Validates the necessity of asymmetric design |
| Different Temperatures $\alpha$ | Optimal $\alpha$ at medium values | Too high leads to over-concentration; too low degrades to GRPO |

### Key Findings
- In successful trajectories, contributions are highly concentrated in a few key rounds—this is a structural feature of search agent tasks.
- Small models (1.7B) gain more from CW-GRPO (+6.3%), possibly because small models require more refined credit assignment to improve search efficiency.
- The 95% consensus rate between the LLM judge and human annotation proves the feasibility of using LLMs for process evaluation.
- Difficulty in failure attribution is a structural challenge—many failures are not due to agent decision errors.

## Highlights & Insights
- **Redefining process supervision as advantage redistribution** is an elegant shift in perspective—not training a value function or directly optimizing process rewards, but rather modulating outcome advantages with process signals.
- The design of the joint contribution signal ($u \cdot v$) reflects the core of search tasks: good retrieval must be accompanied by correct interpretation; both are indispensable.
- The philosophy of asymmetric treatment is profound—"we know success is because we did something right, but we don't necessarily know failure is because we did something wrong."

## Limitations & Future Work
- The LLM judge's own evaluation may be biased, especially regarding its judgment of reasoning correctness.
- Validated only on knowledge-intensive QA tasks; applicability to other agent tasks such as code generation remains to be verified.
- Temperature $\alpha$ is a hyperparameter and needs adjustment for different tasks.
- Binary contribution signals (0/1) may be too coarse; continuous value evaluation might be more refined.

## Related Work & Insights
- **vs Search-R1**: Search-R1 uses standard GRPO for outcome supervision, whereas CW-GRPO adds round-level credit assignment.
- **vs PPO Process Supervision**: PPO requires learning a value function and its training is unstable, whereas CW-GRPO avoids the value function entirely.
- **vs PRM Methods**: PRMs require round-level human annotation, whereas CW-GRPO substitutes this with an LLM judge.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective shift from process supervision to advantage redistribution is novel, and the joint contribution signal design is reasonable.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two model sizes, multiple benchmarks, and judge calibration validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation chain, smooth methodological derivation, and elegant formula design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ACL 2026\] Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents](rerank_before_you_reason_analyzing_reranking_tradeoffs_through_effective_token_c.md)
- [\[ACL 2026\] From Relevance to Authority: Authority-aware Generative Retrieval in Web Search Engines](from_relevance_to_authority_authority-aware_generative_retrieval_in_web_search_e.md)
- [\[ICML 2026\] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards](../../ICML2026/information_retrieval/reseek_a_self-correcting_framework_for_search_agents_with_instructive_rewards.md)

</div>

<!-- RELATED:END -->
