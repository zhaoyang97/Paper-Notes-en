---
title: >-
  [Paper Note] SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies
description: >-
  [NeurIPS 2025][LLM (Other)][multi-agent planning] This paper proposes SYMPHONY, an MCTS-based multi-agent planning framework that leverages diversity-driven search over a heterogeneous LLM pool, UCB-based adaptive scheduling, entropy-modulated confidence scoring, and pool-level memory sharing to substantially improve planning diversity and efficiency.
tags:
  - "NeurIPS 2025"
  - "LLM (Other)"
  - "multi-agent planning"
  - "MCTS"
  - "heterogeneous models"
  - "LLM collaboration"
  - "tree search"
date: 2026-05-08
content_hash: f3463a989f863d18
---

# SYMPHONY: Synergistic Multi-agent Planning with Heterogeneous Language Model Assemblies

**Conference**: NeurIPS 2025  
**arXiv**: [2601.22623](https://arxiv.org/abs/2601.22623)  
**Code**: [https://github.com/ZHUWEI-hub/SYMPHONY](https://github.com/ZHUWEI-hub/SYMPHONY)  
**Area**: LLM/NLP  
**Keywords**: multi-agent planning, MCTS, heterogeneous models, LLM collaboration, tree search

## TL;DR
This paper proposes SYMPHONY, an MCTS-based multi-agent planning framework that leverages diversity-driven search over a heterogeneous LLM pool, UCB-based adaptive scheduling, entropy-modulated confidence scoring, and pool-level memory sharing to substantially improve planning diversity and efficiency.

## Background & Motivation

Combining LLMs with MCTS for complex task planning has attracted significant attention (RAP, LATS, MASTER, etc.), yet existing methods almost exclusively adopt a **single-model paradigm**: repeatedly querying the same LLM to generate search branches. The fundamental problem is that multiple samples from a single LLM tend to be highly similar, reflecting the same dominant reasoning pattern, which leads to:
- Search trees populated with redundant, near-identical branches
- Limited exploration capacity and susceptibility to local optima
- Additional sampling and token overhead to cover the solution space

**Key Challenge**: MCTS requires diverse rollouts for effective exploration, but the intra-model variability of a single LLM is insufficient to support such diversity.

**Key Insight**: The paper replaces the single model with a pool of heterogeneous LLMs (differing in pretraining origin and reasoning style), enhancing branch diversity from the perspective of model-level diversity. Adaptive scheduling and collaborative memory are introduced to ensure efficient coordination.

## Method

### Overall Architecture
SYMPHONY introduces four key components into the standard MCTS framework: (1) a heterogeneous agent pool for diverse branch generation; (2) a UCB-based scheduling policy for dynamic agent selection; (3) Entropy-Modulated Confidence Scoring (EMCS) to calibrate node value estimation; and (4) pool-level memory sharing to enable cross-agent reflective learning.

### Key Designs

1. **Heterogeneous Agent Pool**:

    - Maintains $\mathcal{M}^{(k)} = \{M_1^{(k)}, \cdots, M_n^{(k)}\}$, a pool of heterogeneous LLMs
    - SYMPHONY-S (consumer-grade hardware): Qwen2.5-7B + Mistral-7B + Llama-3.1-8B
    - SYMPHONY-L (API-level): GPT-4 + Qwen-Max + DeepSeek-V3
    - Unified input/output interface supporting modular replacement
    - Theoretically proven: sampling from the ensemble with non-zero probability yields strictly lower expected error than deterministically selecting any single agent

2. **UCB Adaptive Scheduling**:

    - Models agent selection as a multi-armed bandit problem
    - $\text{UCB}(M_i^{(k)}) = \bar{Q}(M_i^{(k)}) + \alpha \cdot \sqrt{\frac{\ln N_{total}}{N(M_i^{(k)})+1}}$
    - Balances exploitation of high-performing agents and exploration of underutilized ones
    - Applied across all three MCTS phases: expansion, evaluation, and reflection

3. **Pool-Level Memory Sharing**:

    - Failed trajectories trigger UCB-selected agents to generate natural-language reflections $\mathcal{R}_i^k$
    - Reflections are broadcast to all agents and injected into prompts as shared memory blocks
    - A fixed-size buffer with FIFO eviction manages memory capacity
    - Achieves behavioral adaptation through prompt-level memory without any parameter updates

4. **Entropy-Modulated Node Evaluation (EMCS)**:

    - Agents evaluate nodes to produce a value $Z(s_t) \in [0,1]$ and confidence $C(s_t) \in (0,1)$
    - Uncertain predictions are penalized via Bernoulli entropy: $R(s_t) = Z(s_t) \cdot (1 - E(s_t))$
    - Where $E(s_t) = -C(s_t)\ln C(s_t) - (1-C(s_t))\ln(1-C(s_t))$
    - Penalty is maximized at $C=0.5$ (maximum uncertainty), preserving high-confidence evaluations

### Loss & Training
- No training is required; SYMPHONY is a purely inference-time collaboration framework.
- Key hyperparameters include the UCB exploration coefficient $\alpha$, MCTS expansion width $n$, and search budget $K$.

## Key Experimental Results

### Main Results

| Task | Metric | SYMPHONY-S | SYMPHONY-L | LATS | MASTER |
|------|------|-----------|-----------|------|--------|
| HotpotQA | EM | 0.59 | **0.79** | 0.71 | 0.76 |
| WebShop | Score/SR | 0.82/0.56 | **0.88/0.72** | 0.76/0.38 | 0.80/– |
| MBPP (Python) | Pass@1 | 0.927 | **0.965** | 0.811 | 0.910 |
| MBPP (Rust) | Pass@1 | 0.946 | **0.974** | – | – |

### Ablation Study

| Configuration | HotpotQA (EM) | WebShop (SR) | MBPP (Pass@1) |
|------|-------------|-------------|-------------|
| SYMPHONY-S (full) | 0.59 | 0.56 | 0.927 |
| w/o Agent Scheduling | 0.51 | 0.48 | 0.906 |
| w/o Memory Sharing | 0.45 | 0.46 | 0.871 |
| w/o EMCS | 0.51 | 0.49 | 0.892 |

### Key Findings
- **Diversity drives performance**: The three-model combination achieves over 80% 4-Unique branch diversity on MBPP, compared to under 20% for a single model, yielding performance gains exceeding 30%.
- **Search efficiency**: SYMPHONY-L requires only 9.47 node expansions on HotpotQA, versus 66.65 for LATS@50—a 7× improvement in efficiency.
- **Cost optimization**: GPT-4 accounts for only 40% of calls in SYMPHONY-L, yet outperforms the GPT-4-only baseline.
- **Consumer-grade feasibility**: SYMPHONY-S (purely open-source 7B/8B models) surpasses single-model GPT-4 baselines on multiple tasks.

## Highlights & Insights
- **Heterogeneous > Homogeneous**: Even a combination of different open-source models at the same scale outperforms repeated sampling from a single strong model—model-level diversity is an underexploited resource.
- **Decentralized design for memory sharing**: Unlike multi-agent systems requiring explicit communication protocols, SYMPHONY achieves lightweight knowledge propagation through natural-language reflections.
- **Elegance of EMCS**: Calibrating LLM self-assessment scores via Bernoulli entropy from information theory is both principled and effective.

## Limitations & Future Work
- The selection of the heterogeneous model pool lacks systematic guidance and currently relies on empirical choice.
- The search budget $K$ and expansion width $n$ require manual tuning.
- The FIFO memory eviction strategy is relatively simple and may discard important reflections.
- Robustness under adversarial or high-noise environments has not been thoroughly validated.

## Related Work & Insights
- **vs. MASTER (Gan et al.)**: MASTER constructs multiple agents from the same LLM and modifies the UCT formula, whereas SYMPHONY fundamentally increases diversity through a heterogeneous model pool.
- **vs. LATS (Zhou et al.)**: LATS performs single-model search; SYMPHONY combines multi-model search with memory sharing to achieve superior results under a smaller budget.
- **vs. AgentCoder (Huang et al.)**: AgentCoder employs fixed role assignments (programmer/tester); SYMPHONY's adaptive scheduling offers greater flexibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of a heterogeneous agent pool and UCB scheduling is novel in the MCTS-LLM setting.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three diverse task types, two configurations, comprehensive ablation and efficiency analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Framework is clearly presented, though the notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides a principled approach to combining multiple LLMs for planning, with strong accessibility on consumer-grade hardware.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Last Vote: A Multi-Stakeholder Framework for Language Model Governance](the_last_vote_a_multi-stakeholder_framework_for_language_model_governance.md)
- [\[ACL 2025\] Planning-Driven Programming: A Large Language Model Programming Workflow](../../ACL2025/llm_nlp/planning-driven_programming_a_large_language_model_programming_workflow.md)
- [\[ACL 2025\] MasRouter: Learning to Route LLMs for Multi-Agent Systems](../../ACL2025/llm_nlp/masrouter_learning_to_route_llms_for_multi-agent_systems.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](../../ACL2025/llm_nlp/agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](../../ACL2025/llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)

</div>

<!-- RELATED:END -->
