---
title: >-
  [Paper Note] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks
description: >-
  [NeurIPS 2025][LLM Evaluation][Multi-agent systems] This paper proposes the Belief-Calibrated Consensus Seeking (BCCS) framework, which incorporates three modules—belief-calibrated consensus judgment…
tags:
  - "NeurIPS 2025"
  - "LLM Evaluation"
  - "Multi-agent systems"
  - "consensus mechanism"
  - "belief calibration"
  - "collaborative reasoning"
  - "LLM collaboration"
date: 2026-05-08
content_hash: 42ae34a8654181d7
---

# Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks

**Conference**: NeurIPS 2025
**arXiv**: [2510.06307](https://arxiv.org/abs/2510.06307)  
**Code**: [https://github.com/dengwentao99/BCCS](https://github.com/dengwentao99/BCCS)  
**Area**: LLM Evaluation
**Keywords**: Multi-agent systems, consensus mechanism, belief calibration, collaborative reasoning, LLM collaboration

## TL;DR
This paper proposes the Belief-Calibrated Consensus Seeking (BCCS) framework, which incorporates three modules—belief-calibrated consensus judgment, conflict-aware collaborator assignment, and leader selection—to enable multi-agent systems to reach more stable consensus on complex NLP tasks, yielding improvements of 2.23% and 3.95% on difficult subsets of MATH and MMLU, respectively.

## Background & Motivation

**Background**: Multi-agent systems (MAS) enhance reasoning by coordinating multiple LLM agents, with consensus-seeking as the core protocol. Existing methods such as MAD, GroupDebate, and DyLAN achieve consensus through voting or debate.

**Limitations of Prior Work**:
   - Existing consensus judgment relies solely on answer-level agreement (e.g., Byzantine Consensus requires >2/3 majority), **neglecting contradictions in agents' internal beliefs (confidence/belief)**—even when answers agree, consensus may be unstable if agents hold low confidence.
   - Agents interact indiscriminately with all others during collaboration, without selectively identifying optimal partners—over-reliance on supporters may trap the system in suboptimal states, while excessive exposure to opposition impedes consensus.

**Key Challenge**: Stable consensus requires not only answer agreement but also belief agreement; existing methods address only the former.

**Goal**: (a) How to incorporate belief calibration into consensus judgment to prevent spurious consensus arising from low-confidence agreement? (b) How to select optimal collaborators for each agent (balancing supporters and opponents)?

**Key Insight**: Drawing on opinion dynamics theory, the paper uses LLM output probabilities as proxy belief measures and formally analyzes the conditions under which a MAS converges to stable consensus.

**Core Idea**: LLM output probabilities are used as beliefs to calibrate consensus judgment; conflict scores are combined with automatic collaborator and leader assignment to stabilize multi-agent reasoning.

## Method

### Overall Architecture
BCCS is an iterative multi-agent consensus framework. Given a question $q$, $n$ LLM agents each generate an answer $x_i^k$ and reasoning chain $e_i^k$. Belief is defined as $b_i^k = P(x_i^k | q, e_i^k)$. Each iteration proceeds as: (1) the BCCJ module assesses consensus state → (2) the state routes to either the CA or LS module → (3) agents update their opinions → repeat until full consensus or maximum rounds are reached.

### Key Designs

1. **Belief-Calibrated Consensus Judgment (BCCJ)**:

    - Function: Classifies MAS state into three levels—full consensus, partial consensus, and no consensus.
    - Mechanism: Beyond requiring answer proportion $p_s^k > 2/3$ (Byzantine), the module also requires belief proportion $p_b^k > 0.8$, i.e., the total belief of the supporting group must exceed that of the opposing group by more than fourfold. Partial consensus requires $p_b^k > 0.5$. Neither condition met implies no consensus.
    - Design Motivation: Prevents spurious consensus where answers agree but beliefs are low, avoiding convergence to suboptimal solutions.
    - Distinction from Prior Work: Byzantine Consensus counts only answer frequency; BCCJ additionally evaluates belief quality.

2. **Collaborator Assignment (CA) Module**:

    - Function: Assigns optimal collaborators to each agent under partial consensus.
    - Mechanism: Conflict between groups is quantified via conflict score $\psi_{pq} = \psi_{pq}^{\mathcal{G}} \cdot \psi_{pq}^{\mathcal{L}}$. Macro-conflict $\psi^{\mathcal{G}}$ measures global disagreement using the complement of belief-weighted Jaccard similarity; micro-conflict $\psi^{\mathcal{L}}$ captures local consistency differences. $\psi_{pq} > 2$ denotes conflicting groups.
    - Key Strategy: The agent with the lowest belief in the most uncertain group collaborates with the highest-belief agent from a conflicting group (for correction); other agents collaborate with the highest-belief agent from the supporting group (to accelerate convergence).
    - Design Motivation: Grounded in Theorem 3.2—collaboration solely with supporters converges but may be suboptimal; a controlled introduction of conflicting opinions is necessary.

3. **Leader Selection (LS) Module**:

    - Function: Selects leaders from each opinion group to guide direction under no-consensus conditions.
    - Mechanism: The $n^l$ agents with the highest beliefs in each group are designated as leaders; remaining agents interact only with leaders to update their opinions.
    - Design Motivation: Grounded in Theorem 3.3—following high-belief leaders accelerates convergence to stable consensus. Leaders are re-selected each round to prevent suboptimal agents from dominating persistently.

### Theoretical Foundation

The paper establishes two key theorems:
- **Theorem 3.2**: Collaborating with supporters tends toward stable consensus (converging to the mean opinion), whereas collaborating with opponents leads to instability (potential oscillation or divergence).
- **Theorem 3.3**: Following intra-group leaders tends toward the leaders' mean state; high-belief leaders accelerate convergence.

These theorems provide the theoretical justification for the CA and LS module designs.

## Key Experimental Results

### Main Results

| Method | MATH Avg | MMLU Avg |
|------|----------|----------|
| CoT (single agent) | 73.33 | 71.87 |
| CoT-SC | 76.67 | 73.13 |
| EoT | 78.40 | 74.33 |
| GroupDebate | 77.93 | 74.87 |
| MAD | 78.87 | 76.13 |
| PARSE | 78.53 | 76.47 |
| CMD | 78.93 | 75.07 |
| DyLAN | 78.80 | 75.00 |
| **BCCS** | **80.60** | **78.47** |

BCCS outperforms the strongest baseline (CMD) by 1.67% on MATH and PARSE by 2.00% on MMLU. Gains are more pronounced on difficult subsets: MATH Intermediate Algebra +2.23%, MMLU Humanities +3.95%.

### Ablation Study

| Configuration | MATH Avg | Notes |
|------|----------|------|
| BCCS (full) | 80.60 | Full model |
| -CA | 78.60 | Remove collaborator assignment, −2.00% |
| -Conflict | 79.33 | Supporters only, −1.27% |
| -LS | 79.20 | Remove leader selection, −1.40% |
| R.Leader | 79.53 | Random leader selection, −1.07% |
| -BCCJ | 79.07 | Remove belief-calibrated judgment, −1.53% |

### Key Findings
- The CA module contributes most (−2.00% when removed), demonstrating that selective collaboration outperforms indiscriminate interaction.
- Introducing conflicting opinions (vs. -Conflict) is critical for avoiding suboptimal solutions (+1.27%).
- Belief calibration (BCCJ) is especially important for difficult tasks, as easy tasks inherently elicit high confidence.
- High-belief leaders outperform randomly selected leaders (+1.07%).
- Performance differences across methods are small on easy tasks; BCCS's advantages emerge primarily on difficult ones.

## Highlights & Insights
- **Belief as a consensus quality signal**: Incorporating LLM output probabilities into consensus judgment elegantly upgrades superficial "answer agreement" to stable "dual agreement on answer and belief." This idea is transferable to any multi-agent voting or debate system.
- **Two-level conflict score design**: Macro-level (Jaccard) combined with micro-level (local consistency differences) conflict scoring provides a more comprehensive characterization of inter-group relationships than a single metric.
- **Theory-driven design**: Establishing opinion dynamics guarantees prior to algorithm design gives the framework theoretical grounding rather than purely empirical motivation.

## Limitations & Future Work
- Belief is approximated by LLM output probabilities; LLMs are known to be poorly calibrated (overconfident), and imperfect belief estimation may degrade performance.
- The computational overhead of 7 agents over 3 iterations is substantial (7 LLM calls/round × 3 rounds = 21 calls), leaving room for efficiency improvement.
- Evaluation is limited to MATH and MMLU; open-ended generation tasks are not assessed.
- The grouping strategy relies on keyword-based distributional similarity, which is coarse and may yield inaccurate groupings in complex reasoning scenarios.
- Gains on easy tasks are limited, indicating the method primarily targets difficult, high-disagreement scenarios.

## Related Work & Insights
- **vs. MAD (Multi-Agent Debate)**: MAD allows free-form debate among agents; BCCS employs belief-guided selective collaboration, offering greater directional control.
- **vs. DyLAN**: DyLAN dynamically selects agents without considering beliefs; BCCS's belief calibration enables more precise selection.
- **vs. GroupDebate**: GroupDebate conducts inter-group debate without selective interaction; the CA module in BCCS addresses this limitation.

## Rating
- Novelty: ⭐⭐⭐⭐ Belief-calibrated consensus judgment is a meaningful contribution, though the overall framework is relatively engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive, but only a single backbone model (Qwen2.5-7B) is evaluated.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though the paper is lengthy.
- Value: ⭐⭐⭐⭐ Offers practical reference value for multi-agent collaborative reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can Large Language Models Master Complex Card Games?](can_large_language_models_master_complex_card_games.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/llm_evaluation/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)
- [\[ICCV 2025\] Neural Multi-View Self-Calibrated Photometric Stereo without Photometric Stereo Cues](../../ICCV2025/llm_evaluation/neural_multi-view_self-calibrated_photometric_stereo_without_photometric_stereo_.md)
- [\[ICLR 2026\] Which LLM Multi-Agent Protocol to Choose?](../../ICLR2026/llm_evaluation/which_llm_multi-agent_protocol_to_choose.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](../../ICML2026/llm_evaluation/multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)

</div>

<!-- RELATED:END -->
