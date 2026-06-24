---
title: >-
  [Paper Note] Voting or Consensus? Decision-Making in Multi-Agent Debate
description: >-
  [Multi-Agent] This work systematically compares 7 decision protocols (voting vs. consensus) in multi-agent debate (MAD). It is found that consensus protocols improve performance by 2.8% on knowledge tasks, while voting protocols improve performance by 13.2% on reasoning tasks. Two new methods, AAD and CI, are proposed to enhance answer diversity, yielding performance gains of 3.3% and 7.4%, respectively.
tags:
  - "Multi-Agent"
date: 2026-05-08
content_hash: 97a9514ba04f6e56
---

# Voting or Consensus? Decision-Making in Multi-Agent Debate

- **Conference**: ACL 2025
- **arXiv**: [2502.19130](https://arxiv.org/abs/2502.19130)
- **Code**: [GitHub](https://github.com/lkaesberg/decision-protocols)
- **Area**: Others
- **Keywords**: Multi-Agent Debate, Decision Protocols, Voting, Consensus, Answer Diversity, AAD, CI

## TL;DR

This work systematically compares 7 decision protocols (voting vs. consensus) in multi-agent debate (MAD). It is found that consensus protocols improve performance by 2.8% on knowledge tasks, while voting protocols improve performance by 13.2% on reasoning tasks. Two new methods, AAD and CI, are proposed to enhance answer diversity, yielding performance gains of 3.3% and 7.4%, respectively.

## Background & Motivation

- **Core Problem**: The success of Multi-Agent Debate (MAD) highly depends on parameter choices. Among these, the **decision protocol**—how multiple agents converge to a final answer from discussion—has a huge impact on outcomes, yet prior studies treat it as a fixed variable rather than a key factor to be optimized.
- **Limitations of Prior Work**:
    - **Lack of Systematic Comparison**: Exchange-of-Thought (Yin et al. 2023) only utilizes consensus methods, Yang et al. 2024 focuses solely on voting protocols, and ReConcile (Chen et al. 2023) mixes both without analyzing the individual contribution of each protocol. Consequently, the fundamental question of "which decision protocol is optimal for a specific task type" remains unanswered.
    - **Parameter Confounding**: Prior works simultaneously vary multiple parameters in experiments (decision protocol + discussion rounds + agent count + response generator), failing to isolate the direct impact of the decision protocol itself on performance, which leads to poor experimental comparability.
    - **Unquantified Task Adaptability**: Intuitively, knowledge tasks and reasoning tasks may require different decision strategies, but this hypothesis has not been quantitatively verified; current methods indiscriminately apply the same protocol across all task types.
- **Goal**: Through rigorous **single-variable controlled experiments**—varying only the decision protocol—this work systematically evaluates the performance differences of 4 voting protocols and 3 consensus protocols across 3 knowledge tasks and 3 reasoning tasks, and proposes new methods to promote answer diversity.

## Method

### Overall Architecture

A multi-agent debate system is constructed based on Llama 3 (8B / 70B): three expert personas are automatically generated, and a final answer is reached via a designated decision protocol after multiple rounds of discussion. The overall framework consists of three core components: the **discussion paradigm** (defining the inter-agent communication structure and round rules), the **decision protocol** (defining when to terminate the discussion and how to select the final solution), and the **response generator** (defining the agent reply style, such as neutral, critical, or reasoning-only). Each agent generates one response per round, keeping only the messages from the last two rounds to control the context length.

### Key Designs

1. **Unified Evaluation Framework for 7 Decision Protocols**: Three consensus protocols (majority consensus >50%, supermajority consensus >66%, and unanimity 100%) and four voting protocols (plurality voting—one vote per agent, Borda count/ranked-choice voting—weighted by ranking, approval voting—multiple votes allowed per agent, and cumulative voting—distributing 25 points) are implemented. Consensus protocols require agents to gradually converge during discussion until reaching the protocol threshold, whereas voting protocols have all agents vote from candidate solutions after 3 rounds of discussion to select the final answer. The key distinction is that consensus is a "negotiation-convergence" process, while voting is an "exploration-selection" process.

2. **All-Agents Drafting (AAD)**: To address the issue in default settings where subsequent agents are biased by the first agent's answer, AAD forces all agents to **independently draft** their initial solutions in the first round without seeing other agents' outputs. Normal discussion resumes from the second round onwards. This ensures diversity in the initial answer pool and avoids groupthink. AAD is compatible with all 7 decision protocols.

3. **Collective Improvement (CI)**: Based on the independent drafting of AAD, CI further restricts communication by eliminating direct message exchanges between agents. At the end of each round, agents can only see the **set of solutions** from the previous round (instead of the discussion history) and must independently improve existing solutions or propose new ones. Decoupled from consensus building (which is why it is designed specifically for voting protocols), CI maintains answer diversity by suppressing excessive interaction, keeping the voting pool rich throughout the entire discussion process.

## Experiments

### Benchmarks

| Dataset | Task Type | Details | Sample Size |
|--------|---------|---------|-------|
| MMLU | Knowledge | Multiple-choice test covering a broad range of subjects | Subset sampling |
| MMLU-Pro | Knowledge | Harder, expert-focused multiple-choice questions | Subset sampling |
| GPQA | Knowledge | Graduate-level, Google-proof Q&A | Subset sampling |
| SQuAD 2.0 | Reasoning | Reading comprehension (containing unanswerable questions) | Subset sampling |
| StrategyQA | Reasoning | Multi-step reasoning yes/no questions | Subset sampling |
| MuSR | Reasoning | Multi-step reasoning over long narratives (e.g., murder mysteries) | Subset sampling |

### Main Results: Comparison of Decision Protocols (Llama 3 8B, Mean ± Std over 3 Runs)

| Decision Protocol Category | MMLU | MMLU-Pro | GPQA | SQuAD 2.0 | StrategyQA | MuSR |
|-------------|------|----------|------|-----------|------------|------|
| **Voting Mean** | Lower | Lower | Lower | **+13.1%** | **+0.2%** | **+26.4%** |
| **Consensus Mean** | **+2.3%** | **+4.9%** | **+1.3%** | Lower | Lower | Lower |
| CoT Baseline | Below MAD | Below MAD | Below MAD | Below MAD | Below MAD | Below MAD |

**Key Findings**: Consensus protocols consistently outperform voting on all 3 knowledge tasks (average +2.8%), while voting protocols significantly outperform consensus on all 3 reasoning tasks (average +13.2%), and both outperform the single-agent CoT baseline. Consensus takes an average of 1.42 rounds to reach a decision, whereas voting requires 3.38 rounds. Approval voting fails to reach a decision in 59% of cases due to excessive agent sycophancy.

### Scaling Analysis (StrategyQA, Plurality Voting Protocol)

| Scaling Dimension | Range of Variation | Trend | Interpretation |
|---------|---------|---------|------|
| Increase Agent Count | 1 → 10 | Accuracy increases linearly ↑ | Similar to self-consistency multi-sampling, larger knowledge base |
| Increase Discussion Rounds | 1 → 10 | Accuracy decreases linearly ↓ | Problem drift leading to deviation from the original task |
| Challenge Round (providing discussion history) | Extra +1 round | Challenge rate decreases by 10%, no positive effect | Agents tend to agree with existing discussions, failing to perform self-correction |

### Answer Diversity Experiments (StrategyQA)

| Method | Answer Cosine Similarity | Average Accuracy | vs Baseline |
|------|--------------|-----------|-------------|
| Baseline | 0.888 | 58.3% | — |
| AAD | 0.870 | 62.8% | **+3.3%** |
| CI | 0.845 | 65.7% | **+7.4%** |
| Critical Response | 0.843 | 59.4% | +1.1% |
| Reasoning Response | 0.916 | 51.9% | -6.4% |

Answer diversity (lower cosine similarity) positively correlates with task accuracy. CI achieves the highest accuracy (65.7%) when similarity is at its lowest (0.845). However, directly altering diversity via prompting styles (critical/reasoning-only) yields unstable and occasionally detrimental results.

### Key Findings

1. **Task Type Dictates the Optimal Decision Protocol**: Knowledge tasks benefit from consensus (multi-agent cross-verification reduces factual errors), whereas reasoning tasks benefit from voting (allowing parallel exploration of multiple paths before selecting the best).
2. **Scaling Agents Outperforms Scaling Rounds**: Increasing the number of agents functions like self-consistency multi-sampling, bringing linear performance gains. Conversely, increasing discussion rounds degrades performance due to problem drift, challenging the intuition that "more discussion equals better results."
3. **Structured Communication Beats Prompt Engineering**: AAD/CI enhances diversity reliably by modifying the communication structure rather than shifting prompt tones. Critical or reasoning-constrained prompts may instead degrade discussion quality.
4. **Consensus is More Decision-Efficient**: Consensus protocols require only 1.42 rounds on average (compared to 3.38 rounds for voting), achieving higher performance on knowledge tasks with lower computational costs.
5. **Approval Voting Fails in LLM Agents**: The sycophantic nature of LLM agents causes approval voting to fail in reaching a decision 59% of the time, revealing the limitations of directly transferring human decision protocols to LLM systems.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 6 | First systematic comparison of voting vs. consensus and proposes AAD/CI, though core ideas (independent sampling, restricted communication) are not entirely new. |
| Experimental Thoroughness | 8 | 6 datasets × 7 protocols × multiple ablations, rigorous control of variables, 3 experimental runs. |
| Value | 8 | Provides clear guidelines on protocol selection and reproducible practical recommendations, with open-source code and data. |
| Writing Quality | 8 | Well-structured with rich tables and figures; clear correspondence between experimental designs and findings. |

## Highlights & Insights

- First systematic comparison of 7 decision protocols across both knowledge and reasoning tasks, establishing a clear task-protocol selection matrix.
- Rigorous single-variable controlled experimental design: changing only the decision protocol at a time, eliminating parameter confounding.
- The AAD and CI methods are simple and elegant—requiring no model fine-tuning or prompt content modifications, achieving significant gains solely by adjusting communication structures.
- Findings reveal that agent scaling (increasing quantity) is more effective than discussion scaling (increasing rounds), providing clear guidance for resource allocation in multi-agent systems.
- Quantitatively reveals the positive correlation between answer diversity and task performance (cosine similarity vs. accuracy).
- Unveils the extreme manifestation of agent sycophancy in approval voting (59% failure to decide), sounding an alarm for protocol design.

## Limitations & Future Work

- Experiments are limited to Llama 3 (8B / 70B), without validating generalization to proprietary models such as GPT-4 or Claude.
- MAD incurs high computational overhead (approx. 5× for consensus and 10× for voting compared to the CoT baseline); the ROI between performance gains and resource consumption needs careful evaluation.
- Due to compute limits, dataset subsets are sampled (95% confidence level); although 3 runs are evaluated, some statistical volatility may remain.
- The focus is restricted to the decision protocol dimension, without exploring the interaction effects between persona design, prompt engineering, and decision protocols.
- Agent sycophancy remains a fundamental limitation; AAD and CI merely mitigate rather than cure the issue.
- The study does not consider the decision dynamics of heterogeneous agents (teams composed of different models), whereas heterogeneous teams might be more common in practical deployments.

## Related Work

- **Multi-Agent Debate**: Du et al. 2023 (Improving Factuality & Reasoning), Exchange-of-Thought (Yin et al. 2023, consensus method), ReConcile (Chen et al. 2023, hybrid voting+consensus), Liang et al. 2024 (encouraging divergent thinking).
- **LLM Agent Enhancement**: Self-consistency (Wang et al. 2023, multi-path sampling voting), CoT reasoning (Wei et al. 2022), persona-based prompting (Jiang et al. 2024), Self-Refine (Madaan et al. 2023).
- **Decision Theory & Voting Mechanisms**: Social choice theory (List 2022), consensus vs. voting (Jones 1994), Yang et al. 2024 (multi-voting protocol comparison of LLMs).
- **MALLM Framework**: Becker et al. 2025 proposes a multi-agent LLM collaborative framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic comparison using controlled variables; solid methodological contribution.
- **Value**: ⭐⭐⭐⭐⭐ — Provides clear task-protocol selection guidelines; high practical value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multiple experimental runs with reported standard deviations, though evaluations are on dataset subsets.
- **Overall**: ⭐⭐⭐⭐

<!-- Systematic comparison of voting vs consensus decision protocols in multi-agent LLM debates -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives](../../ACL2026/multi_agent/social_dynamics_as_critical_vulnerabilities_that_undermine_objective_decision-ma.md)
- [\[ACL 2025\] CortexDebate: Debating Sparsely and Equally for Multi-Agent Debate](cortexdebate_debating_sparsely_and_equally_for_multi-agent_debate.md)
- [\[NeurIPS 2025\] Belief-Calibrated Multi-Agent Consensus Seeking for Complex NLP Tasks](../../NeurIPS2025/multi_agent/belief-calibrated_multi-agent_consensus_seeking_for_complex_nlp_tasks.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](../../ACL2026/multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](../../ACL2026/multi_agent/topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)

</div>

<!-- RELATED:END -->
