---
title: >-
  [Paper Note] Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search
description: >-
  [ACL 2025 Main][LLM Alignment][Multi-turn jailbreak attacks] This paper proposes Tempest (referred to as Siege in its early version), a multi-turn adversarial framework based on breadth-first tree search. By tracking the partial compliance information of the target LLM and re-injecting it into subsequent queries, Tempest achieves a 100% attack success rate against GPT-3.5-turbo and 97% against GPT-4 on JailbreakBench, requiring significantly fewer queries than baselines like…
tags:
  - "ACL 2025 Main"
  - "LLM Alignment"
  - "Multi-turn jailbreak attacks"
  - "tree search"
  - "partial compliance tracking"
  - "LLM safety"
  - "red teaming"
date: 2026-05-08
content_hash: 210aca1d4acecacf
---

# Tempest: Autonomous Multi-Turn Jailbreaking of Large Language Models with Tree Search

**Conference**: ACL 2025 Main  
**arXiv**: [2503.10619](https://arxiv.org/abs/2503.10619)  
**Code**: None  
**Area**: RLHF Alignment / AI Safety  
**Keywords**: Multi-turn jailbreak attacks, tree search, partial compliance tracking, LLM safety, red teaming

## TL;DR

This paper proposes Tempest (referred to as Siege in its early version), a multi-turn adversarial framework based on breadth-first tree search. By tracking the partial compliance information of the target LLM and re-injecting it into subsequent queries, Tempest achieves a 100% attack success rate against GPT-3.5-turbo and 97% against GPT-4 on JailbreakBench, requiring significantly fewer queries than baselines like Crescendo/GOAT.

## Background & Motivation

**Background**: LLM safety evaluation is primarily categorized into single-turn and multi-turn attacks. Single-turn attacks rely on meticulously designed single prompts (e.g., GCG, PAIR), while multi-turn attacks progressively induce violations through back-and-forth conversations (e.g., Crescendo, GOAT).

**Limitations of Prior Work**: (1) Single-turn attacks capture only one aspect of safety evaluation and fail to reflect the real-world behavior of attackers probing boundaries through multiple interactions; (2) current multi-turn attack frameworks (e.g., Crescendo) typically proceed along a single path, lacking systematic exploration of multiple attack paths; (3) there is a lack of quantitative measurement for "partial compliance"—the grey area between complete refusal and total violation remains underutilized.

**Key Challenge**: Existing multi-turn methods either follow a single dialogue path (which may miss more effective attack routes) or require excessive retries (necessitating 10 independent sessions to achieve high success rates), posing a trade-off between efficiency and coverage.

**Goal**: To design a systematic multi-turn attack framework that can (1) efficiently explore multiple attack routes in a single run, (2) precisely quantify and exploit the partial compliance of target models, and (3) achieve the highest attack success rate with minimal queries.

**Key Insight**: The authors observe that LLM safety guardrails exhibit a "progressive erosion" phenomenon during multi-turn dialogues—an isolated, small concession by the model may seem harmless, but cumulative concessions lead to complete violations. This aligns naturally with the concept of "gradually expanding promising branches" in tree search.

**Core Idea**: To model multi-turn jailbreak attacks as a breadth-first tree search (BFS), where each dialogue turn expands multiple attack branches, tracks the partial compliance score of each branch, prunes unproductive paths, and retains promising paths for further exploration.

## Method

### Overall Architecture

Tempest consists of three core components: (1) an Attacker LLM—an adversarial prompt generator equipped with various red-teaming tactics; (2) a Partial Compliance Tracker—which quantifies the degree of violation in each response from the target model; and (3) a BFS Tree Search Engine—which manages the expansion, evaluation, and pruning of dialogue states. Given a description of the target harmful behavior as input, the framework outputs a complete dialogue history that successfully induces a violating response.

### Key Designs

1. **Attacker LLM and Red-Teaming Tactics (Attacker LLM)**:

    - Function: Generates multiple adversarial prompts with different tactics based on the current dialogue history and the target model's response from the previous turn.
    - Mechanism: The Attacker LLM is provided with a system prompt containing (a) the adversarial target description and (b) a list of diverse attack strategies (e.g., persona adoption, hypothetical framing, refusal suppression). It incorporates a chain-of-thought reasoning structure, requiring the attacker to perform four steps in each turn: observe the target model's previous response $\rightarrow$ assess whether it is partially compliant $\rightarrow$ select the next adversarial tactic $\rightarrow$ generate a new user prompt. A general helpful-only LLM is used as the attacker, functioning via in-context learning rather than specialized red-teaming fine-tuning.
    - Design Motivation: Utilizing the reasoning capabilities of LLMs to dynamically adjust attack tactics provides greater flexibility than human-engineered static templates. Since it does not rely on gradient information (black-box attack), it is applicable to API-only commercial models.

2. **Partial Compliance Tracking**:

    - Function: Quantifies the extent to which the target model leaks harmful information in each response, providing guidance signals for branch selection in the tree search.
    - Mechanism: Defines a partial compliance function $\gamma(m_t) \in [0, 10]$, where 0 denotes a safe refusal, 10 indicates a complete violation, and intermediate values represent varying degrees of information leakage. The cumulative compliance score is maintained as $\Gamma = \sum_{t=1}^{k} \gamma(m_t)$. When $\gamma(m_t) > 0$, the system automatically extracts the partially leaked content (e.g., code snippets, disclaimers with harmful details, partial instructions) and re-injects it into subsequent attack prompts.
    - Design Motivation: Traditional methods rely on binary classification (success/failure), losing valuable intermediate state information. Quantifying partial compliance allows the system to identify "near-breakthrough" paths and concentrate search resources on them.

3. **BFS Tree Search Engine (Multi-Turn Tree Search)**:

    - Function: Systematically explores multiple attack paths in parallel, balancing exploration and depth.
    - Mechanism: Each dialogue state is treated as a node in the search tree, containing the complete dialogue history and the cumulative compliance score $\Gamma$. Each turn (corresponding to a layer of the tree) involves three steps: **Expansion**—for each active node, the Attacker LLM generates $B$ different attack prompts; **Evaluation**—each prompt is sent to the target model to calculate the $\gamma$ value of its response, marking nodes with $\gamma = 10$ as successful terminal nodes; **Pruning**—branches with $\gamma = 0$ (completely safe) or extremely low compliance scores are discarded, reserving resources for paths with partial breakthroughs. The search proceeds up to $k$ turns (typically $k=5$) or until all branches either succeed or are pruned.
    - Design Motivation: Compared to DFS, BFS explores different attack strategies more uniformly. Pruning prevents exponential growth, and parallel expansion is more efficient than serial retries. A single run can cover diverse attack routes without restarting sessions from scratch.

### Loss & Training

Tempest is an inference-time framework and does not involve model training. The Attacker LLM utilizes in-context learning, and partial compliance scoring is provided by an independent, open-source safety judge model.

## Key Experimental Results

### Main Results

Comparison of attack success rates and query counts on JailbreakBench (100 harmful behavior prompts):

| Target Model | Method | Runs | ASR (%) | Total Queries |
|----------|------|---------|---------|---------|
| GPT-3.5-Turbo | Crescendo | 1 | 40.0 | 6 |
| GPT-3.5-Turbo | Crescendo | 10 | 80.4 | 60 |
| GPT-3.5-Turbo | GOAT | 1 | 55.7 | 6 |
| GPT-3.5-Turbo | GOAT | 10 | 91.6 | 60 |
| GPT-3.5-Turbo | **Tempest** | **1** | **100.0** | **44.4** |
| GPT-4 | Crescendo | 1 | 31.7 | 6 |
| GPT-4 | Crescendo | 10 | 70.9 | 60 |
| GPT-4 | GOAT | 1 | 46.6 | 6 |
| GPT-4 | GOAT | 10 | 87.9 | 60 |
| GPT-4 | **Tempest** | **1** | **97.0** | **84.2** |
| Llama-3.1-70B | Crescendo | 10 | 77.0 | 60 |
| Llama-3.1-70B | GOAT | 10 | 91.0 | 60 |
| Llama-3.1-70B | **Tempest** | **1** | **97.0** | **51.8** |

### Ablation Study

Contribution analysis of different components (based on the GPT-4 target model):

| Configuration | ASR (%) | Description |
|------|---------|------|
| Tempest Full | 97.0 | Complete framework |
| No BFS (Single-path) | ~75 | Similar to Crescendo's single-path progression |
| No Partial Compliance Tracking | ~70 | Only binary classification used |
| No Tactics Diversification | ~80 | Attacker generates only 1 prompt per turn |
| Reduce max turns to 3 | ~85 | Insufficient exploration depth |

### Key Findings

- **Single run outperforms multi-retry baselines**: A single run of Tempest achieves an ASR (97–100%) that substantially outperforms Crescendo/GOAT across 10 runs (70–92%), while utilizing a comparable or even smaller number of total queries.
- **High value of partial compliance information**: Eliminating partial compliance tracking leads to an ASR drop of approximately 27 percentage points, proving that capturing grey-area information is vital for attack efficiency.
- **GPT-4 is not inherently safer than GPT-3.5**: Under multi-turn attacks, the performance gap between the two is far smaller than in single-turn attacks, indicating that GPT-4's safety enhancements primarily address single-turn scenarios.

## Highlights & Insights

- **Modeling jailbreak attacks as tree search**: This is an elegant abstraction—structured exploration of the attack space is far more efficient than random retries. The same BFS approach can be transferred to any adversarial scenario requiring systematic strategy-space exploration.
- **Quantitative measurement of partial compliance is a key innovation**: The fine-grained 0-10 scoring provides richer search signals than binary feedback. This design concept can be adapted to other safety evaluation tasks, such as grey-scale grading in content moderation.
- **Revealing the fundamental vulnerability of multi-turn safety**: Even models with robust single-turn defenses collapse under multi-turn scenarios that accumulate small concessions. This poses a fundamental challenge to safety training strategies, highlighting the need for alignment training within multi-turn conversation contexts.

## Limitations & Future Work

- **Evaluation limited to JailbreakBench**: The coverage of 100 behavioral prompts is relatively limited, focusing primarily on explicit harmful requests without testing more subtle safety boundaries.
- **Impact of Attacker LLM selection remains undiscussed**: Variations in the capabilities of different attacker models could significantly affect the ASR, but this comparison is not explored in the paper.
- **High computational cost**: BFS tree search requires generating multiple prompts and responses per turn. The GPT-4 target requires an average of 84.2 queries, which is expensive in terms of API costs.
- **Lack of defense strategy discussion**: Although the paper details the threat, it does not propose specific mitigation strategies. Promising directions include compliance monitoring of multi-turn dialogue contexts and using cumulative compliance scores as a trigger for safety-based termination.

## Related Work & Insights

- **vs Crescendo**: Crescendo progresses along a single dialogue path and generates only one prompt at a time, requiring multiple independent runs to cover different strategies. Tempest parallelizes the exploration of multiple paths in a single run using BFS, substantially improving efficiency.
- **vs GOAT**: GOAT utilizes an attacker LLM to dynamically adjust prompts but similarly operates on a single path and lacks partial compliance tracking. Tempest's quantitative tracking mechanism allows it to identify and exploit minimal concessions from the target model more accurately.
- **vs Tree of Attacks (TAP)**: While TAP also employs a tree structure, it primarily focuses on different variants of single-turn attacks. In contrast, Tempest's tree structure spans across dialogue turns, with each layer corresponding to a dialogue round, representing a fundamentally different approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Integrating tree search into multi-turn jailbreaking is an intuitive and elegant combination, and partial compliance tracking represents a key innovation.
- Experimental Thoroughness: ⭐⭐⭐ The results are impressive, but the evaluation dataset is singular, and the ablation study lacks extensive depth.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described, and the algorithmic pseudocode is complete, although the acknowledgments hint at partial assistance from AI systems.
- Value: ⭐⭐⭐⭐ This work provides valuable references for the multi-turn safety evaluation of LLMs, exposing critical safety blind spots.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Red Queen: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking](red_queen_safeguarding_large_language_models_against_concealed_multi-turn_jailbr.md)
- [\[ACL 2025\] M2S: Multi-turn to Single-turn jailbreak in Red Teaming for LLMs](m2s_multiturn_to_singleturn_jailbreak_in.md)
- [\[ACL 2025\] QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language](queryattack_jailbreaking_aligned_large_language_models_using_structured_non-natu.md)
- [\[NeurIPS 2025\] Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency](../../NeurIPS2025/llm_alignment/adjacent_words_divergent_intents_jailbreaking_large_language_models_via_task_con.md)
- [\[ACL 2025\] MTSA: Multi-Turn Safety Alignment for LLMs through Multi-Round Red-Teaming](mtsa_multi-turn_safety_alignment_for_llms_through_multi-round_red-teaming.md)

</div>

<!-- RELATED:END -->
