---
title: >-
  [Paper Note] Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs
description: >-
  [ICML 2026][Multi-Agent][HiddenBench] This paper adapts the Hidden Profile paradigm from social psychology to evaluate multi-agent LLMs, constructing the 65-task HiddenBench. Across 15 frontier LLMs…
tags:
  - "ICML 2026"
  - "Multi-Agent"
  - "HiddenBench"
  - "Hidden Profile"
  - "Distributed Information"
  - "Information Asymmetry"
  - "Collective Reasoning Failure"
date: 2026-05-08
content_hash: bce3359589744f7a
---

# Systematic Failures in Collective Reasoning under Distributed Information in Multi-Agent LLMs

**Conference**: ICML 2026  
**arXiv**: [2505.11556](https://arxiv.org/abs/2505.11556)  
**Code**: HuggingFace + GitHub (Available)  
**Area**: LLM Agent / Multi-Agent / Collective Reasoning Evaluation  
**Keywords**: HiddenBench, Hidden Profile, Distributed Information, Information Asymmetry, Collective Reasoning Failure

## TL;DR
This paper adapts the Hidden Profile paradigm from social psychology to evaluate multi-agent LLMs, constructing the 65-task HiddenBench. Across 15 frontier LLMs, it systematically reveals that while a single agent achieves 80.7% accuracy on tasks under Full Profile conditions, multi-agent groups under distributed information only achieve 30.1%. The fundamental failure mode is the **inability to actively elicit information that others have not disclosed**, which can be significantly mitigated across model families by lightweight structured communication protocols.

## Background & Motivation

**Background**: Multi-agent LLM systems are increasingly deployed in scenarios such as software development, scientific discovery, and social simulation. The **core promise is that "the group can integrate more information than a single agent."** This assumption leads the multi-agent paradigm to be considered naturally superior to single-model approaches.

**Limitations of Prior Work**: Numerous reproduction efforts in reality show that multi-agent systems often underperform single agents. However, **no clean evaluation exists to separate "collective reasoning failure" from "individual reasoning deficiency."** If a group answers incorrectly, is it because the models are incompetent or because the information integration mechanism is poor? Existing benchmarks conflate the two, making attribution impossible.

**Key Challenge**: To measure "collective reasoning" itself, one must ensure: (i) the task is **unsolvable** for individuals in isolation (necessitating a group); (ii) the task is **solvable** if all information is given to a single agent (eliminating the "task too difficult" confounder); and (iii) ground truth must be verifiable.

**Goal**: To engineer the Hidden Profile paradigm from social psychology into a scalable multi-agent benchmark and systematically characterize the failure modes of frontier LLMs under distributed information, while testing if simple protocols can rescue them.

**Key Insight**: Hidden Profile is a classic paradigm used in social psychology for decades to study human group decision-making failures. Each member holds different critical information; they must pool it to find the correct answer, otherwise, shared information leads to a wrong conclusion. Formalizing this as an LLM evaluation naturally satisfies the requirements of "unsolvable by individuals, solvable by the group, with ground truth."

**Core Idea**: Construct HiddenBench (65 tasks, 5 adapted from human studies + 3 handcrafted + 57 automatically generated). Evaluate 15 frontier LLMs under both Hidden and Full Profile conditions. Use ablation to isolate the true bottleneck of failure: agents **can** integrate information that has been stated, but **cannot** actively elicit information that remains unstated.

## Method

### Overall Architecture
Task Structure: Each task consists of several decision options and task-relevant facts. Under the **Hidden Profile condition**, some facts ($\mathcal{I}_s$) are shared by all agents, while the remaining unshared facts ($\mathcal{I}_u$) are uniquely distributed to each agent, i.e., agent $a_i$ receives $I_i=\mathcal{I}_s\cup\{u_i\}$. The shared information is constructed to **support the incorrect option**, and only pooling all unshared facts points to the correct option. Under the **Full Profile condition**, all agents receive $\mathcal{I}_s\cup\mathcal{I}_u$. Agents are not informed if information asymmetry exists. Evaluation compares $Y^{\text{pre}}$ (pre-discussion), $Y^{\text{post}}$ (post-discussion), and $Y^{\text{full}}$ (Full Profile upper bound).

### Key Designs

1.  **HiddenBench Task Construction and Automated Generation Pipeline**:
    *   Function: Extends the social psychology Hidden Profile paradigm to 65 cross-domain tasks (medical, organizational planning, cultural preservation, etc.), ensuring each task satisfies the formal constraint of "unsolvable individually, solvable collectively."
    *   Mechanism: A three-stage pipeline—(i) **Generation**: GPT-4.1 generates candidate tasks based on structured templates (scenario + options + shared facts + unshared facts + correct answer); (ii) **Execution**: Each candidate is run 10 times under both Full and Hidden conditions to measure pre-discussion accuracy; (iii) **Selection**: Only tasks with Full Profile accuracy $\ge 80\%$ and Hidden Profile accuracy $\le 20\%$ are retained. From 200 candidates, 57 were selected (28.5% pass rate), plus 5 adapted and 3 handcrafted tasks for a total of 65.
    *   Design Motivation: Handcrafted tasks are not scalable and prone to subjective bias; pure automated generation cannot guarantee "formal correctness." This "generation-execution-selection" workflow transforms the hard constraints of the social psychology paradigm into machine-verifiable threshold conditions, which is key to turning the soft concept of "collective reasoning" into a reproducible benchmark.

2.  **Evaluation Protocol Across Three Conditions**:
    *   Function: Uses $Y^{\text{full}}$ as the upper bound of individual reasoning, $Y^{\text{pre}}$ as the lower bound of "necessitating a group," and $Y^{\text{post}}$ to evaluate collective reasoning. Compare these to derive clean metrics: "Collective Gain" ($Y^{\text{post}}-Y^{\text{pre}}$) and "Gap to Upper Bound" ($Y^{\text{post}}-Y^{\text{full}}$).
    *   Mechanism: 10 sessions are run for every model on every task. 15 frontier LLMs are evaluated (spanning four families: OpenAI GPT series, Google Gemini series, Alibaba Qwen series, and Meta Llama series), while varying communication depth $T\in\{5,10,15,20\}$ and group size for scaling analysis.
    *   Design Motivation: Traditional benchmarks report only a single accuracy figure, preventing attribution. The three-condition comparison allows direct reading of "whether the model is inadequate or the coordination is poor," representing the paper's most significant methodological contribution.

3.  **Targeted Ablation for Failure Modes**:
    *   Function: Decomposes "collective failure" into three possibilities: aggregation failure, inference failure, and action selection failure, eventually localizing it to action selection.
    *   Mechanism: Varied communication rounds (5/10/15/20), prompting strategies (cooperative / conflictual / CoT / informing asymmetry / share-all), and a forced reveal-all intervention (mechanically forcing disclosure of all info in round-1). It was found that reveal-all significantly closed the gap, proving agents can reason correctly once info is disclosed—the bottleneck is not reasoning, but **the lack of awareness to elicit information from others**.
    *   Design Motivation: Simply stating "multi-agent systems fail" is of little value; identifying the specific failure stage guides future improvements. This ablation upgrades the conclusion from "phenomenon" to "mechanistic diagnosis."

### Loss & Training
Evaluation paper; no training involved. All models are invoked zero-shot via APIs.

## Key Experimental Results

### Main Results
Performance on HiddenBench across 15 frontier LLMs (65 tasks, 10 sessions, average rule, post-discussion accuracy under Hidden Profile):

| Model | $Y^{\text{full}}$ (Full) | $Y^{\text{pre}}$ (Hidden Pre-disc) | $Y^{\text{post}}$ (Hidden Post-disc) | Gain | Gap with Full |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | 0.981 | 0.217 | **0.671** | +0.454 | -0.310 |
| Gemini-2.5-Flash | High | Mid | 0.550 | Large | Mid |
| Gemini-2.5-Flash-Lite | High | Mid | 0.394 | Mid | Large |
| GPT-5 (minimal reasoning) | High | Mid | Mid | Small | **-0.750** |
| GPT-5-Nano | High | Mid | Low | **-0.004** (Almost none) | Extreme |
| **Overall Mean (15 Models)** | **0.807** | 0.082~0.217 | **0.301** | Moderate | ~ -0.5 |

Key lateral comparison facts: (i) Single agents average 80.7% under Full Profile, while multi-agents only achieve 30.1% under Hidden Profile—a 50-point gap; (ii) Model size/individual reasoning ability **does not** reliably predict collective performance (GPT-5 is individually strong but collectively weak); (iii) The Gemini family significantly outperforms other families in collective settings.

### Ablation Study

| Intervention Dimension | Key Phenomenon | Interpretation |
| :--- | :--- | :--- |
| Comm. Depth $T=5/10/15/20$ | $Y^{\text{post}}=0.233$ peaks at $T=15$; drops to 0.133 at $T=20$ | Long discussions reinforce false consensus rather than exploration |
| Cooperative / Constructive prompt | $Y^{\text{post}}=0.20\sim 0.24$ | Cooperative prompts show no significant improvement |
| Conflictual prompt | $Y^{\text{post}}=0.0\sim 0.26$, no majority consensus in most cases | Conflictual prompts lead to failure to converge |
| Zero-shot CoT | 0.222 | Limited improvement |
| Informing asymmetry (telling agents "info may be hidden") | 0.367 | Simply informing helps but is insufficient |
| Share All Information (prompting to disclose) | 0.467 | Only closes about half the gap; disclosure alone isn't enough |
| **Reveal-All (Mechanically forced round-1 disclosure)** | Significantly closes the gap | **Proves the bottleneck is action selection, not inference** |
| Increasing Group size | $Y^{\text{post}}$ decreases | More agents make coordination harder |

### Key Findings
*   **Failure Mode Localization**: Agents can integrate disclosed information but **fail to actively elicit unshared information**—this is the core conclusion attributing the 50-point gap to a specific capability deficit.
*   Model scale / Individual reasoning $\neq$ Collective reasoning. Reasoning-heavy models like GPT-5 showed no significant advantage in collective settings, challenging the "scale up will solve it" assumption.
*   Excessive communication rounds **reinforce premature consensus**, consistent with "groupthink" in human social psychology.
*   A **lightweight structured communication protocol** (asking agents to explicitly list unique evidence before debating) significantly improves $Y^{\text{post}}$ across families, proving the bottleneck is actionable without changing models.

## Highlights & Insights
*   Transforming the soft concept of "collective reasoning" into a hard formal constraint ("solvable collectively, unsolvable individually" via dual-threshold filtering) is a masterful engineering feat that other multi-agent benchmark efforts can adopt.
*   The three-condition comparison protocol (Hidden-pre / Hidden-post / Full) essentially adds **causal counterfactual control** to evaluation—running the same task under different information conditions allows for direct failure attribution. This paradigm can be generalized to other collective reasoning or cooperation evaluations.
*   The contrast between "Reveal-All intervention" and "Share-All prompting" is particularly educational: prompting makes the model "know it should speak" but it still fails to disclose everything; mechanical intervention forces full disclosure and leads to major improvement—indicating elicitation behavior is not a knowledge problem, but a **goal/incentive problem** that should be addressed via RL objective design.
*   Reveals a counter-intuitive fact: **more agents $\neq$ better**. Contrary to the naive "wisdom of the crowd" assumption, this aligns with March's exploration-exploitation theory and Janis's groupthink theory.

## Limitations & Future Work
*   Tasks are restricted to multiple-choice decision-making and do not cover open-ended generation, tool-use, or long-term collaboration.
*   The communication protocol is synchronous all-to-all broadcasting; partial observability, asynchronous messaging, and structured organizational hierarchies were not tested.
*   Prompting interventions prove the bottleneck is elicitation, but **no systemic solution was provided at the training level**—future work requires RL/SFT dataset design to fundamentally change this behavior.
*   The 15 models are mostly closed-source (with some open-source); while broad, no fine-grained attribution based on training data or RLHF recipes was performed.

## Related Work & Insights
*   **vs. Du et al. (Multi-agent debate)**: They assume debate automatically brings benefits; this paper refutes that—more debate can sometimes degrade performance, and the core bottleneck is unrelated to "debate quality."
*   **vs. Cemri et al. (LLM coordination failure)**: They observed coordination problems but lacked controlled variables; HiddenBench isolates information asymmetry as the single variable for cleaner attribution.
*   **vs. SocPsych Hidden Profile studies**: This paper serves as a template for engineering human psychology paradigms into LLM evaluations, proving many AI agent failure modes are isomorphic to human group failures—this path of "leveraging social psychology for AI evaluation" holds significant future potential.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Engineering the Hidden Profile paradigm into a scalable LLM benchmark is a pioneering bridge from social psychology to AI evaluation.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 15 models, 65 tasks, dual conditions, and multi-dimensional ablations provide rare breadth and depth of attribution.
*   Writing Quality: ⭐⭐⭐⭐⭐ The logical chain localizing the failure mode to "action selection" is clear and reproducible.
*   Value: ⭐⭐⭐⭐⭐ Provides the multi-agent LLM community with a clean evaluation tool and a clear research direction (elicit-aware coordination); its influence will be lasting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Collaborative Multi-Agent Scripts Generation for Enhancing Imperfect-Information Reasoning in Murder Mystery Games](../../ACL2026/multi_agent/collaborative_multi-agent_scripts_generation_for_enhancing_imperfect-information.md)
- [\[ICML 2026\] Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information](beyond_majority_voting_llm_aggregation_by_leveraging_higher-order_information.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](../../ACL2026/multi_agent/topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ICML 2026\] MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks](mas-orchestra_understanding_and_improving_multi-agent_reasoning_through_holistic.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)

</div>

<!-- RELATED:END -->
