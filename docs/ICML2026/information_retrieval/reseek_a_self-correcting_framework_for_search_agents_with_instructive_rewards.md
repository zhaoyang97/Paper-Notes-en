---
title: >-
  [Paper Note] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards
description: >-
  [ICML 2026][Information Retrieval & RAG][Search Agent] ReSeek adds a JUDGE action to RL-trained search agents and utilizes a BGE-reranker to calculate "ideal judgments" as process rewards. This enables agents to "softly…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Search Agent"
  - "Self-correction"
  - "JUDGE action"
  - "Process Reward"
  - "Contamination Evaluation"
date: 2026-05-08
content_hash: fa0f544ac7e2a8b6
---

# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards

**Conference**: ICML 2026  
**arXiv**: [2510.00568](https://arxiv.org/abs/2510.00568)  
**Code**: https://github.com/TencentBAC/ReSeek (Available)  
**Area**: Information Retrieval / Search Agent / Reinforcement Learning  
**Keywords**: Search Agent, Self-correction, JUDGE action, Process Reward, Contamination Evaluation

## TL;DR
ReSeek adds a JUDGE action to RL-trained search agents and utilizes a BGE-reranker to calculate "ideal judgments" as process rewards. This enables agents to "softly shield" ineffective information and re-query after each retrieval. Simultaneously, the authors propose FictionalHot, an anti-contamination benchmark based on fictional entities. On Qwen2.5-7B, ReSeek achieves an average EM of 0.377, a +3.1 improvement over ZeroSearch.

## Background & Motivation

**Background**: LLM search agents (e.g., Search-R1, ZeroSearch, DeepResearcher, WebThinker) utilize RL training to allow LLMs to learn a multi-step "think-search-reason" cycle, significantly outperforming single-step RAG on knowledge-intensive tasks. The mainstream approach models the task as an MDP and optimizes policies using RL algorithms like GRPO or PPO.

**Limitations of Prior Work**: **(1) Sparse reward signals**—most methods only provide an EM reward for "answer correctness" at the final step (Search-R1, ZeroSearch), with no feedback for intermediate steps, leading to difficult credit assignment; **(2) Lack of self-correction mechanisms**—if early search queries are poor (e.g., "creator of Saddle Rash" returns only show descriptions without birthdates), agents "stick" to this dead end and chain-produce incorrect answers; **(3) Evaluation contamination**—mainstream benchmarks (NQ, TriviaQA, HotpotQA) appear extensively in LLM pre-training corpora, where high scores may reflect memory rather than true reasoning.

**Key Challenge**: RL agents require dense, instructive intermediate feedback to learn "how useful this clue is and whether to change the query." However, such feedback traditionally requires expensive human annotation or specially trained Process Reward Models (PRMs). Providing reliable process rewards without additional training costs remains an open problem.

**Goal**: To enable search agents to "self-evaluate" retrieved information mid-episode, allowing them to re-plan if information is useless rather than forcing an answer; and to provide a clean, uncontaminated evaluation environment.

**Key Insight**: The authors observe that when humans conduct web research, they naturally "stop after reading each search result, judge its utility, and decide whether to change keywords." This judgment action can be computed using a reranker model (bge-reranker-large) to measure the "semantic similarity between observation and ground truth" as an objective reference—rewarding the agent for correct judgments and punishing errors. Since the reranker is pre-trained, no extra training is needed.

**Core Idea**: "Self-judgment" is explicitly integrated as an agent action (`<judge>Yes/No</judge>`). A reranker calculates the "ideal judgment" to serve as a process reward for dense supervision. Furthermore, the FictionalHot dataset, composed entirely of fictional entities, forces agents to rely on retrieval rather than memory.

## Method

### Overall Architecture
The agent employs an LLM policy $\pi_\theta$ optimized by GRPO, with the action space expanded to include the `<judge>` action alongside standard `<search>` and `<answer>` actions. In each turn, the agent thinks $\rightarrow$ decides whether to search $\rightarrow$ must execute a judge action after searching $\rightarrow$ determines the next step (re-search or answer) based on the judge result. The reward consists of two parts: (1) a terminal EM reward $R_{\text{answer}}$ and (2) a process reward $R_{\text{judge}}$ based on the BGE-reranker's "ideal judgment" reference. The pipeline is trained on 169k NQ + HotpotQA samples, deployed on Qwen2.5-3B/7B-Instruct with Wiki-18 and E5 embeddings for retrieval.

### Key Designs

1.  **JUDGE Action and Soft Shielding Mechanism**:
    *   **Function**: Provides the agent with meta-cognitive abilities to evaluate retrieved results, transforming the reasoning chain from static linearity into a dynamic self-correcting loop.
    *   **Mechanism**: After each `<search>`, the agent is forced to output `<judge>Yes</judge>` or `<judge>No</judge>`. This label $j_t$ is concatenated to the context: $\mathcal{C}_t = \tau_{t-1} \oplus a_t \oplus o_t \oplus j_t$, and the agent samples the next action accordingly. "No" does not physically delete the observation; it is a "soft shield"—marking the irrelevant information so the policy "notices the rejection" and tends to change the query rather than proceed with incorrect info. The observation remains in context to provide reflection (e.g., "I have already tried this path"). Strict if-then rules (Table 1) in the prompt mandate: "if judge=No, you must search again and cannot answer directly," providing a structured starting point for RL.
    *   **Design Motivation**: Physical deletion causes the same errors to repeat. Soft shielding prevents error propagation while retaining meta-information of failed paths. Treating JUDGE as an internal action rather than an external module allows the policy to learn "when to re-evaluate" in an end-to-end manner.

2.  **Dense Process Reward based on Reranker**:
    *   **Function**: Converts sparse EM signals into dense, step-by-step feedback specifically rewarding the agent for correctly judging observation utility.
    *   **Mechanism**: The "ideal judgment" $j^*_t$ is defined: if the BGE-reranker similarity between observation $o_t$ and ground truth (gt) is $> 0.7$, then $j^*_t = \text{Yes}$, else No. Process reward $R_{\text{judge}}$ is given based on the alignment of $j_t$ with $j^*_t$: **Correct** judgments receive +0.3 (for both Yes=Yes and No=No); **Incorrect** judgments receive asymmetric penalties—"falsely accepting useless info" is penalized -0.6 (double weight), while "falsely rejecting useful info" is penalized -0.3. Total target $R(\tau) = \sum_t \gamma^{t-1} r_t$, where $r_{t<T} = R_{\text{judge}}$ and $r_T = R_{\text{answer}} = \text{EM}(A_p, A_g)$, optimized via GRPO.
    *   **Design Motivation**: Rerankers are more lightweight and stable than LLM-as-judge without introducing new biases. Asymmetric penalties reflect real-world costs: accepting junk sways the entire trajectory (high cost), while discarding useful info merely wastes one query (low cost).

3.  **FictionalHot Anti-Contamination Benchmark**:
    *   **Function**: Isolates gains from LLM memory and accurately tests reasoning and retrieval capabilities.
    *   **Mechanism**: 10% (5,116 samples) are sampled from 6 mainstream QA datasets (NQ, TriviaQA, etc.) as seeds and rewritten via GPT-5. Real entities (e.g., Taylor Swift) are replaced with fictional ones (e.g., Lila Starling) while maintaining the reasoning structure. Supporting Wiki-style documents and new ground truths (e.g., album year 2007) are generated and inserted into the 2018 Wikipedia corpus to form a closed-world testbed. "Memory-based" methods fail FictionalHot (Direct Inference drops to ~0.001); only methods capable of retrieval and reasoning score well.
    *   **Design Motivation**: Comparing 7 prior works (Table 2) revealed fragmented evaluation settings (different corpora, test sets, metrics). FictionalHot provides a standardized, reproducible benchmark that eliminates memory bias.

### Loss & Training
*   **RL Algorithm**: GRPO is the default; uses the same unified NQ+HotpotQA training set as Search-R1 (169,615 pairs).
*   **Hyperparameters**: Retrieval top-k=3, max 4 turns, 16×H20 GPUs, E5 embedding + Wiki-18.
*   **Reward Parameters**: Reranker threshold 0.7, $R_{\text{match}} = +0.3$, $R_{\text{mismatch}}^{\text{accept-wrong}} = -0.6$, $R_{\text{mismatch}}^{\text{reject-right}} = -0.3$.
*   **Structured Prompt**: Enforces the flow: `<think>` $\rightarrow$ `<search>` $\rightarrow$ `<judge>` $\rightarrow$ conditional branch $\rightarrow$ `<answer>` (Table 1) to ensure trajectory transparency.

## Key Experimental Results

### Main Results

| Method (Qwen2.5-7B-Instruct) | NQ | TriviaQA | PopQA | HotpotQA | 2Wiki | Musique | Bamboogle | FictionalHot | Avg |
|---|---|---|---|---|---|---|---|---|---|
| Direct Inference | 0.134 | 0.408 | 0.140 | 0.183 | 0.250 | 0.031 | 0.120 | 0.001 | 0.158 |
| CoT | 0.048 | 0.185 | 0.054 | 0.092 | 0.111 | 0.022 | 0.232 | 0.001 | 0.093 |
| RAG | 0.349 | 0.585 | 0.392 | 0.299 | 0.235 | 0.058 | 0.208 | 0.012 | 0.267 |
| Search-o1 | 0.151 | 0.443 | 0.131 | 0.187 | 0.176 | 0.058 | 0.296 | 0.020 | 0.183 |
| Search-R1 | 0.393 | 0.610 | 0.397 | 0.370 | 0.414 | 0.146 | 0.368 | 0.034 | 0.342 |
| ZeroSearch | 0.436 | 0.652 | 0.488 | 0.346 | 0.352 | 0.184 | 0.278 | 0.031 | 0.346 |
| **ReSeek** | **0.469** | 0.640 | **0.501** | **0.389** | 0.382 | **0.185** | **0.392** | **0.061** | **0.377** |

### Ablation Study

| Component (Qwen2.5-7B) | Avg EM | Gain | Description |
|---|---|---|---|
| $R_{\text{answer}}$ only (=Search-R1) | 0.288 | baseline | Terminal EM reward only |
| + judge Action (rule only) | 0.297 | +3.1% | Judge action + forced prompt only |
| + $R_{\text{judge}}$ (full ReSeek) | **0.312** | **+8.3%** | Process reward via Reranker added |

| Reranker-only vs RL-trained (Qwen2.5-7B) | Avg EM |
|---|---|
| Search-R1 baseline | 0.342 |
| + Reranker-only intervention (no RL) | 0.354 (+1.2) |
| + Prompt-only judge (no RL) | 0.349 (+0.7) |
| **ReSeek (GRPO full)** | **0.377 (+3.5)** |

### Key Findings
*   **Greater gains for multi-hop tasks**: ReSeek shows significant improvement (+5-12%) on tasks requiring 2-3 hops (HotpotQA, Bamboogle, Musique). Single-hop tasks (TriviaQA) occasionally show slight inferiority to ZeroSearch, confirming self-correction's primary value for long reasoning chains.
*   **FictionalHot exposed base model memory**: Direct Inference dropped to essentially 0; even ReSeek reached only 0.061. The 7B vs 3B gap on TriviaQA (+12%) disappeared on FictionalHot (0.061 vs 0.059), verifying that it tests reasoning over memory.
*   **Turn Budget Analysis**: Other baselines saturate after 2 turns. ReSeek increases monotonically up to 4 turns, suggesting it correctly utilizes extra budget for re-querying rather than wasting it.
*   **Asymmetric penalty is crucial**: The "falsely accept (-0.6) > falsely reject (-0.3)" setting outperformed symmetric penalties—the agent learns to prefer more queries over blind answering.
*   **Reranker vs RL Decoupling**: Reranker-only intervention adds +1.2, but adding RL training adds another +2.3. RL teaches the agent to **properly use** the judgment within the context, rather than just relaying the reranker signal.

## Highlights & Insights
*   **Leveraging pre-trained rerankers as PRMs**: Instead of training an expensive, unstable PRM, using bge-reranker-large as an "ideal judgment" reference provides immediate, bias-free feedback. This is applicable to any RL task where intermediate steps can be assessed by a pre-trained discriminator.
*   **JUDGE as an internal action**: Unlike ReAct's "Thought $\rightarrow$ Action $\rightarrow$ Observation" loop, ReSeek embeds judgment as a policy component. This enables end-to-end learning of meta-cognitive actions ("when to stop and evaluate").
*   **Asymmetric Reward Design**: Matching penalty magnitude to real task costs (misinterpretation vs. inefficiency) is a refined design choice often overlooked in symmetric reward RL works.
*   **FictionalHot Construction**: The pipeline of "LLM rewriting + synthetic Wiki entries" provides a constructive solution to data contamination—arguably more valuable than simply pointing it out.
*   **"Soft Shielding" vs "Hard Deletion"**: Retaining rejected observations in context as "known dead ends" prevents the agent from repeating the same mistakes within an episode.

## Limitations & Future Work
*   There is a gap between training (using GT for ideal judgment) and deployment (where GT is absent). While the agent learns to judge independently, the robustness of the 0.7 threshold across domains remains unexplored.
*   Evaluation is limited to QA; generalizability to open-ended tasks (research reports, reviews) is untested.
*   FictionalHot quality depends on the rewriter (GPT-5); human sanity checks may be needed for long-term use.
*   The impact of reranker performance on the agent's ceiling was not fully ablated—the JUDGE learning process itself might be the bottleneck rather than the reranker strength.
*   Integration with latest reasoning models (R1/o1) was not addressed.

## Related Work & Insights
*   **vs Search-R1 (Jin 2025)**: Search-R1 uses sparse EM rewards, prone to early errors. ReSeek adds JUDGE actions and process rewards, showing a +3.5 EM improvement under the same baseline.
*   **vs ReAct (Yao 2022)**: ReAct uses free-form thoughts; ReSeek uses typed actions with rewarded supervision, making it more structured and trainable.
*   **vs AgentPRM (Xi 2026)**: While AgentPRM trains a standalone PRM, ReSeek replaces it with a reranker to lower costs and integrates judgment directly into the action space.
*   **vs Reflexion / Self-Refine**: These perform "post-hoc reflection" for the next trajectory; ReSeek performs "mid-task correction" within the same episode.

## Rating
*   Novelty: ⭐⭐⭐⭐ (Combining JUDGE action, reranker-as-PRM, and FictionalHot is innovative)
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (8 datasets, 4 baselines, 6 ablations, and budget analysis)
*   Writing Quality: ⭐⭐⭐⭐ (Figure 1 and Table 2 are highly persuasive)
*   Value: ⭐⭐⭐⭐⭐ (Provides both a method and a benchmark; the latter is a significant community contribution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multi-Faceted Self-Consistent Preference Alignment for Query Rewriting in Conversational Search](../../ACL2026/information_retrieval/multi-faceted_self-consistent_preference_alignment_for_query_rewriting_in_conver.md)
- [\[ACL 2026\] Enhancing LLM-based Search Agents via Contribution Weighted Group Relative Policy Optimization](../../ACL2026/information_retrieval/enhancing_llm-based_search_agents_via_contribution_weighted_group_relative_polic.md)
- [\[ACL 2026\] Rerank Before You Reason: Analyzing Reranking Tradeoffs through Effective Token Cost in Deep Search Agents](../../ACL2026/information_retrieval/rerank_before_you_reason_analyzing_reranking_tradeoffs_through_effective_token_c.md)
- [\[ACL 2026\] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities](../../ACL2026/information_retrieval/can_compact_language_models_search_like_agents_distillation-guided_policy_optimi.md)
- [\[ICML 2026\] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
