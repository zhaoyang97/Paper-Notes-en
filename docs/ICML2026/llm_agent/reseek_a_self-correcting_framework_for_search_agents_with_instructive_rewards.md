---
title: >-
  [Paper Note] ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards
description: >-
  [ICML 2026][LLM Agent][Search Agent] ReSeek augments RL-trained search agents with a JUDGE action and uses BGE-reranker to compute an "ideal judgment" as a process reward…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Search Agent"
  - "Self-Correction"
  - "JUDGE action"
  - "Process Reward"
  - "Data Contamination Evaluation"
date: 2026-05-08
content_hash: de4243390a83172d
---

# ReSeek: A Self-Correcting Framework for Search Agents with Instructive Rewards

**Conference**: ICML 2026  
**arXiv**: [2510.00568](https://arxiv.org/abs/2510.00568)  
**Code**: https://github.com/TencentBAC/ReSeek (available)  
**Area**: Information Retrieval / Search Agent / Reinforcement Learning  
**Keywords**: Search Agent, Self-Correction, JUDGE action, Process Reward, Data Contamination Evaluation

## TL;DR
ReSeek augments RL-trained search agents with a JUDGE action and uses BGE-reranker to compute an "ideal judgment" as a process reward, enabling the agent to softly "mask" irrelevant information and re-query after each retrieval. It also introduces FictionalHot, a contamination-resistant benchmark based on fictional entities. On Qwen2.5-7B, the average EM reaches 0.377, +3.1 higher than ZeroSearch.

## Background & Motivation

**Background**: LLM-based search agents (e.g., Search-R1, ZeroSearch, DeepResearcher, WebThinker) are trained via RL to perform multi-step "think-search-reason" cycles, significantly outperforming single-step RAG on knowledge-intensive tasks. The mainstream approach models the task as an MDP and optimizes the policy using RL algorithms such as GRPO/PPO.

**Limitations of Prior Work**: (1) **Sparse reward signals**—most methods only provide an EM reward ("is the answer correct") at the final step (e.g., Search-R1, ZeroSearch), with no feedback for intermediate steps, making credit assignment difficult; (2) **Lack of self-correction**—if early search queries are poor (e.g., "creator of Saddle Rash" returns only a show description, not the birthdate), the agent gets stuck and produces chained errors; (3) **Evaluation contamination**—main benchmarks (NQ, TriviaQA, HotpotQA) are heavily present in LLM pretraining corpora, so high scores may reflect memorization rather than genuine reasoning.

**Key Challenge**: RL agents require dense, instructive intermediate feedback to learn "is this clue useful, should I change the query," but such feedback traditionally requires costly human annotation or a specially trained PRM (process reward model). How to provide reliable process rewards without extra training cost remains an open problem.

**Goal**: Enable search agents to "self-evaluate" the usefulness of retrieved information mid-episode, replanning if necessary rather than stubbornly answering; also, provide a clean, contamination-free evaluation bed.

**Key Insight**: The authors observe that humans naturally "pause after reading each search result, judge its usefulness, and decide whether to change keywords" during web research. This judgment can be objectively approximated using a reranker model (bge-reranker-large) to compute the semantic similarity between the observation and ground truth—rewarding correct judgments and penalizing incorrect ones. The reranker is pretrained and requires no additional training.

**Core Idea**: Make "self-judgment" an explicit agent action (<judge>Yes/No</judge>), use the reranker to compute the "ideal judgment" as a process reward, and provide dense supervision for this ability. Simultaneously, construct the fully fictional FictionalHot dataset to force agents to rely on retrieval rather than memorization.

## Method

### Overall Architecture
The agent uses a GRPO-optimized LLM policy $\pi_\theta$, with an action space that adds a `<judge>` action to the standard `<search>` / `<answer>`. Each round: the agent thinks → decides whether to search → after searching, must execute judge → based on the judge result, decides the next step (search again / answer). The reward has two parts: (1) terminal EM reward $R_{\text{answer}}$, (2) process reward $R_{\text{judge}}$ at each judge step, computed by BGE-reranker as the reference "ideal judgment." The pipeline is trained on 169k NQ + HotpotQA samples, deployed on Qwen2.5-3B/7B-Instruct, with Wiki-18 + E5 embedding as the retrieval corpus.

### Key Designs

1. **JUDGE Action and Soft Masking Mechanism**:

    - **Function**: Empowers the agent with metacognitive ability to "evaluate whether the just-retrieved result is useful," turning the reasoning chain from static linear to dynamic self-correcting loops.
    - **Mechanism**: After each `<search>`, the agent is forced to output `<judge>Yes</judge>` or `<judge>No</judge>`. This label $j_t$ is appended to the context: $\mathcal{C}_t = \tau_{t-1} \oplus a_t \oplus o_t \oplus j_t$, and the agent samples the next action based on this labeled context. "No" does not physically delete the observation but "softly masks" it—marking the irrelevant information so the policy is more likely to change the query rather than continue based on wrong info. The observation remains in context, providing a record of "already tried paths" for reflection. The prompt uses strict if-then rules (Table 1) to enforce: "If judge=No, you must search again and cannot answer directly," producing structured trajectories even on untrained LLMs and providing a clean starting point for RL training.
    - **Design Motivation**: Physically deleting observations leads to repeated mistakes; soft masking prevents error propagation while retaining meta-information about failed paths. Making judge an internal action (not an external module) allows the policy to learn end-to-end when to re-evaluate.

2. **Reranker-Based Dense Process Reward**:

    - **Function**: Converts sparse EM signals into dense stepwise feedback, specifically rewarding the agent for "correctly judging whether an observation is useful."
    - **Mechanism**: Defines the "ideal judgment" $j^*_t$ as: if BGE-reranker computes the semantic similarity between observation $o_t$ and ground truth gt as $> 0.7$, then $j^*_t = \text{Yes}$, otherwise No (threshold 0.7 validated by grid search). Each time the agent executes a judge action, $R_{\text{judge}}$ is given based on whether $j_t$ matches $j^*_t$: **correct** judgments are rewarded +0.3 (whether correctly accepting Yes=Yes or correctly rejecting No=No), **incorrect** judgments are penalized asymmetrically—"wrongly accepting useless info" is penalized -0.6 (double strength), "wrongly discarding useful info" is penalized -0.3. The full training objective is $R(\tau) = \sum_t \gamma^{t-1} r_t$, with $r_{t<T} = R_{\text{judge}}$ for intermediate steps and $r_T = R_{\text{answer}} = \text{EM}(A_p, A_g)$ for the terminal step, optimized by GRPO.
    - **Design Motivation**: Why use a reranker instead of LLM-as-judge? The reranker is lighter, more stable, and introduces no new biases. Why asymmetric penalties? Accepting useless info derails the entire trajectory (high cost), while discarding useful info only wastes a query (lower cost)—this asymmetry matches the real-world error cost distribution.

3. **FictionalHot Contamination-Resistant Benchmark**:

    - **Function**: Removes "pseudo-high scores" from LLM memorization, truly testing reasoning and retrieval ability.
    - **Mechanism**: Randomly samples 10% (5,116) from six mainstream QA datasets (NQ/TriviaQA/PopQA/HotpotQA/2Wiki/Musique, total 51,588 samples) as seeds, and rewrites them using GPT-5—replacing real entities (e.g., Taylor Swift) with fictional ones (e.g., Lila Starling), preserving the original reasoning structure. GPT-5 also generates Wikipedia-style support documents for these fictional entities, setting new fictional facts (e.g., album release year 2007) as new GT. These fictional samples are inserted into the 2018 Wikipedia corpus to form a closed-world evaluation bed. Any "memory-based" method will fail on FictionalHot (Direct Inference ≈ 0.001); only methods that truly retrieve and reason can score.
    - **Design Motivation**: Table 2 compares evaluation setups of seven prior works, revealing inconsistencies in corpus (static wiki vs live internet), test set, training set, and metrics. The authors argue this fragmentation obscures real capability differences; FictionalHot is both contamination-resistant and reproducible, providing a standard bed for the search agent community.

### Loss & Training

- **RL Algorithm**: GRPO by default, using the same unified NQ+HotpotQA training set as Search-R1 (169,615 pairs).
- **Hyperparameters**: retrieval top-k=3, max 4 turns, 16×H20 GPU, E5 embedding + Wiki-18.
- **Reward Parameters**: reranker threshold 0.7, $R_{\text{match}} = +0.3$, $R_{\text{mismatch}}^{\text{accept-wrong}} = -0.6$, $R_{\text{mismatch}}^{\text{reject-right}} = -0.3$.
- **Structured Prompt**: Enforces the `<think>` → `<search>` → `<judge>` → conditional branch → `<answer>` flow (Table 1), ensuring parseable trajectories.

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

| Component (Qwen2.5-7B) | Avg EM | Gain | Note |
|---|---|---|---|
| $R_{\text{answer}}$ only (=Search-R1) | 0.288 | baseline | Only terminal EM reward |
| + judge Action (rule only) | 0.297 | +3.1% | Adds judge action + forced prompt |
| + $R_{\text{judge}}$ (full ReSeek) | **0.312** | **+8.3%** | Adds reranker process reward |
| Reranker Type: None | 0.259 | - | No reranker at all |
| Regex-based | 0.301 | - | Keyword-matching heuristic |
| Qwen-Reranker | 0.311 | - | Replaced with Qwen reranker |
| BGE-Reranker (ReSeek) | **0.312** | - | Default choice |

| Reranker-only vs RL-trained (Qwen2.5-7B) | Avg EM |
|---|---|
| Search-R1 baseline | 0.342 |
| + Reranker-only intervention (no RL) | 0.354 (+1.2) |
| + Prompt-only judge (no RL) | 0.349 (+0.7) |
| **ReSeek (GRPO full)** | **0.377 (+3.5)** |

### Key Findings
- **Greater gains on multi-hop tasks**: ReSeek shows significant improvements (+5-12%) on tasks requiring 2-3 hops (HotpotQA, Bamboogle, Musique), while on single-hop tasks (TriviaQA) it occasionally lags behind ZeroSearch—demonstrating that self-correction is most valuable for long reasoning chains.
- **FictionalHot exposes all methods**: Direct Inference is nearly 0; even the best ReSeek achieves only 0.061. On TriviaQA, 7B vs 3B differs by 0.408 vs 0.288 (+12%), but on FictionalHot, scores are nearly identical (0.061 vs 0.059), confirming that FictionalHot truly tests reasoning, not memorization.
- **Turn budget analysis (Figure 4)**: Other baselines benefit most from 1→2 turns, saturating at 3-4; ReSeek increases monotonically up to 4 turns, indicating that extra budget is used for "re-querying," while other methods waste it.
- **Asymmetric penalty is crucial**: Empirically, "wrongly accepting (-0.6) > wrongly discarding (-0.3)" outperforms symmetric penalties—agents learn "better to over-query than answer blindly."
- **Reranker vs RL decoupling**: Reranker-only yields +1.2, but adding RL training brings an additional +2.3, showing that RL not only amplifies the reranker signal but also teaches the agent to **use judgment appropriately** in context—a meaningful decoupling experiment.

## Highlights & Insights
- **Using pretrained reranker as PRM (Process Reward Model)**: Traditionally, a separate PRM is trained at high cost and instability; the authors use the off-the-shelf bge-reranker-large as the "ideal judgment" reference, plug-and-play, introducing no new bias—this approach is generalizable to any RL task where "intermediate steps can be evaluated by a pretrained discriminator."
- **JUDGE as an internal action, not an external module**: Unlike ReAct's "Thought → Action → Observation" loop, ReSeek makes judge part of the policy, enabling RL to learn end-to-end "when to pause and evaluate, what evidence counts as useful." This "explicit metacognitive action" is a reusable paradigm for agentic LLM design.
- **Asymmetric reward design**: The penalty for wrongly accepting vs wrongly rejecting differs by a factor, matching real-world cost asymmetry—many RL works use symmetric rewards, but this nuanced design is instructive.
- **FictionalHot's construction method is itself a contribution**: Standardizing "LLM rewriting + synthetic Wikipedia entries" as a pipeline allows any subfield to create its own "contamination-resistant" version—much more constructive than merely complaining about data contamination.
- **"Soft masking" instead of "hard deletion"**: Observations judged as No remain in context as "failed attempts," preventing the agent from repeatedly trying the same dead end—this explicit failure trace design is clever.

## Limitations & Future Work
- Training uses ground truth to compute "ideal judgment," but at deployment, the agent has no GT for reference—fortunately, the agent has already learned to judge and does not need to compute reranker online; however, whether the reranker threshold 0.7 is robust across domains or needs adaptation is not discussed.
- Only validated on QA benchmarks; whether the judge ability generalizes to open-ended tasks (e.g., writing reviews, conducting research) is untested.
- FictionalHot is rewritten by GPT-5, so rewrite quality depends on GPT-5; long-term, manual sanity checks may be needed to prevent new biases.
- Training data is limited to NQ + HotpotQA; transferability to corpora with greater domain differences or the need for in-domain training data is not discussed.
- BGE-reranker and Qwen-reranker differ minimally (0.312 vs 0.311); no ablation on reranker strength's impact on the ceiling—possibly, reranker is not the bottleneck, and judge learning itself is.
- Only compared up to Qwen2.5; integration with the latest R1/o1-type reasoning models is not explored.

## Related Work & Insights
- **vs Search-R1 (Jin 2025)**: Search-R1 trains GRPO with sparse EM rewards, prone to early error chains; ReSeek adds judge action and reranker process reward, directly improving EM by +3.5—demonstrating the value of process rewards under the same baseline.
- **vs ReAct (Yao 2022)**: ReAct uses a Thought/Action/Observation loop, but thought is free-form and unsupervised; ReSeek makes self-evaluation an explicit typed action with reward, enabling end-to-end RL learning—more structured and trainable.
- **vs AgentPRM (Xi 2026)**: AgentPRM trains a separate PRM to score the agent; ReSeek uses a reranker instead, reducing training cost and embedding judge as an action rather than an external scorer—lower cost, tighter integration.
- **vs Reflexion / Self-Refine**: These methods perform "post-hoc reflection" to correct the next trajectory; ReSeek does "in-episode correction," changing queries within the same episode for more timely response.
- **vs S2R (Ma 2025)**: S2R trains self-verification for mathematical reasoning; ReSeek applies this idea specifically to search (reranker signal + judge action), making it more targeted.

## Rating
- Novelty: ⭐⭐⭐⭐ judge action + reranker as PRM + FictionalHot are all relatively novel, though each has predecessors
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 datasets + 4 baselines + 6 ablations + turn budget analysis, very comprehensive
- Writing Quality: ⭐⭐⭐⭐ Figure 1's comparative example is intuitive, Table 2 convincingly illustrates evaluation fragmentation
- Value: ⭐⭐⭐⭐⭐ Provides both a method (judge action + process reward) and an evaluation bed (FictionalHot), with the latter potentially contributing even more to the community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICLR 2026\] InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios](../../ICLR2026/llm_agent/infiagent_self-evolving_pyramid_agent_framework_for_infinite_scenarios.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](../../ICLR2026/llm_agent/mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](../../ACL2026/llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](../../ICLR2026/llm_agent/your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)

</div>

<!-- RELATED:END -->
