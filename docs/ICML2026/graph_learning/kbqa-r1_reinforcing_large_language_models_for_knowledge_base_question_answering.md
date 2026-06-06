---
title: >-
  [Paper Note] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering
description: >-
  [ICML 2026][Graph Learning][KBQA] Redefines KBQA from "one-shot logical expression generation" into a "multi-turn decision process." It utilizes Referenced Rejection Sampling guided by gold action sequences to generate e…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "KBQA"
  - "Multi-turn Reinforcement Learning"
  - "GRPO"
  - "Referenced Rejection Sampling"
  - "Action Space"
date: 2026-05-08
content_hash: 63f092aef1c2ab1a
---

# KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering

**Conference**: ICML 2026  
**arXiv**: [2512.10999](https://arxiv.org/abs/2512.10999)  
**Code**: https://github.com/sunxin000/KBQA-R1 (Available)  
**Area**: LLM Reasoning / Reinforcement Learning / Knowledge Base Question Answering  
**Keywords**: KBQA, Multi-turn Reinforcement Learning, GRPO, Referenced Rejection Sampling, Action Space

## TL;DR
Redefines KBQA from "one-shot logical expression generation" into a "multi-turn decision process." It utilizes Referenced Rejection Sampling guided by gold action sequences to generate executable reasoning trajectories for SFT cold-starting, followed by GRPO to optimize the policy based on F1 outcome rewards. This allows an 8B Llama model to simultaneously outperform GPT-4 prompting methods and graph retrieval SOTA on three benchmarks: WebQSP, GrailQA, and GraphQ.

## Background & Motivation

**Background**: Knowledge Base Question Answering (KBQA) requires models to translate natural language questions into executable logical forms (SPARQL / S-Expression) for large-scale graphs like Freebase/Wikidata and return answer sets. Current LLM-based KBQA primarily follows three paths: (i) end-to-end one-shot generation (KB-BINDER / KB-Coder / ChatKBQA); (ii) prompt-driven step-by-step graph exploration (ToG / RoG, relying on commercial APIs like GPT-4); (iii) supervised or search-augmented agentic methods (KBQA-o1 using MCTS + synthetic trajectories).

**Limitations of Prior Work**: The authors summarize the failures of current methods as "dichotomous failures." One class (end-to-end, prompting) tends to **hallucinate schemas**—generating queries that "look executable" but actually reference non-existent or irrelevant relations. The other class (supervised agents) exhibits **templated repetition**—mechanically mimicking actions in synthetic trajectories without truly understanding the KB feedback, while search augmentation introduces massive inference overhead.

**Key Challenge**: Fundamentally, there is a mismatch between "static supervision" and the "dynamic environment." Gold truth (gold S-Expression) only tells the model what the "final look" should be, not "how to decide when seeing a specific neighbor set from the KB at each step." LLMs lack grounded experience with KB executors, leading them to "guess" what to write via text imitation, which naturally results in hallucinations or mechanization.

**Goal**: To enable an 8B-scale open-source LLM to learn "autonomous exploration on KB" without relying on external commercial APIs or large-scale retrieval pipelines, outperforming both LLM prompting methods and graph retrieval SOTA in zero-shot and compositional generalization scenarios.

**Key Insight**: Reformulate KBQA as a **multi-turn sequential decision problem**. The LLM acts as a policy $\pi_\theta$ operating on a compact, validated discrete action space, making decisions at each step based on real KB feedback. Consequently, the ground truth is no longer the "final query" but the "final answer F1," allowing the model to derive "which action to choose in each context" from outcome feedback via reinforcement learning.

**Core Idea**: Replace the **imitation of static logical forms** with **RL on a typed KB action space**, and solve the cold-start challenge for RL training using Referenced Rejection Sampling.

## Method

### Overall Architecture
KBQA-R1 consists of two parts. **At inference time**: A ReAct-style multi-turn agent executes a Think-Action-Information cycle. The LLM first reasons within `<think>`, issues an atomic action (e.g., `Find_relation`, `Merge`, `Order`, `Compare`, `Time_constraint`, `Count`) within `<action>`. The system translates the action into S-Expression fragments, then into SPARQL to execute on Freebase, and writes the retrieved entities or diagnostic information back into `<information>` until the model outputs `<answer>`. A **Relation Retrieval and Confidence Gating (RRCG)** module uses dense retrieval to verify if the relation proposed by the LLM exists in the current entity's neighbor schema. **During training**: High-quality SFT data is synthesized via Referenced Rejection Sampling for cold-starting, followed by policy optimization via GRPO based on a composite "F1 outcome + format" reward. The entire pipeline is trained on Llama-3.1-8B-Instruct.

### Key Designs

1. **Typed Atomic Action Space + RRCG Schema Validation**:

    - **Function**: Decomposes the generation of long, potentially syntactically incorrect S-Expressions into a sequence of individually verifiable atomic operations and aligns the LLM's proposed text relations with actual schema relations in the KB before each `Find_relation` action.
    - **Mechanism**: The action space includes 6 atomic ops, each strictly defining a `(arguments, target functional update, S-Expression template)` triple (e.g., `Find_relation(entity, relation)` maps to `JOIN(relation, START(entity))`). RRCG scores the agent's proposed $r_{\text{agent}}$ against all neighbor relations $R(e_c)$ of the current entity $e_c$ using a dense retriever $Sim(\cdot,\cdot)$. Based on the maximum score $s_{\max}$ and double thresholds $\tau_{\text{high}}, \tau_{\text{low}}$, it processes them in three tiers: $s_{\max} \geq \tau_{\text{high}}$ executes automatically with the nearest neighbor $r_s^*$; $\tau_{\text{low}} \leq s_{\max} < \tau_{\text{high}}$ executes but provides top-k candidates in the observation noting uncertainty; $s_{\max} < \tau_{\text{low}}$ rejects and returns the neighbor list for the model to re-select.
    - **Design Motivation**: A single token misspelling making the entire query unexecutable is the primary fragility of end-to-end methods. Changing "writing a long S-Expression correctly once" to "deciding the next atomic action + validating every relation" transforms hallucination risks into recoverable feedback. This is the foundation for stable RL training—ablations show removing RRCG drops F1 by 18% on average, and removing multi-turn drops it by 25% (36.3% on GrailQA).

2. **Referenced Rejection Sampling (RRS) Cold Start**:

    - **Function**: Provides the policy with an initial checkpoint that already knows how to traverse the KB, bypassing the extremely low acceptance rate of standard rejection sampling in KBQA.
    - **Mechanism**: Training samples are expanded to $(q, \mathcal{A}, S^*)$. The gold S-Expression $S^*$ is first parsed into an atomic action sequence $\mathbf{a}^* = (a_1^*, \ldots, a_k^*)$. During rollout, $a_t^*$ is explicitly injected into the prompt as a "reference action" at step $t$, forcing the model to generate a `<think>` explanation for how this step leads toward the answer and observe real KB feedback. A trajectory is accepted only if $\text{F1}(\hat{\mathcal{A}}, \mathcal{A}) \geq \tau$ and the tag structure is valid. "Reference action prompts" are stripped before SFT to ensure the model does not rely on hidden gold signals during inference.
    - **Design Motivation**: Standard rejection sampling has an acceptance rate of only ~40% in KBQA, often producing "syntactically correct but semantically weak" trajectories. RRS constrains generation to the "skeleton" of gold actions, preventing post-hoc explanations and aligning reasoning with executable steps. Table 7 shows RRS increases acceptance rates from ~40% to 67% on GrailQA/GraphQ, with SFT initialization F1 significantly higher than standard RS (73.8 $\rightarrow$ 80.2).

3. **GRPO + Outcome-Gated Composite Reward**:

    - **Function**: After SFT cold-starting, uses outcome-driven RL to push the policy toward "proactive exploration + adaptive reasoning" rather than just mimicking demonstrations.
    - **Mechanism**: The composite reward is $R = \lambda_{\text{outcome}} \cdot r_{\text{outcome}} + \lambda_{\text{format}} \cdot \mathbb{I}[r_{\text{outcome}} > 0] \cdot r_{\text{format}}$, where $r_{\text{outcome}}$ is the F1 between predicted $\hat{\mathcal{A}}$ and gold $\mathcal{A}$. $r_{\text{format}}$ rewards tag completeness and order. **Crucially, the format reward is only granted if the outcome is non-zero**, preventing the agent from learning "valid format but wrong answer." GRPO samples $n$ rollouts per prompt; the group mean serves as a baseline for the advantage $\hat{A}_i = r_i - \frac{1}{n}\sum_{j=1}^n r_j$, removing the need for a separate value function. The objective is clipped PPO with KL regularization:
    $$\max_\theta \mathbb{E}[\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)\hat{A}_t)] - \beta D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$
    - **Design Motivation**: KBQA ground truth is naturally "whether the answer set is correct," a sparse but reliable signal. Using outcome F1 as the primary reward + KL anchoring to the reference policy allows freedom to explore action combinations while preventing drift into unexecutable areas. Ablations show w/o GRPO drops ~5-10% F1, and w/o SFT warm-start drops 8.6%, proving they are complementary.

### Loss & Training
Two-stage training. **Stage 1 (SFT)**: Each RRS accepted trajectory is sliced into independent training samples by turn. Context acts as input and model response as target; loss is computed only on response tokens. **Stage 2 (GRPO)**: RL is performed on the SFT checkpoint. Llama-3.1-8B-Instruct is the backbone. RRS rollout uses Qwen-2.5-72B-Instruct to generate trajectories for distillation into the 8B model.

## Key Experimental Results

### Main Results

| Dataset | Metric | KBQA-R1 (Llama-3.1-8B) | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| GrailQA Overall | F1 | **86.1** | 78.5 (KBQA-o1, Llama-3.1-8B) / 81.9 (TIARA, T5-large) | +7.6 |
| GrailQA Zero-shot | EM / F1 | **83.6 / 85.2** | 68.1 / 76.1 (KBQA-o1) | +15.5 / +9.1 |
| WebQSP | F1 | **83.4** | 78.2 (SubgraphRAG + GPT-4o) / 76.0 (MCTS-KBQA) | +5.2 / +7.4 |
| GraphQ | F1 | **53.8** | 48.7 (KBQA-o1) / 47.5 (CoTKR) | +5.1 |

Of particular note is the zero-shot dimension—EM improves by 15.5% where relations/combinations were unseen in training, indicating that RL learns policy-level generalization rather than just distribution fitting.

### Ablation Study

| Configuration | WebQSP F1 | GraphQ F1 | GrailQA F1 | Description |
|------|-----------|-----------|------------|------|
| Full KBQA-R1 | 83.4 | 53.8 | 86.1 | Full Model |
| w/o RRCG | 64.1 | 37.7 | 67.1 | No schema validation, avg. −18% |
| w/o Multi-turn | 63.2 | 34.1 | 49.8 | One-shot generation, avg. −25% (GrailQA −36) |
| w/o RRS (Std RS) | 78.9 | 49.2 | 78.3 | Lower cold-start quality, avg. −5.6% |
| w/o SFT warm-start | 75.2 | 47.3 | 75.1 | Direct RL, avg. −8.6% |
| w/o GRPO (SFT only) | 72.1 | 47.8 | 80.2 | No RL optimization stage |
| w/o Format Reward | 81.1 | 51.6 | 84.2 | Format reward as a stabilizer; minor impact |

### Key Findings
- **Multi-turn and RRCG are the Foundation**: Removing either drops F1 by 18-25%, showing that the "executable environment" provided by structured actions + schema validation is the prerequisite for RL to learn.
- **RRS vs. Standard RS Data Efficiency**: On GrailQA, the acceptance rate increases from 39.3% to 67.0%, and SFT initialization F1 increases from 73.8 to 80.2. Cold-start quality directly determines the final upper bound.
- **Efficiency Gains**: Compared to GPT-4 prompting (ToG / PoG), KBQA-R1 (Llama-3.1-8B) reduces LLM calls by over 70% while maintaining higher accuracy.
- **Frontier Agent Comparison**: Even when the same KBQA-R1 harness (action space + feedback) is provided to frontier models like GLM-4 / Kimi-K2.5, accuracy remains lower than the trained 8B policy, using more turns/tokens.

## Highlights & Insights
- **The RRS paradigm (skeleton gold actions + rationalized explanation + stripping)** is clever. It converts the sparse success probability of KBQA into a dense supervision problem. This is applicable to any task where the final answer is verifiable but the reasoning process is hard to supervise.
- **Outcome-gated format reward**: The $\mathbb{I}[r_{\text{outcome}} > 0]$ term prevents the agent from learning "pretty but wrong" behaviors, a crucial detail often missed in LLM agent RL.
- **Schema validation as "soft-hard" layering** (auto-validate / tentative / reject) is an elegant solution for the tension between LLM hallucinations and KG ground truth. It feeds uncertainty back to the model for self-correction instead of just rejecting.
- **The "Aha!" moment**: In KBQA, a task thought to be dominated by GPT-4 + RAG, the authors prove an 8B model can outperform GPT-4o + retrieval using the right training paradigm.

## Limitations & Future Work
- **Dependency on linked topic entities**: Assumes entities are pre-linked to the KB, excluding entity linking errors from evaluation—EL is often a bottleneck in real deployment.
- **Validated only on Freebase**: It is unclear if the action space and RRCG scale to Wikidata, which has significantly larger and more complex schemas.
- **Freebase-specific Action Design**: The 6 actions target S-Expression operations. Migration to Cypher, SQL, or higher-order reasoning requires redesign.
- **RRS Requires Gold S-Expressions**: Cold-start synthesis relies on gold logical forms. For datasets with only weak supervision (answers only), extra steps are needed.
- **Future Directions**: Modularizing action spaces for KB-agnostic interfaces, combining with self-play for weak supervision scenarios, and extending RRS for process reward training.

## Related Work & Insights
- **vs KBQA-o1 (Luo et al., 2025c)**: Both use 8B Llama and atomic actions, but KBQA-o1 uses MCTS + incremental finetuning. KBQA-R1 internalizes search into policy weights via GRPO, achieving higher accuracy without test-time search (WebQSP F1 83.4 vs 57.8).
- **vs ToG / RoG / PoG**: These rely on GPT-4 multi-turn graph exploration at test time. KBQA-R1 trains an 8B model to do the same with 70%+ fewer calls.
- **vs SubgraphRAG / GNN-RAG**: These rely on offline subgraphs to reduce hallucination, but retrieval isn't end-to-end optimized. KBQA-R1 learns "how to explore," providing an advantage in compositional and zero-shot scenarios.
- **vs Standard Rejection Sampling**: Constraining RS to gold action skeletons is a fundamental improvement for structural tasks, more effective than just tuning temperature or sampling budget.

## Rating
- Novelty: ⭐⭐⭐⭐ — RRS's "reference action skeleton" is a genuine innovation. RL on action space has precedents like KBQA-o1, but this is the first to succeed with pure RL over SFT+search.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, seven baselines, seven ablations, frontier agent harness comparisons, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ — Narrative follows "dichotomous failures" clearly. Method is cross-described with Algorithms, tables, and formulas.
- Value: ⭐⭐⭐⭐⭐ — Provides hard evidence that "Small Model + RL Internalized Reasoning > Big Model + Test-time Search." The RRS technique is a major contribution to structured tasks like tool-calling and code generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)
- [\[AAAI 2026\] PathMind: A Retrieve-Prioritize-Reason Framework for Knowledge Graph Reasoning with Large Language Models](../../AAAI2026/graph_learning/pathmind_a_retrieve-prioritize-reason_framework_for_knowledge_graph_reasoning_wi.md)
- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](../../NeurIPS2025/graph_learning/deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)
- [\[ICML 2026\] Beyond Model Base Retrieval: Weaving Knowledge to Master Fine-grained Neural Network Design](beyond_model_base_retrieval_weaving_knowledge_to_master_fine-grained_neural_netw.md)

</div>

<!-- RELATED:END -->
