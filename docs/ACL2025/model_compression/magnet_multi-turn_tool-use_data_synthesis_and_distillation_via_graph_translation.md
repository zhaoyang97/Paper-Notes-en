---
title: >-
  [Paper Note] Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation
description: >-
  [ACL 2025][Model Compression][Function Calling] The Magnet framework is proposed to construct high-quality multi-turn function calling (FC) training trajectories based on random walks and node operations (Insert/Merge/Split) on a function dependency graph. Combined with prompt-based context distillation to generate positive/negative contrastive trajectories for SFT + mDPO training, the 14B model Magnet-14B-mDPO achieves a score of 68.01 on BFCL-v3 (ranking 4th)…
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Function Calling"
  - "Multi-turn Dialogue"
  - "Graph Translation"
  - "Data Synthesis"
  - "Context Distillation"
  - "DPO"
  - "Preference Optimization"
date: 2026-05-08
content_hash: 0cedcb2c81350083
---

# Magnet: Multi-turn Tool-use Data Synthesis and Distillation via Graph Translation

**Conference**: ACL 2025  
**arXiv**: [2503.07826](https://arxiv.org/abs/2503.07826)  
**Code**: Not publicly released  
**Authors**: Fan Yin, Zifeng Wang, I-Hung Hsu, Jun Yan, Ke Jiang, Yanfei Chen, Jindong Gu, Long T. Le, Kai-Wei Chang, Chen-Yu Lee, Hamid Palangi, Tomas Pfister  
**Affiliations**: Google, UCLA  
**Area**: Model Distillation / Tool-use Agents  
**Keywords**: Function Calling, Multi-turn Dialogue, Graph Translation, Data Synthesis, Context Distillation, DPO, Preference Optimization  

## TL;DR

The Magnet framework is proposed to construct high-quality multi-turn function calling (FC) training trajectories based on random walks and node operations (Insert/Merge/Split) on a function dependency graph. Combined with prompt-based context distillation to generate positive/negative contrastive trajectories for SFT + mDPO training, the 14B model Magnet-14B-mDPO achieves a score of 68.01 on BFCL-v3 (ranking 4th), significantly outperforming the teacher model Gemini-1.5-pro-002 in multi-turn scenarios.

## Background & Motivation

**Background**: LLM Agents need to invoke external tools (APIs/functions) to complete complex tasks. Current models perform well in single-step FC, but still face challenges in **multi-turn, multi-step** interactions.

**Three key challenges of multi-turn FC**:
   - **Nested FCs**: Certain turns require multiple or even nested function calls, which are not explicitly mentioned in the query.
   - **Long Dependency**: Certain turns require utilizing information from distant parts of the conversation history to assemble the FC.
   - **Irrelevance**: Certain turns have missing functionalities or parameters, requiring the model to ask clarifying questions.

**Data Bottleneck**: The multi-turn success rate of the best open-source models on BFCL-v3 is only ~10%. The low performance of existing models in multi-turn scenarios makes it extremely difficult to collect high-quality training trajectories.

**Core Motivation**: Design a principled data synthesis pipeline to construct reliable multi-turn FC training data from a graph perspective.

## Method

### Overall Architecture

Four-stage pipeline: **Function Pool & Dependency Graph Construction** $\rightarrow$ **Node Operation Augmentation** $\rightarrow$ **Back-and-Forth Translation for Query-FC Pairs** $\rightarrow$ **Context Distillation for Trajectory Generation**

### Phase 1: Function Dependency Graph Construction

1. **Function Collection**: 5,011 executable APIs across 49 categories were collected from StableToolBench and BFCL-v3.
2. **Local Dependency Graph**: Functions are treated as graph nodes. For each node, 30 neighbor candidates from the same category are sampled, and LLMs are used to determine if an input-output dependency exists. If the output of a source node is related to the input of a target node, a directed edge is established.
3. **Initial FSP Sampling**: Start from each node and perform random walks of $S=7$ steps along the dependency edges to sample an initial Function Signature Path $\tilde{\phi} = (\tilde{f}_1, \tilde{f}_2, \cdots, \tilde{f}_H)$.

### Phase 2: Node Operation Augmentation

Three graph-level operations are designed to address the three major challenges of multi-turn FC:

**Operation 1: Insert — Resolving Nested FCs and Long Dependencies**
- Traverse the last function signature $\tilde{f}_{hk}$ of each turn in the FSP, and use an LLM to check if neighbor functions satisfy the conditions for nested calls.
- If such functions exist, append the nested function to the current turn or insert it at a subsequent random position (simulating long dependency).
- Example: Query "How many **kilometers** from San Francisco to San Mateo?" $\rightarrow$ Requires calling `get_distance()` (returns miles) + `convert_unit()` (implicit nesting).

**Operation 2: Merge — Creating Multi-FC within a Single Turn**
- Merge two adjacent turns into a single turn with probability $p=0.3$. The model must understand the output of previous functions to assemble subsequent functions.
- Difference from Insert: Functions merged by Merge are related, but not necessarily nested.

**Operation 3: Split — Simulating Missing Information**
- Randomly select a turn and insert an empty node `{}` after it, marked as 'miss params' or 'miss func'.
- The model must identify the missing information and ask clarifying questions.

**Execution Order**: First Merge $\rightarrow$ then Insert $\rightarrow$ generate augmented FSP $\phi$; additionally, apply Split to $\phi$ to generate FSP $\hat{\phi}$ with missing information.

### Phase 3: Back-and-Forth Translation

Iteratively translate the augmented FSP into query-FC pairs:
- **Back translation** $\mathcal{M}_b(f_h) = q_h$: Translates function signatures into simulated user queries.
- **Forward translation** $\mathcal{M}_f(q_h, f_h, t_{h-1}) = fc_h$: Translates queries into executable FCs (leveraging the previous turn's output $t_{h-1}$).
- Iterate turn-by-turn to ensure that the outputs of previous turns are ready before passing them to subsequent turns.

### Phase 4: Context Distillation for Trajectory Generation

**Positive Trajectory Generation**:
- Inspired by Context Distillation, when the teacher model (Gemini-1.5-pro-002) generates trajectories, FC references are appended to the query as a `[Hint]`.
- Ensure the teacher model generates actions as accurately as possible.

**Negative Trajectory Generation**:
- Collect 10 inference trajectories from the SFT model on each data point.
- Use LLMs to judge whether each turn contains erroneous FCs.
- Provide the erroneous FCs as misleading `[Hint]` inputs to the SFT model to regenerate negative trajectories.
- Form positive/negative trajectory pairs for mDPO training.

### Loss & Training

$$\mathcal{L}(x; \tau_w, \tau_l) = \mathcal{L}_{\text{SFT}}(x; \tau_w) + \lambda \cdot \mathcal{L}_{\text{mDPO}}(x; \tau_w, \tau_l)$$

mDPO (multi-turn DPO) compares the action of each turn individually against the reference policy, rather than comparing the entire sequence as a whole.

### Data Statistics

| Category | SFT Count | mDPO Count |
|------|----------|-----------|
| Single-turn | 20,000 | 1,556 |
| Multi-turn | 7,800 | 2,250 |
| Irrelevant | 6,200 | 750 |
| Avg. Turns (Multi-turn) | 4.71 | 5.22 |
| Avg. FCs (Multi-turn) | 15.13 | 14.98 |

The total training set consists of 38,556 samples, which is only about half of APIGen (60K) and Hammer (67.5K).

## Experiments

### Main Results

| Model | Overall | Single-turn | Multi-turn | Irrelevance |
|------|---------|------|------|--------|
| watt-tool-70B | 74.31 | — | 58.75 | 76.32 |
| GPT-4o (Prompt) | 72.08 | — | 47.62 | 83.76 |
| GPT-4o (FC) | 69.58 | — | 41.00 | 83.15 |
| o1 | 66.73 | — | 28.25 | 89.62 |
| Gemini-1.5-pro-002 | 62.19 | — | 20.75 | 78.15 |
| **Magnet-14B-mDPO** | **68.01** | — | **37.88** | 84.78 |
| Magnet-14B-SFT | 66.83 | — | 33.38 | 82.59 |
| Magnet-7B-mDPO | 64.64 | — | 27.75 | 78.51 |
| Qwen2.5-Coder-14B (base) | 51.88 | — | 5.38 | 44.58 |
| Qwen2.5-Coder-7B (base) | 53.13 | — | 8.25 | 65.39 |

**Key Findings**:
- Magnet-14B-mDPO ranks **4th** on BFCL-v3, outperforming o1 and the teacher model Gemini-1.5-pro-002.
- Multi-turn scenarios: 14B improves from the base model's 5.38 $\rightarrow$ **37.88** (+32.5), and 7B improves from 8.25 $\rightarrow$ 27.75 (+19.5).
- All Magnet models outperform the teacher model Gemini-1.5-pro-002 (20.75) in multi-turn scenarios.
- Compared with pure SFT, mDPO improves performance in multi-turn scenarios by about 2.5 to 4.5 percentage points.

### ToolQuery Results

| Model | Success Rate | Progress Rate |
|------|--------|--------|
| Magnet-14B-mDPO | **73.3** | **78.7** |
| Gemini-1.5-pro-002 | 68.3 | 74.6 |
| GPT-4o | 63.3 | 80.1 |
| Magnet-7B-mDPO | 67.7 | 73.4 |
| Qwen-Coder-14B | 51.7 | 68.7 |

The 14B model also achieves the highest success rate on ToolQuery.

### Ablation Study

| Component | Overall | Multi-turn |
|------|---------|------|
| Initial Graph (No Node Operations) | 58.54 | 12.75 |
| +Merge | 60.83 | 20.63 |
| +Merge+Insert | 64.39 | 29.25 |
| +All Operations (SFT) | **66.83** | **33.38** |
| -Positive Trajectory Context Distillation | 60.26 | 18.88 |
| SFT+mDPO | **68.01** | **37.88** |
| -Negative Trajectory Context Distillation | 67.35 | 36.25 |

**Key Findings**:
1. Each node operation brings significant improvement: Merge (+7.9 on multi-turn) $\rightarrow$ Insert (+8.6) $\rightarrow$ Split (+4.1).
2. Positive trajectory context distillation contributes enormously: removing it drops the multi-turn performance from 33.38 to 18.88 (-14.5).
3. Negative trajectory distillation contributes to mDPO, but to a lesser extent (-1.63 on multi-turn).
4. Data synthesis is generalizable to different base models and self-training scenarios.

### Data Source Comparison

- The 7B model trained on APIGen+ToolAce struggles with a multi-turn score of only 7.13, which is far lower than Magnet-7B's 26.50.
- Including irrelevant category data improves the overall score (57.24) but slightly decreases multi-turn performance.

## Highlights & Insights

1. **Novel Graph-based Perspective**: Abstracting multi-turn FC challenges as graph node operations (Insert/Merge/Split) provides an elegant, structured approach.
2. **Achieving More with Less**: Achieving the 4th rank with only 38K training data points (about half of APIGen/Hammer).
3. **Student Outperforming the Teacher**: All Magnet models outperform the teacher Gemini-1.5-pro-002 in multi-turn scenarios, indicating that the synthesis pipeline introduces beneficial extra supervision.
4. **Crucial Contribution of Positive Prompt Distillation**: Ablation reveals that removing positive trajectory hints almost halves multi-turn performance.
5. **Multi-turn FC Remains an Active Challenge**: Even Magnet-14B-mDPO only reaches a 37.88% success rate in multi-turn scenarios, and the strongest model (watt-tool-70B) only achieves 58.75%.

## Limitations & Future Work

1. Reliance on the quality of the teacher model (Gemini-1.5-pro-002) — although the student outperforms the teacher, the initial data quality remains constrained by the teacher.
2. The function pool is mainly derived from StableToolBench and BFCL-v3; the API variety and complexity might not fully represent real-world scenarios.
3. Random walk sampling might miss certain complex function combination patterns.
4. Negative trajectory generation depends on mistakes made by the SFT model itself — if the SFT model is too weak or too strong, the quality of negative samples may suffer.
5. Experiments are only validated on the Qwen2.5-Coder series.

## Related Work & Insights

- **FC Evaluation Benchmarks**: BFCL-v3 (Comprehensive), ToolBench / StableToolBench (Multi-step), ToolQuery (Multi-turn multi-step)
- **FC Training Data Synthesis**: Toolformer (Inserting APIs into texts) $\rightarrow$ APIGen/xLAM (Unified format + automated query generation) $\rightarrow$ Hammer (Function masking + irrelevant functions) $\rightarrow$ Magnet (Graph structure + node operations + contrastive distillation)
- **Preference Optimization**: DPO $\rightarrow$ mDPO (Multi-turn extension)

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — The combination of a graph perspective, node operations, and contrastive distillation is highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covered two benchmarks, multiple model scales, detailed ablations, and data source comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear pipeline explanation with consistent notation.
- **Value**: ⭐⭐⭐⭐⭐ — A directly deployable data synthesis pipeline that outperforms proprietary models.
- **Limitations**: The absolute success rate of multi-turn FC is still not high enough, leaving a gap for practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Offline Policy Evaluation of Multi-Turn LLM Health Coaching with Real Users](../../NeurIPS2025/model_compression/offline_policy_evaluation_of_multi-turn_llm_health_coaching_with_real_users.md)
- [\[ACL 2026\] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation](../../ACL2026/model_compression/find_your_optimal_teacher_personalized_data_synthesis_via_router-guided_multi-te.md)
- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](../../NeurIPS2025/model_compression/atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[ACL 2025\] AlignDistil: Token-Level Language Model Alignment as Adaptive Policy Distillation](aligndistil_token_level_alignment.md)

</div>

<!-- RELATED:END -->
