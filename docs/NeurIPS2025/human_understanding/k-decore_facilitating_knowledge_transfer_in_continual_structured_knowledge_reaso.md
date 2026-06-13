---
title: >-
  [Paper Note] K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning
description: >-
  [NeurIPS 2025][Human Understanding][Continual Learning] This paper proposes K-DeCore, a framework that decouples structured knowledge reasoning into two stages — task-agnostic schema filtering and task-specific query con…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Continual Learning"
  - "Structured Knowledge Reasoning"
  - "Knowledge Decoupling"
  - "Memory Replay"
  - "Text-to-SQL"
date: 2026-05-08
content_hash: c3cd4185557eb731
---

# K-DeCore: Facilitating Knowledge Transfer in Continual Structured Knowledge Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.16929](https://arxiv.org/abs/2509.16929)  
**Code**: None  
**Area**: Human Understanding
**Keywords**: Continual Learning, Structured Knowledge Reasoning, Knowledge Decoupling, Memory Replay, Text-to-SQL

## TL;DR

This paper proposes K-DeCore, a framework that decouples structured knowledge reasoning into two stages — task-agnostic schema filtering and task-specific query construction — and combines dual-perspective memory construction with structure-guided pseudo-data synthesis to enable effective knowledge transfer across heterogeneous SKR tasks under a fixed parameter budget.

## Background & Motivation

Structured Knowledge Reasoning (SKR) involves translating natural language questions into structured queries (SQL, SPARQL, TOP, etc.) over diverse structured knowledge sources, including databases, knowledge graphs, and dialogue states. Key limitations of existing methods:

**Unrealistic static assumptions**: Existing methods assume a single, fixed SKR task, whereas real-world systems (e.g., Siri/Alexa) must continually adapt to new reasoning tasks.

**Poor heterogeneous generalization**: Continual learning methods focused on a single type (e.g., text-to-SQL only) fail to generalize across heterogeneous structured knowledge.

**Parameter growth problem**: PEFT-based methods allocate independent parameters per task, causing parameter count to grow linearly with the number of tasks.

The authors' core observation is that **schema filtering** — selecting schema elements relevant to a query from the full schema — is a shared and reusable component across SKR tasks, with relatively stable input/output formats, making it a suitable bridge for knowledge transfer.

## Method

### Overall Architecture

K-DeCore consists of a frozen backbone LLM and three lightweight PEFT modules (LoRA):

- **$\mathbf{P}_a$**: Schema filter (task-agnostic, shared across tasks)
- **$\mathbf{P}_b$**: Query constructor (task-specific, fixed parameter count)
- **$\mathbf{P}_c$**: Pseudo-question generator (for synthesizing training data)

### Key Designs

#### 1. Knowledge Decoupling: Schema Filtering + Query Construction

**Schema Filtering** (task-agnostic):

All heterogeneous schemas are unified into a DB-like format. For example:

- Database: table → $\phi$, column → $\psi$
- Knowledge graph: entity type → $\phi$, relation → $\psi$
- Dialogue state: intent → $\phi$, slot → $\psi$

The unified text representation is: $\widetilde{\Omega} = \phi_1: \psi_1^1, \ldots | \phi_2: \psi_2^1, \ldots$

$\mathbf{P}_a$ is trained to predict the relevant schema subset $\Omega^*$ and is initialized from the previous task's checkpoint at each new task.

**Query Construction** (task-specific):

$\mathbf{P}_b$ generates the final query $\mathcal{Y}$ from the original schema $\mathcal{S}$ and the filtered $\mathcal{S}^*$, while preserving the original format to produce executable queries.

Advantages of the two-stage design:
- Schema filtering shares a unified format across tasks → facilitates forward/backward knowledge transfer
- Query construction retains task specificity → preserves query generation accuracy

#### 2. Dual-Perspective Memory Construction

**Schema-guided memory $\mathcal{M}_a^k$**:

Samples are clustered by schema similarity, and the sample closest to each cluster centroid is selected. The distance is defined as: $d_1(\mathcal{X}_1, \mathcal{X}_2) = \cos(g(\mathcal{S}_1^*), g(\mathcal{S}_2^*))$

This ensures memory coverage over diverse schema patterns.

**Structure-guided memory $\mathcal{M}_b^k = \mathcal{M}_{\text{real}}^k \cup \mathcal{M}_{\text{pseudo}}^k$**:

- $\mathcal{M}_{\text{real}}^k$: Representative samples selected by clustering on query structural similarity
- $\mathcal{M}_{\text{pseudo}}^k$: Pseudo-data synthesis introduces novel query structures absent from the training set

#### 3. Structure-Guided Pseudo-Data Synthesis (Algorithm 1)

Core pipeline:
1. Randomly sample $T$ query structures from the training set
2. Use the backbone LLM to synthesize new query structures (e.g., complex SQL or multi-hop S-expressions)
3. Randomly select schemas to fill placeholders and generate concrete queries
4. **Execution validation**: retain only synthesized queries that execute successfully
5. Use $\mathbf{P}_c$ to generate corresponding natural language questions

### Loss & Training

**Schema filtering loss**:

$$\mathcal{L}(\mathcal{D}^k; \theta, \mathbf{P}_a) = -\sum_i \sum_j \log P(\omega_j^* | \mathcal{Q}_i, \widetilde{\Omega}_i, \omega_{<j}^*) + \sum_{k'=1}^{k-1} \mathcal{L}(\mathcal{M}_a^{k'})$$

**Query construction loss**:

$$\mathcal{L}(\mathcal{D}^k; \theta, \mathbf{P}_b) = -\sum_i \sum_j \log P(y_j | \mathcal{Q}_i, \mathcal{S}_i^*, \mathcal{S}_i, y_{<j}) + \sum_{k'=1}^{k-1} \mathcal{L}(\mathcal{M}_b^{k'})$$

Both losses combine the current task with historical memory replay. Memory size: $|\mathcal{M}_a^k| = |\mathcal{M}_b^k| = 5$; real-to-pseudo ratio = 4:1.

Training hyperparameters: batch size 12, lr $5 \times 10^{-5}$, 5 epochs, single RTX 4090.

## Key Experimental Results

### Main Results (3 SKR task streams × 3 backbone models)

| Backbone | Method | Stream1 AA | Stream1 BWT | Stream1 FWT |
|----------|--------|-----------|-------------|-------------|
| T5-Large | Fine-Tuning | 2.9 | -31.1 | 6.3 |
| T5-Large | SFNet | 27.3 | -14.7 | 3.6 |
| T5-Large | C3 | 26.6 | — | -1.9 |
| T5-Large | **K-DeCore** | **31.8** | **-9.6** | **8.6** |
| Llama3-8B | C3 | 39.7 | — | 2.3 |
| Llama3-8B | **K-DeCore** | **40.5** | -16.7 | **5.9** |
| QWEN2.5-7B | C3 | 38.9 | — | 1.9 |
| QWEN2.5-7B | **K-DeCore** | **43.2** | **-8.2** | **6.9** |

### Ablation Study (Llama3-8B, Stream1)

| Variant | AA | BWT | FWT |
|---------|----|-----|-----|
| **K-DeCore** | **40.5** | -16.7 | **5.9** |
| w/o Decoupling | 38.8 | **-13.8** | 2.6 |
| w/o Unification | 37.8 | -17.7 | 3.4 |
| w/o Replay | 20.5 | -41.2 | 4.3 |
| w/o $\mathcal{M}_b^k$ | 20.8 | -42.1 | 5.3 |
| w/o $\mathcal{M}_a^k$ | 39.4 | -17.1 | 5.1 |
| w Random Memory | 39.9 | -17.4 | 5.6 |

### Key Findings

1. **Memory replay is essential**: removing replay (w/o Replay) causes AA to drop sharply from 40.5 to 20.5, with BWT falling to -41.2.
2. **Query memory is far more important than schema memory**: removing $\mathcal{M}_b^k$ (AA=20.8) causes substantially greater degradation than removing $\mathcal{M}_a^k$ (AA=39.4).
3. **Both knowledge decoupling and unified format contribute**: removing decoupling (AA−1.7) or unified representation (AA−2.7) both lead to performance drops.
4. **Optimal pseudo-data ratio is 20%**: excessive pseudo-data degrades performance.
5. **Training efficiency**: K-DeCore's training time is only marginally higher than EMAR and far lower than C3 and SAPT.

## Highlights & Insights

1. **First heterogeneous SKR continual learning framework**: unlike methods restricted to text-to-SQL, K-DeCore spans databases, knowledge graphs, and dialogue states.
2. **Elegant schema unification**: all structured knowledge is mapped to a DB-like format via minimal mapping rules, bridging heterogeneity concisely.
3. **Execution-validated pseudo-data synthesis**: synthesized queries are filtered by actual execution to ensure semantic validity.
4. **Fixed parameter count**: no parameter growth with the number of tasks, outperforming methods such as C3 and SAPT in this regard.

## Limitations & Future Work

1. The low-resource setting (1,000 training and 300 test samples per task) has not been validated at larger scales.
2. Memory size is fixed at 5 samples, which may be insufficient for more complex task streams.
3. Reasoning-oriented LLMs (e.g., QWQ-32B) are not used as backbones, potentially missing stronger reasoning capabilities.
4. Unifying schemas into DB-like format may discard semantic information unique to certain structured knowledge types.
5. Pseudo-data synthesis relies on the LLM's understanding of query structures, which may generalize poorly to novel query languages.

## Related Work & Insights

- **C3**: trains independent PEFT modules per task; competitive AA but parameters grow with tasks and FWT is poor.
- **SAPT**: uses soft prompts for continual learning but incurs high training overhead.
- **SFNet**: a replay-based method that does not decouple the reasoning stages.
- **Insight**: the knowledge decoupling principle is generalizable to other continual learning scenarios involving multi-stage reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of knowledge decoupling, dual-perspective memory, and structure-guided synthesis is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets × 3 backbones × 3 task streams with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear and method motivation is well-grounded.
- Value: ⭐⭐⭐⭐ — Fills a clear gap in heterogeneous SKR continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] One-Shot Knowledge Transfer for Scalable Person Re-Identification](../../ICCV2025/human_understanding/one-shot_knowledge_transfer_for_scalable_person_re-identification.md)
- [\[NeurIPS 2025\] Foundation Cures Personalization: Improving Personalized Models' Prompt Consistency via Hidden Foundation Knowledge](foundation_cures_personalization_improving_personalized_models_prompt_consistenc.md)
- [\[ICCV 2025\] CleanPose: Category-Level Object Pose Estimation via Causal Learning and Knowledge Distillation](../../ICCV2025/human_understanding/cleanpose_category-level_object_pose_estimation_via_causal_learning_and_knowledg.md)
- [\[NeurIPS 2025\] Part-Aware Bottom-Up Group Reasoning for Fine-Grained Social Interaction Detection](part-aware_bottom-up_group_reasoning_for_fine-grained_social_interaction_detecti.md)
- [\[NeurIPS 2025\] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild](omnigaze_reward-inspired_generalizable_gaze_estimation_in_the_wild.md)

</div>

<!-- RELATED:END -->
