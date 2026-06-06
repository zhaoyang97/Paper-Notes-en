---
title: >-
  [Paper Note] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning
description: >-
  [ACL 2026][Recommender Systems][Recommendation Assistant] This paper proposes ReRec, a framework for recommendation assistants based on Reinforcement Fine-tuning (RFT). It provides fine-grained reward signals through dua…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Recommendation Assistant"
  - "Reinforcement Fine-tuning"
  - "Reasoning Augmentation"
  - "Reward Shaping"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: bad35f80c34b20c0
---

# ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning

**Conference**: ACL 2026  
**arXiv**: [2604.07851](https://arxiv.org/abs/2604.07851)  
**Code**: [GitHub](https://github.com/jiani-huang/ReRec)  
**Area**: Reinforcement Learning  
**Keywords**: Recommendation Assistant, Reinforcement Fine-tuning, Reasoning Augmentation, Reward Shaping, Curriculum Learning

## TL;DR

This paper proposes ReRec, a framework for recommendation assistants based on Reinforcement Fine-tuning (RFT). It provides fine-grained reward signals through dual-graph enhanced reward shaping, applies differentiated supervision to reasoning steps via Reasoning-Aware Advantage Estimation (RAAE), and dynamically adjusts training difficulty using an online curriculum scheduler. ReRec enables LLMs to handle complex multi-step reasoning queries, significantly outperforming existing methods on the RecBench+ benchmark.

## Background & Motivation

**Background**: Traditional recommendation systems (Matrix Factorization, GNNs, etc.) rely on historical interaction data and cannot process natural language queries. The emergence of LLMs has introduced new possibilities for intelligent recommendation assistants. Recent research explored LLM-based Conversational Recommendation Systems (CRS), but most only handle simple, direct queries such as "recommend a sci-fi movie."

**Limitations of Prior Work**: (1) Real-world user queries are often complex and require multi-step reasoning. For example, "Recommend other movies starring the lead actor from that film about a man trapped on an island" requires first inferring *Cast Away*, then identifying Tom Hanks, and finally recommending his other films. Existing LLM recommendation systems lack this deep reasoning capability. (2) Applying RFT directly to recommendation tasks faces two major challenges: reward signals are too coarse (NDCG only considers exact matches; reasonable but non-exact recommendations receive zero rewards), and the reasoning process lacks supervision (all tokens share the same reward score, making it impossible to distinguish between correct and incorrect reasoning steps).

**Key Challenge**: Exact match metrics like NDCG are too harsh and sparse as reward signals—recommendations that satisfy query constraints but do not match the ground truth receive the same zero reward as completely irrelevant ones. This leads to low exploration efficiency for the policy model.

**Goal**: Design an RFT framework tailored for recommendation tasks to address the issues of insufficient reward granularity and unsupervised reasoning processes.

**Key Insight**: Improve RFT for recommendation from three dimensions: (1) Enrich the reward space using item attribute graphs and collaborative filtering signals; (2) Segment the reasoning process and impose penalties on erroneous steps; (3) Adjust the training curriculum based on the model's dynamic capabilities.

**Core Idea**: Expand the recommendation reward from coarse exact matches to fine-grained signals merging query alignment and preference alignment. Simultaneously provide differentiated supervision at the reasoning step level to ensure the LLM learns to reason rather than memorize.

## Method

### Overall Architecture

ReRec is built on the GRPO reinforcement learning framework. Given a user query $q$, the LLM policy model generates multiple candidate responses $\{o_1, ..., o_G\}$, each containing a reasoning process and recommended items. Three core modules handle reward calculation, advantage estimation, and training scheduling: Dual-Graph Enhanced Reward Shaping computes the reward $r_i$ for each response; Reasoning-Aware Advantage Estimation allocates rewards to token-level advantages $A_{i,t}$ at the reasoning step level; and an Online Curriculum Scheduler dynamically adjusts the training data order and difficulty for each epoch.

### Key Designs

1. **Dual-Graph Enhanced Reward Shaping**:
    - **Function**: Provides fine-grained rewards merging exact match, query constraint satisfaction, and user preference alignment.
    - **Mechanism**: Introduces two auxiliary scores alongside NDCG@K. **Query Alignment Score (QAS)** uses an item-attribute graph $G_{attr}$ to calculate the overlap ratio of attributes between the recommended item and the ground truth: $S_{QAS}(p_i, gt) = |R_{p_i}^{G_{attr}} \cap R_{gt}^{G_{attr}}| / |R_{gt}^{G_{attr}}|$. **Preference Alignment Score (PAS)** uses a pre-trained lightweight recommendation model (e.g., LightGCN) to generate item embeddings from the user-item interaction graph and calculates cosine similarity: $S_{PAS}(p_i, gt) = \mathcal{M}(p_i) \cdot \mathcal{M}(gt) / (\|\mathcal{M}(p_i)\| \|\mathcal{M}(gt)\|)$. The final reward is $r_i = \text{NDCG} + w_1 S_{QAS} + w_2 S_{PAS}$.
    - **Design Motivation**: QAS rewards recommendations that satisfy query constraints (e.g., genre, actors) but do not match the ground truth, avoiding zero penalties for reasonable suggestions. PAS captures implicit preferences from collaborative filtering signals, preventing over-reliance on attribute matching while ignoring personalized preferences.

2. **Reasoning-Aware Advantage Estimation (RAAE)**:
    - **Function**: Provides step-level differentiated supervision for the reasoning process and penalizes erroneous reasoning steps.
    - **Mechanism**: Decomposes the LLM output $o_i$ into $K$ reasoning segments $\mathcal{S}_i = \{s_{i,1}, ..., s_{i,K}\}$. If a reasoning segment discusses an item that is ultimately incorrectly recommended (i.e., $p_i \neq gt$ and $p_i \in s_{i,k}$), it indicates the model failed to exclude that item. The reward for that segment is penalized as $r_{s_{i,k}} = (1 - w_{penalty}) \cdot r_i$. Other segments retain the original reward. These segment rewards are then mapped to token-level normalized advantages $A_{i,t} = (r_{i,t} - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$.
    - **Design Motivation**: Traditional RFT assigns the same reward to all tokens, which fails to help the model identify which reasoning steps are incorrect. RAAE achieves step-level supervision through lightweight segment decomposition and conditional penalties without the high cost of training a dedicated process reward model.

3. **Online Curriculum Scheduler**:
    - **Function**: Dynamically evaluates query difficulty and organizes training from easy to hard to ensure stable convergence.
    - **Mechanism**: A three-step process—**Adaptive Difficulty Assessment**: At the start of epoch $t$, difficulty is assessed based on the average inverse reward of each query from the previous round $d^{t-1} = \frac{1}{G}\sum_{i=1}^G (1 - r_i)$; **Sample Filtering & Sorting**: "Mastered" samples with difficulty below threshold $\tau$ are filtered out, and the remainder are sorted by difficulty to form a new dataset $\mathcal{D}^t$; **Iterative Update**: This process repeats each epoch to adapt to the model's improving capabilities.
    - **Design Motivation**: Difficulty in recommendation tasks is hard to pre-define (unlike math/code with clear hierarchies), and static curricula cannot reflect the model's dynamic progress. The online scheduler updates difficulty estimates using existing rollout data at zero cost.

### Loss & Training

The objective follows the GRPO loss with a clipped probability ratio:
$$\mathcal{J}(\theta) = \mathbb{E}[\frac{1}{N}\sum_{i=1}^G \sum_{t=1}^{|o_i|} \min(h_{i,t}(\theta) A_{i,t}, \text{clip}(h_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_{i,t})]$$
The backbone models are Qwen-2.5-3B-Instruct and Llama-3.2-3B-Instruct.

## Key Experimental Results

### Main Results

**Recommendation Accuracy on RecBench+ (Backbone: Llama-3.2-3B-Instruct)**

| Method | Movie-Simple | Movie-Medium | Movie-Hard | Movie-Interest | Book-Simple | Book-Hard | Book-Interest |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Original Llama | 0.107 | 0.052 | 0.029 | 0.097 | 0.215 | 0.106 | 0.254 |
| GPT-4o | 0.554 | 0.519 | 0.188 | 0.550 | 0.554 | 0.160 | 0.458 |
| GRPO | 0.686 | 0.600 | 0.644 | 0.651 | 0.664 | 0.713 | 0.786 |
| REINFORCE++ | 0.699 | 0.623 | 0.597 | 0.676 | 0.661 | 0.697 | 0.795 |
| **ReRec** | **0.748** | **0.700** | **0.729** | **0.719** | **0.671** | **0.750** | **0.811** |

### Ablation Study

| Configuration | Impact |
| :--- | :--- |
| ReRec (Full) | Best; highest accuracy across all dimensions |
| w/o RAAE | Largest drop; reasoning step supervision is the core contribution |
| w/o QAS | Significant drop; query constraint alignment is crucial for Hard queries |
| w/o PAS | Slight drop; implicit preference is more critical for personalized scenarios |
| w/o Curriculum | Moderate drop; training stability is affected |

### Key Findings

- ReRec improves over the untrained model by approximately 2414% (0.029 $\rightarrow$ 0.729) on the Movie-Hard task, showing that RFT greatly enhances the LLM's reasoning for misleading queries.
- The 3B-parameter ReRec outperforms GPT-4o and DeepSeek-R1 on most tasks.
- The removal of RAAE caused the most significant impact in ablation studies, proving that reasoning process supervision is the core contributor.
- **Cross-domain generalization**: A model trained on Movies achieved 0.494 on Books (Llama baseline 0.168, a 194% increase), surpassing GPT-4o (0.453).
- **Cross-task generalization**: ReRec reached 88.4% of the performance of the specialized SASRec model on sequential recommendation tasks.
- Compared to SFT, ReRec maintained reasoning capabilities (+21.6%), whereas SFT logic capabilities dropped by 80%.

## Highlights & Insights

- The dual-graph reward shaping is highly pragmatic: QAS uses attribute graphs to solve the "harsh exact match" problem, while PAS uses collaborative filtering embeddings to capture implicit preferences. These enrich the reward signal from explicit constraints and implicit preferences respectively, making them transferable to other RFT tasks requiring fine-grained rewards.
- The segment-level penalty mechanism in RAAE provides a lightweight alternative for process supervision: it does not require training a dedicated reward model, simply assigning differentiated rewards by checking if reasoning segments contain incorrectly recommended items.
- The online curriculum scheduler cleverly utilizes rollout data already available in RFT to estimate difficulty with zero additional inference cost.

## Limitations & Future Work

- Currently only supports single-turn dialogue recommendation, ignoring context accumulation and dynamic requirement adjustments in multi-turn settings.
- The candidate set is limited to a multiple-choice format (1 positive, 19 negatives), which differs from open-ended recommendation scenarios.
- RAAE's segment decomposition relies on simple paragraph splitting, which may be inaccurate for unstructured or mixed reasoning outputs.
- Backbones are limited to 3B parameters; performance on larger models has not been verified.
- The weights $w_1, w_2$ for QAS and PAS require manual tuning, and sensitivity was not fully discussed.

## Related Work & Insights

- **vs GRPO/REINFORCE++**: These general RFT methods use exact matches as rewards, which are too sparse for recommendation tasks. ReRec provides richer learning signals via dual-graph reward shaping, showing clear advantages on Hard queries.
- **vs TallRec/InteRecAgent**: These systems are based on SFT or tool-calling, which handle simple queries well but lack deep reasoning. ReRec stimulates reasoning via RFT, leading to significant gains on complex queries.
- **vs Process Reward Models**: Traditional PRMs require additional training or expensive LLM calls to score reasoning steps. RAAE implements lightweight step supervision through simple item-matching rules.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically adapts RFT for recommendation with highly targeted modules.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers cross-domain, cross-task, personalization, and capability maintenance.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, complete method description, and standard formula derivation.
- **Value**: ⭐⭐⭐⭐ A practical guide for RFT in recommendation systems; dual-graph rewards and RAAE are reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](../../NeurIPS2025/recommender/transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[AAAI 2026\] TraveLLaMA: A Multimodal Travel Assistant with Large-Scale Dataset and Structured Reasoning](../../AAAI2026/recommender/travellama_a_multimodal_travel_assistant_with_large-scale_dataset_and_structured.md)
- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)

</div>

<!-- RELATED:END -->
