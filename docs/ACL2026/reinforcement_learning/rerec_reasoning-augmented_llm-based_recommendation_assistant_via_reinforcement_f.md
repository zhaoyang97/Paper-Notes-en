---
title: >-
  [Paper Note] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning
description: >-
  [ACL 2026][Reinforcement Learning][Recommendation Assistant] This paper proposes ReRec, a reinforcement fine-tuning (RFT)-based recommendation assistant framework that addresses the limitations of coarse reward signals and unsupervised reasoning processes through three components: dual-graph enhanced reward shaping for fine-grained reward signals, reasoning-aware advantage estimation for step-level differentiated supervision, and an online curriculum scheduler for dynamic training difficulty adjustment. ReRec enables LLMs to handle complex multi-step reasoning recommendation queries and significantly outperforms existing methods on the RecBench+ benchmark.
tags:
  - ACL 2026
  - Reinforcement Learning
  - Recommendation Assistant
  - Reinforcement Fine-tuning
  - Reasoning Augmentation
  - Reward Shaping
  - Curriculum Learning
date: 2026-05-08
content_hash: 260bc55733c4dd7a
---

# ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning

**Conference**: ACL 2026
**arXiv**: [2604.07851](https://arxiv.org/abs/2604.07851)
**Code**: [GitHub](https://github.com/jiani-huang/ReRec)
**Area**: Reinforcement Learning
**Keywords**: Recommendation Assistant, Reinforcement Fine-tuning, Reasoning Augmentation, Reward Shaping, Curriculum Learning

## TL;DR

This paper proposes ReRec, a reinforcement fine-tuning (RFT)-based recommendation assistant framework that addresses the limitations of coarse reward signals and unsupervised reasoning processes through three components: dual-graph enhanced reward shaping for fine-grained reward signals, reasoning-aware advantage estimation for step-level differentiated supervision, and an online curriculum scheduler for dynamic training difficulty adjustment. ReRec enables LLMs to handle complex multi-step reasoning recommendation queries and significantly outperforms existing methods on the RecBench+ benchmark.

## Background & Motivation

**Background**: Traditional recommendation systems (matrix factorization, GNNs, etc.) rely on historical interaction data and cannot process natural language queries. The emergence of LLMs has opened new possibilities for intelligent recommendation assistants. Recent studies have explored LLM-based conversational recommendation systems (CRS), but most handle only simple and direct queries such as "recommend a sci-fi movie."

**Limitations of Prior Work**: (1) Real-world user queries are often complex and require multi-step reasoning. For example: "Recommend other movies starring the lead actor of that film about a man stranded on an island"—which requires first inferring the film is *Cast Away*, then identifying Tom Hanks, and then recommending his other films. Existing LLM-based recommendation systems lack this deep reasoning capability. (2) Directly applying RFT to recommendation tasks faces two key challenges: reward signals are too coarse (NDCG only considers exact matches, giving zero reward to reasonable but non-exact recommendations); and the reasoning process lacks supervision (all tokens share the same reward score, making it impossible to distinguish correct from incorrect reasoning steps).

**Key Challenge**: Exact-match metrics such as NDCG are overly strict and sparse as reward signals—recommendations that satisfy query constraints but do not match the ground truth receive the same zero reward as completely irrelevant ones, leading to poor exploration efficiency for the policy model.

**Goal**: To design an RFT framework tailored for recommendation tasks, addressing the two core issues of insufficient reward granularity and unsupervised reasoning processes.

**Key Insight**: RFT for recommendation is improved along three dimensions: (1) enriching the reward space using item attribute graphs and collaborative filtering signals; (2) segmenting the reasoning process and penalizing erroneous reasoning steps; and (3) dynamically adjusting the training curriculum based on the model's evolving capability.

**Core Idea**: Extend the recommendation reward from coarse-grained exact matching to a fine-grained signal that integrates query alignment and preference alignment, while providing differentiated supervision at the reasoning step level, enabling LLMs to learn to reason rather than memorize.

## Method

### Overall Architecture

ReRec is built upon the GRPO reinforcement learning framework. Given a user query $q$, the LLM policy model generates multiple candidate responses $\{o_1, ..., o_G\}$, each containing a reasoning process and recommended items. Three core modules operate on reward computation, advantage estimation, and training scheduling respectively: dual-graph enhanced reward shaping computes the reward $r_i$ for each response; reasoning-aware advantage estimation distributes rewards to step-level token advantages $A_{i,t}$; and the online curriculum scheduler dynamically adjusts the ordering and difficulty of training data at each epoch.

### Key Designs

1. **Dual-Graph Enhanced Reward Shaping**:

    - **Function**: Provides fine-grained rewards integrating exact matching, query constraint satisfaction, and user preference alignment.
    - **Mechanism**: Two auxiliary scores are introduced on top of NDCG@K. The **Query Alignment Score (QAS)** uses an item-attribute graph $G_{attr}$ to compute the overlap ratio of attribute relations between the recommended item and ground truth: $S_{QAS}(p_i, gt) = |R_{p_i}^{G_{attr}} \cap R_{gt}^{G_{attr}}| / |R_{gt}^{G_{attr}}|$. The **Preference Alignment Score (PAS)** uses a pre-trained lightweight recommendation model (e.g., LightGCN) to generate item embeddings from the user-item interaction graph, computing cosine similarity $S_{PAS}(p_i, gt) = \mathcal{M}(p_i) \cdot \mathcal{M}(gt) / (\|\mathcal{M}(p_i)\| \|\mathcal{M}(gt)\|)$. The final reward is $r_i = \text{NDCG} + w_1 S_{QAS} + w_2 S_{PAS}$.
    - **Design Motivation**: QAS rewards recommendations that satisfy query constraints (e.g., genre, actor) but do not match the ground truth, avoiding the blanket penalization of reasonable recommendations as zero; PAS captures implicit preferences encoded in collaborative filtering signals, preventing over-reliance on attribute matching at the expense of personalized user preferences.

2. **Reasoning-Aware Advantage Estimation (RAAE)**:

    - **Function**: Provides step-level differentiated supervision over the reasoning process, penalizing erroneous reasoning steps.
    - **Mechanism**: The LLM output $o_i$ is decomposed into $K$ reasoning steps $\mathcal{S}_i = \{s_{i,1}, ..., s_{i,K}\}$ by paragraph. If a reasoning paragraph discusses an item that is ultimately recommended incorrectly (i.e., $p_i \neq gt$ and $p_i \in s_{i,k}$), indicating the model failed to exclude that item, the reward for the corresponding paragraph is penalized to $r_{s_{i,k}} = (1 - w_{penalty}) \cdot r_i$; other paragraphs retain the original reward. Paragraph rewards are then mapped to the token level to compute normalized advantages $A_{i,t} = (r_{i,t} - \text{mean}(\mathbf{r})) / \text{std}(\mathbf{r})$.
    - **Design Motivation**: Conventional RFT assigns the same reward to all tokens, providing no signal to identify which reasoning steps are erroneous. RAAE achieves step-level supervision through lightweight paragraph decomposition and conditional penalization, avoiding the high cost of training dedicated process reward models.

3. **Online Curriculum Scheduler**:

    - **Function**: Dynamically assesses query difficulty and organizes training in an easy-to-hard order to ensure stable convergence.
    - **Mechanism**: A three-step process is employed. **Adaptive difficulty assessment**: at the start of epoch $t$, the difficulty of each query is estimated using the mean inverse reward from the previous round: $d^{t-1} = \frac{1}{G}\sum_{i=1}^G (1 - r_i)$. **Sample filtering and ranking**: samples with difficulty below threshold $\tau$ (deemed "mastered") are filtered out, and the remaining samples are sorted in ascending order of difficulty to form the new dataset $\mathcal{D}^t$. **Iterative update**: this process is repeated each epoch to adapt to the model's continuously improving capability.
    - **Design Motivation**: Query difficulty in recommendation tasks is hard to define a priori (unlike math or code, which have clear difficulty levels), and static curricula cannot reflect the model's dynamic learning progress. The online scheduler leverages existing rollout data to update difficulty estimates at zero additional inference cost.

### Loss & Training

The GRPO objective with clipped probability ratios is adopted: $\mathcal{J}(\theta) = \mathbb{E}[\frac{1}{N}\sum_{i=1}^G \sum_{t=1}^{|o_i|} \min(h_{i,t}(\theta) A_{i,t}, \text{clip}(h_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon) A_{i,t})]$. Backbone models are Qwen-2.5-3B-Instruct and Llama-3.2-3B-Instruct.

## Key Experimental Results

### Main Results

**Recommendation accuracy on RecBench+ benchmark (Llama-3.2-3B-Instruct backbone)**

| Method | Movie-Simple | Movie-Medium | Movie-Hard | Movie-Interest | Book-Simple | Book-Hard | Book-Interest |
|--------|-------------|-------------|-----------|---------------|-------------|----------|--------------|
| Llama (original) | 0.107 | 0.052 | 0.029 | 0.097 | 0.215 | 0.106 | 0.254 |
| GPT-4o | 0.554 | 0.519 | 0.188 | 0.550 | 0.554 | 0.160 | 0.458 |
| GRPO | 0.686 | 0.600 | 0.644 | 0.651 | 0.664 | 0.713 | 0.786 |
| REINFORCE++ | 0.699 | 0.623 | 0.597 | 0.676 | 0.661 | 0.697 | 0.795 |
| **ReRec** | **0.748** | **0.700** | **0.729** | **0.719** | **0.671** | **0.750** | **0.811** |

### Ablation Study

| Configuration | Effect |
|---------------|--------|
| ReRec (Full) | Best performance across all dimensions |
| w/o RAAE | Largest drop; step-level reasoning supervision is the core contribution |
| w/o QAS | Notable drop; query constraint alignment is especially important for Hard queries |
| w/o PAS | Minor drop; implicit preferences are more critical in personalized scenarios |
| w/o Curriculum | Moderate drop; training stability is affected |

### Key Findings

- ReRec achieves approximately 2414% improvement on Movie-Hard over the untrained model (0.029→0.729), demonstrating that RFT substantially enhances LLMs' ability to handle misleading queries requiring deep reasoning.
- ReRec with 3B parameters outperforms GPT-4o and DeepSeek-R1 on most tasks.
- Removing RAAE causes the largest performance drop in the ablation study, confirming that reasoning process supervision is the core contribution.
- Cross-domain generalization: a model trained on Movie data achieves 0.494 on Book (vs. Llama baseline 0.168, a 194% improvement), surpassing GPT-4o (0.453).
- Cross-task generalization: ReRec achieves 88.4% of the performance of the dedicated model SASRec on sequential recommendation tasks.
- Compared to SFT, ReRec preserves reasoning capability (+21.6%), whereas SFT incurs an 80% degradation in general knowledge capability.

## Highlights & Insights

- The dual-graph reward shaping design is highly pragmatic: QAS leverages item attribute graphs to address the problem of "exact matching being overly strict," while PAS uses collaborative filtering embeddings to capture implicit preferences. Together, they enrich the reward signal from both explicit constraint and implicit preference dimensions, and are transferable to other RFT tasks requiring fine-grained rewards.
- The paragraph-level penalization mechanism in RAAE provides a lightweight alternative to process supervision: it requires no dedicated reward model training, and achieves differentiated reward assignment simply by checking whether reasoning paragraphs contain incorrectly recommended items—simple yet effective.
- The online curriculum scheduler cleverly reuses existing rollout data from RFT to estimate query difficulty, incurring zero additional inference cost.

## Limitations & Future Work

- Only single-turn conversational recommendation is supported; multi-turn dialogue with contextual accumulation and dynamic preference updates is not considered.
- The candidate set is formulated as a multiple-choice setting with 1 positive and 19 negatives, which differs substantially from open-ended recommendation scenarios.
- Paragraph decomposition in RAAE relies on simple paragraph segmentation, which may be inaccurate for unstructured or mixed-reasoning outputs.
- The backbone model is limited to 3B parameters; performance on larger models remains unverified.
- The weights $w_1, w_2$ for QAS and PAS require hyperparameter tuning, and sensitivity analysis is not sufficiently discussed in the paper.

## Related Work & Insights

- **vs. GRPO/REINFORCE++**: These general-purpose RFT methods use exact matching as reward, resulting in sparse reward signals for recommendation tasks. ReRec provides richer learning signals via dual-graph reward shaping, with particularly pronounced advantages on Hard queries.
- **vs. TallRec/InteRecAgent**: These LLM-based recommendation systems rely on SFT or tool invocation and can handle simple queries but lack deep reasoning capability. ReRec elicits reasoning ability through RFT, achieving substantial improvements on complex queries.
- **vs. Process Reward Models**: Conventional process reward models require additional training or calls to large models to score reasoning steps, incurring high cost and limited scalability. RAAE achieves lightweight step-level supervision through simple item-matching rules.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Systematically adapts RFT to recommendation tasks with three targeted module designs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers cross-domain, cross-task, personalization, and capability preservation dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ Problem motivation is clear, method description is complete, and mathematical derivations are rigorous.
- **Value**: ⭐⭐⭐⭐ Serves as a practical guide for applying RFT to recommendation systems; dual-graph reward shaping and RAAE are reusable components.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Optimizing User Profiles via Contextual Bandits for Retrieval-Augmented LLM Personalization](optimizing_user_profiles_via_contextual_bandits_for_retrieval-augmented_llm_pers.md)
- [\[NeurIPS 2025\] Parameter Efficient Fine-tuning via Explained Variance Adaptation](../../NeurIPS2025/reinforcement_learning/parameter_efficient_fine-tuning_via_explained_variance_adaptation.md)
- [\[ACL 2026\] Semantic-Space Exploration and Exploitation in RLVR for LLM Reasoning](semantic-space_exploration_and_exploitation_in_rlvr_for_llm_reasoning.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[CVPR 2026\] ReAG: Reasoning-Augmented Generation for Knowledge-based Visual Question Answering](../../CVPR2026/reinforcement_learning/reag_reasoning-augmented_generation_for_knowledge-based_visual_question_answerin.md)

<!-- RELATED:END -->
