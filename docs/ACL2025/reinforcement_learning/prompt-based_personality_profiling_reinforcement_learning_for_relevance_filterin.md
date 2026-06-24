---
title: >-
  [Paper Note] Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering
description: >-
  [ACL 2025][Reinforcement Learning][personality profiling] This paper proposes RL-Profiler, which trains a post relevance filter (SelNet) using reinforcement learning to select a small subset of posts relevant to personality traits from a user's large profile. These selected posts are then passed to an LLM for zero-shot personality prediction, thereby significantly reducing context length while maintaining prediction performance close to using all posts.
tags:
  - "ACL 2025"
  - "Reinforcement Learning"
  - "personality profiling"
  - "reinforcement-learning"
  - "relevance filtering"
  - "LLM prompting"
  - "Big Five"
date: 2026-05-08
content_hash: cc5984f3eca5ee67
---

# Prompt-based Personality Profiling: Reinforcement Learning for Relevance Filtering

**Conference**: ACL 2025  
**arXiv**: [2409.04122](https://arxiv.org/abs/2409.04122)  
**Code**: [GitHub](https://github.com/bluzukk/rl-profiler)  
**Area**: Author Profiling / Personality Prediction  
**Keywords**: personality profiling, reinforcement-learning, relevance filtering, LLM prompting, Big Five

## TL;DR
This paper proposes RL-Profiler, which trains a post relevance filter (SelNet) using reinforcement learning to select a small subset of posts relevant to personality traits from a user's large profile. These selected posts are then passed to an LLM for zero-shot personality prediction, thereby significantly reducing context length while maintaining prediction performance close to using all posts.

## Background & Motivation
**Background**: Author profiling aims to infer user attributes (age, gender, personality, etc.) from user-generated content, typically using supervised learning methods, though deep learning methods struggle on this task.

**Limitations of Prior Work**: A user profile may contain hundreds of posts, which far exceeds the input length limits of Transformers. Even if they can be processed, using the entire content leads to high API costs and suffers from the "needle in a haystack" problem—not all posts are relevant to a specific personality dimension, and irrelevant posts introduce noise.

**Key Challenge**: There is a need to filter relevant posts, but post-level relevance labels are unavailable. Retrieval methods like RAG are general ad-hoc retrieval techniques not optimized for specific tasks.

**Goal**: Learn a task-specific post filter without post-level relevance annotations to improve both the efficiency and accuracy of personality prediction.

**Key Insight**: Train a filter using reinforcement learning (RL) by using the correctness of the LLM's zero-shot prediction as a reward signal.

**Core Idea**: Use reinforcement learning to teach the model to "select posts" so that the LLM prediction on a small set of selected posts is better than or comparable to that using all posts.

## Method

### Overall Architecture
RL-Profiler consists of two components:
- **SelNet (Selection Network)**: A BERT-based RL agent that performs binary select/reject decisions for each post.
- **CNet (Classification Network)**: A zero-shot classifier based on Llama-2-13B-Chat that predicts personality using the selected posts and preset prompts.

### Key Designs
1. **SelNet — RL Agent**: Utilizing a `bert-base-uncased` backbone with a binary classification head, the policy $\pi(a|s,\theta)$ outputs the selection/rejection probability for each post. During training, actions are sampled based on probabilities (exploration), while during inference, posts are sorted by probability to select the top-N (exploitation), ensuring a fixed number of posts are chosen.
2. **Reward Function Design**: $R(y,\hat{y},C) = -2 + \text{sign}(|C|)(3 - 2|y-\hat{y}|) - \lambda|C|$. Three cases exist: correct prediction yields $+1-\lambda|C|$, incorrect prediction yields $-1-\lambda|C|$, and selecting no posts yields $-2$. Here, $\lambda$ penalizes choosing too many posts, encouraging correct predictions with minimum posts.
3. **NPMI Pretraining**: Normalized Pointwise Mutual Information (NPMI) is used to compute the association between words and labels. Post-level relevance scores within each profile are calculated, and the top-M posts are labeled as "relevant" for supervised pretraining of the RL agent to improve training stability.
4. **CNet Prompt Design**: Prompts are designed for each of the Big Five dimensions, containing system instructions (single-word answer), trait descriptions (adapted from the BFI-44 questionnaire), selected posts, and prediction instructions.

### Loss & Training
- Policy optimization is performed using the REINFORCE algorithm: $\theta \leftarrow \theta + \alpha \sum(R-b) \ln \nabla_{\theta} \pi(a_t|s_t,\theta)$
- Baseline $b$ is calculated as the moving average reward of the last 10 steps.
- NPMI pretraining runs for 2 epochs, while RL training runs for up to 200 epochs.
- $\lambda=0.05$, learning rate is $10^{-6}$, optimized using AdamW.
- Validation is conducted across top-N $\in \{5,10,20,30,50\}$ to select the optimal model.

## Key Experimental Results

### Main Results — PAN-AP-2015 Dataset (macro-F1 / weighted-F1)

| System | Openness | Consc. | Extrav. | Agree. | Neurot. |
|------|----------|--------|---------|--------|---------|
| Baseline-R (All posts) | 49.8/98.9 | 47.9/88.0 | 82.5/96.7 | 59.8/88.2 | 66.8/73.6 |
| Baseline-B (All posts) | 56.5/99.0 | 58.8/88.8 | 76.9/93.6 | 66.2/88.2| 67.8/73.6 |
| ALL+CNet (All posts) | 49.8/98.9 | 47.0/70.5 | 48.4/92.1 | 52.5/75.8 | 42.7/57.0 |
| **RL-Profiler** | 47.7/94.6 | 44.6/63.9 | **57.0/92.3** | 43.1/70.8 | 39.3/47.0 |
| RND+CNet (Random selection) | 49.6/98.5 | 33.4/45.7 | 48.3/91.8 | 41.8/58.1 | 38.8/46.1 |

### Ablation Study — Comparison of Variants
- RL-Profiler vs ALL+CNet: The average macro-F1 is only 1.8 percentage points lower, but the number of posts is reduced from 92.9 to ~10.
- RL-Profiler vs RND+CNet: The average macro-F1 is 3.9 percentage points higher, with RL using only 5 posts compared to 50 posts used by random selection.
- PMI+CNet and PT+CNet (statistical/pretraining only) underperform compared to the full RL training.

### Key Findings
- Using RL to filter down to only ~5 posts yields prediction performance close to using all ~93 posts, reducing inference time by 76% (1.65s $\rightarrow$ 0.38s).
- In balanced data augmentation experiments, RL filtering significantly improves prediction accuracy, demonstrating that relevance filtering has greater superiority when classes are balanced.
- Pure statistical methods (direct NPMI selection) are inferior to the agent trained via RL, proving that RL successfully learns a selection strategy beyond word frequency statistics.
- Severe labels skewness in the dataset (e.g., Openness: 119 high vs 1 low) limits the performance of zero-shot methods.

## Highlights & Insights
- **Elegant Problem Formulation**: Modeling "which posts to select for prediction" as an RL selection problem elegantly leverages profile-level labels as weak supervision.
- **High Practicality**: Only a small number of posts need to run through the LLM during inference, dramatically reducing API costs and latency.
- **No Post-Level Annotation Required**: Relies on the RL reward mechanism to bypass the need for tedious manual post-level relevance annotations.
- **Generalizability**: The method does not depend on a specific task and can theoretically be extended to other scenarios requiring information filtering from large collections of texts.

## Limitations & Future Work
- The performance improvement on the raw data (where labels are heavily skewed) is limited, as zero-shot methods themselves scale poorly in highly imbalanced classes.
- There is still a 15-19pp macro-F1 gap compared to supervised learning baselines (Ridge/BERT).
- Fixed CNet to Llama-2-13B; whether stronger LLMs (like GPT-4) can further push the performance ceiling remains unexplored.
- Validated only on English Twitter data; capability to generalize across languages/platforms is unknown.
- Performance varies dramatically across the Big Five dimensions, with relatively weaker results on Neuroticism.

## Related Work & Insights
- Difference from RAG: RAG serves as an ad-hoc retriever, whereas this method is a task-specific filter optimized directly for the downstream goal.
- Utilizing questionnaire items from BFI-44 as trait descriptions in the prompt elegantly marries psychological domain expertise with NLP methods.
- Insight: In long-context/multi-document input scenarios, a two-stage "filter-then-predict" pipeline could serve as a general paradigm for managing context length restrictions.

## Rating
⭐⭐⭐ (3.5/5)
- **Novelty**: ⭐⭐⭐⭐ — Applying RL for post relevance filtering is novel, with a well-designed reward function.
- **Experimental Thoroughness**: ⭐⭐⭐ — Ablation and comparisons are relatively thorough, but the single dataset and high label imbalance hurt the strength of the findings.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear methodology description and structured experimental analysis.
- **Value**: ⭐⭐⭐ — The proposed method has practical value, though realistic improvements are constrained by experimental data quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Knowledge-based Visual Question Answer with Multimodal Processing, Retrieval and Filtering](../../NeurIPS2025/reinforcement_learning/knowledge-based_visual_question_answer_with_multimodal_processing_retrieval_and_.md)
- [\[NeurIPS 2025\] Prompt Tuning Decision Transformers with Structured and Scalable Bandits](../../NeurIPS2025/reinforcement_learning/prompt_tuning_decision_transformers_with_structured_and_scalable_bandits.md)
- [\[ICLR 2026\] Prompt Curriculum Learning for Efficient LLM Post-Training](../../ICLR2026/reinforcement_learning/prompt_curriculum_learning_for_efficient_llm_post-training.md)
- [\[ACL 2025\] Learning to Generate Structured Output with Schema Reinforcement Learning](learning_to_generate_structured_output_with_schema_reinforcement_learning.md)
- [\[AAAI 2026\] MARS: Multi-Agent Adaptive Reasoning with Socratic Guidance for Automated Prompt Optimization](../../AAAI2026/reinforcement_learning/mars_multi-agent_adaptive_reasoning_with_socratic_guidance_f.md)

</div>

<!-- RELATED:END -->
