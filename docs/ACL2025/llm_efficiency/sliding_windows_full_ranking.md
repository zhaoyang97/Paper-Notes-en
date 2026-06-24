---
title: >-
  [Paper Note] Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models
description: >-
  [ACL 2025 (Long Paper)][LLM Efficiency][LLM Reranking] This paper systematically investigates the application of long-context LLMs in passage ranking, proposing the use of full ranking (ranking all passages at once) to replace traditional sliding window strategies. By designing a multi-pass sliding window label construction method and an importance-aware loss function to fine-tune the full ranking model, this approach achieves comprehensive improvements in ranking performance…
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM Efficiency"
  - "LLM Reranking"
  - "Passage Ranking"
  - "Long-Context LLM"
  - "Listwise Ranking"
  - "Importance-Aware Loss"
date: 2026-05-08
content_hash: dc4361e3643d32ec
---

# Sliding Windows Are Not the End: Exploring Full Ranking with Long-Context Large Language Models

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2412.14574](https://arxiv.org/abs/2412.14574)  
**Code**: [https://github.com/8421BCD/fullrank](https://github.com/8421BCD/fullrank)  
**Area**: LLM Efficiency  
**Keywords**: LLM Reranking, Passage Ranking, Long-Context LLM, Listwise Ranking, Importance-Aware Loss  

## TL;DR
This paper systematically investigates the application of long-context LLMs in passage ranking, proposing the use of full ranking (ranking all passages at once) to replace traditional sliding window strategies. By designing a multi-pass sliding window label construction method and an importance-aware loss function to fine-tune the full ranking model, this approach achieves comprehensive improvements in ranking performance while enhancing efficiency by approximately 30-65%.

## Background & Motivation
LLMs perform exceptionally well on listwise passage ranking tasks. However, constrained by input length limits, existing methods commonly employ a sliding window strategy, moving from back to front with a window size of 20 and a stride of 10 to gradually bubble up relevant passages to the top. This strategy suffers from two core pain points:
1. **Redundant Inference**: Neighboring windows contain a large number of overlapping passages that are repeatedly evaluated, causing API costs to scale linearly with the number of inference tokens.
2. **Sequential Dependence**: Sequential dependencies exist between windows, preventing parallel inference and creating an efficiency bottleneck.

Armed with the development of long-context LLMs (such as Mistral-7B 32k and LLaMA 3.1 128k), inputting all candidate passages at once and outputting a complete ranked list (full ranking) has become feasible, yet its performance and efficiency have not been systematically investigated.

## Core Problem
1. How does the full ranking strategy of long-context LLMs compare to the traditional sliding window strategy in terms of efficiency and effectiveness?
2. How can full ranking models be effectively fine-tuned? Existing methods face two limitations: sliding windows cannot generate complete ranking labels, and standard language modeling losses fail to distinguish the importance of high-ranking versus low-ranking passages.

## Method

### Overall Architecture
The pipeline is divided into two phases:
1. **Label Construction**: A teacher model (GPT-4o/GPT-4o-mini) is utilized to generate complete ranking labels for 100 passages via a multi-pass sliding window method.
2. **Model Fine-Tuning**: An importance-aware loss is used to fine-tune Mistral-7B-Instruct-v0.3, taking 100 passages as input at once and outputting the complete ranked list.

### Key Designs
1. **Multi-pass Sliding Window (Label Construction)**: Mathematically, a single-pass sliding window (window size 20, stride 10) can only guarantee the correctness of the top-10 ranking (analogous to how a single pass of bubble sort only determines the maximum value). This work proposes an iterative method: the first pass applies the sliding window to all 100 passages to get the top-10; the second pass ranks the remaining 90 passages to determine positions 11-20; this process repeats until a complete 100-passage ranked list is constructed as training labels.
2. **Full Ranking Strategy**: All 100 candidate passages are fed into the long-context LLM at once, directly outputting the complete ranking sequence like `[99] > [100] > ...`, thus bypassing the redundant computation and sequential dependencies of sliding windows.

### Loss & Training
**Importance-Aware Loss**: While the standard language modeling loss penalizes all passage ID tokens equally, only a few of the 100 IDs in the full ranking labels are actually relevant. This work introduces a position-weighted loss:

$$\mathcal{L}_{ia} = -\sum_{i=1}^{|y|} w_i \log P_\theta(y_i | x, y_{<i})$$

where the weight is defined as:
- For passage ID tokens: $w_i = 1 + \frac{1}{\log_2(p_i + 1)}$, where $p_i$ is the rank position of the passage.
- For non-ID tokens (e.g., `>`): $w_i = \alpha$ ($\alpha \leq 1$).

Consequently, passages ranked higher receive larger weights in the loss, aligning with evaluation metrics that focus on top-ranked results.

Training details: Using Mistral-7B-Instruct-v0.3 as the backbone, training labels were generated from 1k MS MARCO queries. Training was conducted for 4 epochs with lr=5e-6, batch size=1, using 4×A100-40GB GPUs.

## Key Experimental Results

| Dataset | Metric | RankMistral100 (Full) | RankMistral20 (Sliding) | Gain |
|--------|------|------|----------|------|
| DL19 | NDCG@10 | **73.17** | 69.08 | +4.09 |
| DL20 | NDCG@10 | **70.16** | 66.31 | +3.85 |
| BEIR Avg | NDCG@10 | **52.63** | 50.45 | +2.18 |
(The above are results from SFT with GPT-4o-mini)

| Dataset | Metric | RankMistral100 (Full) | RankMistral20 (Sliding) | Gain |
|--------|------|------|----------|------|
| DL19 | NDCG@10 | **72.55** | 70.34 | +2.21 |
| DL20 | NDCG@10 | **71.29** | 69.58 | +1.71 |
| BEIR Avg | NDCG@10 | **52.40** | 51.85 | +0.55 |
(The above are results from SFT with GPT-4o)

Efficiency aspects:
- Compared to the sliding window, full ranking **reduces latency by 29.3%** (DL19).
- On the Signal dataset, when outputting only the top-10 IDs, full ranking takes just 3.8s compared to 29.9s for the sliding window, representing an approximate **8x speedup**.
- Full ranking **reduces API costs by approximately 50%**.

### Ablation Study
- Removing the importance-aware loss causes a drop of approximately 0.7 points in BEIR Avg for both RankMistral100 and RankMistral20.
- Even with the standard LM loss, RankMistral100 still outperforms RankMistral20 (on DL19/DL20/BEIR Avg), demonstrating the inherent advantage of full ranking.
- Under the zero-shot setting, full ranking performs worse than the sliding window (e.g., Mistral-v0.3 BEIR Avg: 40.14 vs 45.16), though the gap is smallest with GPT-4o.
- Although RankMistral100 is trained with 100 passages, it generalizes well and outperforms RankMistral20 across various candidate sizes (N=20/40/60/80).
- The initial order of passages significantly impacts ranking performance (random or reversed ordering drastically degrades performance).
- Performance gains tend to converge after the number of ranking passes increases to 3-4.

## Highlights & Insights
- **First systematic study on applying long-context LLMs to ranking tasks**, filling a critical gap in this field.
- **Win-win in efficiency and effectiveness**: The fine-tuned full ranking model simultaneously outperforms the sliding window approach in both performance and speed.
- **Multi-pass sliding window label construction** is an ingenious concept, analogous to the multi-pass process of bubble sort, successfully addressing the issue of single-pass methods being unable to generate complete rankings.
- **Importance-aware loss** design is intuitive and effective, aligning with the preferences of ranking evaluation metrics (such as NDCG focusing on top-ranked items).
- Extremely comprehensive evaluation: covering both zero-shot and SFT settings, multiple open-source/closed-source models, multi-dimensional analysis of efficiency/effectiveness/costs, and generalization validation across different values of N.

## Limitations & Future Work
- Experiments are restricted to 7B/8B scale models; larger models like 30B/70B are not explored, leaving the impact of model scale on full ranking unknown.
- No dedicated long-context LLM architecture is designed specifically for ranking tasks.
- The cost of multi-pass sliding window label construction is relatively high (generating labels for 1k queries using GPT-4o costs $261, much higher than $29 for the standard sliding window).
- Sensitive to the initial passage order; random or reversed order significantly degrades performance, indicating a need for improved robustness.
- Only evaluated scenarios where BM25 acts as the retriever.

## Related Work & Insights
- **RankZephyr (Pradeep et al., 2023b)**: A listwise reranker distilled from GPT-3.5/GPT-4 that uses a sliding window strategy. RankMistral100 in this paper achieves 72.55 (NDCG@10) on DL19, notably surpassing RankZephyr's 73.39, and 52.40 on BEIR Avg, surpassing 51.15.
- **RankVicuna (Pradeep et al., 2023a)**: Also a distillation method but yields weaker performance (BEIR Avg 48.95); the proposed method achieves comprehensive dominance.
- **Sun et al., 2023 (RankGPT)**: A pioneering work that proposed the sliding window ranking prompt framework. Building upon this, this paper proposes the full ranking alternative, proving that sliding windows are not the end of the road.

## Insights
1. **Long-context capabilities shift the ranking paradigm**: Transitioning from sliding windows that require multiple inferences to one-pass global ranking. This paradigm shift could also be valuable in other tasks that require global comparisons (e.g., document summarization, multi-candidate selection).
2. **Quality of training labels > Quantity**: Fine-tuning a powerful model requires only 1k queries, with no gains observed upon increasing to 1.5k. This indicates that carefully constructed high-quality labels are far more important than sheer volume.
3. **Position-weighted loss is a general philosophy**: Any task where different positions in the output sequence carry different levels of importance could benefit from a similar importance-aware loss design.
4. Global interaction (where full ranking allows mutual comparison among all passages) can be fully utilized after fine-tuning, suggesting that models can learn to utilize long-context information more effectively through training.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic study of full ranking vs. sliding window; however, the core idea (using long context to replace sliding windows) is relatively intuitive. Technical contributions lie mainly in label construction and loss function design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive experiments in both zero-shot/SFT settings across various models and datasets, including multi-dimensional analysis of efficiency, effectiveness, and cost, as well as detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear, fluent writing. The problem motivation is well-articulated, figures and tables are instinctively designed, and the experiment section is well-organized.
- **Value**: ⭐⭐⭐⭐ Offers direct reference value for designing LLM-based reranking systems. The design philosophy of importance-aware loss is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CNNSum: Exploring Long-Context Summarization with Large Language Models in Chinese Novels](cnnsum_exploring_long-context_summarization_with_large_language_models_in_chines.md)
- [\[ACL 2025\] LongSafety: Evaluating Long-Context Safety of Large Language Models](longsafety_evaluating_long-context_safety_of_large_language_models.md)
- [\[ACL 2025\] LongReward: Improving Long-context Large Language Models with AI Feedback](longreward_improving_long-context_large_language_models_with_ai_feedback.md)
- [\[ICLR 2026\] The End of Manual Decoding: Towards Truly End-to-End Language Models](../../ICLR2026/llm_efficiency/the_end_of_manual_decoding_towards_truly_end-to-end_language_models.md)
- [\[ACL 2025\] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing](scaling_context_not_parameters_training_a_compact_7b_language_model_for_efficien.md)

</div>

<!-- RELATED:END -->
