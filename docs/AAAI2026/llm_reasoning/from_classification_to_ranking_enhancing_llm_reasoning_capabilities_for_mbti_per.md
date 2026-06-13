---
title: >-
  [Paper Note] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection
description: >-
  [AAAI 2026][LLM Reasoning][MBTI] This paper reformulates MBTI personality detection from four independent binary classifications into a listwise ranking task over all 16 personality types…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "MBTI"
  - "learning to rank"
  - "GRPO"
  - "NDCG"
  - "personality detection"
date: 2026-05-08
content_hash: b73ca13d5d252e0f
---

# From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection

**Conference**: AAAI 2026
**arXiv**: [2601.18582](https://arxiv.org/abs/2601.18582)  
**Code**: None  
**Area**: LLM/NLP
**Keywords**: MBTI, learning to rank, GRPO, NDCG, personality detection

## TL;DR
This paper reformulates MBTI personality detection from four independent binary classifications into a listwise ranking task over all 16 personality types, training a 7B model via SFT cold-start followed by GRPO reinforcement learning with a dual reward (NDCG + dimension similarity), achieving state-of-the-art results on the Kaggle and PANDORA datasets.

## Background & Motivation
**Background**: The dominant approach to MBTI personality detection decomposes the 16-type taxonomy into four independent binary classifications (E/I, S/N, T/F, J/P), encoding user posts with pretrained models such as BERT/RoBERTa. More recently, methods such as PsyCoT have explored chain-of-thought reasoning with LLMs for personality analysis.

**Limitations of Prior Work**: (a) Treating the four dimensions as independent classifiers ignores psychologically meaningful interactions among dimensions — INTJ and ENTJ differ by only one letter yet have fundamentally different dominant cognitive functions (Ni vs. Te); (b) prompt-based methods rely heavily on expert-designed prompts, while large-model inference incurs high latency and cost, rendering them unsuitable for real-time deployment.

**Key Challenge**: Personality traits are inherently relative preferences on a continuous spectrum rather than discrete categories. The classification paradigm cannot capture fine-grained affinity differences among personality types.

**Goal**: Enable LLMs to understand the relative affinity relationships among all 16 MBTI types, rather than simply assigning a post to one category.

**Key Insight**: Psychometrics characterizes personality through relative trait preferences rather than absolute categories, making ranking a more natural modeling choice than classification.

**Core Idea**: Redefine personality detection as a listwise ranking task over 16 MBTI types, and train the LLM via GRPO with an NDCG-based reward to learn optimal rankings.

## Method

### Overall Architecture
The input is a collection of user social media posts $P=\{p_1, ..., p_n\}$; the output is a Top-K ranking list of the 16 MBTI types ordered by affinity. A two-stage post-training pipeline is adopted: Stage 1 SFT cold-start → Stage 2 GRPO reinforcement learning.

### Key Designs

1. **Training Data Construction (Ranking Data Distillation)**:

    - Function: Construct ranked training data with reasoning traces for the SFT stage.
    - Mechanism: Qwen-plus is used as the teacher model to generate `<think>...</think>` reasoning chains and Top-K ranked answers `<answer>[type1, type2, ...]</answer>` given user posts. Rejection sampling filters out low-quality samples in which the ground-truth type does not appear in the Top-K predictions, yielding 1,000 high-quality SFT instances.
    - Design Motivation: 1,000 curated samples are sufficient to cold-start ranking capability while avoiding overfitting in the SFT stage. Rejection sampling ensures that reasoning chains are both correct and interpretable.

2. **SFT Cold-Start**:

    - Function: Teach the 7B model the ranked output format and basic reasoning patterns.
    - Mechanism: Qwen2.5-7B-Instruct is fine-tuned with LoRA (rank=32) for 3 epochs to learn the comparative reasoning and formatted output capabilities distilled from the teacher model.
    - Design Motivation: Provides a well-initialized policy for the subsequent RL stage. As shown in Figure 4, omitting SFT results in negligible performance gains during RL training.

3. **GRPO Reinforcement Learning + Ranking Reward Function**:

    - Function: Further optimize ranking quality through RL and enhance generalization.
    - Mechanism: The GRPO algorithm (no value network required) is employed, with reward design as the core contribution. Total reward = NDCG reward + Dimension Similarity (DS) reward. NDCG@k measures overall ranking list quality via position discounting (Equations 4–6). Dimension similarity $s(\hat{t}, t^*) = \sum_{i=1}^{4}(1+\epsilon \cdot i) \cdot \delta(\hat{t}^{(i)}, t^{*(i)})$ compares the predicted and ground-truth types character-by-character across four dimensions; the DS reward is computed for the Top-1 prediction to reinforce top-position accuracy.
    - Design Motivation: Classification reward signals are sparse and undifferentiated (minimal within-group variance), providing insufficient gradient signal for GRPO. The NDCG+DS dual reward stabilizes training — using either reward alone leads to training collapse (Figure 4, red curves), while their combination prevents reward hacking and produces consistent performance gains.

### Loss & Training
- GRPO objective (Equation 3): relative advantage normalization over $G$ sampled responses within a group, $A_i = (r_i - \mu_r)/\sigma_r$, with PPO-clip-style updates and KL regularization.
- The RL stage trains for 2,000 steps on the original dataset (excluding SFT data), with batch size 128 and 16 responses sampled per group.
- Each post is truncated to 128 tokens; each user contributes at most 50 posts; tokens matching the personality-type label are replaced with `<MASK>` to prevent information leakage.

## Key Experimental Results

### Main Results

| Method | Kaggle Macro-F1 | Kaggle F1 (16-class) | PANDORA Macro-F1 | PANDORA F1 (16-class) |
|--------|----------------|----------------------|------------------|------------------------|
| D-DGCN | 71.35 | 30.32 | 61.40 | 29.69 |
| ETM | 77.79 | 32.55 | 65.77 | 30.09 |
| Qwen-plus | 76.43 | 38.63 | 61.34 | 25.08 |
| **PerDet-R1** | **80.57** | **41.34** | **66.10** | **35.08** |

- Compared to the strongest baseline ETM: Macro-F1 improves by 2.78%/0.83%; 16-class F1 improves by 8.79%/4.99%.
- The advantage on 16-class F1 is especially pronounced, as the ranking paradigm models the four MBTI dimensions holistically.

### Ablation Study

| Configuration | Binary F1 | Multi F1 | NDCG@3 |
|---------------|-----------|----------|--------|
| Full (SFT+GRPO) | 80.57 | 41.34 | 74.31 |
| w/o GRPO | 65.03 | 15.82 | 64.09 |
| w/o SFT | 68.11 | 17.21 | 67.58 |
| w/o NDCG reward | 68.90 | 27.28 | 66.92 |
| w/o DS reward | 70.06 | 28.64 | 69.38 |
| Classification instead of Ranking | 63.07 | 33.67 | — |

### Key Findings
- Removing GRPO reduces NDCG by 10.22%; removing both SFT and GRPO reduces it by 24.85%, confirming that both training stages are indispensable.
- Replacing ranking with classification reduces Binary F1 by 17.5%, strongly validating the ranking paradigm.
- Removing either the NDCG or DS reward alone causes the other to collapse during training, verifying the mutual stabilization of the dual reward.
- Top-5 vs. Top-3: a larger answer space yields marginal gains on multi-class F1 and NDCG (+2.24%/+0.60%) but decreases Binary F1 by 4.13%.

## Highlights & Insights
- The **classification-to-ranking paradigm shift** is the core contribution: ranking naturally models the four-dimensional trait interactions among 16 MBTI types without artificially decomposing the dimensions. This idea is transferable to other multi-dimensional classification tasks (e.g., multi-dimensional sentiment scoring).
- The **dual-reward mutual stabilization mechanism** is elegant: NDCG focuses on overall list quality while DS enforces dimensional alignment at the top position; the two serve as mutual regularizers, preventing reward hacking and addressing practical RL training instability.
- The **1,000-sample SFT cold-start** is practically efficient: teacher distillation combined with rejection sampling enables cold-starting with a small amount of high-quality data, lowering the barrier to application.

## Limitations & Future Work
- Only textual posts are utilized; multimodal signals such as profile pictures and emoji are not incorporated (noted by the authors in the Future Work section).
- The Kaggle dataset is sourced from an MBTI-dedicated forum where posts implicitly contain personality-type abbreviations (e.g., "xNTP"), potentially introducing data bias and inflating results on this dataset.
- The improvement on PANDORA is relatively modest (+0.83% F1), indicating that generalization to non-MBTI-specific forum data warrants further investigation.
- Experiments are conducted only on 7B models; whether larger models further amplify the advantages of the ranking paradigm remains unknown.

## Related Work & Insights
- **vs. ETM**: ETM leverages LLM text embeddings and generation capabilities to enhance user vector representations, yet remains a classification paradigm; the proposed method's reframing as ranking yields substantially larger gains on 16-class F1 (+8.79%).
- **vs. PsyCoT**: PsyCoT simulates psychological questionnaire completion via multi-turn dialogue, relying on expert-designed CoT prompts; the proposed method autonomously learns reasoning patterns through RL without requiring expert knowledge.
- Takeaway: The classification-to-ranking paradigm shift is broadly applicable to other subjective evaluation tasks, including sentiment analysis, aesthetic scoring, and product recommendation.

## Rating
- Novelty: ⭐⭐⭐⭐ — The classification-to-ranking paradigm shift is creative, though the underlying technical components (SFT+GRPO) are well-established.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Ablations are comprehensive, but only two datasets are used and the Kaggle dataset exhibits known bias.
- Writing Quality: ⭐⭐⭐⭐ — Structure is clear, though notation is dense.
- Value: ⭐⭐⭐ — Application scope is relatively narrow, but the ranking paradigm idea is transferable.

## Additional Notes
- The methodology and experimental design of this work offer useful reference for related research areas.
- Future work may validate the generalizability and scalability of the approach across broader scenarios and larger model scales.
- Potential research value exists in combining this work with recent related methods (e.g., RL/MCTS or multimodal approaches).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On The Fragility of Benchmark Contamination Detection in Reasoning Models](../../ICLR2026/llm_reasoning/on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)
- [\[ICLR 2026\] Scaf-GRPO: Scaffolded Group Relative Policy Optimization for Enhancing LLM Reasoning](../../ICLR2026/llm_reasoning/scaf-grpo_scaffolded_group_relative_policy_optimization_for_enhancing_llm_reason.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[AAAI 2026\] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation](evaluating_synthesizing_and_enhancing_for_customer_support_conversation.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](../../NeurIPS2025/llm_reasoning/srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)

</div>

<!-- RELATED:END -->
