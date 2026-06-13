---
title: >-
  [Paper Note] SenseJudge: Human-Centric Preference-Driven Judgment Framework
description: >-
  [ACL 2026][Recommender Systems][LLM Evaluation] SenseJudge is proposed as a customizable LLM judgment framework based on explicit human preferences. Coupled with SenseBench—a benchmark of real-world multi-turn dialogues—…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "LLM Evaluation"
  - "Personalized Judgment"
  - "Preference-Driven"
  - "Multi-turn Dialogue"
  - "Model Ranking"
date: 2026-05-08
content_hash: d294799a43ecd874
---

# SenseJudge: Human-Centric Preference-Driven Judgment Framework

**Conference**: ACL 2026 Findings  
**arXiv**: [2606.03189](https://arxiv.org/abs/2606.03189)  
**Code**: [GitHub](https://github.com/qiongrenpiqida/SenseJudge)  
**Area**: recommender  
**Keywords**: LLM Evaluation, Personalized Judgment, Preference-Driven, Multi-turn Dialogue, Model Ranking

## TL;DR
SenseJudge is proposed as a customizable LLM judgment framework based on explicit human preferences. Coupled with SenseBench—a benchmark of real-world multi-turn dialogues—it achieves an average accuracy 16.08% higher than baselines in personalized evaluation tasks, yielding model rankings consistent with actual human performance.

## Background & Motivation
**Background**: The LLM-as-a-Judge paradigm is increasingly popular for evaluating model responses, generating preference data, and ranking models.

**Limitations of Prior Work**: (1) Existing judgment methods (PandaLM, Auto-j, reward models) rely on training with fixed preference data, learning homogenized standards while ignoring the diversity of user preferences; (2) current benchmarks (MT-Bench, Auto-j) focus primarily on single or double-turn dialogues, disconnecting from real-world multi-turn human-computer interaction scenarios; (3) trained reward models exhibit limited generalization when facing diverse real-world scenarios.

**Key Challenge**: User preferences are diverse and scenario-dependent (some prioritize creativity, others format, and others accuracy), yet existing judges learn only a single fixed preference standard.

**Goal**: To build a customizable LLM judgment framework adaptable to different user preferences, alongside an evaluation benchmark that truly reflects the complexity of human-computer interaction.

**Key Insight**: Extract explicit preference text from a small number of human annotations and utilize a multi-preference voting mechanism to enable small models to provide accurate personalized judgments.

**Core Idea**: Preference Extraction + Preference Set Selection + Multi-Preference Voting = Personalized LLM judgment without retraining.

## Method

### Overall Architecture
SenseBench constructs a multi-turn evaluation benchmark from real user dialogues via quality and challenge filtering. SenseJudge extracts preference text from a small set of human-annotated pairs, selects the optimal preference subset, and generates final judgments through multi-preference voting during inference.

### Key Designs
1.  **SenseBench Benchmark Construction**:
    - **Function**: Provides a multi-turn, multi-domain evaluation benchmark (8 categories × 125 questions) close to real human-computer interactions.
    - **Mechanism**: Two-stage filtering—(1) Quality filtering: using Qwen3-14B for de-noising and classification (Math/Logic/Code/Creative Writing/Roleplay/Translation/QA/NLU); (2) Challenge filtering: multi-model response comparison (strong vs. weak models) + GPT-4 automatic screening + human verification to ensure discriminative questions.
    - **Design Motivation**: Most existing benchmarks involve single-turn simple tasks, failing to reflect the complexity and multi-turn context dependencies of real user scenarios.

2.  **Preference Extraction and Selection**:
    - **Function**: Distills a generalizable set of explicit preferences from minimal annotations.
    - **Mechanism**: (1) Preference generation: using DeepSeek-R1 to generate explicit preference text from annotated pairs $(q, \text{chosen}, \text{rejected})$; (2) Preference set selection: traversing all preference subsets $\mathcal{P}_k \subseteq P$, performing multi-preference voting on the annotation set, and selecting the subset $\mathcal{P}_k^*$ with the highest accuracy; (3) Preference application: using each preference in $\mathcal{P}_k^*$ to judge independently on the test set, followed by majority voting.
    - **Design Motivation**: Different preference texts capture various aspects of user annotation decisions (e.g., "valuing logical rigor" vs. "valuing completeness"); ensemble use is more robust than a single preference.

3.  **Input/Output Format and Voting Mechanism**:
    - **Function**: Standardizes the judgment process and reduces position bias.
    - **Mechanism**: Input $I = \{q, (r_1, r_2), p\}$, output judgment + analysis; "tie" options are disallowed to force model discrimination; both forward and reverse orders are evaluated to detect position bias; final stable judgments are produced via multi-preference voting.
    - **Design Motivation**: Position bias is a known issue for LLM-as-a-Judge (models tend to select the first or last response); dual-order evaluation and multi-preference voting effectively mitigate this.

## Key Experimental Results

### Main Results (LLM-as-a-Personalized-Judge Accuracy %)

| Method | Math | Code | Logic | QA | Write | Role | NLU | Trans | Overall |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 66.00 | 61.60 | 65.47 | 72.93 | 60.80 | 63.20 | 65.47 | 56.40 | 63.98 |
| DeepSeek-V3 | 72.80 | 62.27 | 66.67 | 77.07 | 62.67 | 64.40 | 64.80 | 61.87 | 66.57 |
| Skywork-Reward-Gemma2-27B | 70.40 | 61.60 | 66.10 | 74.10 | 64.00 | 60.00 | 62.70 | 58.40 | 64.70 |
| Qwen2.5-14B + **Ours** | **73.45** | **80.90** | **72.44** | **85.67** | **72.89** | **75.24** | **76.80** | **74.21** | **76.88** |
| Qwen2.5-72B + **Ours** | **82.30** | **89.01** | **79.76** | **89.87** | **79.82** | **82.12** | **78.10** | **75.23** | **81.99** |
| Qwen3-14B + **Ours** | **86.53** | **87.96** | **83.69** | **92.24** | **75.27** | **81.04** | **78.72** | **75.78** | **82.65** |

### Consistency and Position Bias

| Model | Original Consistency | +Ours Consistency |
| :--- | :--- | :--- |
| Qwen2.5-14B-Instruct | 69.97% | **74.17%** |
| Llama3.1-8B-Instruct | 60.36% | **68.19%** |
| Qwen2.5-72B-Instruct | 78.86% | 78.79% |
| Qwen3-14B-Instruct | 81.23% | 81.30% |

### Key Findings
- SenseJudge improves performance by an average of +16.08% over baselines; even with 8B/14B small models, it outperforms direct judgments from strong models like GPT-4o.
- Improvements are observed across all 8 categories, with the largest gains in Code (+20.10) and Trans (+18.84).
- Reward models (INF-ORM-70B, QRM-27B) achieve accuracy <65% on personalized datasets, indicating that fixed preferences struggle to generalize.
- SenseJudge significantly mitigates position bias, particularly benefiting smaller models.
- It achieves 90.55% on RewardBench, close to the specially trained Skywork-Critic (92.2%), verifying its general effectiveness.
- Model ranking results align with Arena human rankings: DeepSeek-R1 > Claude-3-7-Sonnet > GPT-4o > Qwen2.5-72B > GPT-3.5.

## Highlights & Insights
- The three-step process of preference extraction, subset selection, and voting is elegant and simple, achieving personalization without retraining judge models.
- The concept of "learning from failure"—inferring preferences backward from a small number of labels—is more data-efficient than direct reward model training.
- The SenseBench construction method (strong/weak model comparison + human verification) ensures the discriminative power of the evaluation benchmark.
- The study demonstrates that **Small Model + Good Preferences > Large Model + No Preferences**, providing a new path for low-cost deployment.

## Limitations & Future Work
- Preference construction depends on strong models like DeepSeek-R1; preference quality is constrained by the generation model's capability (ablations confirm weaker models generate less effective preferences).
- Only 3 annotators were used with a limited scale (1000 items each); larger-scale annotation might reveal more diverse preference patterns.
- Preference subset selection requires traversing the combination space, leading to exponential computational growth as the preference set increases.
- Cross-domain preference transfer results are inconsistent (Math → Logic 78.62% vs. Math → Translation 61.83%).

## Related Work & Insights
- While training-based judges like Auto-j and PandaLM learn fixed preferences, the explicit preference text in SenseJudge is more flexible and interpretable.
- Personalized LLMs (OPPU / Multi-granularity interest prediction) focus on response personalization; SenseJudge focuses on judgment personalization—making them complementary.
- The preference voting mechanism can be extended to any evaluation scenario requiring multi-perspective aggregation (e.g., code review, content moderation).

## Rating
- **Novelty**: ⭐⭐⭐⭐ Explicit preference-driven personalized judgment is a meaningful new direction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model comparison + consistency/position bias analysis + ablation + cross-domain + RewardBench verification.
- **Writing Quality**: ⭐⭐⭐ The structure is complete, though some formula expressions could be more concise.
- **Value**: ⭐⭐⭐⭐ High practicality and implementation value for low-cost personalized evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)

</div>

<!-- RELATED:END -->
