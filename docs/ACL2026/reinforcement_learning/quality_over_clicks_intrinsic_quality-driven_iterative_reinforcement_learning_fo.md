---
title: >-
  [Paper Note] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion
description: >-
  [ACL 2026][Reinforcement Learning][Cold-Start] This paper proposes Cold-EQS, a query suggestion framework for cold-start e-commerce scenarios. It leverages answerability, factual accuracy…
tags:
  - "ACL 2026"
  - "Reinforcement Learning"
  - "Cold-Start"
  - "Query Suggestion"
  - "Quality-Driven Reward"
  - "Uncertainty Sampling"
  - "E-Commerce Dialogue"
date: 2026-05-08
content_hash: 263174c8f733696c
---

# Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion

**Conference**: ACL 2026
**arXiv**: [2603.22922](https://arxiv.org/abs/2603.22922)
**Code**: [GitHub](https://github.com/QiSun123/Cold-EQS)
**Area**: E-Commerce / Reinforcement Learning
**Keywords**: Cold-Start, Query Suggestion, Quality-Driven Reward, Uncertainty Sampling, E-Commerce Dialogue

## TL;DR

This paper proposes Cold-EQS, a query suggestion framework for cold-start e-commerce scenarios. It leverages answerability, factual accuracy, and information gain as intrinsic quality rewards, and employs iterative reinforcement learning to continuously optimize query suggestion quality, achieving a 6.81% online chatUV improvement.

## Background & Motivation

**Background**: Query Suggestion (QS) is a core component of e-commerce dialogue systems—in multi-turn interactions, AI assistants proactively provide clickable suggested queries to help users refine their needs with minimal effort. Existing generative approaches typically use LLMs to generate queries and align them via CTR models.

**Limitations of Prior Work**: (1) Generative methods rely heavily on large volumes of online click data (e.g., 20M+ interaction records) to train effective CTR models, which are unavailable during cold-start; (2) existing methods focus primarily on query **relevance and diversity**, neglecting the **intrinsic quality** of generated queries—suggestions may be unanswerable (beyond downstream agent capabilities), contain hallucinated facts (e.g., "buy an iPhone for one dollar"), or merely rephrase the user's original query without adding new information.

**Key Challenge**: During cold-start, click data is unavailable for training CTR models, yet query suggestion quality must still be continuously improved. Click signals are also inherently noisy—high-click queries do not necessarily lead to high-quality multi-turn interactions.

**Goal**: To continuously optimize query suggestion quality during the cold-start phase without relying on large-scale click data.

**Core Idea**: Instead of using CTR as reward, three intrinsic quality dimensions—answerability, factual accuracy, and information gain—are used as RL reward signals. Uncertainty sampling is combined to select hard samples from online data lacking click signals for iterative training.

## Method

### Overall Architecture

Cold-EQS is a four-stage iterative framework: (1) warm-start SFT with a small amount of click data; (2) GRPO-based reinforcement learning using quality-driven rewards; (3) uncertainty sampling to select hard samples from click-free online data; (4) multi-round iterative RL training for continuous optimization. The backbone model is Qwen3-4B.

### Key Designs

1. **Quality-Driven Reward**:

    - **Function**: Replaces CTR signals by providing intrinsic quality evaluation independent of click data.
    - **Mechanism**: Each rollout generates $k$ query suggestions; each suggestion $s_i$ is evaluated along three dimensions—answerability $r_{\mathrm{ans}}(s_i)$ (whether the downstream agent can answer it), factual accuracy $r_{\mathrm{fact}}(s_i)$ (absence of hallucinations), and information gain $r_{\mathrm{info}}(s_i)$ (whether new information is introduced). The overall reward is $r_q = r_f \cdot \frac{1}{k}\sum_{i=1}^k r_{\mathrm{ans}}(s_i) \cdot r_{\mathrm{fact}}(s_i) \cdot r_{\mathrm{info}}(s_i)$, with Qwen-30B-A3B serving as the reward evaluator.
    - **Design Motivation**: Click signals are unavailable in cold-start and are inherently noisy (e.g., "buy an iPhone for one dollar" may attract high clicks but yield low quality); intrinsic quality evaluation is more reliable.

2. **Uncertainty Sampling**:

    - **Function**: Selects the most informative hard samples from click-free online data.
    - **Mechanism**: For each online query $q$, the model generates $k$ suggestions and scores them. The reward variance $u_q = \frac{1}{k}\sum_{i=1}^k (R(s_i) - \frac{1}{k}\sum_j R(s_j))^2$ is computed. Queries with high uncertainty are selected as hard samples for the next round of RL training.
    - **Design Motivation**: Training exclusively on click data introduces bias; uncertainty sampling increases data diversity and directs attention to the model's weaknesses.

3. **Iterative RL Training**:

    - **Function**: Continuously optimizes query suggestion quality.
    - **Mechanism**: Each iteration proceeds as follows: the current model generates suggestions on click-free data → uncertainty sampling selects hard samples → RL training updates the model → repeat. Each round mixes click data and hard samples.
    - **Design Motivation**: A single round of RL training cannot cover all scenarios; iterative training continuously identifies and addresses model weaknesses.

### Loss & Training

The GRPO algorithm is used for RL training. Rewards adopt a multiplicative combination across three dimensions (answerability × factual accuracy × information gain × format), ensuring all dimensions are simultaneously satisfied. The SFT stage uses click-positive samples; the RL stage uses quality-driven rewards. The backbone is Qwen3-4B and the reward evaluator is Qwen-30B-A3B.

## Key Experimental Results

### Main Results

| Model | Strict Accuracy | Valid Rate | Notes |
|-------|----------------|------------|-------|
| GPT-4.1-mini | 70.5 | 79.2 | Among the best closed-source models |
| Qwen-flash | 75.4 | 81.3 | Best closed-source model |
| Qwen3-4B (base) | 36.7 | 53.2 | Open-source baseline |
| Cold-EQS (Qwen3-4B) | Significant improvement | Significant improvement | Approaches or surpasses closed-source models |
| **Online Metric** | **+6.81% chatUV** | | Validated in real deployment |

### Key Findings
- Offline metrics and online performance are strongly positively correlated, validating the effectiveness of intrinsic quality evaluation.
- In cold-start scenarios, quality-driven rewards are more reliable than click-driven rewards.
- Uncertainty sampling effectively mitigates the bias introduced by relying solely on click data.
- Iterative RL training yields continuous performance improvements.
- A 4B model trained with Cold-EQS can approach or surpass large closed-source models.

## Highlights & Insights
- **Paradigm Shift from Clicks to Quality**: Replacing CTR signals with intrinsic quality rewards is better suited for cold-start and more robust.
- **Rationale of Three-Dimensional Quality Evaluation**: Answerability, factual accuracy, and information gain precisely cover the core quality dimensions of query suggestions.
- **Offline–Online Consistency**: The strong positive correlation between offline quality metrics and online chatUV makes offline iteration reliable.
- **Industrial Deployment Validation**: Beyond academic experimentation, effectiveness is validated on Alibaba's international e-commerce system.
- **Contribution of EQS-Benchmark**: Provides a standardized evaluation benchmark for e-commerce query suggestion.

## Limitations & Future Work
- The reward evaluator (Qwen-30B-A3B) may itself introduce bias; its quality directly affects training outcomes.
- Experiments are primarily conducted on Alibaba's international e-commerce platform; generalizability across platforms and languages remains unknown.
- Uncertainty sampling relies on a simple variance metric; more sophisticated uncertainty estimation methods (e.g., Bayesian approaches) may be more effective.
- EQS-Benchmark is relatively small (16,949 samples); larger-scale benchmarks remain to be constructed.

## Related Work & Insights
- **vs. CTR-based QS (Min et al., 2025)**: CTR-based methods require 20M+ online records; Cold-EQS operates effectively during cold-start.
- **vs. Retrieval-based Methods**: Retrieval-based methods are constrained by historical query pools and cannot generate novel suggestions; Cold-EQS can produce diverse new queries.
- **vs. Standard SFT**: Pure SFT amplifies click noise (e.g., "buy an iPhone for one dollar"); RL with quality rewards avoids this problem.

## Rating
- Novelty: ⭐⭐⭐⭐ An intrinsic quality-driven cold-start RL framework for e-commerce QS is novel in this domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes both offline and online experiments with multi-model comparisons, though ablation details could be richer.
- Writing Quality: ⭐⭐⭐ Structure is reasonable, but author anonymization is incomplete and some descriptions could be clearer.
- Value: ⭐⭐⭐⭐ Addresses a practical industrial problem; the 6.81% online improvement validates practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](../../ICLR2026/reinforcement_learning/from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[ICLR 2026\] AutoQD: Automatic Discovery of Diverse Behaviors with Quality-Diversity Optimization](../../ICLR2026/reinforcement_learning/autoqd_automatic_discovery_of_diverse_behaviors_with_quality-diversity_optimizat.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](../../ICLR2026/reinforcement_learning/reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[ICLR 2026\] Post-training Large Language Models for Diverse High-Quality Responses](../../ICLR2026/reinforcement_learning/post-training_large_language_models_for_diverse_high-quality_responses.md)
- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](../../ICLR2026/reinforcement_learning/metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)

</div>

<!-- RELATED:END -->
