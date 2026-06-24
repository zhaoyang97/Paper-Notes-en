---
title: >-
  [Paper Note] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization
description: >-
  [ACL 2025][LLM Alignment][Reward Model Evaluation] This paper revisits reward model evaluation through the lens of reward overoptimization, finding that existing benchmarks correlate weakly with downstream policy performance. To address this, it proposes three key design principles for building reliable benchmarks: minimizing non-correctness confounding differences in positive/negative samples, using multiple comparisons to cover a wide response spectrum…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Reward Model Evaluation"
  - "Reward Overoptimization"
  - "RLHF"
  - "Benchmark Design"
  - "Human Preference Alignment"
date: 2026-05-08
content_hash: 8a05202734da0c1a
---

# Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization

**Conference**: ACL 2025  
**arXiv**: [2505.12763](https://arxiv.org/abs/2505.12763)  
**Code**: None  
**Area**: RLHF Alignment  
**Keywords**: Reward Model Evaluation, Reward Overoptimization, RLHF, Benchmark Design, Human Preference Alignment

## TL;DR

This paper revisits reward model evaluation through the lens of reward overoptimization, finding that existing benchmarks correlate weakly with downstream policy performance. To address this, it proposes three key design principles for building reliable benchmarks: minimizing non-correctness confounding differences in positive/negative samples, using multiple comparisons to cover a wide response spectrum, and sampling responses from diverse models.

## Background & Motivation

**Background**: Reward Models (RMs) play a central role in RLHF, aligning language model behavior with human preferences. Currently, the community relies on various RM evaluation benchmarks, such as RewardBench, which typically measure RM quality through classification accuracy over chosen/rejected pairs.

**Limitations of Prior Work**: Existing RM benchmarks show a weak correlation with the performance of policy models optimized through RLHF in practice. This implies that an RM scoring highly on benchmarks does not necessarily yield a better policy during reinforcement learning, casting doubt on the "validity" of these benchmarks.

**Key Challenge**: Benchmark evaluations focus on the RM's ability to discriminate within a single chosen-rejected pair, whereas in actual RLHF, the RM must provide stable and accurate learning signals throughout the policy optimization process. Static, single-pair discrimination accuracy fails to reflect the RM's true performance in dynamic optimization, where RM robustness becomes critical as the policy continuously "chases" high rewards.

**Goal**: (1) Identify the design flaws that cause existing benchmarks to fail; (2) propose evaluation design principles based on the perspective of reward overoptimization; and (3) construct an RM evaluation methodology highly correlated with downstream performance.

**Key Insight**: The authors analyze the phenomenon of reward overoptimization—where optimizing a policy too heavily against a proxy reward model causes the proxy reward to increase while actual human satisfaction decreases. This phenomenon captures both the alignment quality between the RM and human preferences, and the dynamic nature of the learning signals provided by the RM.

**Core Idea**: Use the degree of reward overoptimization as a proxy metric for measuring RM quality, thereby deriving the design principles that evaluation benchmarks should satisfy.

## Method

### Overall Architecture

The research framework systematically explores the design space of RM evaluation benchmarks. The process is as follows: (1) Select multiple RMs; (2) obtain optimized policies for each RM using Best-of-N sampling or RLHF training; (3) evaluate the true downstream performance of these policies; (4) compare the correlation (Spearman/Kendall correlation) between the RM rankings under different benchmark designs and the downstream rankings; and (5) derive effective benchmark design principles from the results.

### Key Designs

1. **Minimal Confounding Differences**:

    - **Function**: Ensure that differences between chosen and rejected samples only reflect "correctness".
    - **Mechanism**: If chosen responses are systematically longer, better styled, or formatted more beautifully than rejected ones, the RM might exploit these "shortcuts" instead of truly understanding semantic correctness. The authors construct pairs with minimal confounding factors by controlling for response length and source models.
    - **Design Motivation**: In traditional benchmarks, chosen samples often originate from stronger models (thus being naturally longer and more fluent), while rejected ones come from weaker models. This systematic difference allows RMs to cheat based on superficial features, distorting benchmark rankings.

2. **Multiple Comparisons Across Diverse Responses**:

    - **Function**: Construct multiple pairwise comparisons by generating historical responses of varying quality for the same prompt.
    - **Mechanism**: Instead of relying on a single chosen-rejected pair, multiple responses are collected for each prompt (either from different models or different samplings of the same model). The overall accuracy is computed across all pairwise comparisons. This closely simulates the actual operational scenario of RMs in RLHF, where the policy generates diverse responses.
    - **Design Motivation**: Single comparisons exhibit high variance and instability. Multiple comparisons test the RM's ranking capacity more comprehensively and reduce random noise.

3. **Diverse Model Sources**:

    - **Function**: Sample responses from multiple different LLMs to construct evaluation data.
    - **Mechanism**: In practical RLHF training, policies at different stages produce responses with distinct styles. If evaluation relies solely on outputs from one or two models, the RM may overfit to specific styles. Mixing responses from various models (e.g., GPT-4, Llama, Mistral) tests the RM's generalization ability across diverse "textual representations".
    - **Design Motivation**: RMs must maintain stable discriminative power within a diverse response space, which cannot be adequately captured by single-source evaluation data.

### Loss & Training

This paper does not propose a new training algorithm but is a study on evaluation methodology. The core experiments employ Best-of-N sampling and PPO-based RLHF to obtain optimized policies, which are then evaluated using metrics like AlpacaEval 2.0 and MT-Bench to calculate rank correlation with various benchmark designs.

## Key Experimental Results

### Main Results

The authors compare the correlation (Spearman $\rho$) between different evaluation benchmark designs and downstream policy performance:

| Benchmark Design | Correlation with BoN Performance | Correlation with PPO Performance | Notes |
|----------|------------------|------------------|------|
| RewardBench (Original) | ~0.3 | ~0.2 | Existing benchmark, weak correlation |
| Single Source + Single Comparison | ~0.4 | ~0.3 | Limited improvement |
| Multi-source + Multiple Comparisons + Controlling Confounding | ~0.8 | ~0.7 | Recommended design in this paper |
| Extremely High Overoptimization Correlation Design | ~0.9 (with overoptimization) | ~0.5 (with downstream) | Over-optimization actually reduces downstream correlation |

### Ablation Study

| Design Variable | Correlation Change | Notes |
|---------|----------|------|
| Controlling Response Length Differences | +15-20% | Significant improvement after eliminating the length confounding factor |
| Increasing Number of Comparisons (1→10) | +10-15% | Multiple comparisons reduce variance |
| Diverse Model Sources (1→5) | +8-12% | Source diversity improves generalization |
| Using Overoptimization Rank Only | High correlation with overoptimization but moderate with downstream | Shows that overoptimization level is a tool, not the goal |

### Key Findings

- Controlling confounding differences is the most critical single factor—superficial traits like length differences severely mislead RM evaluation rankings.
- Although diverse model sources and multiple comparisons contribute less individually than controlling confounding variables, their combination yields the best results.
- Excessively chasing a high correlation with the degree of overoptimization can conversely degrade the correlation with some downstream tasks, indicating that the overoptimization degree should serve as an auxiliary tool rather than the sole objective.
- These insights directly guide RM benchmark designs: existing benchmarks like RewardBench need to be redesigned under these principles.

## Highlights & Insights

- **Revisiting evaluation through the lens of overoptimization is a clever point of entry**: Examining the behavior of RMs within optimization dynamics reflects actual RLHF performance much better than static accuracy. This "begin with the end in mind" thinking is instructive.
- **The three design principles are clear and practical**: Controlling confounding, multiple comparisons, and multi-source sampling. These principles are not only suitable for RM evaluation, but also benchmark designs involving any model scoring or ranking.
- **The observation that overoptimization is "useful but should not be the ultimate goal" is insightful**: It reveals the non-linear relationship between proxy indicators and ultimate objectives, reminding the community to avoid the trap of Goodhart's Law.

## Limitations & Future Work

- The experiments are primarily based on Best-of-N and PPO, without covering newer alignment methods like DPO and KTO, which rely differently on RMs and might require extended validation.
- Downstream evaluations rely on automatic metrics like AlpacaEval 2.0 and MT-Bench, which themselves may contain bias.
- The principles found in this paper are highly qualitative; there is a lack of a unified, ready-to-use new benchmark dataset.
- Future work can construct a standardized, next-generation RM benchmark based on these principles and investigate the applicability across different alignment algorithms.

## Related Work & Insights

- **vs RewardBench**: RewardBench uses fixed chosen-rejected pairs for classification accuracy evaluation. This paper points out its severe confounding issues, which lead to low correlation with downstream performance.
- **vs RLHF-Reward-Bench**: Similar evaluation benchmarks face the same issues; the three principles proposed here offer a systematic improvement path.
- **vs Overoptimization Theoretical Work (Gao et al., 2023)**: This paper repurposes overoptimization from "an issue to avoid" into "a useful tool for evaluating RMs," representing a valuable shift in perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ Examining evaluation via overoptimization is a novel entry point, though the core findings (e.g., controlling confounding) are somewhat intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematically ablates multiple dimensions of evaluation design, but lacks validation on more alignment algorithms.
- Writing Quality: ⭐⭐⭐⭐ Logical exposition with a smooth flow from the problem identification to findings and principles.
- Value: ⭐⭐⭐⭐ Directly instructive to the RM evaluation benchmark community, but needs to be further instantiated as a concrete benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RewardBench 2: Advancing Reward Model Evaluation](../../ICLR2026/llm_alignment/rewardbench_2_advancing_reward_model_evaluation.md)
- [\[ACL 2025\] HAF-RM: A Hybrid Alignment Framework for Reward Model Training](haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)
- [\[ACL 2025\] Rethinking Table Instruction Tuning](rethinking_table_instruction_tuning.md)
- [\[ACL 2025\] Towards Reward Fairness in RLHF: From a Resource Allocation Perspective](reward_fairness_rlhf.md)
- [\[ACL 2025\] Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch](cheems_chinese_reward_models.md)

</div>

<!-- RELATED:END -->
