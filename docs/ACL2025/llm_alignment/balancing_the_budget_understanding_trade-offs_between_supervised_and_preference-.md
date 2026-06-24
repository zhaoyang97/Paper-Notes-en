---
title: >-
  [Paper Note] Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning
description: >-
  [ACL 2025][LLM Alignment][Supervised Finetuning] This paper systematically investigates how to optimally allocate resources between Supervised Finetuning (SFT) and Preference Finetuning (PFT/DPO) under a fixed data annotation budget. It reveals that pure SFT is optimal at low data regimes, while a combined approach performs best at high budgets. Furthermore, allocating less than 10% of the budget to SFT can resolve the cold start problem of DPO…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Supervised Finetuning"
  - "Preference Optimization"
  - "Data Budget Allocation"
  - "Cold Start Problem"
  - "DPO"
date: 2026-05-08
content_hash: fae10a61c8a4b09f
---

# Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning

**Conference**: ACL 2025  
**arXiv**: [2502.11284](https://arxiv.org/abs/2502.11284)  
**Code**: None  
**Area**: Others  
**Keywords**: Supervised Finetuning, Preference Optimization, Data Budget Allocation, Cold Start Problem, DPO

## TL;DR

This paper systematically investigates how to optimally allocate resources between Supervised Finetuning (SFT) and Preference Finetuning (PFT/DPO) under a fixed data annotation budget. It reveals that pure SFT is optimal at low data regimes, while a combined approach performs best at high budgets. Furthermore, allocating less than 10% of the budget to SFT can resolve the cold start problem of DPO, bringing a 15-20% performance gain in mathematical reasoning.

## Background & Motivation

- **Background**: The post-training of large language models (LLMs) typically follows a two-stage "SFT $\rightarrow$ PFT" pipeline—first teaching the model to follow instructions via supervised finetuning, and then improving output quality and alignment via preference finetuning (e.g., DPO, RLHF). This has become the standard paradigm for LLM development.
- **Limitations of Prior Work**: SFT data (instruction-response pairs) and PFT data (preference pairs/ranking data) fundamentally differ in structure, acquisition cost, and annotation difficulty. In practical development, annotation budgets are limited, with high-quality data annotation constituting a significant portion of the total cost of LLM development. However, systematic research and guidance are lacking on how to allocate this limited budget between the two stages.
- **Key Challenge**: Intuitively, both SFT and PFT have their respective advantages, but how do their marginal return curves vary with data volume? Under what conditions should SFT data be prioritized? Under what conditions is preference data more valuable? Clear empirical answers to these questions are missing.
- **Goal**: Through large-scale experiments, this work aims to provide a practical guide for SFT and PFT data allocation, helping researchers and practitioners make optimal decisions under budget constraints.
- **Key Insight**: Systematic experiments are designed to span four diverse tasks, various model scales, and different annotation cost settings, keeping the total budget constant and varying only the SFT/PFT allocation ratio to directly measure performance changes.
- **Core Idea**: The optimal data budget allocation strategy depends on the data scale: SFT dominates when annotations are scarce, while preference data should be gradually increased with larger budgets. Furthermore, direct DPO training on base models suffers from a severe "cold-start problem," which can be effectively resolved with a minimal amount of SFT data (<10%).

## Method

### Overall Architecture

This work is a systematic empirical study rather than proposing a new method. The core experimental design is as follows: given a fixed budget $N$, it is allocated into an SFT budget $N_s$ and a PFT budget $N_p$ ($N_s + N_p = N$). The performance is evaluated across various tasks and model scales by sweeping through different allocation ratios (e.g., 100:0, 90:10, 70:30, 50:50, 30:70, 10:90, 0:100).

### Key Designs

1. **Multi-Task Multi-Scale Design**: The experiments cover four types of tasks—mathematical reasoning (GSM8k), general instruction following, summarization, and dialogue generation. The model sizes range from small to large (e.g., 1B, 3B, 7B). The annotation cost settings account for different unit prices of SFT and preference data (preference data usually requires generating multiple candidates and manual comparison, making it more expensive). This comprehensive experimental matrix ensures the generalizability of the findings.
2. **Cold Start Problem Analysis**: Directly running DPO on a base model (skipping the SFT stage) is found to cause severe performance degradation, particularly in mathematical reasoning tasks. Analysis shows that this is caused by "distribution shift"—the base model's output distribution differs drastically from the distribution of the chosen/rejected pairs in the DPO training data, preventing the contrastive learning signals of DPO from propagating effectively.
3. **Minimum SFT Budget Threshold**: Sampler experiments show that allocating less than 10% of the total budget to SFT can effectively resolve the cold-start problem. This small amount of SFT acts as a "distribution bridge"—shifting the model's output distribution from the pre-training distribution closer to the instruction-following distribution, which enables effective subsequent DPO training.

### Loss & Training

- **SFT Stage**: Standard causal language modeling loss, trained on instruction-response pairs: $\mathcal{L}_{SFT} = -\sum_t \log p_\theta(y_t | x, y_{<t})$
- **PFT Stage**: DPO (Direct Preference Optimization) loss: $\mathcal{L}_{DPO} = -\mathbb{E}_{(x,y_w,y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$
- **Reference Model**: The reference model for DPO is the model trained after the SFT stage (or the base model itself when the SFT ratio is 0).
- **Hyperparameters**: Consistent learning rate schedules and regularization settings are used across experiments, with the $\beta$ parameter of DPO being tuned.

## Key Experimental Results

### Main Results

Comparison of the optimal performance under different SFT/PFT allocation ratios across various data budget scales.

| Total Budget | Optimal SFT:PFT Ratio | Optimal Task Performance | Pure SFT Performance | Pure PFT Performance | Notes |
|---|---|---|---|---|---|
| <500 | 100:0 | Baseline | Optimal | Very Poor | Low data regime, pure SFT is optimal |
| 500-1000 | 80:20~70:30 | +5-8% vs Pure SFT | Baseline | Poor | Starts benefiting from a small amount of preference data |
| 1000-5000 | 50:50~30:70 | +10-15% vs Pure SFT | Baseline | Catching up | Preference data ratio gradually increases |
| >5000 | 20:80~10:90 | +15-25% vs Pure SFT | Saturated | Near-optimal | More budget should be allocated to preference data |
| >5000 (No SFT) | 0:100 | Lower than 10:90 | - | Cold start problem | Skipping SFT entirely leads to degradation |

### Ablation Study

Taking the GSM8k mathematical reasoning task as an example to demonstrate the impact of the cold start problem:

| Configuration | GSM8k Accuracy | Notes |
|------|------------|------|
| Pure DPO (0% SFT) | ~25% | Severe cold start problem |
| 5% SFT + 95% DPO | ~42% | Tiny amount of SFT significantly improves performance |
| 10% SFT + 90% DPO | ~45% | Near-optimal |
| 30% SFT + 70% DPO | ~43% | Too much SFT, insufficient preference data |
| 100% SFT (0% DPO) | ~35% | No preference alignment |
| Model scale impact (7B vs 3B) | +8-12% | Larger models leverage preference data better |

### Key Findings

- **SFT dominates in low-data regimes** (<1000 annotated samples): When annotated data is extremely scarce, investing the entire budget into SFT is the optimal strategy. Preference data inherently requires a sufficient sample size to provide effective contrastive signals.
- **Preference data proportion should gradually increase at higher budgets**: As the budget grows, the marginal returns of SFT diminish, while the benefits of preference finetuning gradually emerge. The optimal strategy is to allocate more (70-90%) of the budget to preference data.
- **The cold start problem is real**: On tasks requiring step-by-step reasoning like GSM8k, directly running DPO on the base model leads to a severe performance drop. The root cause is that the base model lacks the "step-by-step reasoning" format, causing the reference distribution of DPO to deviate severely.
- **A minimal amount of SFT resolves the cold start**: Allocating less than 10% of the SFT budget (even just dozens of high-quality samples) allows the model to learn basic instruction-following formats. This lays the foundation for subsequent DPO, yielding a 15-20% improvement in mathematical reasoning.
- **Task sensitivity**: The cold start problem is most severe in analytical tasks (math, logic) and milder in open-ended dialogue tasks, which correlates with the tasks' dependency on specific output formats.

## Highlights & Insights

- **Highly practical guidance**: This work directly addresses the practical question of "how to allocate annotation budgets," which both industry and academia frequently face, providing clear and actionable conclusions.
- **In-depth analysis of the cold start problem**: The paper not only identifies the problem but also traces it back to the root cause of distribution shift, offering an extremely simple solution (a micro-dose of SFT).
- **Comprehensive experimental design**: Cross-experiments spanning multiple tasks, scales, and budget levels guarantee the robustness and generalizability of the findings.
- **Challenging the popular view that "SFT is optional"**: Some recent studies claim that preference alignment can be performed directly on base models. This paper provides empirical data showing that a minimal amount of SFT remains essential, at least for certain key tasks.

## Limitations & Future Work

- The experiments are based on DPO as the preference finetuning method; other methods (such as KTO, IPO, ORPO) may exhibit different behaviors.
- The annotation cost model is relatively simplified and does not consider the heterogeneity of data quality (the same quantity of annotations may differ greatly in quality).
- Multi-turn iterative strategies (e.g., SFT $\rightarrow$ DPO $\rightarrow$ SFT $\rightarrow$ DPO) were not explored.
- The choice of base models may affect the optimal allocation ratio—stronger base models with extensive pre-training might benefit from preference data earlier.
- Future work could construct a predictive model for data allocation, automatically recommending the optimal budget ratio based on task types and model scales.

## Related Work & Insights

- **vs training recipes like Zephyr/Tulu**: These works demonstrate the effectiveness of the SFT+DPO pipeline but do not systematically study the budget allocation between the two stages. This paper fills this gap.
- **vs SPIN/Self-Play methods**: These methods attempt to use synthetic data to reduce reliance on human preference annotations, alleviating budget pressure from another angle.
- **vs unified methods like ORPO/SimPO**: These approaches attempt to merge SFT and preference alignment into a single stage, but the findings in this paper suggest that the two-stage decoupled design still holds significant value.

## Rating

- Novelty: ⭐⭐⭐ The research question is important, but methodologically it is a systematic empirical study rather than a new method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The comprehensive experimental matrix of four tasks $\times$ multiple scales $\times$ multiple budgets is highly impressive.
- Writing Quality: ⭐⭐⭐⭐ The conclusions are clearly formulated, and experimental charts are highly informative.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for LLM development practices; the discovery and solution for the cold start problem are directly applicable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization](fine-grained_video_dubbing_duration_alignment_with_segment_supervised_preference.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2026\] MAESTRO: Meta-learning Adaptive Estimation of Scalarization Trade-offs for Reward Optimization](../../ACL2026/llm_alignment/maestro_meta-learning_adaptive_estimation_of_scalarization_trade-offs_for_reward.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ICLR 2026\] Cognitive models can reveal interpretable value trade-offs in language models](../../ICLR2026/llm_alignment/cognitive_models_can_reveal_interpretable_value_trade-offs_in_language_models.md)

</div>

<!-- RELATED:END -->
