---
title: >-
  [Paper Note] DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization
description: >-
  [ACL 2025][Model Compression][structured pruning] DRPruning introduces distributionally robust optimization (DRO) to LLM structured pruning. By leveraging scaling laws to predict the final loss of each domain as a reference and dynamically adjusting the training data distribution to balance post-pruning domain performance, it surpasses Sheared LLaMA by -5.59% PPL and +2.95% on downstream tasks in monolingual and multilingual settings, respectively.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "structured pruning"
  - "distributionally robust optimization"
  - "data scheduling"
  - "scaling law"
  - "domain balance"
date: 2026-05-08
content_hash: f8bb15015d1038a2
---

# DRPruning: Efficient Large Language Model Pruning through Distributionally Robust Optimization

**Conference**: ACL 2025  
**arXiv**: [2411.14055](https://arxiv.org/abs/2411.14055)  
**Code**: [https://github.com/hexuandeng/DRPruning](https://github.com/hexuandeng/DRPruning)  
**Area**: Model Compression  
**Keywords**: structured pruning, distributionally robust optimization, data scheduling, scaling law, domain balance  

## TL;DR
DRPruning introduces distributionally robust optimization (DRO) to LLM structured pruning. By leveraging scaling laws to predict the final loss of each domain as a reference and dynamically adjusting the training data distribution to balance post-pruning domain performance, it surpasses Sheared LLaMA by -5.59% PPL and +2.95% on downstream tasks in monolingual and multilingual settings, respectively.

## Background & Motivation
**Background**: Structured pruning (e.g., Sheared LLaMA) can compress large models into smaller ones, but usually requires continued pretraining afterward to restore capabilities.

**Limitations of Prior Work**:
   - Performance degradation across different domains after pruning is uneven—some domains recover quickly, while others are severely damaged, leading to bias.
   - Standard DRO requires manually setting crucial hyperparameters (reference loss and reference data ratio), and poor settings lead to suboptimal performance.
   - The dynamic scheduling strategy of Sheared LLaMA relies on the loss ratios of domains in the large model, which becomes ineffective in scenarios with large distribution shifts, such as multilingual settings.

**Key Challenge**: How to automatically balance the performance of various domains during post-pruning continued pretraining without extensive hyperparameter tuning?

**Goal**: Automate the determination of the reference loss and reference data ratio in DRO.

**Key Insight**: Use scaling laws to predict the loss at the end of training as the reference loss, and use the EMA (Exponential Moving Average) of DRO weights to update the reference data ratio.

**Core Idea**: Automatically predict the optimal attainable loss for each domain using scaling laws, combined with progressive data ratio adjustment, to achieve a balanced recovery across all domains after pruning.

## Method

### Overall Architecture
Built upon the structured pruning + continued pretraining framework of Sheared LLaMA, DRPruning incorporates three improvements: (1) dynamically adjusting training data ratios using DRO; (2) predicting the reference loss with scaling laws; and (3) progressively updating the reference data ratio.

### Key Designs

1. **Dynamic Reference Loss based on Scaling Laws**:

    - **Function**: Predict the lowest loss each domain can achieve by the end of training, serving as the benchmark reference for DRO.
    - **Mechanism**: Fit the loss curve of each domain using $\hat{\ell}(P, T) = A \cdot P^{-\alpha} \cdot T^{-\beta} + E$, and use the predicted value at the total training steps from the fitted curve as the reference loss. This curve is refitted after each evaluation.
    - **Design Motivation**: Setting the reference loss manually is difficult and unreliable. Scaling laws provide a data-driven prediction, avoiding manual parameter tuning. Predictions begin after 20% of the training is complete (to ensure sufficient data points).

2. **Progressive Reference Data Ratio Update**:

    - **Function**: Allow the constraint center of the data distribution to progressively shift toward high-loss domains.
    - **Mechanism**: $\mathbf{p}_R^{t+1} = \delta \cdot \mathbf{q}^t + (1-\delta) \cdot \mathbf{p}_R^t$, where $\mathbf{q}^t$ represents the optimal weights computed by DRO. An EMA is used for smooth updates to prevent drastic fluctuations.
    - **Design Motivation**: A fixed reference ratio is overly conservative (Zhou et al., 2021), limiting DRO's focus on difficult domains. Progressive updates allow gradual expansion of coverage to challenging distributions.
    - **Safety Constraint**: The ratio of each domain is restricted to within $[1/n, n]$ times its initial ratio to prevent degenerating into training only the worst-performing domain.

3. **DRO Weight Update**:

    - **Function**: Adjust data sampling weights based on the "excess loss" (current loss minus reference loss) of each domain.
    - **Mechanism**: Assign higher weights to domains whose loss deviates more from their reference, solving a maximization problem constrained within a $\chi^2$-divergence ball.
    - **Difference from Sheared LLaMA**: Sheared LLaMA allocates weights based on the absolute loss rankings of the large model, whereas DRPruning aligns with the model's own attainable optimal values.

### Loss & Training
- Pruning is performed on the first 0.4B tokens (to learn the pruning mask), followed by 50B tokens of continued pretraining.
- Pruning is conducted from Llama2-7B to two target scales: 1.3B and 2.7B.
- DRO updates are executed during each evaluation.

## Key Experimental Results

### Main Results

| Method | 7B→1.3B PPL | 7B→1.3B Task Avg. | 7B→2.7B PPL | 7B→2.7B Task Avg. |
|------|:----------:|:-----------------:|:----------:|:-----------------:|
| Sheared LLaMA | 10.05 | 34.89 | 7.64 | 39.75 |
| ReSheared (Replication) | 10.42 | 34.85 | 7.83 | 39.98 |
| **DRPruning** | **9.83** | **35.60** | **7.40** | **40.18** |

PPL decreases by -5.59%, and downstream tasks gain +1.52%.

**Instruction Tuning Win Rate (vs Sheared LLaMA, evaluated by GPT-4o)**: 55.4% win rate.

### Ablation Study

| Config | PPL | Task | Description |
|------|:---:|:----:|------|
| DRPruning (full) | 9.83 | 35.60 | Full method |
| w/o dynamic ref loss | 10.15 | 35.20 | Scaling law prediction is effective |
| w/o dynamic ref ratio | 10.02 | 35.35 | Progressive ratio update is effective |
| Fixed ref ratio | 10.20 | 35.10 | Fixed ratio is significantly inferior to dynamic |

### Key Findings
- The advantage is even more pronounced in multilingual settings (+2.95% on downstream tasks), demonstrating that DRPruning is particularly effective in scenarios with large distribution shifts.
- The reference loss predicted by scaling laws remains numerically stable, which is a key prerequisite for the method's effectiveness.
- Domain-level evaluations show that DRPruning yields the greatest improvement in the worst-performing domain (+17.9%), aligning with the worst-case optimization objective of DRO.

## Highlights & Insights
- **Scaling Law-Driven DRO**: Using scaling laws to automatically predict reference loss provides an elegant automated solution, shifting DRO from "requiring expert knowledge to set hyperparameters" to being completely data-driven.
- **Progressive Constraint Relaxation**: Gradually shifting the reference ratio toward difficult domains balances exploration (trying new distributions) and exploitation (training on known good distributions).
- **Not Limited to Pruning**: The reference loss and data ratio optimization methods can be used independently of pruning and are applicable to any multi-domain continued pretraining.

## Limitations & Future Work
- **Evaluated Only on Llama2-7B**: The effectiveness on larger models and other architectures (e.g., Mistral, Qwen) remains unknown.
- **Scaling Law Fitting Requires Sufficient Data Points**: The dynamic reference loss cannot be utilized during the initial 20% of training, which may be insufficient for short training schedules.
- **Predefined Domain Division**: The data must be split into explicit domains in advance, which is inapplicable to unlabeled data.

## Related Work & Insights
- **vs Sheared LLaMA**: Also built on structured pruning + continued pretraining, but with a superior data scheduling strategy—automated DRO vs. heuristic scheduling.
- **vs Group DRO**: DRPruning resolves the two core hyperparameter issues of Group DRO (reference loss + reference ratio), making it practically applicable in LLM settings.
- The method of automatically determining reference loss can be generalized to multi-task training in DPO/RLHF.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of scaling law + DRO is novel, and automated hyperparameter configuration has practical value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-dimensional evaluation across monolingual/multilingual, PPL, downstream, and instruction tuning settings.
- Writing Quality: ⭐⭐⭐⭐ Clear method description and intuitive diagrams.
- Value: ⭐⭐⭐⭐ Provides best practices for structured pruning, and automated DRO holds broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[ACL 2025\] BlockPruner: Fine-grained Pruning for Large Language Models](blockpruner_fine-grained_pruning_for_large_language_models.md)
- [\[ICLR 2026\] GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching](../../ICLR2026/model_compression/gptailor_large_language_model_pruning_through_layer_cutting_and_stitching.md)
- [\[ACL 2025\] Quantification of Large Language Model Distillation](quantification_of_large_language_model_distillation.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)

</div>

<!-- RELATED:END -->
