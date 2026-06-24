---
title: >-
  [Paper Note] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs
description: >-
  [ACL 2025][LLM Safety] This paper proposes LIBU (LoRA-enhanced Influence-Based Unlearning), which achieves machine unlearning for LLMs in two phases: Phase 1 utilizes influence function updates weighted by the diagonal Fisher information matrix for precise unlearning, and Phase 2 stabilizes training using the Sophia second-order optimizer. On OLMo-7B in SemEval-2025 Task 4, this method achieves an unlearning rate of 0.283 while maintaining an MMLU accuracy of 0.469.
tags:
  - "ACL 2025"
  - "LLM Safety"
date: 2026-05-08
content_hash: 3a4647e153fbcaba
---

# Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs

**Conference**: ACL 2025  
**arXiv**: [2506.04044](https://arxiv.org/abs/2506.04044)  
**Code**: None  
**Area**: Model Compression  

## TL;DR

This paper proposes LIBU (LoRA-enhanced Influence-Based Unlearning), which achieves machine unlearning for LLMs in two phases: Phase 1 utilizes influence function updates weighted by the diagonal Fisher information matrix for precise unlearning, and Phase 2 stabilizes training using the Sophia second-order optimizer. On OLMo-7B in SemEval-2025 Task 4, this method achieves an unlearning rate of 0.283 while maintaining an MMLU accuracy of 0.469.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: 1. **GDPR and other privacy regulations mandate the "right to be forgotten"**: Post-deployment LLMs must selectively remove the influence of specific training data, but full retraining is computationally prohibitive for large models.
2. **Existing unlearning methods compromise general capabilities**: Techniques such as gradient ascent successfully unlearn target data but severely degrade model performance on benchmarks like MMLU.
3. **SemEval-2025 Task 4 Challenges**: It requires completing the unlearning process for OLMo models within 1 hour while maintaining MMLU > 0.371, demanding a balance between unlearning efficacy and model utility.

## Method

### Phase 1: Influence Function Updates

1. Approximate the diagonal elements of the Fisher information matrix $F_{ii}$ using the mean of the squared gradients on the retain set.
2. Compute the parameter importance weight: $w_\theta = \frac{1}{F_{ii} + \lambda}$ ($\lambda=10^{-3}$ to prevent division by zero).
3. Compute gradients $g_{forget}$ on the forget set.
4. Update parameters: $\theta_{t+1} \leftarrow \theta_t - \eta \cdot w_{\theta_t} \cdot g_{forget}$

**Core Idea**: Parameters critical to the retain set (high $F_{ii}$) receive minor updates, while non-critical parameters are dynamically adjusted more aggressively to erase the influence of the forget set. LoRA is integrated for parameter-efficient fine-tuning throughout the process.

### Phase 2: Stabilization via Sophia Second-Order Optimizer

Fine-tune on the forget set using the Sophia optimizer, utilizing the diagonal Hessian estimation for an adaptive learning rate:
$$\Delta\theta_t = -\eta \cdot \frac{g_t}{\max(\gamma \cdot h_t, \epsilon)}$$

- $h_t$: Estimated curvature represented by the randomly sampled squared gradients.
- $\gamma$: Controls the level of conservatism (a higher $\gamma$ prevents over-updating critical parameters).
- Gradient accumulation ($k = 4-8$ steps) simulates larger batch sizes to stabilize training.

## Key Experimental Results

### Table 2: Comparison with Baseline Methods (OLMo-7B, Setup 3 Optimal Configuration)

### Main Results

| Method | Total Score | Task Aggregation Score | MIA Score | MMLU Mean |
|:---|:---:|:---:|:---:|:---:|
| Gradient Ascent | 0.394 | 0 | 0.912 | 0.269 |
| KL Minimization | 0.395 | 0 | 0.916 | 0.269 |
| Gradient Difference | 0.243 | 0 | 0.382 | 0.348 |
| NPO | 0.188 | 0.021 | 0.080 | 0.463 |
| **LIBU (Setup 3)** | **0.254** | **0.280** | **0.0** | **0.483** |

- LIBU achieves the best performance (0.280) on task aggregation score (actual unlearning effectiveness) while maintaining the highest MMLU (0.483).
- Although gradient ascent and KL minimization yield high total scores, their MMLU scores collapse severely (0.269), and their task aggregation scores drop to 0.

### Hyperparameter Sensitivity

### Ablation Study

| Configuration | Aggressiveness | MMLU Results |
|:---|:---|:---|
| Setup 1 (High LR 4e-5) | Aggressive | MMLU < threshold, Failed |
| Setup 2 (Medium) | Balanced | Best on Subtasks 1 and 3 |
| Setup 3 (Conservative 2e-5) | Conservative | Overall Best |

## Highlights & Insights

- **Two-Stage Decoupled Design**: Accurate localization via influence functions followed by stabilization via second-order optimization, avoiding approximation drift caused by joint training.
- **Diagonal Fisher Matrix Replacing Full Hessian**: Reduces computational complexity from $O(d^2)$ to $O(d)$, making the approach highly feasible for LLMs.
- **LoRA-Based Parameter-Efficient Unlearning**: Updates only low-rank adapters, further lowering memory and computational demands.
- **Optimal MMLU Performance Preservation**: Achieves the highest MMLU across all comparison methods, indicating minimal disruption to general capabilities.

## Limitations & Future Work

- **MIA Score of 0**: LIBU completely fails on Member Inference Attack (MIA) metrics, suggesting the unlearning is incomplete.
- **Validated Only on Small Models**: The method is evaluated only on OLMo-1B/7B, leaving its scalability to larger models unknown.
- **Extreme Sensitivity to Hyperparameters**: Significant performance variation across the three configurations leads to high tuning costs in LLM scenarios.
- **Competition System Paper**: Lacks deeply thorough ablation studies and theoretical analyses.

## Rating

- ⭐⭐⭐ Novelty: Combining influence functions with Sophia is a reasonable innovation for unlearning tasks, though individual components are existing techniques.
- ⭐⭐⭐ Practicality: Lightweight and runnable on a single A100; however, the failure in MIA limits its use in actual privacy preservation scenarios.
- ⭐⭐ Experimental Thoroughness: Written in a competition paper format, lacking thorough ablation studies and individual component contribution analysis.
- ⭐⭐⭐ Writing Quality: The methodology is clearly described, but the experimental analysis remains superficial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)
- [\[ACL 2025\] SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?](seuf_is_unlearning_one_expert_enough_for_mixture-of-experts_llms.md)
- [\[ACL 2025\] PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative Prompts](pig_privacy_jailbreak.md)
- [\[ICLR 2026\] Unlearning Evaluation through Subset Statistical Independence](../../ICLR2026/llm_safety/unlearning_evaluation_through_subset_statistical_independence.md)

</div>

<!-- RELATED:END -->
