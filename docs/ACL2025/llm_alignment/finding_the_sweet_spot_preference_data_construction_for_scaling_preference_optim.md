---
title: >-
  [Paper Note] Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization
description: >-
  [ACL2025][LLM Alignment][DPO] It is discovered that the traditional DPO preference data construction strategy (max-min) suffers from performance degradation as the sample size increases. Through a systematic exploration based on reward distribution, it is found that the rejected response should be selected at $\mu-2\sigma$ instead of the minimum. Based on this finding, a preference data construction method is proposed that consistently scales with the sample size.
tags:
  - "ACL2025"
  - "LLM Alignment"
  - "DPO"
  - "preference optimization"
  - "preference pair construction"
  - "reward distribution"
  - "scaling"
  - "on-policy sampling"
date: 2026-05-08
content_hash: abd1c3e6844e6e5b
---

# Finding the Sweet Spot: Preference Data Construction for Scaling Preference Optimization

**Conference**: ACL2025  
**arXiv**: [2502.16825](https://arxiv.org/abs/2502.16825)  
**Code**: [XYaoooo/DPO_Pair](https://github.com/XYaoooo/DPO_Pair)  
**Area**: LLM Alignment  
**Keywords**: DPO, preference optimization, preference pair construction, reward distribution, scaling, on-policy sampling  

## TL;DR

It is discovered that the traditional DPO preference data construction strategy (max-min) suffers from performance degradation as the sample size increases. Through a systematic exploration based on reward distribution, it is found that the rejected response should be selected at $\mu-2\sigma$ instead of the minimum. Based on this finding, a preference data construction method is proposed that consistently scales with the sample size.

## Background & Motivation

Direct Preference Optimization (DPO) is one of the most popular LLM alignment methods, directly optimizing the policy model via preference pairs (chosen-rejected) to avoid the complexity of training reward models and PPO in RLHF.

The commonly used workflow for constructing preference data is:
1. Sample $n$ on-policy responses from the policy model for each prompt.
2. Score them using a reward model.
3. Select the response with the highest reward as chosen, and the lowest as rejected.

Intuitively, increasing the sample size $n$ should yield higher-quality chosen responses and poorer rejected responses, thereby improving the alignment performance. **However, experiments show that this is not the case—performance either degrades or fluctuates as $n$ increases**.

This counter-intuitive finding motivates the authors to thoroughly investigate the optimal strategy for constructing preference pairs, specifically the critical impact of the chosen position of the rejected response on DPO training.

## Method

### 1. Problem Discovery: Failure of the Max-Min Strategy

Across four models (Llama-3-8B, Llama-3-8B-Instruct, Mistral-7B-v0.1, and Mistral-7B-Instruct-v0.2), using prompts from the UltraFeedback dataset, the sample size was scaled from 5 to 200. AlpacaEval 2 evaluation results show:
- Llama Base: Performance fluctuates and is unstable.
- Llama Instruct: Performance drops significantly as $n$ increases.
- Mistral Base: Shows a similar declining trend.
- Mistral Instruct: Increases slightly at first, then declines.

### 2. Preference Pair Construction Based on Reward Distribution

**Mechanism**: Instead of partitioning samples by absolute ranking, preference pairs are selected based on the reward distribution (approximated as normal) of each prompt.

For the rewards of $n$ responses of each prompt, the distribution is approximated as $N(\mu_i, \sigma_i^2)$, and samples are selected at key positions of the distribution:

**7 representative sampling points**:
$$\{min, \mu-2\sigma, \mu-\sigma, \mu, \mu+\sigma, \mu+2\sigma, max\}$$

By constructing all $C_7^2 = 21$ combinations of preference pairs, 21 × 4 = 84 policy models are trained, systematically evaluating the performance of each combination.

### 3. Key Findings

Based on the evaluation of 84 models on AlpacaEval 2, the following conclusions are drawn:

**Finding 1: The optimal position for the rejected response is $\mu-2\sigma$, not the minimum.**

This is the most crucial discovery. The traditional approach of selecting the sample with the minimum reward as rejected is actually too extreme, which can lead the training to trigger "shortcut learning"—where the model only learns to avoid extremely bad outputs, rather than learning to distinguish between high-quality and medium-quality responses.

**Finding 2: The chosen response should be as good as possible (provided that the rejected response is within a reasonable range).**

When the rejected response is fixed at $\mu-2\sigma$, the performance continuously improves as the chosen response shifts from $\mu$ to $\mu+\sigma$, then to $\mu+2\sigma$, and finally to max. The $(\mu+2\sigma, \mu-2\sigma)$ combination is optimal under most settings. For example, Llama-3-8B-Instruct achieves an LC win rate of 48.18% under this combination, which is about 3 percentage points higher than the max-min strategy.

**Finding 3: Small reward margins lead to poor performance.**

Small-margin preference pairs such as $(\mu+2\sigma, \mu+\sigma)$ only achieve a 34.63% LC win rate, as the model struggles to distinguish tiny quality differences.

**Finding 4: DPO training is robust.**

None of the 21 preference pair configurations result in performance below the SFT baseline, indicating that DPO has fundamental robustness to different preference pairs.

### 4. Scalable Preference Data Construction Strategy

Based on the aforementioned findings, a simple and practical strategy is proposed:

- **Rejected Selection**: Select the lowest reward from 5 random samples (as an effective approximation of $\mu-2\sigma$).
- **Chosen Selection**: Select the highest reward from all $n$ samples.

In this way, as $n$ increases, the quality of the chosen response naturally scales up, while the rejected response remains around $\mu-2\sigma$ (since it is always the minimum of 5 samples), achieving continuous performance improvement.

### Loss & Training

DPO Objective Function:

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \Big[\log \sigma(r(x,y_w) - r(x,y_l))\Big]$$

where $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$

Different preference pairs lead to different training dynamics:
- (max, min): Lowest training loss, but prone to **overfitting**.
- (max, $\mu-2\sigma$): Moderate loss, yielding the best generalization performance.
- (max, $\mu+2\sigma$): Loss stagnates and fails to decrease, leading to **underfitting**.

## Experiments

### Experimental Setup

- **Models**: Llama-3-8B, Llama-3-8B-Instruct, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.2
- **Data**: UltraChat-200k (SFT), UltraFeedback (DPO prompts)
- **Reward Models**: ArmoRM (main experiment), Skywork (validation of generalization)
- **Sampling**: Temperature 0.8, 5-400 samples per prompt, accelerated with vLLM
- **Evaluation**: AlpacaEval 2 (LC win rate + win rate), Arena-Hard

### Main Results: Comprehensive Exploration of 21 Preference Pairs

Under the 200 samples/prompt setting, the heatmap of the 84 models (Figure 4) clearly indicates:
- In each column (fixed chosen), the performance is optimal when rejected is at $\mu-2\sigma$.
- In each row (fixed rejected), higher chosen rewards lead to better performance.
- Performance is worst near the diagonal (small margins).

### Extension Experiments

**More Sampling Points**: Extending the reward positions to $\mu\pm3\sigma$ and $\mu\pm4\sigma$, it is observed that:
- $\mu+3\sigma$/$\mu+4\sigma$ show no significant difference from max.
- $\mu-3\sigma$/$\mu-4\sigma$ show no significant difference from min.
- This suggests that $\mu\pm2\sigma$ is sufficient to cover the meaningful reward space.

**Scaling to 400 samples/prompt**: The conclusion remains consistent on Llama-3-8B-Instruct.

### Effects of the Scalable Strategy

| n (Sample Size) | Traditional Max-Min LC | Traditional Max-Min WR | Ours LC | Ours WR |
|-----------|--------|--------|--------|--------|
| 5 | ~45% | ~47% | ~45% | ~47% |
| 50 | ~43% | ~44% | ~47% | ~49% |
| 100 | ~42% | ~43% | ~48% | ~50% |
| 200 | ~41% | ~41% | ~49% | ~50% |

Traditional methods degrade as $n$ increases, whereas the proposed method (Ours) continuously improves.

### Related Work & Insights

| Data (Method) | #Sample | AE LC | AE WR | AH WR |
|-----------|---------|-------|-------|-------|
| Baseline* (SimPO) | 5 | 53.7 | 47.5 | 36.5 |
| Baseline* (DPO) | 5 | 48.2 | 47.5 | 35.2 |
| Baseline† (DPO) | 400 | 42.0 | 42.0 | 34.5 |
| **Ours (DPO)** | **400** | **49.1** | **50.2** | **37.3** |

At 400 samples, the DPO of the proposed method outperforms the 5-sample DPO baseline of Meng et al. (2024), and even beats their SimPO baseline on WR and AH.

### Cross-Reward Model Validation

Replacing ArmoRM with the Skywork reward model, validation on Llama-3-8B-Instruct shows an identical trend—performance scales up as $n$ increases and then plateaus.

### Academic Benchmark Evaluation

No performance degradation is observed on ARC, HellaSwag, TruthfulQA, and GSM8K, demonstrating that alignment improvement does not come at the expense of general capabilities.

## Highlights & Insights

1. **Value of Counter-intuitive Discovery**: The max-min strategy degrades as sampling scales up. This finding serves as an important warning to the entire DPO community: more sampling does not automatically equate to better training data.
2. **Statistical Distribution Perspective**: Systematically partitioning the reward space based on key positions of the normal distribution provides an interpretable and reproducible analytical framework. The effectiveness of $\mu-2\sigma$ can be understood from the training dynamics: it provides sufficient contrastive signals while avoiding overfitting caused by overly extreme samples.
3. **High Practicality**: The final strategy is extremely simple—electing the worst from 5 samples as the rejected, and the best from all samples as the chosen. No extra models or complex computations are required.
4. **Precise Diagnosis of Overfitting**: Loss curve analysis explains the failure mechanism of the max-min strategy: the rapid decline of training loss leads to overfitting, whereas $\mu-2\sigma$ provides a "sweet spot" difficulty for contrastive learning.

## Limitations & Future Work

1. **Dependence on Strong Reward Models**: The effectiveness of the method depends on the quality of the reward model; low-quality reward models may yield suboptimal results.
2. **Computational Cost**: Generating a large number of samples (e.g., 200-400 per prompt) requires significant inference compute resources.
3. **Evaluated only on DPO**: The generalizability of these findings to other preference optimization methods, such as SimPO, IPO, and KTO, has not been explored.
4. **Normal Distribution Assumption**: The reward distribution is not always normally distributed; under skewed or multimodal distributions, partitioning by $\mu\pm n\sigma$ might not be optimal.

## Related Work & Insights

- **DPO and Variants**: DPO by Rafailov et al. (2023), SimPO by Meng et al. (2024), KTO by Ethayarajh et al. (2024)
- **Preference Data Construction**: Self-generated preference data by Dong et al. (2023), preference pair studies by Kim et al. (2025)
- **RLHF**: InstructGPT by Ouyang et al. (2022), PPO by Schulman et al. (2017)
- **Inference Scaling**: Repeated sampling by Brown et al. (2024), inference scaling by Snell et al. (2024)

## Rating

⭐⭐⭐⭐⭐ (5/5)

Highly focused topic, thorough experimentation (systematic analysis of 84 models), and findings with direct practical value. It uncovers the long-ignored problem of preference pair construction in the DPO community, presenting a strategy that is both simple and effective. Particularly in the context of the current widespread use of on-policy DPO training, this work holds high practical significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](../../NeurIPS2025/llm_alignment/preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)
- [\[ACL 2025\] Boosting Vulnerability Detection of LLMs via Curriculum Preference Optimization with Synthetic Reasoning Data](boosting_vulnerability_detection_of_llms_via_curriculum_preference_optimization_.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](otpo_token_weighting.md)
- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)

</div>

<!-- RELATED:END -->
