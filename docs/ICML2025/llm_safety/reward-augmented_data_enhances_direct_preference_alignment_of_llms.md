---
title: >-
  [Paper Note] Reward-Augmented Data Enhances Direct Preference Alignment of LLMs
description: >-
  [ICML2025][LLM Safety][DPO] Proposes a **reward-augmented data relabeling method** that constructs an augmented dataset by conditioning preference pairs on reward scores. This enables DPO to perceive the full spectrum of response quality, mitigating the issues where high-quality rejected responses are forgotten and low-quality chosen responses are blindly learned, consistently and significantly enhancing DPO performance across multiple benchmarks.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "DPO"
  - "preference alignment"
  - "reward conditioning"
  - "data augmentation"
  - "RLHF"
  - "LLM alignment"
date: 2026-05-08
content_hash: 1a0ffe9d6e7e88c5
---

# Reward-Augmented Data Enhances Direct Preference Alignment of LLMs

**Conference**: ICML2025  
**arXiv**: [2410.08067](https://arxiv.org/abs/2410.08067)  
**Code**: [shenao-zhang/reward-augmented-preference](https://github.com/shenao-zhang/reward-augmented-preference)  
**Area**: Signal Communication  
**Keywords**: DPO, preference alignment, reward conditioning, data augmentation, RLHF, LLM alignment

## TL;DR

Proposes a **reward-augmented data relabeling method** that constructs an augmented dataset by conditioning preference pairs on reward scores. This enables DPO to perceive the full spectrum of response quality, mitigating the issues where high-quality rejected responses are forgotten and low-quality chosen responses are blindly learned, consistently and significantly enhancing DPO performance across multiple benchmarks.

## Background & Motivation

### Three Major Limitations of Direct Preference Alignment

Existing direct alignment algorithms (such as DPO) focus only on **relative preferences** (chosen vs. rejected) and ignore the **absolute quality scores** of responses, leading to three major issues:

**Unlearning of high-quality rejected responses**: When the quality gap between chosen and rejected is extremely small (e.g., $r=9$ vs. $r=8$), DPO still maximizes the implicit reward gap, leading to an unnecessary decrease in the probability of generating high-quality rejected responses.

**Blind learning of low-quality chosen responses**: DPO indiscriminately boosts the probability of all chosen responses, even if some chosen responses are of very low quality (e.g., $r=1$), merely because they outperform even worse rejected ones ($r=0$).

**Reward Sparsity**: Optimal responses ($r = r_{\max}$) are highly sparse in the training data, making DPO unable to distinguish different quality levels of chosen responses, which hinders generalization to optimal responses.

### Key Insight

In the RLAIF pipeline, the judge model (such as GPT-4 or a reward model) already provides the **quality score** for each response, but DPO completely ignores this information. By conditioning the LLM on target rewards to generate responses, the entire quality spectrum information can be leveraged.

## Method

### 1. Reward-Conditioned Policy

Define the target-conditioned reward function:

$$R(x, y, g) = -(g - r(x, y))^2$$

where $g$ is the target reward, and $r(x,y)$ is the actual quality score given by the judge model. The optimization objective of the policy $\pi(y|x,g)$ is:

$$\min_{\pi} \mathbb{E}_{g, x \sim \mathcal{D}_N, y \sim \pi(\cdot|x,g)} \left[(g - r(x,y))^2\right]$$

### 2. Data Relabeling for constructing Reward-Augmented Dataset

For each pair in the original preference dataset $\mathcal{D}_N = \{(x^i, y_w^i, y_l^i)\}$, relabeling is performed using two target reward values:

- **Target $g = r_w^i$ (reward of the chosen)**: $R(x,y_w,g)=0 > R(x,y_l,g)=-(r_w-r_l)^2$, maintaining the original ranking $y_w \succ y_l$.
- **Target $g = r_l^i$ (reward of the rejected)**: $R(x,y_l,g)=0 > R(x,y_w,g)=-(r_w-r_l)^2$, **reversing the ranking** to $y_l \succ y_w$.

This generates two new pairs from each original pair, expanding the dataset from $N$ to $2N$.

### 3. Implementation

- Conditioning is achieved via **system prompts**, such as "generate responses of score $g$".
- During inference, setting $g^* = r_{\max}$ (the maximum reward, e.g., 10) guides the model to generate the optimal response.
- It can be directly integrated with any direct alignment algorithm (DPO, IPO, etc.) **without modifying the algorithm itself**.

### 4. Theoretical Guarantees

The paper provides a convergence proof (Theorem 4.1): the suboptimality of reward-augmented DPO decays at a rate of $O(N^{-1/2})$, guaranteeing global convergence to the optimal strategy. This outperforms previous target-conditioned RL work, which could only prove local improvements.

## Key Experimental Results

### Instruction Following Benchmarks (UltraFeedback + DPO)

| Model | AlpacaEval 2.0 LC WR | MT-Bench | Arena-Hard |
|------|----------------------|----------|------------|
| Qwen2-7B-Instruct | 20.93 | 7.90 | 24.3 |
| + DPO (UF) | 21.46 | 8.33 | 21.9 |
| + DPO (**RA**, Ours) | **31.17** | **8.47** | **30.1** |
| Gemma-2-9B-It | 49.20 | 8.54 | 42.8 |
| + DPO (UF) | 50.70 | 8.54 | 35.8 |
| + DPO (**RA**, Ours) | **59.27** | **8.59** | **43.9** |
| SPPO | 55.60 | 8.40 | 47.6 |
| + DPO (UF) | 52.75 | 8.41 | 40.4 |
| + DPO (**RA**, Ours) | **60.97** | **8.73** | **49.0** |

### Academic Benchmark Average Scores

| Model | GSM8K | GPQA | TruthfulQA | Average |
|------|-------|------|------------|------|
| Llama-3.1-8B + DPO (UF) | 78.47 | 33.72 | 56.61 | 53.50 |
| Llama-3.1-8B + DPO (**RA**) | **78.77** | 32.55 | **63.32** | **54.37** |
| Gemma-2-9B + DPO (UF) | 83.32 | 34.14 | 65.12 | 59.22 |
| Gemma-2-9B + DPO (**RA**) | **83.62** | **35.74** | **65.27** | **59.75** |

### Key Ablation Study

- **Half RA** (using only half of the data for augmentation to match the original data size) still significantly outperforms original DPO, proving that the improvement is not solely due to doubling the data size.
- **Implicit Reward Augmentation (IRA)**: Relabeling using the implicit rewards of the DPO model performs even better than RA with GPT-4 scores, indicating that DPO does not fully exploit the data.
- **Multi-Attribute Reward Conditioning**: Conditioning on 5-dimensional attribute rewards from ArmoRM achieves SOTA performance on Llama-3-8B (LC WR 56.57), outperforming SimPO (53.70).

## Highlights & Insights

1. **Simplicity with Zero Algorithm Modifications**: By only applying data relabeling and adding system prompts, the DPO algorithm itself remains unchanged, offering plug-and-play capability.
2. **Effective Even When Reusing SPPO Data**: SPPO trained on UF degrades when trained again on UF using standard DPO, but using RA data leads to a substantial improvement, demonstrating that the method extracts more information.
3. **Mitigating Unlearning**: Experiments show that the log probability drop for high-quality rejected responses (reward $\geq 5$) is far smaller than that of standard DPO.
4. **Compatibility with Data Lacking Reward Scores**: Can use DPO implicit rewards for relabeling, making it applicable to datasets with only binary preferences.
5. **Theoretical Rigor**: Provides theoretical proof of global convergence to the optimal strategy, surpassing the local guarantees from previous target-conditioned RL.

## Limitations & Future Work

1. **Dependency on Reward Score Quality**: The performance of the method depends on the accuracy of the judge model's scoring; if the reward scores are highly noisy, the augmentation benefit may degrade.
2. **Rough Policy Conditioning via System Prompts**: Using text prompts to achieve reward conditioning might be less efficient than embedding-level conditioning, leaving room for optimization.
3. **Validation Only on 7B-9B Models**: Lack of experimental validation on larger-scale models (70B+).
4. **Manual Setting of $g^*$ at Inference**: The target reward requires manual selection of the highest value; dynamic or adaptive targets were not explored.
5. **Insufficient Comparison with PPO/REINFORCE**: The main points of comparison are direct alignment methods, with limited comparison against explicit RL methods.

## Related Work & Insights

- **SteerLM / DPA**: Pioneers in conditional fine-tuning, but focus on multi-attribute user customization, whereas this paper addresses the inherent limitations of DPO itself.
- **RPO (Nemotron-4)**: Also addresses the unlearning issue but requires algorithm modifications, whereas this paper only requires data modification.
- **SimPO**: A strong on-policy alignment baseline, which this paper surpasses under the same settings.
- **Decision Transformer**: Shares a similar conditional sequence modeling concept, but this paper combines it with DPO instead of SFT.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of reward conditioning and data relabeling is simple, elegant, and insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive coverage with 5 base models across multiple benchmarks and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and well-integrated theory and experiments.
- Value: ⭐⭐⭐⭐ — A practical, plug-and-play approach with direct inspiration for the preference learning community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] POPri: Private Federated Learning using Preference-Optimized Synthetic Data](popri_private_federated_learning_using_preference-optimized_synthetic_data.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](../../NeurIPS2025/llm_safety/simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)
- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)

</div>

<!-- RELATED:END -->
