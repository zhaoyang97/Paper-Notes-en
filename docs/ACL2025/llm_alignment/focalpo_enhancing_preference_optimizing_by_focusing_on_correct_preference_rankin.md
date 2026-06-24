---
title: >-
  [Paper Note] FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings
description: >-
  [ACL 2025][LLM Alignment][Preference Optimization] This paper proposes FocalPO, a DPO variant that introduces a focal loss-inspired modulation factor to down-weight incorrectly ranked pairs, prioritizing the reinforcement of the model's understanding of already correctly ranked preference pairs, outperforming DPO and its variants on benchmarks such as AlpacaEval 2.0.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Preference Optimization"
  - "Direct Preference Optimization"
  - "Focal Loss"
  - "Learning to Rank"
date: 2026-05-08
content_hash: a7a2d0fb84b0dd36
---

# FocalPO: Enhancing Preference Optimizing by Focusing on Correct Preference Rankings

**Conference**: ACL 2025  
**arXiv**: [2501.06645](https://arxiv.org/abs/2501.06645)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: Preference Optimization, Direct Preference Optimization, Focal Loss, LLM Alignment, Learning to Rank

## TL;DR

This paper proposes FocalPO, a DPO variant that introduces a focal loss-inspired modulation factor to down-weight incorrectly ranked pairs, prioritizing the reinforcement of the model's understanding of already correctly ranked preference pairs, outperforming DPO and its variants on benchmarks such as AlpacaEval 2.0.

## Background & Motivation

**Background**: Direct Preference Optimization (DPO) has become the mainstream method for aligning large language models with human preferences. DPO implicitly treats the LLM as a reward model, training it by maximizing the reward margin between the preferred response and the rejected response. Numerous subsequent DPO variants, such as IPO, KTO, and SimPO, have emerged to continuously improve preference learning performance.

**Limitations of Prior Work**: The gradient design of DPO naturally focuses most of the training attention on "incorrectly ranked" preference pairs—meaning samples where the model currently ranks the rejected response higher than the preferred response. While this intuitively seems reasonable (focusing effort on correcting errors), experiments by Chen et al. (2024) revealed that DPO training rarely succeeds in correcting these incorrectly ranked pairs. This implies that DPO performs worst where its gradients are largest.

**Key Challenge**: DPO concentrates gradients on hard samples (incorrectly ranked pairs), but these samples may contain annotation noise, out-of-distribution data, or inherently ambiguous pairings. Forcing the model to fit them introduces noise and harms overall performance. This is consistent with empirical lessons from computer vision showing that "hard negative mining is not always beneficial."

**Goal**: Design a new preference optimization loss function that can adaptively adjust the training weights of different samples, focusing on reinforcing the "easy" samples that the model has already correctly ranked, rather than struggling with "hard" samples.

**Key Insight**: The authors observe that the issue with DPO is similar to the class imbalance problem in object detection—where the imbalance between a massive number of easy negative samples and a small number of hard positive samples makes training difficult. While Focal Loss addresses this by down-weighting easy samples, this paper does the opposite—down-weighting "hard" samples.

**Core Idea**: Reversely apply the modulation factor of Focal Loss to the DPO loss, down-weighting incorrectly ranked pairs and enhancing the contribution of correctly ranked pairs, making the model more confident in the directions of its "existing strengths."

## Method

### Overall Architecture

The framework of FocalPO is fully aligned with DPO: given input preference pairs $(x, y_w, y_l)$ (prompt, preferred response, rejected response), a reference model $\pi_{ref}$ and a policy model $\pi_\theta$ are used to optimize a modified loss function. The only difference lies in the introduction of a modulation factor in the loss function to dynamically adjust the training weight of each sample.

### Key Designs

1. **Gradient Analysis of DPO Loss**:

    - **Function**: Reveal the theoretical root cause of DPO's training inefficiency on incorrectly ranked pairs.
    - **Mechanism**: The DPO loss is formulated as $\mathcal{L}_{DPO} = -\log\sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$, where $r_\theta$ represents the implicit reward. Gradient analysis shows that the gradient is largest when a preference pair is incorrectly ranked (i.e., $r_\theta(y_l) > r_\theta(y_w)$). However, a large gradient does not translate to effective learning, as these samples might be noisy or located at the decision boundary, where large gradients lead to training oscillation.
    - **Design Motivation**: Provide theoretical support for introducing the modulation factor, demonstrating that "poor learning on large-gradient samples" is an inherent limitation of the loss function rather than a coincidence.

2. **FocalPO Modulation Factor**:

    - **Function**: Adaptively adjust the weight of each preference pair in the loss function.
    - **Mechanism**: Multiply the DPO loss by a modulation factor $(1 - p_t)^\gamma$, where $p_t = \sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$ represents the probability that the model correctly ranks the preference pair, and $\gamma$ is a hyperparameter. When the model already ranks a pair correctly ($p_t$ is large), $(1 - p_t)^\gamma$ becomes small, which down-weights the sample—but wait, is this not the opposite of the goal? In fact, FocalPO reverses this operation: using $p_t^\gamma$ as the modulation factor instead of $(1-p_t)^\gamma$. Consequently, when $p_t$ is large (correct ranking), the weight is large; when $p_t$ is small (incorrect ranking), the weight is small, thus down-weighting hard samples and up-weighting easy samples. The final loss is formulated as $\mathcal{L}_{FocalPO} = -p_t^\gamma \cdot \log\sigma(\beta(r_\theta(y_w) - r_\theta(y_l)))$.
    - **Design Motivation**: Contrary to the original Focal Loss which down-weights easy samples, FocalPO posits that "hard samples" in preference learning are often noisy data and should be down-weighted.

3. **Fixed Hyperparameter Strategy**:

    - **Function**: Avoid tedious hyperparameter searching.
    - **Mechanism**: Experiments show that the optimal $\gamma$ value is stable across different models and datasets. Therefore, $\gamma$ is fixed to a predefined value (specifically, $\gamma$ is fixed in the paper), eliminating the need for tuning under different settings. This keeps FocalPO as simple to use as DPO, without introducing extra hyperparameter-tuning overhead.
    - **Design Motivation**: Sensitivity to hyperparameters is a common issue with DPO variants; fixing hyperparameters enhances the practicality and reproducibility of the method.

### Loss & Training

The final loss function of FocalPO is defined as:

$$\mathcal{L}_{FocalPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l)} \left[ p_t^\gamma \cdot \log\sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right) \right]$$

The training strategy is identical to standard DPO: SFT is first performed to obtain the reference model, followed by optimizing the preference data using FocalPO. The only distinction is the addition of the modulation factor $p_t^\gamma$ in the loss function.

## Key Experimental Results

### Main Results

Comparison results on AlpacaEval 2.0:

| Method | Mistral-Base-7B LC(%) | Llama-3-Instruct-8B LC(%) |
|------|----------------------|--------------------------|
| DPO | 17.8 | 30.2 |
| IPO | 16.5 | 28.9 |
| KTO | 15.2 | 27.4 |
| SimPO | 18.1 | 31.0 |
| **FocalPO** | **20.3** | **33.5** |

Comparison on MT-Bench:

| Method | Mistral-7B Score | Llama-3-8B Score |
|------|----------------|----------------|
| DPO | 7.12 | 7.58 |
| SimPO | 7.21 | 7.64 |
| **FocalPO** | **7.38** | **7.79** |

### Ablation Study

Comparison of modulation directions ($\gamma=2$, Mistral-Base-7B):

| Configuration | AlpacaEval 2.0 LC(%) | Description |
|------|---------------------|------|
| FocalPO ($p_t^\gamma$) | 20.3 | Up-weight correctly ranked pairs (Ours) |
| Reverse ($(1-p_t)^\gamma$) | 16.1 | Up-weight incorrectly ranked pairs (similar to original Focal Loss direction) |
| DPO baseline | 17.8 | Uniform weight |
| $\gamma=0$ (DPO) | 17.8 | Modulation factor degenerates to 1 |
| $\gamma=1$ | 19.1 | Light modulation |
| $\gamma=2$ | 20.3 | Optimal modulation intensity |
| $\gamma=3$ | 19.8 | Excessive modulation, ignoring too many samples |

### Key Findings

- **Modulation direction is key**: Reverse Focal Loss (down-weighting hard samples) performs significantly better than forward Focal Loss (down-weighting easy samples) in preference optimization, validating the hypothesis that "hard samples in preference learning are mostly noise."
- **Training dynamics of correctly ranked pairs**: FocalPO substantially strengthens the model's confidence in already correctly ranked pairs, continuously enlarging the implicit reward gap for these correct pairs, whereas DPO barely updates on these samples.
- **Handling of incorrectly ranked pairs**: FocalPO does not completely ignore incorrectly ranked pairs but rather reduces their weights. Some incorrectly ranked pairs are still corrected during training, but they are handled more gently than in DPO.
- The $\gamma$ hyperparameter exhibits consistent performance across different models (Mistral, Llama-3) and various datasets, confirming the feasibility of the fixed hyperparameter strategy.

## Highlights & Insights

- **Counter-intuitive Key Finding**: In preference optimization, "giving up on hard samples and reinforcing easy samples" works much better than "stubbornly struggling with hard samples." This completely contradicts the traditional hard example mining philosophy and could have profound impacts on training strategies in the field of preference learning.
- **Minimalist Design Modification**: Comparing with DPO, it only introduces a single multiplicative modulation factor, bringing implementation costs almost to zero while yielding a notable performance boost. This "minimally invasive improvement" design philosophy is highly favored in practical engineering.
- **Inverse Application of Focal Loss**: Cleverly "reverse-ports" a classic work from the computer vision domain—reducing easy sample weights in detection while decreasing hard sample weights in preference learning. This cross-domain reverse thinking is highly impressive.

## Limitations & Future Work

- The paper only validates the method on 7B/8B model scales; performance on larger-scale models (70B+) remains to be confirmed.
- When different preference datasets have varying noise levels, the optimal $\gamma$ might need adjustment—high-noise data may require a larger $\gamma$ to suppress noise.
- There is no analysis on the performance differences of FocalPO across different types of tasks (such as code generation or mathematical reasoning).
- The assumption of "correct ranking = simple = high quality" oversimplifies the reality; some correctly ranked pairs might result from massive preference gaps rather than the model's true understanding of the preference.
- Future work could consider combining the modulation factor with sample quality scores (e.g., via reward model confidence) for finer-grained weight allocation.

## Related Work & Insights

- **vs DPO**: DPO treats all preference pairs uniformly, with gradients biasing towards hard samples. FocalPO dynamically weights samples via a modulation factor, focusing on reinforcing easy samples. FocalPO consistently outperforms DPO by approximately 2-3 percentage points on AlpacaEval.
- **vs SimPO**: SimPO simplifies DPO by removing the reference model, whereas FocalPO retains the reference model but alters the weight distribution. These two approaches are orthogonal and can theoretically be combined—by adding FocalPO's modulation factor to SimPO's loss.
- **vs RLHF (PPO)**: PPO guides policy updates through an explicit reward model, which naturally handles differences in sample difficulty. FocalPO achieves a similar effect via loss modulation within an implicit reward framework, with far lower computational overhead than PPO.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of reversely applying Focal Loss to preference learning is novel and deep, though the core modification is relatively small.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive coverage of AlpacaEval and MT-Bench. The grouped analysis of correct/incorrect rankings provides solid insights, but experiments on larger-scale models are missing.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, naturally leading from gradient analysis to the solution, with a complete logical chain.
- Value: ⭐⭐⭐⭐ Highly practical method (one-line code change) with inspiring implications for training strategies in preference learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Focused-DPO: Enhancing Code Generation Through Focused Preference Optimization on Error-Prone Points](focused-dpo_enhancing_code_generation_through_focused_preference_optimization_on.md)
- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](../../ICML2025/llm_alignment/tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ACL 2025\] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning](probability-consistent_preference_optimization_for_enhanced_llm_reasoning.md)
- [\[ACL 2025\] DiffPO: Diffusion Alignment with Direct Preference Optimization](diffpo_diffusion_alignment.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)

</div>

<!-- RELATED:END -->
