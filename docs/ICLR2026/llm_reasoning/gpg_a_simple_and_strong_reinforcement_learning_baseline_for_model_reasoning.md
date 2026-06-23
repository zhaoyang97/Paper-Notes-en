---
title: >-
  [Paper Note] GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Policy Gradient] GPG (Group Policy Gradient) returns to the most basic policy gradient, directly optimizing the original RL objective—eliminating the critic, reference model, KL constraints, and surrogate loss, while retaining only group-mean normalization and a gradient debiasing correction, consistently outperforming GRPO in math and
tags:
  - ICLR 2026
  - LLM Reasoning
  - Policy Gradient
  - GRPO
date: 2026-05-08
content_hash: 8b2e161c4c26cfdb
---
# GPG: A Simple and Strong Reinforcement Learning Baseline for Model Reasoning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=inccdtfx8x](https://openreview.net/forum?id=inccdtfx8x)  
**Code**: [https://github.com/AMAP-ML/GPG](https://github.com/AMAP-ML/GPG)  
**Area**: LLM Reasoning / RL Post-training  
**Keywords**: Policy Gradient, GRPO, Reasoning RL, Gradient Estimation Bias, RLHF Simplification  

## TL;DR
GPG (Group Policy Gradient) returns to the most basic policy gradient, directly optimizing the original RL objective—eliminating the critic, reference model, KL constraints, and surrogate loss, while retaining only group-mean normalization and a gradient debiasing correction, consistently outperforming GRPO in math and multimodal reasoning tasks.

## Background & Motivation
**Background**: RL post-training (RFT) is the core engine for reasoning models such as OpenAI o1 and DeepSeek R1. The mainstream approaches are PPO and GRPO. PPO requires maintaining both critic and reference models, incurring massive resource overhead; GRPO removes the critic and uses group-normalized rewards to estimate advantage but still retains the reference model, KL constraints, and clipped surrogate loss.

**Limitations of Prior Work**: The community has been performing "PPO subtraction" (ReMax removes the critic, GRPO uses group normalization, Dr. GRPO studies reward/loss normalization details), but these methods either still carry redundant components or, despite pointing out reward bias in advantage functions, fail to deliver significant performance gains (reproductions show Dr. GRPO does not significantly outperform GRPO).

**Key Challenge**: PPO was originally designed for tasks like Atari that require learning vision representations and control policies from scratch. In the LLM era, the policy itself is a pre-trained and SFT-tuned LLM with strong representation capabilities. Applying complex mechanisms designed for general RL (surrogate loss, trust region constraints) becomes a burden on scalability. The primary weakness of policy gradients is high variance, which can be mitigated via "baseline advantage estimation + multi-trajectory sampling"—both of which are already standard in LLM post-training.

**Goal**: Construct a streamlined method that retains minimal RL components and directly optimizes the original objective, matching or exceeding GRPO without relying on any auxiliary tricks.

**Core Idea**: **Return to the roots of policy gradients**. Since LLM post-training naturally satisfies the two conditions for variance reduction in policy gradients, there is no need for surrogate losses or constraints. By removing the implicit reward bias in advantage estimation and correcting the gradient estimation bias, a simple policy gradient is the strongest baseline.

## Method

### Overall Architecture
Within a group, GPG samples $G$ responses for the same question, uses the group reward mean to perform advantage normalization for each response, and then directly minimizes a policy gradient objective of the form $-\log\pi_\theta \cdot \hat{A}$. This process does not involve a critic, reference model, KL constraints, or clipping. Two key corrections are superimposed on this minimalist skeleton: removing the implicit reward bias in the advantage function and applying a scaling correction (AGE) for gradient estimation bias caused by all-correct or all-wrong samples, further auxiliary by a threshold resampling mechanism to control variance.

```mermaid
flowchart LR
    A[Question q] --> B[Sample G responses o_i]
    B --> C[Rule-based Reward R_i<br/>Correct 1.0/Wrong 0.0]
    C --> D[Group Mean Normalization<br/>Â = (r - mean R)/F_norm]
    D --> E[AGE Gradient Debiasing<br/>×α = B/(B-M)]
    E --> F{Effective Sample Ratio<br/>≥ β_th?}
    F -- No --> G[Accumulated Resampling]
    G --> F
    F -- Yes --> H[loss = -log π · Â · α]
```

### Key Designs

**1. Group Policy Gradient: Direct optimization of the original objective by removing all redundant components.** The objective function of GPG (Eq 5) directly writes the advantage-weighted log-likelihood from the policy gradient theorem as a loss: $J_{GPG}(\theta)=\mathbb{E}\big[\frac{1}{\sum|o_i|}\sum_i\sum_t -\log\pi_\theta(o_{i,t}\mid q,o_{i,<t})\hat{A}_{i,t}\big]$. It does not optimize a surrogate with importance sampling ratios and clipping like PPO/GRPO but performs gradient ascent directly on the original RL objective. Advantage is obtained via group mean normalization $\hat{A}_{i,t}=\frac{r_i-\text{mean}(\{R_i\}_{i=1}^G)}{F_{norm}}$ (Eq 6). As shown in Table 2, compared to PPO (value+reference+surrogate+constraints all present), GRPO (removes critic but retains the other three), and TRPO, GPG leaves all four items blank, representing the simplest form that is both easy to implement and efficient.

**2. Removing Reward Bias: $F_{norm}=1$ instead of group standard deviation.** GRPO sets $F_{norm}=\text{std}\{R(o)\}$. The authors point out that this is essentially a function of the state $s_t$ (Eq 2), which introduces explicit reward bias and shifts the optimization objective away from the original problem. GPG advocates solving the original problem directly without surrogates or biases, thus setting $F_{norm}=1$. However, simply removing this bias term (43.9%) does not significantly outperform GRPO (43.7%), which contradicts observations in Dr. GRPO—indicating that the true source of improvement lies elsewhere and must be paired with the next design. The authors further find that GRPO's std normalization (where group std fluctuates between 0.10~0.35) implicitly acts as a gradient correction, leading to AGE.

**3. AGE Accurate Gradient Estimation: $\alpha=\frac{B}{B-M}$ scaling for all-correct/all-wrong samples.** In a batch, if all $G$ responses in certain groups are entirely correct or entirely wrong, the intra-group advantages are all zero, contributing nothing to the gradient. However, standard backpropagation still averages using $B$ as the denominator, effectively diluting the gradient with ineffective samples. Given $M$ such ineffective samples in a batch, the true gradient should be $\hat{g}=\frac{\sum_{i=M+1}^B g_i}{B-M}=g\cdot\frac{B}{B-M}=\alpha g$ (Eq 7). Thus, the corrected objective is $\hat{J}_{GPG}(\theta)=\alpha J_{GPG}(\theta)$ (Eq 8). $\alpha$ is not a constant; it changes dynamically with batch difficulty (measured between 1.5~4.0), automatically amplifying effective signals. Adding AGE increases the average score from 43.9% to 47.8%, which is the true source of the performance leap. The paper also provides an equivalent form for multi-GPU training that requires no additional communication (Appendix proof), avoiding the overhead of collecting non-zero gradients across GPUs.

**4. Threshold Resampling: Ensuring a minimum effective sample ratio to reduce variance.** AGE provides an unbiased estimate, but when the effective sample ratio is too low, $\alpha$ becomes very large, and gradient variance explodes. GPG introduces a threshold $\beta_{th}=\frac{1}{\alpha_{th}}$. When the effective sample ratio is below this value, effective samples are accumulated into subsequent resampling batches until the ratio exceeds the threshold before updating. Compared to DAPO's approach of "resampling until $M=0$ (i.e., $\alpha=1$)", GPG is not held back by the slowest worker, is more efficient, and automatically adjusts loss intensity based on batch performance. Adding $\beta_{th}=0.6$ further improves the average score to 48.3%.

## Key Experimental Results

### Main Results: Ablation of Components (Qwen2.5-Math-7B, MATH-lighteval)

| Configuration | Average | AIME24 | MATH-500 | AMC23 |
|------|---------|--------|----------|-------|
| GRPO | 43.7 | 16.7 | 73.4 | 62.5 |
| Dr. GRPO† | 43.7 | 26.7 | 74.6 | 50.0 |
| GPG ($F_{norm}{=}1,\alpha{=}1$) | 43.9 | 23.3 | 76.3 | 52.5 |
| GPG ($F_{norm}{=}\text{std},\alpha{=}1$) | 45.3 | 23.3 | 73.6 | 60.0 |
| GPG ($F_{norm}{=}1,\alpha{=}\tfrac{B}{B-M}$) | 47.8 | 30.0 | 75.0 | 62.5 |
| **GPG (+$\beta_{th}{=}0.6$)** | **48.3** | **30.0** | **76.2** | 62.5 |

### 1.5B Distilled Models (Zero-shot pass@1 on five math benchmarks)

| Model | Average | AIME24 | MATH-500 | AMC23 |
|------|---------|--------|----------|-------|
| DeepSeek-R1-Distill-Qwen-1.5B | 48.9 | 28.8 | 82.8 | 62.9 |
| Open-RS1† | 53.1 | 33.3 | 83.8 | 67.5 |
| **Ours-RS1** | **55.7** | 33.3 | **87.6** | 77.5 |
| **Ours-RS3** | 55.5 | 33.3 | 85.0 | **80.0** |

### Key Findings
- **Debiasing gradients is key, not rewards**: Simply removing std normalization barely matches GRPO; adding AGE results in a ~4% jump, proving that performance gains stem from gradient estimation correction rather than reward normalization.
- **GRPO's std normalization implies gradient correction**: The std varies with difficulty (0.10~0.35) within groups, effectively acting as a partial $\alpha$ scaling, which explains why it is slightly better than pure $F_{norm}{=}1$.
- **Multimodal Generalization**: GPG consistently outperforms other RL methods in visual reasoning (SAT/CV-Bench), geometric reasoning (GEOQA), few-shot classification, and grounding (Flower102/LISA, etc.), verifying that the method is not task-modality dependent.
- **Computational Efficiency**: Removing the reference model saves one forward pass, and avoiding the strategy of resampling until completion prevents being bottlenecked by slow workers, resulting in lower training costs than GRPO.

## Highlights & Insights
- **Extreme "Subtraction"**: Table 2 intuitively shows GPG is the only method to zero out all four items: value, reference, surrogate, and constraints. It provides a definitive answer to "Can GRPO be simpler?".
- **Precise Diagnosis**: By decomposing GRPO's improvements into "reward bias removal" and "gradient bias correction," and proving the former is ineffective while the latter is critical, the paper corrects the attribution made by Dr. GRPO.
- **Simple yet Sound AGE**: A one-line formula for $\alpha=\frac{B}{B-M}$ is intuitive for unbiased estimation and offers an equivalent implementation with zero multi-card communication overhead, making it engineering-friendly.
- **Dynamic Adaptation**: $\alpha$ scales automatically with batch difficulty, acting as an internal difficulty-adaptive learning rate without extra hyperparameter tuning.

## Limitations & Future Work
- **Simplistic Reward Settings**: Experiments focus mainly on rule-based rewards (1.0 for correct / 0.0 for wrong) for math/vision tasks. The paper admits that intermediate step rewards are difficult to design and uses a simplified approach. Whether removing std normalization remains robust in open tasks with continuous and noisy reward models remains to be verified.
- **Variance Control Dependency**: $\beta_{th}$ is a newly introduced hyperparameter, and resampling accumulation might alter the data distribution of batches. Stability boundaries in long-term training are not fully discussed.
- **Risks of Removing KL Constraints**: Completely removing reference models and KL constraints might lead to distribution drift or reward hacking in long-term training; the paper does not deeply analyze behavior under extreme scaling.
- **Theory vs. Empirics**: AGE is an intuitive correction for all-correct/all-wrong samples but lacks more rigorous theoretical guarantees for convergence and optimality.

## Related Work & Insights
- **PPO/GRPO Lineage**: GPG follows the trajectory of "reverse simplification" along PPO→TRPO→Policy Gradient, advocating for a return to PG roots in the LLM era. This is consistent with the history where PPO simplified TRPO, which was built upon PG.
- **Response to Dr. GRPO**: A concurrent work, Dr. GRPO, identifies reward bias in GRPO and its tendency to generate more tokens. GPG's reproduction suggests these gains are not significant and attributes the real cause to gradient estimation bias, providing a different perspective.
- **Comparison with DAPO**: Methods like DAPO eliminate ineffective samples by resampling until $M=0$. GPG achieves the same effect through thresholding and AGE scaling but more efficiently, serving as a lightweight alternative to "dynamic sampling."
- **Insight**: When performing RL on a foundation with existing strong representations, the rule of "fewer components are better" may be universal. Precisely attributing performance gains to a single mechanism (gradient debiasing) carries more methodological value than stacking tricks.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a brand-new algorithm, but the combination of "returning to basic PG + precise attribution + AGE debiasing" is concise and powerful, correcting misinterpretations in concurrent works.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 1.5B/7B models, five math benchmarks, and multimodal tasks, with ablations clearly isolating the contribution of each component.
- Writing Quality: ⭐⭐⭐⭐ The motivation chain (PPO history → subtraction trend → root regression) is clear, and the component comparison in Table 2 is highly persuasive.
- Value: ⭐⭐⭐⭐ As a "simple and strong" reasoning RL baseline, it has low engineering costs and high reproducibility, offering direct reference value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models](a_simple_motivation_can_enhance_reinforcement_finetuning_of_large_reasoning_mode.md)
- [\[ICLR 2026\] NFT: Bridging Supervised Learning and Reinforcement Learning in Math Reasoning](nft_bridging_supervised_learning_and_reinforcement_learning_in_math_reasoning.md)
- [\[ICLR 2026\] Learning to Reason over Continuous Tokens with Reinforcement Learning (HyRea)](learning_to_reason_over_continuous_tokens_with_reinforcement_learning.md)
- [\[ICLR 2026\] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning](emergent_hierarchical_reasoning_in_llms_through_reinforcement_learning.md)
- [\[ICLR 2026\] Process-Verified Reinforcement Learning for Theorem Proving via Lean](process-verified_reinforcement_learning_for_theorem_proving_via_lean.md)

</div>

<!-- RELATED:END -->
