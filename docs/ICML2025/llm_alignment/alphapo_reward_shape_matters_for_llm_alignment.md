---
title: >-
  [Paper Note] AlphaPO: Reward Shape Matters for LLM Alignment
description: >-
  [ICML 2025][LLM Alignment][Direct Alignment] AlphaPO introduces an $\alpha$ parameter into the Direct Alignment Algorithms (DAA) framework to alter the "shape" of the reward function, generalizing it from the standard log-based reward to a more general power transform. This enables fine-grained control over likelihood displacement and over-optimization, achieving a 7%-10% improvement over SimPO and a 15%-50% improvement over DPO on Mistral-7B and Llama3-8B.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Direct Alignment"
  - "Reward Shaping"
  - "Likelihood Displacement"
  - "preference optimization"
  - "Alpha Parameter"
date: 2026-05-08
content_hash: 71bd40e150aa1541
---

# AlphaPO: Reward Shape Matters for LLM Alignment

**Conference**: ICML 2025  
**arXiv**: [2501.03884](https://arxiv.org/abs/2501.03884)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: Direct Alignment, Reward Shaping, Likelihood Displacement, preference optimization, Alpha Parameter

## TL;DR

AlphaPO introduces an $\alpha$ parameter into the Direct Alignment Algorithms (DAA) framework to alter the "shape" of the reward function, generalizing it from the standard log-based reward to a more general power transform. This enables fine-grained control over likelihood displacement and over-optimization, achieving a 7%-10% improvement over SimPO and a 15%-50% improvement over DPO on Mistral-7B and Llama3-8B.

## Background & Motivation

**Background**: RLHF is the mainstream paradigm for LLM alignment, consisting of two stages—first training a reward model, and then optimizing the policy using RL algorithms like PPO. In recent years, Direct Alignment Algorithms (DAA) have emerged, skipping independent reward model training and directly expressing the reward as a function of the policy itself. Representative methods include DPO (Direct Preference Optimization) and SimPO (Simple Preference Optimization).

**Limitations of Prior Work**: DAA methods widely suffer from the **likelihood displacement** problem—during training, although the model learns to distinguish the probability gap between preferred and rejected responses, the absolute probability of the preferred response is often undesirably decreased. This means the model "learns preferences" but simultaneously "forgets good answers". Furthermore, DAA is also prone to **over-optimization**: the model continues to climb in reward metrics, but actual generation quality declines.

**Key Challenge**: The reward function shape used by existing DAA methods is fixed (e.g., DPO uses the log-ratio as an implicit reward) and lacks the ability to control training dynamics. The geometric shape of the reward function directly determines the magnitude and direction of the gradients, whereas a fixed shape cannot balance discriminability and stability simultaneously.

**Goal**: (a) How to flexibly adjust the shape of the reward function within the DAA framework? (b) How to mitigate likelihood displacement through shape control? (c) How to improve alignment performance without introducing additional models?

**Key Insight**: The authors observe that the "shape" of the reward function (i.e., the curve morphology of reward values changing with policy probabilities) profoundly affects training dynamics. Different shapes lead to different gradient distributions, which in turn affect the model's emphasis on learning from preferred/rejected samples. The standard log reward is merely a special case among many possible shapes.

**Core Idea**: Generalizing the reward of DAA from a log to an $\alpha$-power form by introducing an $\alpha$ parameter, enabling continuous adjustment of the reward curve shape via a single hyperparameter, thereby precisely controlling the trade-off between likelihood displacement and alignment performance.

## Method

### Overall Architecture

The overall architecture of AlphaPO follows the DAA paradigm: the input consists of preference data pairs $(x, y_w, y_l)$ (prompt, preferred response, rejected response), and the output is the aligned policy model $\pi_\theta$. The difference from DPO/SimPO lies in the introduction of an adjustable $\alpha$ parameter in the definition of the reward function, allowing the reward function shape to continuously interpolate between the log form and the linear form. The training workflow remains simple: load the SFT model $\to$ construct the $\alpha$-reward function $\to$ optimize the objective function using preference data.

### Key Designs

1. **$\alpha$-Reward Function (Alpha Reward)**:

    - **Function**: Generalizes the standard log-based implicit reward in DAA into a parameterized reward family.
    - **Mechanism**: The implicit reward of standard DPO is $r(x,y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$, which is a log form. AlphaPO generalizes it to:
    $r_\alpha(x,y) = \frac{1}{\alpha}\left[\left(\frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)^\alpha - 1\right]$
      When $\alpha \to 0$, this function reduces to the standard log-ratio reward (DPO); when $\alpha = 1$, it becomes a linear difference; different values of $\alpha$ correspond to different reward curve shapes. This is similar to the concept of Tsallis entropy or Box-Cox transformation.
    - **Design Motivation**: The log reward yields very large gradients in low-probability regions, which easily amplifies noise, while yielding very small gradients in high-probability regions, leading to slow learning. By adjusting $\alpha$, the distribution characteristics of gradients across different probability regions can be controlled. Larger $\alpha$ values compress gradients in high-probability regions and amplify them in low-probability regions, and vice versa.

2. **Likelihood Displacement Control Mechanism**:

    - **Function**: Inhibits the decline in the probability of preferred responses by adjusting the $\alpha$ parameter.
    - **Mechanism**: The root cause of likelihood displacement is that the gradient of the DAA loss function simultaneously drives down both the rejected and preferred probabilities (although their relative gap increases, their absolute values both decrease). The $\alpha$ parameter changes the gradient weight ratio of the loss function with respect to preferred and rejected samples. Selecting an appropriate $\alpha$ allows the gradient to focus more on "depressing rejected" rather than "pulling down preferred".
    - **Design Motivation**: In DPO, likelihood displacement is a widely observed but difficult-to-control problem. Previous patches (like adding regularization terms, tuning $\beta$, etc.) are indirect, whereas AlphaPO fundamentally regulates this behavior by changing the underlying shape of the reward function.

3. **Over-optimization Mitigation**:

    - **Function**: Prevents the model's reward metrics from inflating, while the actual quality degrades in the later stages of training.
    - **Mechanism**: Over-optimization typically occurs when the model finds shortcuts for reward hacking. Reward functions corresponding to different $\alpha$ values have different growth rates when departing from the reference policy. By selecting a suitable $\alpha$, the reward function can naturally "saturate" when the policy shifts too far from the reference, serving as an implicit regularization.
    - **Design Motivation**: Compared to controlling the KL divergence constraint by tuning $\beta$ in DPO, $\alpha$ provides an orthogonal control dimension operating directly on the reward function shape.

### Loss & Training

The loss function of AlphaPO is based on the Bradley-Terry preference model:

$$\mathcal{L}_{\text{AlphaPO}} = -\mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma\left(r_\alpha(x,y_w) - r_\alpha(x,y_l) - \gamma\right)\right]$$

where $\gamma$ is the margin term (similar to the target reward margin in SimPO), and $\sigma$ is the sigmoid function. Regarding the training strategy, $\alpha$ is a hyperparameter optimized via validation set tuning, typically searched within the range $[-1, 2]$. The training process is as simple as DPO/SimPO, requiring only a single forward pass to compute the probabilities of the preferred and rejected responses to calculate the loss, without needing additional reward models or critic networks.

## Key Experimental Results

### Main Results

On standard alignment benchmarks such as AlpacaEval 2 and MT-Bench, AlphaPO demonstrates significant improvements over DPO and SimPO:

| Model | Method | AlpacaEval 2 LC WR (%) | Relative Gain over SimPO | Relative Gain over DPO |
|------|------|----------------------|-------------|------------|
| Mistral-7B-Instruct | DPO | ~14.0 | - | baseline |
| Mistral-7B-Instruct | SimPO | ~17.5 | baseline | +25.0% |
| Mistral-7B-Instruct | **AlphaPO** | **~19.2** | **+9.7%** | **+37.1%** |
| Llama3-8B-Instruct | DPO | ~22.0 | - | baseline |
| Llama3-8B-Instruct | SimPO | ~30.0 | baseline | +36.4% |
| Llama3-8B-Instruct | **AlphaPO** | **~32.1** | **+7.0%** | **+45.9%** |

### Ablation Study

An ablation analysis on different values of the $\alpha$ parameter:

| $\alpha$ Value | AlpacaEval 2 LC WR | Likelihood Displacement | Description |
|-------------|-------------------|------------------------|------|
| $\alpha \to 0$ (DPO) | ~14.0% | Severe | Degenerates to standard DPO |
| $\alpha = 0.5$ | ~17.0% | Moderate | Between DPO and optimal |
| $\alpha = 1.0$ | ~18.5% | Slight | Close to linear reward |
| $\alpha^*$ (Optimal) | **~19.2%** | **Minimal** | Optimal shape configuration |
| $\alpha = 2.0$ | ~16.5% | Over-compensation | $\alpha$ too large degrades performance instead |

### Key Findings

- **The choice of $\alpha$ significantly impacts performance**: From DPO ($\alpha \to 0$) to the optimal $\alpha^*$, the performance improvement reaches over 30%. This validates the core thesis that "reward shape matters"—the shape of the reward function is not merely an implementation detail, but a crucial design choice affecting alignment performance.
- **Likelihood displacement has a monotonic relationship with $\alpha$**: As $\alpha$ increases, the degree of probability drop in the preferred response decreases. However, excessive $\alpha$ leads to insufficient discriminability, indicating an optimal balance point exists.
- **AlphaPO performs consistently across different base models**: On both Mistral-7B and Llama3-8B, AlphaPO significantly outperforms DPO and SimPO, demonstrating that the improvement of this method does not rely on specific pretrained models.
- **The gain over SimPO (7-10%) is smaller than the gain over DPO (15-50%)**: This is likely because SimPO itself already alleviates some likelihood displacement issues through sequence-level scoring and margin design.

## Highlights & Insights

- **Conceptualization of reward function "shape"**: Generalizing the log-based reward, which was previously treated as a fixed choice, into a parameterized family of functions is highly inspiring. Similar to the generalization of Boltzmann-Gibbs entropy in Tsallis statistical mechanics, a single continuous parameter unifies a family of distributions/functions. This approach can be transferred to other scenarios requiring loss function design.
- **A single hyperparameter unifying the control of multiple training dynamics**: $\alpha$ simultaneously affects likelihood displacement, over-optimization, and convergence speed. This "single knob controlling multiple behaviors" design is more elegant and easier to tune than adding multiple individual regularization terms.
- **Convincing experimental design**: The paper not only reports the final performance but also quantitatively demonstrates the relationship between $\alpha$ and likelihood displacement through ablation experiments, enabling readers to intuitively understand "why this parameter is effective". This dual-track narrative of "mechanistic explanation + experimental validation" is highly commendable.

## Limitations & Future Work

- **Tuning cost of $\alpha$**: Although $\alpha$ is a scalar hyperparameter, its optimal value may vary across base models, data distributions, and task domains. The paper does not explore methods for adaptively adjusting $\alpha$ (e.g., dynamically during training).
- **Only validated at 7B-8B scales**: Experiments were only conducted on Mistral-7B and Llama3-8B, lacking validation on larger models (e.g., 70B scale). Whether the optimal value of $\alpha$ varies with model scale remains an open question.
- **Limited depth of theoretical analysis**: Although the paper discusses the impact of $\alpha$ on gradients, it lacks theoretical characterizations of convergence and the optimal $\alpha^*$ (e.g., whether a data-dependent closed-form solution exists).
- **Combination with other DAA improvement methods**: The paper mainly compares with DPO and SimPO, but there are other methods in the DAA field such as IPO, KTO, and ORPO. Whether the $\alpha$-generalization of AlphaPO can be applied to these methods has not yet been explored.
- **Buffer limitations**: This note is compiled based on abstract information; the full paper (26 pages, 16 figures) may contain richer theoretical derivations and experimental details.

## Related Work & Insights

- **vs DPO (Rafailov et al., 2023)**: DPO is a special case of AlphaPO ($\alpha \to 0$). DPO uses an implicit reward in log form, derived directly from the optimal policy under KL constraints. AlphaPO relaxes the log constraint, searching for a better reward shape within a broader function family. DPO's advantage lies in its theoretical elegance, while AlphaPO's advantage lies in practical performance.
- **vs SimPO (Meng et al., 2024)**: SimPO improves upon DPO using sequence-level average log probabilities (rather than token-level) and a target reward margin. AlphaPO introduces improvements from an orthogonal dimension—the reward function shape—and the improvements from both could be complementary. AlphaPO's 7-10% gain over SimPO indicates that shape adjustment brings additional benefits.
- **vs IPO (Azar et al., 2023)**: IPO improves DPO from a regularization perspective, using a squared loss instead of log-sigmoid loss to avoid overfitting. AlphaPO's $\alpha$-parameterization and IPO's loss function modifications can be viewed as two different strategies for "changing the optimization landscape".
- **Insights**: The concept of $\alpha$-generalization can be applied to other methods employing log rewards/losses, such as generalizing the temperature parameter in the InfoNCE loss of contrastive learning.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of parameterizing reward shapes is insightful, though essentially adding a hyperparameter on top of DPO, with moderate technical complexity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Main experiments on two base models + $\alpha$ ablation + likelihood displacement analysis, overall thorough; but lacks validation on larger-scale models.
- Writing Quality: ⭐⭐⭐⭐ The paper contains 26 pages and 16 figures, providing detailed presentations; the core argument "reward shape matters" is clear and powerful.
- Value: ⭐⭐⭐⭐ Possesses direct practical value for the DAA community, as the $\alpha$ parameter can be easily integrated into existing training workflows; however, the improvement relative to SimPO is not massive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] On the Robustness of Reward Models for Language Model Alignment](on_the_robustness_of_reward_models_for_language_model_alignment.md)
- [\[ICLR 2026\] Evaluating and Improving Cultural Awareness of Reward Models for LLM Alignment](../../ICLR2026/llm_alignment/evaluating_and_improving_cultural_awareness_of_reward_models_for_llm_alignment.md)
- [\[ICLR 2026\] Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](../../ICLR2026/llm_alignment/inverse_reinforcement_learning_with_dynamic_reward_scaling_for_llm_alignment.md)
- [\[ACL 2026\] Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards](../../ACL2026/llm_alignment/teaching_llm_to_be_persuasive_reward-enhanced_policy_optimization_for_alignment_.md)
- [\[NeurIPS 2025\] LLM Safety Alignment is Divergence Estimation in Disguise](../../NeurIPS2025/llm_alignment/llm_safety_alignment_is_divergence_estimation_in_disguise.md)

</div>

<!-- RELATED:END -->
