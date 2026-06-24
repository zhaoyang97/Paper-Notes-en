---
title: >-
  [Paper Note] Dynamic Important Example Mining for Reinforcement Finetuning
description: >-
  [CVPR 2026][Reasoning][Reinforcement Finetuning] In each training step of RFT (GRPO/PPO, etc.), DIEM uses the "inner product between single-sample gradients and the total batch gradient" to estimate the marginal contribution of each sample to current policy improvement in real-time. It then solves a constrained optimization problem to reweight samples while maintaining the gradient magnitude. With nearly zero extra overhead (+1.3% time), it improves multimodal reasoning bench…
tags:
  - "CVPR 2026"
  - "Reasoning"
  - "Reinforcement Finetuning"
  - "Data Selection"
  - "Gradient Alignment"
  - "Example Reweighting"
  - "Curriculum Learning"
date: 2026-05-08
content_hash: 0961fd5958646a62
---

# Dynamic Important Example Mining for Reinforcement Finetuning

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Dynamic_Important_Example_Mining_for_Reinforcement_Finetuning_CVPR_2026_paper.html)  
**Code**: https://github.com/hrtan/DIEM  
**Area**: LLM Reasoning / Reinforcement Finetuning (RFT)  
**Keywords**: Reinforcement Finetuning, Data Selection, Gradient Alignment, Example Reweighting, Curriculum Learning  

## TL;DR
In each training step of RFT (GRPO/PPO, etc.), DIEM uses the "inner product between single-sample gradients and the total batch gradient" to estimate the marginal contribution of each sample to current policy improvement in real-time. It then solves a constrained optimization problem to reweight samples while maintaining the gradient magnitude. With nearly zero extra overhead (+1.3% time), it improves multimodal reasoning benchmarks by 1–6 points on average.

## Background & Motivation
**Background**: Reinforcement Fine-Tuning (RFT) has become a mainstream post-training paradigm for enhancing the reasoning capabilities of large models, especially multimodal large models (MLMs). Methods like GRPO and PPO allow models to learn directly from reward signals rather than simple imitation of supervised data. The effectiveness of RFT highly depends on "how training data is used": which samples are selected and what weights are assigned directly determines optimization stability and final generalization.

**Limitations of Prior Work**: The vast majority of data-centric RFT methods treat sample importance as **fixed**. Static methods (e.g., LIMR focusing on reward trends, HVS on reward variance) select subsets once before training starts. Dynamic methods (e.g., PCL training an auxiliary value model for difficulty, SPEED-RL using pass rates to prioritize medium difficulty) adjust the sequence during training but rely on **external heuristic metrics**.

**Key Challenge**: RFT is inherently non-stationary—the value of the same sample to the policy differs significantly between early and late training stages. Heuristic metrics have two fundamental flaws: (1) They score from **outside** the policy, failing to reflect the policy's own "preference/alignment" with the sample at that moment; (2) They cannot measure the **true marginal contribution** of a sample to the current policy update. Treating a value that drifts during training as a constant leads to suboptimal updates.

**Goal**: To transform data selection from a "one-time preprocessing" step into an endogenous component embedded within the optimization loop that dynamically adapts with the policy, while ensuring it (a) is theoretically grounded, (b) adds almost no computational overhead, and (c) is plug-and-play compatible with various RFT algorithms.

**Key Insight**: Instead of external scoring, one should ask—"If sample $z$ is removed from this update, will the total batch reward improve or worsen?" This "leave-one-out" marginal contribution naturally aligns with the policy, but calculating it precisely requires $|B_t|$ full gradient updates, which is infeasible. The key observation of the authors is that this marginal contribution can be approximated via a first-order Taylor expansion using **already calculated gradients** at no extra backpropagation cost.

**Core Idea**: Use the "inner product of the single-sample gradient $G_z$ and the total batch gradient $G_{B_t}$" as a low-cost proxy for instantaneous sample importance. Reweighting is then formulated as a constrained optimization problem: "maximize weighted importance while locking the total gradient magnitude." Solving this in closed-form reshapes the gradient direction with zero additional training.

## Method

### Overall Architecture
DIEM does not modify the backbone of any RFT algorithm. Instead, it inserts two lightweight steps after the standard forward-backward pass and before parameter update: first, **estimate the dynamic importance of each sample**, then **reweight by importance** to obtain a new gradient direction for policy update. The inputs are the "per-sample gradient $G_z$ + total batch gradient" already computed in standard RFT for the current minibatch $B_t$, and the output is the reweighted total gradient $G_{\text{weighted}}$. Since it only reuses existing gradients and inverts a small $N \times N$ matrix ($N$ being minibatch size), the overhead per step is negligible.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["minibatch B_t<br/>Standard RFT Forward + Backward"] --> B["Per-sample Gradient G_z<br/>+ Batch Total Gradient G_Bt"]
    B --> C["Gradient Alignment Importance Estimation<br/>Î(z)=η⟨G_z, G_Bt⟩"]
    C --> D["Constrained Reweighting<br/>max IᵀW s.t. Constant Magnitude<br/>Closed-form Solution W*=P⁻¹I·√C/√(IᵀP⁻¹I)"]
    D -->|Clip Negative Weights max(0,·)| E["Weighted Gradient<br/>G_weighted = W*ᵀG"]
    E --> F["Policy Update θ_{t+1}"]
```

### Key Designs

**1. Leave-one-out Marginal Contribution: Defining "Sample Value" as Interpretable Policy Improvement**

To measure sample value dynamically and intrinsically, the authors provide a **precise but expensive** definition: The importance $I_t(z)$ of sample $z$ at step $t$ equals the "total reward after updating with the whole batch $B_t$" minus the "total reward after updating with $B_t \setminus \{z\}$":

$$I_t(z) = J\!\left(\theta^{\text{update}}_{B_t},\, B_t\right) - J\!\left(\theta^{\text{update}}_{B_t\setminus\{z\}},\, B_t\right)$$

$I_t(z) > 0$ indicates that including $z$ improves the policy's performance on the current batch (beneficial), while $< 0$ indicates $z$ worsens it (harmful at this moment), and values near $0$ are negligible. The benefit of this definition is that it is no longer an external heuristic like "difficulty/entropy" but a direct, directional measure of the "true marginal utility of the sample for policy improvement," which naturally evolves during training.

**2. Gradient Alignment Estimator: Reducing Expensive Leave-one-out to a Single Inner Product**

Calculating $I_t(z)$ directly would require one full gradient update and evaluation for every sample in the batch (at least $|B_t|$ times), which is unaffordable in high-throughput training. DIEM provides a first-order approximation (Proposition 1): The instantaneous value of a sample $\approx$ how much its gradient direction "supports" the collective update direction of the batch, measured by the inner product:

$$\hat I_t(z) = \eta_t\,\big\langle G^{(t)}_z,\, G^{(t)}_{B_t} \big\rangle$$

where $G^{(t)}_z$ is the policy gradient contributed by a single sample $z$, and $G^{(t)}_{B_t}$ is the aggregated gradient of the whole batch. Since both are already computed in standard backpropagation, this step is nearly free. The intuition is clear: a large positive inner product means the sample gradient is highly consistent with the collective direction, marking it as a "representative, high-utility" sample that accelerates convergence; a negative value means the gradient points opposite to the collective, indicating noise or currently harmful samples. The authors also provide an error bound (Proposition 2): assuming the log-likelihood is $\ell$-Lipschitz and the advantage is bounded by $A_{\max}$, $|I_t(z) - \hat I_t(z)| \le O(\eta_t\ell^2 + 2\eta_t\ell A_{\max})$. Crucially, this does not rely on convexity or "proximity to a stationary point," making it suitable for the highly non-convex, non-stationary early stages of RFT.

**3. Magnitude-Preserving Constrained Reweighting: Amplifying High-Value Samples without Altering Update Magnitude**

Given the importance vector $I \in \mathbb{R}^N$, how should it be used? A naive approach (simple scaling by $I$) would change the L2 magnitude of the total gradient, effectively altering the equivalent learning rate and undermining optimization stability. DIEM formulates reweighting as a constrained optimization: maximize weighted utility while forcing the L2 magnitude of the reweighted total gradient to equal that of the original unweighted gradient:

$$\max_{W}\; I^\top W \quad \text{s.t.}\quad \big\|W^\top G\big\|_2 = \big\|\mathbf{1}^\top G\big\|_2$$

The objective $I^\top W$ implicitly tilts weights toward high-impact samples; the constraint locks the update magnitude, ensuring "direction changes, step size remains." Solving for the stationary point using Lagrange multipliers yields a quasi-closed-form solution (where $P = GG^\top$ is the Gram matrix of gradients and $C = \|\mathbf{1}^\top G\|^2$ is the squared magnitude of the original total gradient):

$$W^* = \frac{P^{-1}I\,\sqrt{C}}{\sqrt{I^\top P^{-1}I}}$$

This calculation only requires inverting a small $N \times N$ matrix $P$, which is negligible compared to the cost of an RFT step. Since $W^*$ may contain negative values (indicating truly harmful samples or estimation errors), a non-negative post-processing step $W^* \leftarrow \max(0, W^*)$ is applied, ensuring only constructive contributions enter the update. The final weighted gradient $G_{\text{weighted}} = W^{*\top}G$ is used for the policy update $\theta_{t+1} = \theta_t + \eta_t G_{\text{weighted}}$. This step fuses "importance measurement" with "optimized stability."

### Loss & Training
DIEM does not introduce new loss functions; it follows the underlying RFT algorithm (primarily GRPO, where advantages use group-relative normalization $A(s,a_i) = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$, with PPO-style clipping and KL regularization). Its only change is reshaping the gradient per step using $W^*$. The overall process is detailed in Algorithm 1: compute per-sample gradients $\rightarrow$ inner product for $I$ $\rightarrow$ solve $W^*$ $\rightarrow$ clip negatives $\rightarrow$ weighted update. The mechanism is agnostic to policy update rules (PPO/GRPO/TRPO/Reinforce++).

## Key Experimental Results

### Main Results
Using Qwen2.5-VL-7B / 32B as backbones and 52K multimodal samples randomly sampled from the MM-Eureka corpus for RFT, DIEM is compared against static (LIMR, HVS) and dynamic (PCL, SPEED-RL) data selection baselines across 6 multimodal reasoning benchmarks (average scores):

| Backbone | Method | MathVista | MathVerse | MMMU | Average |
|------|------|-----------|-----------|------|---------|
| 7B | Vanilla RFT | 74.1 | 51.7 | 57.1 | 59.1 |
| 7B | SPEED-RL (Runner-up) | 74.9 | 47.1 | 59.2 | 60.0 |
| 7B | **DIEM** | **76.9** | **53.0** | 59.2 | **61.8** |
| 32B | Vanilla RFT | 75.6 | 52.7 | 69.0 | 64.9 |
| 32B | SPEED-RL (Runner-up) | 75.1 | 53.9 | 70.6 | 65.6 |
| 32B | **DIEM** | **76.9** | **58.0** | **71.9** | **67.3** |

On the 7B model, DIEM achieves an average of 61.8%, +3.6 points over Vanilla RFT and +1.8 points over the strongest baseline SPEED-RL, even slightly exceeding GPT-4o (60.9%). On the 32B model, it reaches 67.3%, ranking first across all 6 benchmarks.

Regarding generalization across RFT algorithms (MathVerse / Qwen2.5-VL-32B), DIEM is plug-and-play and consistently leads:

| Method | GRPO | PPO | Reinforce++ | TRPO |
|------|------|-----|-------------|------|
| Vanilla RFT | 52.7 | 53.6 | 51.2 | 50.5 |
| PCL | 54.1 | 55.0 | 54.8 | 55.1 |
| **DIEM** | **58.0** | **57.3** | **56.9** | **57.2** |

Under GRPO, it outperforms the runner-up PCL by 3.9 points (58.0 vs. 54.1), with stable leads across other optimizers.

### Ablation Study
MathVerse / Qwen2.5-VL-32B, full model score 58.0:

| Configuration | Performance | Note |
|------|------|------|
| DIEM (Full) | 58.0 | — |
| Importance replaced by Random value | 53.0 | Gain -5.0, proves score effectiveness |
| Importance replaced by Pass@k score | 53.2 | Gain -4.8 |
| Importance replaced by Difficulty score (PCL) | 52.1 | Gain -5.9, largest drop |
| Importance replaced by dist. to Pass@k median | 54.9 | Improvement but still -3.1 |
| Reweighting replaced by NULL-operation | 55.4 | Gain -2.6 |
| Reweighting replaced by Softmax normalization | 56.4 | Gain -1.6 |

### Key Findings
- **Gradient Alignment Score > All Heuristic Metrics**: Replacing the dynamic influence score with Random/Pass@k/Difficulty results in drops of 4.8–5.9 points. Even converting these heuristics into "distance to median" only recovers part of the performance (up to 54.9), still falling short of DIEM. This indicates that scoring based on internal policy gradients is fundamentally superior to external heuristics.
- **Magnitude-Preserving Reweighting is Essential**: Removing reweighting (NULL) drops 2.6 points; replacing it with standard Softmax normalization drops 1.6 points. The custom-designed constrained reweighting is necessary to maximize the benefits of importance scoring.
- **Almost Zero Overhead**: Vanilla RFT (GRPO) takes 70.3 hours; adding DIEM takes only 71.2 hours (+ approx. 1.3%). In contrast, PCL takes 79.1h, SPEED-RL 94.6h, and LIMR/HVS take 122.0h. This is because DIEM trains no proxy models and purely reuses computed gradients.
- **Emergent Curriculum Learning**: By partitioning samples into Hard/Medium/Easy via Pass@k and tracking DIEM's weight trajectories (Fig. 3), it was found that Easy/Medium samples have high weights early on. As training progresses, Easy weights drop rapidly while Hard weights continue to rise (with oscillations). This "easy-to-hard" curriculum emerges spontaneously from the model rather than being hardcoded via heuristics as in PCL or SPEED-RL.

## Highlights & Insights
- **"Free Marginal Contribution Estimation" is the most ingenious part**: The leave-one-out method originally requiring $O(|B_t|)$ updates is reduced to a single inner product of per-sample and batch gradients. Since these are available from the standard backprop, influence functions are obtained at zero cost, with error bounds even tighter than classic influence results in supervised learning.
- **Portability of "Direction Change, Constant Magnitude"**: Decoupling the "adjustment of update direction" from "control of effective learning rate" by constraining the L2 magnitude is a concept that can be transferred to any scenario involving gradient reweighting where optimization stability is a concern (e.g., training with noisy labels or multi-task gradient merging).
- **Curriculum Learning: From "Human-designed" to "Spontaneous"**: DIEM includes no explicit easy-to-hard rules, yet the curriculum emerges naturally. This demonstrates that "marginal contribution aligned with the policy itself" inherently encodes "what should be learned at this moment."

## Limitations & Future Work
- **Applicability of First-order Approximation**: The error bound depends on the learning rate $\eta_t$ and the Lipschitz constant $\ell$. In scenarios with extremely large learning rates or highly unsmooth rewards, the approximation might degrade. The paper lacks empirical evidence for such extreme cases.
- **Scalability of Gram Matrix Inversion**: The closed-form solution requires inverting an $N \times N$ matrix $P$. While negligible for standard minibatches, the cost and numerical stability of $P^{-1}$ would need re-evaluation if reweighting were applied at massive batch sizes or token-level granularity.
- **Narrow Evaluation Domain**: Experiments are confined to multimodal math and general reasoning benchmarks. Performance on pure-text LLM reasoning (e.g., code, long-chain mathematical proofs) or alignment-oriented RLHF tasks has not been verified.
- **Coarseness of Negative Weight Clipping**: The $\max(0, W^*)$ operation crudely zeros out all negative-weight samples, which might accidentally discard useful samples misidentified by estimation errors. A more refined soft-weighting approach might yield further gains.

## Related Work & Insights
- **vs. Static Selection (LIMR / HVS)**: These select subsets before RFT based on reward trends/variance, assuming importance is constant. DIEM reassesses importance every step, invalidating the "constant importance" assumption and saving the 122-hour overhead of training their proxy models.
- **vs. Dynamic Heuristics (PCL / SPEED-RL)**: PCL trains an external value model for difficulty, while SPEED-RL uses pass rates; both are **external** to the policy. DIEM uses internal gradient alignment scores, capturing what the policy "truly prefers."
- **vs. Classic Influence Functions**: Traditional influence measures often require convexity or proximity to a stationary point. DIEM's estimator does not, being specifically designed for the non-convex, non-stationary early stages of RFT, resulting in tighter error bounds.

## Rating
- Novelty: ⭐⭐⭐⭐ Approximating the leave-one-out influence function as a "free inner product" combined with magnitude-preserving reweighting is a novel and self-consistent angle, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two backbone scales × 6 benchmarks × 4 RFT algorithms + speed tests + curriculum visualization. Lacks only pure-text/alignment task verification.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain from motivation to definition to approximation to constrained solution. Complete algorithms/formulas, though core derivations are in the appendix.
- Value: ⭐⭐⭐⭐ Plug-and-play, near-zero overhead, and stable 1–6% gains. Directly useful for anyone performing data-centric optimization for RFT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A Simple "Motivation" Can Enhance Reinforcement Finetuning of Large Reasoning Models](../../ICLR2026/llm_reasoning/a_simple_motivation_can_enhance_reinforcement_finetuning_of_large_reasoning_mode.md)
- [\[ICLR 2026\] Dynamics-Predictive Sampling for Active RL Finetuning of Large Reasoning Models](../../ICLR2026/llm_reasoning/dynamics-predictive_sampling_for_active_rl_finetuning_of_large_reasoning_models.md)
- [\[ICLR 2026\] Dynamic Early Exit in Reasoning Models](../../ICLR2026/llm_reasoning/dynamic_early_exit_in_reasoning_models.md)
- [\[CVPR 2026\] Scaling Agentic Reinforcement Learning for Tool-Integrated Reasoning in VLMs](scaling_agentic_reinforcement_learning_for_tool-integrated_reasoning_in_vlms.md)
- [\[ICLR 2026\] WavefrontDiffusion: Dynamic Decoding Schedule for Improved Reasoning](../../ICLR2026/llm_reasoning/wavefrontdiffusion_dynamic_decoding_schedule_for_improved_reasoning.md)

</div>

<!-- RELATED:END -->
