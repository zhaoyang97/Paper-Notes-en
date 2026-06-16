---
title: >-
  [Paper Note] Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance
description: >-
  [CVPR 2026][Multimodal VLM][Paper Note] This work redefines catastrophic forgetting in Multimodal Continual Instruction Tuning (MCIT) as the "absence of gradients from old tasks during new task training." DGG approximates the old task gradients using a "direction vector from current parameters to the optimal parameters of previous tasks," adds this to the re
tags:
  - CVPR 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 8c052325f5beac8c
---
# Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html)  
**Code**: https://github.com/lisongze/DGG  
**Area**: Multimodal VLM / Continual Learning  
**Keywords**: Multimodal Continual Instruction Tuning, Catastrophic Forgetting, Gradient Approximation, Replay, Bernoulli Sampling

## TL;DR
This work redefines catastrophic forgetting in Multimodal Continual Instruction Tuning (MCIT) as the "absence of gradients from old tasks during new task training." DGG approximates the old task gradients using a "direction vector from current parameters to the optimal parameters of previous tasks," adds this to the real gradients from a limited replay buffer, and dynamically regulates the update frequency using Bernoulli sampling. Without **expanding the model**, DGG achieves SOTA on VQAv2 and UCIT.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs), after large-scale pre-training and instruction tuning, often require continual fine-tuning on new instruction datasets (MCIT). Mainstream MCIT methods mostly rely on LLaVA + LoRA, using Mixture-of-Experts (MoE) or prompt tuning to capture task-specific knowledge (CoIN, CL-MoE, HiDE, DISCO, ModalPrompt).

**Limitations of Prior Work**: (1) Methods learning task-specific components **inevitably lead to model expansion**, incurring extra parameter storage and computational overhead during both training and inference; (2) Regularization methods that do not expand the model (e.g., RegLoRA in SEFE) constrain parameter updates to preserve old knowledge but mostly use **static regularization terms**, which remain fixed throughout the learning process and fail to adapt to the evolving optimization landscape.

**Key Challenge**: Choosing between expanding the model for memory (at the cost of bloat) or using static regularization (at the cost of inflexibility). Can old task memory be consolidated **dynamically** without **expanding the model**?

**Goal**: To provide an old knowledge preservation mechanism that adjusts dynamically during training and can be integrated with replay, without adding any model components.

**Key Insight**: The authors view catastrophic forgetting from a new perspective—the loss for joint training of all tasks satisfies additivity $L(\theta;T_1\cup T_2)=L(\theta;T_1)+L(\theta;T_2)$, and gradients are also additive $\nabla L(\theta;T_1\cup T_2)=\nabla L(\theta;T_1)+\nabla L(\theta;T_2)$. When learning $T_2$ in a continual setting, the data for task $T_1$ is unavailable, resulting in the **absence of the $\nabla L(\theta;T_1)$ term**. Gradient descent then converges to the new task optimum rather than the joint optimum $\theta^*_{1:2}$, which is the source of forgetting.

**Core Idea**: Reformulate "preserving old knowledge" as a problem of "approximating the missing old task gradients"—using the direction vector "current parameters $\rightarrow$ old task optimal parameters" to approximate $\nabla L(\theta;T_1)$ and compensate for the missing gradient term.

## Method

### Overall Architecture
DGG does not modify the MLLM architecture (LLaVA-7B + LoRA) but focuses on the **optimization level**. When learning a task sequence $\{T_1,\dots,T_T\}$, for each new task $T_t$, the model can calculate the real gradient of the current task (including the replay buffer) $\nabla L(\theta;T_t\cup M)$, but lacks the old task gradients. DGG performs three actions: (1) Approximates the old task gradient $\hat g$ using the direction vector from the "current parameters $\theta$ to the cumulative optimal parameters $\theta^*_{1:t-1}$" (Gradient Guidance); (2) Adds $\hat g$ to the real gradient of the replay buffer $M$ to obtain a more accurate approximation of the old task gradient; (3) Uses Bernoulli sampling with probability $\alpha$ to decide whether to inject $\hat g$ at each step, dynamically balancing stability (remembering the old) and plasticity (learning the new). This method is a "plug-and-play" modification to the optimizer gradients (see Algorithm 1 in the original paper) and introduces no new parameters. As it is a pure optimization-level gradient approximation and regulation rather than a multi-stage pipeline, no framework diagram is provided.

### Key Designs

**1. Gradient Guidance Approximation: Compensating missing gradients using a direction vector pointing to the old task optimum**

Caching a small number of old samples for replay is common, but a small buffer cannot represent the expected gradient of the entire $T_1$ distribution throughout the gradient descent process, leading to approximations biased toward the current task. The authors' insight is: when learning $T_1$, gradient descent eventually converges to $\theta^*_1=\arg\min_\theta L(\theta;T_1)$; thus, the **"direction pointing to $\theta^*_1$" reflects the expected gradient direction of the entire optimization trajectory to some extent**. Accordingly, $\theta-\theta^*_1$ is used to approximate $\nabla L(\theta;T_1)$ while learning $T_2$. To prevent the magnitude of the pure direction vector from being too large, it is **normalized** using the current task gradient magnitude:
$$\hat g=\begin{cases}\dfrac{\theta-\theta^*_{1:t-1}}{\|\theta-\theta^*_{1:t-1}\|}\cdot\|\nabla L(\theta;T_t)\|, & \text{if }\|\theta-\theta^*_{1:t-1}\|>\|\nabla L(\theta;T_t)\|\\[2mm] \theta-\theta^*_{1:t-1}, & \text{otherwise}\end{cases}$$
The direction vector is scaled only when its norm exceeds that of the current gradient. For subsequent tasks where $t\ge 2$, all old tasks are treated as a joint task, and $\hat g$ is computed using the continuously accumulated optimal parameters $\theta^*_{1:t-1}$. The key to this design is **using parameter space geometry (displacement to the old optimum) to replace unavailable old data gradients**, eliminating the need to save full old task data.

**2. Fusion with Replay Buffer: Mutual compensation of directional approximation and real gradients**

$\hat g$ provides a geometric approximation, while the replay gradient $\nabla L(\theta;M)$ provides real but sparse samples. The two are complementary:
$$\nabla L(\theta;T_1)\approx \hat g+\nabla L(\theta;M),\qquad \nabla L(\theta;\textstyle\bigcup_{i=1}^t T_i)\approx \hat g+\nabla L(\theta;T_t\cup M).$$
Their relative importance **switches automatically based on data distribution differences**: Experiments show that on the in-domain VQAv2 (where 10 sub-tasks come from the same visual domain), $\hat g$ dominates, and using $\hat g$ alone without replay achieves 64.61% FAA, surpassing the strongest baseline. Conversely, on UCIT, which spans 6 highly diverse datasets, replay is more important, and $\hat g$ alone (53.71%) is inferior to replay alone (57.13%). The authors' explanation: when the distribution shift is large, the alignment between the "direction to the old optimum" and the true old gradient decreases, damaging the accuracy of $\hat g$, thus increasing reliance on replay—though $\hat g$ still retains valuable gradient information.

**3. Dynamic Gradient Update via Bernoulli Sampling: Randomly regulating the injection frequency of $\hat g$**

Injecting $\hat g$ at every step would bias the model excessively toward old tasks and weaken its ability to learn new ones (plasticity). The authors introduce a Bernoulli random variable $B(\alpha)$ to decide whether to inject the old task gradient at each optimization step:
$$\nabla L\big(\theta;\textstyle\bigcup_{i=1}^t T_i\big)=\begin{cases}\hat g+\nabla L(\theta;T_t\cup M), & B(\alpha)=1\\ \nabla L(\theta;T_t\cup M), & B(\alpha)=0\end{cases}$$
$\alpha$ directly controls the balance between "stability vs. plasticity": a larger $\alpha$ means more frequent $\hat g$ injection, leading to higher old task accuracy but lower new task accuracy. This stochastic injection simulates the inherent randomness of mini-batch gradient descent while preventing over-fitting to old knowledge. The algorithmic implementation is minimal: when the sample is 1, the (scaled) direction vector is added to the parameters that have gradients.

### Loss & Training
Base model is LLaVA-7B + LoRA. VQAv2: LoRA rank 128, 0.5k replay samples per task, constant $\alpha=0.2$; task sequence is Recognition $\rightarrow$ Location $\rightarrow$ Judge $\rightarrow$ Commonsense $\rightarrow$ Count $\rightarrow$ Action $\rightarrow$ Color $\rightarrow$ Type $\rightarrow$ Subcategory $\rightarrow$ Causal. UCIT: LoRA rank 48, 2k replay samples per task, $\alpha$ set per task (ArxivQA 0.1 / VizWiz 0.1 / IconQA 0.05 / CLEVR 0.05 / Flickr30k 0.1).

## Key Experimental Results

### Main Results
Two MCIT datasets are used, with the metric being **Final Average Accuracy (FAA)** (weighted by the actual test sample count of each task, $\text{FAA}=\sum_i \frac{|T_i|}{|T_{1:T}|}a^T_i$, where $a^T_i$ is the accuracy of the $i$-th task after learning the final task). MultiTask (joint training) serves as the upper performance bound; all baselines are evaluated using the MCITlib framework.

| Dataset | MultiTask (Bound) | Best Baseline | Ours (DGG) | Gain vs. Best |
| :--- | :---: | :---: | :---: | :---: |
| VQAv2 (10 sub-tasks, in-domain) | 66.26 | SEFE 63.57 | **65.17** | +1.60 |
| UCIT (6 datasets, strong shift) | 74.78 | DISCO 69.66 | **73.82** | +4.16 |

DGG's gap from the upper bound is extremely small (only 1.09 on VQAv2 and 0.96 on UCIT), and it even exceeds the MultiTask bound on sub-tasks like Recognition (55.55), Commonsense (76.12), and Type (61.19). Crucially, **most baselines use MoE to expand the model, while DGG does not expand at all**, solving the problem at the optimization level.

### Ablation Study
Decomposition of the two core operations (gradient scaling and Bernoulli sampling); data reported as FAA (%). Note: Subsequent ablations on VQAv2 use the $\hat g$-only setting (no replay $M$), thus the full VQAv2 value below is 64.61 instead of the 65.17 in the main results.

| Configuration | VQAv2 | UCIT | Description |
| :--- | :---: | :---: | :--- |
| Full (Scaling + Bernoulli) | 64.61 | 73.82 | full (VQAv2 is $\hat g$-only) |
| w/o Gradient Scaling | 64.01 (↓0.60) | 65.24 (↓8.58) | Uses unscaled direction vector |
| w/o Bernoulli Sampling | 62.75 (↓1.86) | 59.02 (↓14.80) | Injects $\hat g$ every step |

Gradient approximation ablation (original paper Figure 3): On VQAv2, $\hat g$-only reaches 64.61, far exceeding M-only (57.73 with 1k samples). On UCIT, M-only (57.13 with 0.5k) outperforms $\hat g$-only (53.71).

### Key Findings
- **Bernoulli sampling is more critical than gradient scaling**: Removing Bernoulli causes a drop of 14.80 on UCIT and 1.86 on VQAv2, both exceeding the impact of removing scaling. Stochastic regulation of injection frequency is vital to avoid "excessive remembering."
- **Operations have a larger impact in strong distribution shift scenarios**: On UCIT (cross-domain), removing scaling drops 8.58 and removing Bernoulli drops 14.80, far higher than the 0.60 / 1.86 drops on in-domain VQAv2. This indicates that larger distribution differences are more sensitive to fine-grained gradient regulation.
- **The dominance of $\hat g$ vs. replay switches with distribution**: In-domain relies more on $\hat g$, while cross-domain relies more on replay. Their combination is always superior, validating the "geometric approximation + real sampling" complementarity.
- **$\alpha$ regulates the plasticity-stability trade-off**: As $\alpha$ increases, old task accuracy rises while new task accuracy falls; FAA is optimal near $\alpha=0.2$ for VQAv2 and $\alpha=0.05$ for UCIT. Excessively small $\alpha$ consistently harms old task stability.

## Highlights & Insights
- **The "forgetting = missing gradient" reformulation is insightful**: It shifts catastrophic forgetting from a vague narrative of "weights being overwritten" to a precise missing term $\nabla L(\theta;T_1)$ in the joint gradient. The solution naturally follows.
- **Replacing unavailable data gradients with parameter space geometry**: The direction vector $\theta-\theta^*_{1:t-1}$ acts as a proxy for the expected old gradient direction using the "displacement to the old optimum." This requires only the storage of previous optimal parameters (not data), resulting in extremely low storage costs and natural complementarity with replay.
- **Approaching the joint bound without model expansion**: In an MCIT field dominated by MoE, DGG proves that pure optimization-level gradient regulation can reduce the gap with the MultiTask bound to ~1%. This is deployment-friendly (no inference-time bloat) and the "optimizer-level rather than architecture-level" approach is transferable to other continual learning tasks.

## Limitations & Future Work
- **Reliance on the availability and representativeness of the "old task optimal parameters"**: The quality of $\hat g$ depends on whether $\theta^*_{1:t-1}$ is truly the optimum. If old tasks are under-fitted or the optimum shifts significantly, the directional approximation will be distorted, as evidenced by the decreased accuracy of $\hat g$ under strong distribution shifts (UCIT).
- **Hyperparameter $\alpha$ requires questing per task**: Different $\alpha$ values (0.05~0.1) were used for different tasks in UCIT, and sensitivity analysis shows $\alpha$ has a significant impact; cross-dataset transfer might require re-searching for parameters.
- **Replay buffer is still required**: Replay is the primary driver in cross-domain scenarios (UCIT uses 2k samples per task), so DGG is not entirely storage-free; its utility in privacy/storage-constrained scenarios remains to be validated.
- Scalability to larger models, longer task sequences, or full parameter fine-tuning has not yet been demonstrated, as evaluations were limited to LLaVA-7B + LoRA on two MCIT datasets.

## Related Work & Insights
- **vs. MoE types (CL-MoE / HiDE / DISCO)**: These use experts/routers to learn task-specific knowledge, which provides strong memory but **expands the model**, increasing both training and inference costs. DGG adds no components and solves forgetting at the optimization level, keeping the architecture compact.
- **vs. Regularization types (SEFE / RegLoRA)**: SEFE uses static regularization to constrain updates of critical weights. DGG's gradient guidance can be viewed as a **dynamic regularization term** that adjusts during training and integrates with replay to adapt to the evolving optimization landscape.
- **vs. Purereplay (replay-based)**: Pure replay is limited by buffer size and the approximation is biased toward the current task. DGG supplements this with geometric direction information from $\hat g$, even outperforming the strongest baseline without replay in in-domain scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The "forgetting = missing gradient" reformulation + parameter space direction vector approximation is a fresh perspective and simple to implement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough multi-dimensional ablations on components/buffers/sequences/hyperparameters across two datasets, though task sequences and base model scale remain limited.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from gradient additivity derivation to approximation and regulation; Figure 1/2 are intuitive.
- Value: ⭐⭐⭐⭐ Provides an optimizer-level MCIT solution that approaches the joint bound without model expansion; deployment-friendly and open-source for reproducibility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1"></div>

## Related Papers

- [\[CVPR 2026\] Towards Dynamic Modality Alignment in Multimodal Continual Learning](towards_dynamic_modality_alignment_in_multimodal_continual_learning.md)
- [\[CVPR 2026\] Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs](harmonious_parameter_adaptation_in_continual_visual_instruction_tuning_for_safet.md)
- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](../../ACL2025/multimodal_vlm/branchlora_continual_instruction.md)
- [\[CVPR 2026\] Octopus: History-Free Gradient Orthogonalization for Continual Learning in Multimodal Large Language Models](octopus_history-free_gradient_orthogonalization_for_continual_learning_in_multim.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)

</div>

<!-- RELATED:END -->
