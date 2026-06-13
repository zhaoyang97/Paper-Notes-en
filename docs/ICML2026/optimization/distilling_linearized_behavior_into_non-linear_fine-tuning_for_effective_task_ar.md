---
title: >-
  [Paper Note] Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic
description: >-
  [ICML 2026][Optimization][task arithmetic] This paper proposes DELTA: online distillation of intermediate activations from a "tangent space linearized teacher" into a standard non-linear student…
tags:
  - "ICML 2026"
  - "Optimization"
  - "task arithmetic"
  - "linearization"
  - "knowledge distillation"
  - "weight disentanglement"
  - "EK-FAC"
date: 2026-05-08
content_hash: fc9320ce6bec3c8d
---

# Distilling Linearized Behavior into Non-Linear Fine-Tuning for Effective Task Arithmetic

**Conference**: ICML 2026  
**arXiv**: [2605.18993](https://arxiv.org/abs/2605.18993)  
**Code**: https://github.com/apanariello4/merge-and-rebase  
**Area**: Model Merging / Model Compression / Task Arithmetic  
**Keywords**: task arithmetic, linearization, knowledge distillation, weight disentanglement, EK-FAC

## TL;DR
This paper proposes DELTA: online distillation of intermediate activations from a "tangent space linearized teacher" into a standard non-linear student, combined with EK-FAC curvature regularization and sampling along the interpolation path. This ensures that task vectors from conventional non-linear fine-tuning possess the "additive, low-interference, and scale-robust" properties of linearized models without introducing any inference overhead.

## Background & Motivation

**Background**: Task arithmetic (Ilharco 2022) utilizes the weight difference $\bm\tau_t=\bm\theta_t-\bm\theta_0$ from task-specific fine-tuning as a task vector, performing addition (merging tasks) or subtraction (unlearning) in weight space via $\bm\theta_0+\sum_t\alpha_t\bm\tau_t$. Its efficacy relies heavily on weight disentanglement: applying $\bm\tau_t$ should leave predictions for other tasks nearly unchanged. Ortiz-Jimenez et al. found that fine-tuning in the tangent space (linearized model $f_{\mathrm{lin}}(\bm x;\bm\theta)=f(\bm x;\bm\theta_0)+\mathrm J_{\bm\theta}f(\bm x;\bm\theta_0)(\bm\theta-\bm\theta_0)$) naturally yields more decoupled task vectors.

**Limitations of Prior Work**: The linearized path incurs three significant costs: (i) Jacobian-vector products double both training and inference costs; (ii) constraining optimization to the tangent space impairs expressivity, leading to lower single-task accuracy ceilings; (iii) existing interference-reduction regularizations (e.g., τJp requires training data from other tasks, TAK requires KFAC factors from other tasks) assume a closed set of known tasks, requiring full re-computation when a new task arrives. Non-linear fine-tuning is expressive but performs poorly in task arithmetic (e.g., 32% absolute accuracy on ViT-B/32 8-Vision vs. 77% for the linear version).

**Key Challenge**: There is a seemingly fundamental trade-off between expressivity (non-linear) and composability (linear) for task arithmetic, where merging capabilities depend strictly on tangent space structures.

**Goal**: To enable a standard non-linear fine-tuned student to satisfy two core conditions—"near-linearity relative to weight perturbations" and "support localization (in-domain changes, out-of-domain stability)"—thereby achieving high merging performance without inference costs or access to other tasks' data/statistics.

**Key Insight**: This paper observes that "near-linearity in weight space" is a parameter-space property that can be induced via "activation-space objectives." By matching the hidden activations of a non-linear student to those of a linearized teacher, optimization is biased toward solutions that are near-linear regarding weight perturbations.

**Core Idea**: Jointly train a non-linear student using a linearized teacher + online feature distillation + sampling along the interpolation path + EK-FAC curvature regularization. This embeds "linearization benefits" into a model that remains a standard non-linear forward pass at inference time.

## Method

### Overall Architecture
For each task $t$, two models are maintained simultaneously: a teacher $f_{\mathrm{lin}}(\bm x;\bm\theta_t^T)$ undergoing tangent space linearization and a student $f(\bm x;\bm\theta_t^S)$ undergoing standard non-linear fine-tuning. Both share the same pre-trained initialization $\bm\theta_0$, are optimized jointly in a single backward pass (not sequential teacher-then-student training), and incorporate EK-FAC curvature regularization. The teacher provides "low-interference target activations," while the student captures "linearized behavior" through MSE distillation of teacher features across multiple snapshots sampled along the interpolation path.

### Key Designs

1.  **Linearized Teacher + Task-Agnostic EK-FAC Curvature Regularization**:
    - **Function**: Pushes the teacher toward directions that minimize perturbations for arbitrary other input distributions, avoiding interference with future tasks.
    - **Mechanism**: Teacher loss is $\mathcal L^T_t = \mathcal L_{\text{task}} + \beta^T\,\mathcal L_{\text{drift}}(\bm\theta_t^T)$; representation drift under linearization has a closed form $\mathcal L_{\text{drift}}(\bm\theta_t)\propto (\bm\theta_t-\bm\theta_0)^\top \bm G_t(\bm\theta_0)(\bm\theta_t-\bm\theta_0)$, where $\bm G_t$ is the GGN matrix. Instead of computing GGN on known task sets, the authors pre-compute an EK-FAC approximation $\mathrm{GGN}_{\mathrm{EK\text{-}FAC}}^l=(U_A^l\otimes U_G^l)S^l(U_A^l\otimes U_G^l)^\top$ once using a third-party reference dataset $\mathcal D_\Omega$ (e.g., a 15% subset of ImageNet-21k for vision, $10^5$ samples from C4 for text).
    - **Design Motivation**: Methods like τJp and TAK require data or KFAC factors from other tasks, assuming a closed set. Using a reference dataset allows new tasks to be added without retraining old task vectors and protects private user data. EK-FAC improves curvature estimation over KFAC by modeling eigenvalues under the Kronecker feature basis.

2.  **Online Feature-level Distillation + Along-Path Knowledge Distillation (APKD)**:
    - **Function**: Transfers the "near-linear weight perturbation" property to the non-linear student and ensures this holds along the entire interpolation path, providing robustness to the scaling factor $\alpha$.
    - **Mechanism**: Instead of logits, the MSE is computed on hidden activations before the final projection head. Rather than distilling at a single point, each SGD step samples an interpolation point $\alpha\sim\mathcal U(0.5,1)$. Activations for both teacher and student are computed at $\bm\theta_0+\alpha\bm\tau$ and aligned: $\mathcal L_{\text{KD}}=\mathbb E_{\alpha}\big[\frac{1}{B}\sum_i\|f(\bm x_i;\bm\theta_0+\alpha\bm\tau_t^S)-\mathrm{SG}[f_{\mathrm{lin}}(\bm x_i;\bm\theta_0+\alpha\bm\tau_t^T)]\|_2^2\big]$. A stop-gradient is applied to the teacher.
    - **Design Motivation**: Traditional KD at $\alpha{=}1$ only aligns a single point, allowing the student to drift from linear behavior elsewhere. APKD feeds the entire linear trajectory to the student, acting as an ensemble distillation of a "linearized teacher family," significantly improving $\alpha$-sweep robustness on T5.

3.  **Student-side Joint Curvature Regularization + Student-side Linearization Induction**:
    - **Function**: Directs the student toward support localization (major in-domain changes, pre-training proximity for out-of-domain), disentangling "linearization" and "decoupling."
    - **Mechanism**: Student loss is $\mathcal L^S_t=\mathcal L_{\text{task}}(\bm\theta_t^S)+\beta_1\mathcal L_{\text{KD}}+\beta_2\mathcal L_{\text{drift}}(\bm\theta_t^S)$. Distillation constrains the student to a near-linear region, while curvature regularization provides explicit control within that region. This setup supports Full FT students or LoRA students paired with Full FT teachers; the latter allows the teacher to find directions in an expressive space while the student replicates them in a low-rank subspace.
    - **Design Motivation**: Diagnostics (Fig. 6) show distillation mainly contributes to linearization while curvature regularization drives decoupling. Both are required for effective task addition. This also explains how the student can outperform the teacher by retaining non-linear expressivity while constrained to the teacher's linear activation space.

## Key Experimental Results

### Main Results
Absolute accuracy for task addition across 8-Vision / 14-Vision / 6-NLI benchmarks using $\alpha{=}1$. DELTA outperforms across 4 different backbones:

| Method | 8V ViT-B/32 Abs. | 14V ViT-L/14 Abs. | 14V ViT-B/32 Abs. | 6-NLI T5-Base Abs. |
|------|------|------|------|------|
| Pre-trained | 48.4 | 65.0 | 57.8 | 61.7 |
| Individual fine-tune | 92.8 | 95.8 | 90.2 | 85.9 |
| Non-Linear FT (Ilharco 2022) | 32.0 | 45.3 | 15.6 | 42.0 |
| Linear FT (Ortiz-Jimenez 2023) | 77.4 | 88.0 | 73.7 | 76.0 |
| τJp (Yoshida 2025) | 85.0 | 90.9 | 85.3 | 82.5 |
| TAK (Porrello 2025b) | 86.0 | 91.6 | 84.3 | 79.1 |
| **DELTA (Ours)** | **88.3** | **92.7** | **85.9** | 82.3 |

In the LoRA student + Full FT teacher configuration, DELTA achieves 87.5 / 99.5 normalized on 8V ViT-B/32, surpassing the runner-up Core+TSV-M by 9.6 points.

### Ablation Study

| Configuration | 8V ViT-B/32 Task Arithmetic | Description |
|------|------------------------------------|------|
| Non-linear FT baseline | 32.0 abs | Lacks both linearization and decoupling; failure. |
| Student + Distill + Curvature (DELTA full) | 88.3 abs | Both components present. |
| Student + Distill only (no curvature) | Close to DELTA | Low linearization error, weak support localization. |
| Student + Curvature only (no distill) | Closest to DELTA | Strongest decoupling, but linearization error increases. |
| APKD off (Fixed $\alpha{=}1$ distill) | Error increases | Single-point alignment loses path-wise properties; worse T5 $\alpha$-sweep. |
| Task negation 9.6% target / 62.1% control | DELTA > Non-linear FT | DELTA outperforms non-linear methods but trails pure linear τJp/TAK in subtraction. |

### Key Findings
- "Distillation manages linearization, while curvature regularization manages support localization"—these are independent pathways, both necessary to approach the performance ceiling in task addition.
- **Ours** student outperforms the linearized teacher: On T5, every single-task student scores higher than the teacher, and the merged average accuracy is also higher. This indicates non-linear expressivity is not lost but guided into a "near-linear but more expressive" intermediate state.
- LoRA student + Full FT teacher is a surprisingly strong combination, reaching 97.9 normalized accuracy on 8V ViT-B/32, far exceeding post-hoc merging methods (Iso-C / TSV-M / Core Space).
- $\alpha$-sweep robustness: DELTA shows an almost flat curve for $\alpha\in[0.5,1]$, whereas other non-linear methods collapse when $\alpha \neq 1$. This reduces reliance on validation sets for coefficient tuning.
- Generalization to LLMs: Using LLaMA-3.2-1B + DPO to combine helpfulness and verbosity vectors via $\bm\theta_{\text{mix}}=\bm\theta_0+\bm\tau_{\text{help}}+\lambda_2\bm\tau_{\text{verb}}$, Distilled DPO approaches the reward Pareto frontier of Linear DPO and exceeds the preference accuracy of Non-Linear DPO.

## Highlights & Insights
- Inducing parameter-space properties via function-space targets is a noteworthy perspective. This paper provides empirical evidence that activation-level MSE + curvature regularization makes non-linear models behave "near-linearly" regarding weights. This is likely because optimization is constrained near $\bm\theta_0$ (where Taylor expansions are valid) and simplification biases favor simpler (near-linear) mechanisms.
- The diagnostic partitioning of distillation vs. curvature regularization (Fig. 4/5) is an elegant experimental design, attributing model performance to explainable properties.
- Replacing task-specific statistics with a reference dataset $\mathcal D_\Omega$ allows the method to scale to new tasks incrementally without damaging existing task vectors—a key breakthrough for deploying task arithmetic.
- The LoRA student + Full FT teacher pairing fits industrial pipelines ("train expressive, deploy efficient") and outperforms post-hoc merging by significant margins.

## Limitations & Future Work
- Training costs are roughly tripled and VRAM usage is doubled (teacher + student + along-path sampling + EK-FAC pre-computation), which the authors acknowledge as a primary bottleneck.
- Performance on task negation still trails pure linear methods like τJp/TAK, suggesting that "strict linearity" still offers benefits for subtraction that distillation hasn't fully captured.
- Curvature regularization depends on the representativeness of $\mathcal D_\Omega$; while sensitivity ablations are included, cross-domain performance (e.g., using vision reg for medical tasks) remains to be validated.
- Increased merging efficiency is a double-edged sword, potentially making it easier for unsafe behaviors to be combined and propagated.
- Distilled DPO is preliminary and lacks curvature regularization; an end-to-end version for LLMs is a clear future direction.

## Related Work & Insights
- **vs. Linear FT (Ortiz-Jimenez 2023)**: They train directly in tangent space; DELTA treats the tangent space model as a teacher to transfer properties to a non-linear student, saving 50% inference cost and improving task addition accuracy.
- **vs. τJp (Yoshida 2025)**: τJp regularizes drift using other tasks' training data; DELTA replaces this with a reference dataset + EK-FAC, making it task-agnostic.
- **vs. TAK (Porrello 2025b)**: TAK also achieves dataless merging using KFAC but still requires KFAC factors from all tasks; DELTA uses a shared reference matrix to allow incremental task addition.
- **vs. Iso-C / TSV-M / Core Space**: These are post-hoc merging methods that correct bias after fine-tuning; DELTA pushes the task vector into a decoupled region during training, yielding a 14+ point gain in LoRA settings.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Grouping activation-space constraints to induce parameter-space properties with along-path distillation is a novel combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 8/14-task vision, 6-NLI, ViT-B/32, ViT-L/14, T5-Base, LoRA, and DPO for LLMs. Diagnostic ablations clearly differentiate component functions.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear articulation of the benefits and costs of linearization; Tab. 1 effectively highlights DELTA's completeness.
- **Value**: ⭐⭐⭐⭐⭐ Transitions task arithmetic from research demo to deployable stage (zero inference overhead + incremental tasks). LoRA experiments show industrial utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICML 2026\] Bayesian Gated Non-Negative Contrastive Learning](bayesian_gated_non-negative_contrastive_learning.md)
- [\[ICML 2026\] On the Expressive Power of GNNs to Solve Linear SDPs](on_the_expressive_power_of_gnns_to_solve_linear_sdps.md)
- [\[ICML 2026\] SyMerge: From Non-Interference to Synergistic Merging via Single-Layer Adaptation](symerge_from_non-interference_to_synergistic_merging_via_single-layer_adaptation.md)

</div>

<!-- RELATED:END -->
