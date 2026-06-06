---
title: >-
  [Paper Note] Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning
description: >-
  [ICML 2026][Model Compression][LoRA] LoDA decomposes the LoRA down-projection matrix by "projection energy" into a general subspace shared across tasks and an isolated subspace that is only activated by new tasks. It the…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "LoRA"
  - "Continual Learning"
  - "Subspace Decomposition"
  - "Projection Energy"
  - "Feature-level Recalibration"
date: 2026-05-08
content_hash: 5363e8bc2f9e2d45
---

# Task-Driven Subspace Decomposition for Knowledge Sharing and Isolation in LoRA-based Continual Learning

**Conference**: ICML 2026  
**arXiv**: [2603.00191](https://arxiv.org/abs/2603.00191)  
**Code**: None  
**Area**: Model Compression / LoRA / Continual Learning  
**Keywords**: LoRA, Continual Learning, Subspace Decomposition, Projection Energy, Feature-level Recalibration

## TL;DR
LoDA decomposes the LoRA down-projection matrix by "projection energy" into a general subspace shared across tasks and an isolated subspace that is only activated by new tasks. It then uses gradient alignment to train the up-projection and applies a closed-form recalibration to the general branch during merging, thereby consistently outperforming existing LoRA-CL methods on multiple continual learning benchmarks.

## Background & Motivation
**Background**: Continual learning (CL) based on pretrained ViT is now dominated by PEFT approaches: either prompt pool (L2P / DualPrompt / CODAPrompt) or LoRA-based (O-LoRA, InfLoRA, Bi-LoRA, PLAN, SD-LoRA, etc.), all aiming to maintain the stability-plasticity trade-off with a small number of trainable parameters.

**Limitations of Prior Work**: The mainstream LoRA-CL approach restricts LoRA updates for new tasks to the "null space of previous tasks" to prevent forgetting, but this has two issues: (1) There are naturally shared directions between tasks, and hard-constraining to the null space cuts off transferable information, reducing transferability; (2) When task distributions are highly correlated (the norm in real CL), the "null space of previous tasks" is almost inactive for new tasks, making the so-called "isolated basis" a dead zone that cannot learn new tasks. The authors empirically show on 10S-ImageNetA that $r^t(\mathbf{U}_{\text{null}})\approx 1.0$, proving that the null space is inactive for both old and new tasks.

**Key Challenge**: Existing LoRA-CL methods treat "isolation" and "transfer" as opposites and use a negative space defined by previous tasks to decide "where to learn new tasks," which is inherently conflicting—shared directions are discarded, and the task direction is misidentified.

**Goal**: (i) Explicitly retain transferable directions across tasks to promote knowledge transfer; (ii) Identify truly isolated directions that are highly responsive to new tasks and minimally interfere with old tasks; (iii) When merging LoRA increments back to the backbone, approach the joint optimum for all tasks as closely as possible.

**Key Insight**: The authors approach from the gradient perspective of "LoRA learning capacity," proving that when only the up-projection $\mathbf{B}$ is updated, the first-order loss reduction is fully determined by the projection energy $E=\|\mathbf{A}\mathbf{X}^\top\|_2^2$, i.e., the down-projection $\mathbf{A}$ acts as an "energy gate" that decides which input features are updated. Thus, "how to design $\mathbf{A}$" becomes an energy optimization problem.

**Core Idea**: Fix the LoRA down-projection $\mathbf{A}$ as two sets of data-driven orthogonal bases—"high energy across tasks" for the general branch, "maximal new/old task energy ratio" for the isolated branch—combined with gradient alignment training and closed-form recalibration.

## Method

### Overall Architecture
At task $t$, the backbone $\mathbf{W}^{t-1}$ is frozen, and each ViT layer is equipped with a dual-branch LoRA: the general branch $(\mathbf{A}_G,\mathbf{B}_G)$ for knowledge sharing, and the isolated branch $(\mathbf{A}_I,\mathbf{B}_I)$ for task-specific increments. The process is: (A) Use the new data second-order moment $\mathbf{S}^t$ and accumulated old data moment $\mathbf{S}^{1:t-1}$ to solve two energy objectives and obtain $\mathbf{U}_G,\mathbf{U}_I$; (B) Freeze down-projections $\mathbf{A}_G\leftarrow\mathbf{U}_G^\top$, $\mathbf{A}_I\leftarrow\mathrm{QR}(\mathbf{U}_I^\top)$, and train up-projections on new data using the GAO algorithm; (C) At task end, solve a closed-form recalibration matrix $\mathbf{\Lambda}_G$ for the general branch before merging into the backbone, and directly merge the isolated branch.

### Key Designs

1. **Task-Driven Subspace Decomposition (Core)**:

    - Function: Decompose the LoRA update space into two data-driven low-rank subspaces: "general $\mathcal{U}_G$ + isolated $\mathcal{U}_I$".
    - Mechanism: The general subspace maximizes $E_{\text{old}}+E_{\text{new}}$, with a closed-form solution as the top-$r$ singular vectors of $(\mathbf{S}^{1:t-1}+\mathbf{S}^t)$; the isolated subspace maximizes the "new/old task energy ratio" $\mathrm{tr}(\mathbf{U}^\top\mathbf{S}^t\mathbf{U})/\mathrm{tr}(\mathbf{U}^\top\mathbf{S}^{1:t-1}\mathbf{U})$. The authors use Cholesky decomposition $\mathbf{S}^{1:t-1}=\mathbf{L}\mathbf{L}^\top$ to convert the generalized Rayleigh quotient problem into an SVD of $\tilde{\mathbf{S}}^t=\mathbf{L}^{-1}\mathbf{S}^t\mathbf{L}^{-\top}$, yielding $\mathbf{U}_I=(\mathbf{L}^{-1})^\top\tilde{\mathbf{U}}_I$.
    - Design Motivation: This directly avoids the dead zone problem of "null space approximation"—true isolated directions should have high energy for new tasks and low for old tasks, not just low for old tasks; meanwhile, general directions are preserved.

2. **Gradient Alignment Optimization (GAO)**:

    - Function: When training up-projections $\mathbf{B}_G,\mathbf{B}_I$, align gradient directions across different class subsets to improve robustness to future new classes.
    - Mechanism: Split a batch $\mathcal{B}$ into two label-disjoint subsets $\mathcal{B}_1,\mathcal{B}_2$. For each step, perturb parameters with the gradient from $\mathcal{B}_2$ (step size $\rho\sim U(0,\rho_{\max})$ randomized), then update with the gradient from $\mathcal{B}_1$ on the perturbed parameters, and swap the subsets in the next step. This essentially applies the SAM idea to "inter-class gradient conflict".
    - Design Motivation: In CL, when learning a task, the up-projection does not know what future classes will look like; inter-class gradient conflicts make features vulnerable to future tasks. GAO treats "the other group's gradient" as a perturbation source, forcing the model to move in directions agreed upon by both, thus suppressing class-related fragile directions.

3. **Feature-level Closed-form Recalibration $\mathbf{\Lambda}_G$**:

    - Function: Before merging LoRA updates into the backbone, compute a correction for the general branch that is near-optimal for all tasks.
    - Mechanism: While general updates bring transfer benefits, they inevitably cause feature drift for old tasks. The authors formulate the "feature-level error for old + new tasks after merging" as a least squares problem and solve for the correction matrix $\mathbf{\Lambda}_G$ in closed form, avoiding the local linear approximation and gradient estimation errors relied upon by existing methods (e.g., BECAME). The isolated branch, having low energy for old tasks, can be merged directly.
    - Design Motivation: Previous model merging methods (CoMA / BECAME) mostly perform EMA or Fisher-based mixing in weight space; the authors directly solve for the exact optimum at the feature level, which is theoretically tighter.

### Loss & Training
The training loss is standard cross-entropy, but the GAO dual-perturbation structure implicitly embeds gradient alignment regularization. Key hyperparameters include subspace rank $r$, general branch weight $w_G$, and GAO's $\rho_{\max}$. The statistic $\mathbf{S}^{1:t-1}$ is incrementally accumulated at the end of each task, with no need to store old data.

## Key Experimental Results

### Main Results
Compared with 9+ baselines (covering CVPR'22 ~ NIPS'25) under the 10-task CL protocol on ImageNetR / ImageNetA / CIFAR100 / CUB benchmarks:

| Dataset | Metric | LoDA | Prev. SOTA (CoSO/LoRA-P&M) | Gain |
|---------|--------|------|----------------------------|------|
| 10S-ImageNetR | $\mathcal{A}_{Last}$ | **81.93** | 81.10 | +0.83 |
| 10S-ImageNetA | $\mathcal{A}_{Last}$ | **62.59** | 56.57 | **+6.02** |
| 10S-CIFAR100 | $\mathcal{A}_{Last}$ | **90.47** | 88.77 | +1.70 |
| 10S-CUB | $\mathcal{A}_{Last}$ | **81.74** | 78.29 | +3.45 |
| 20S-ImageNetA | $\mathcal{A}_{Last}$ | **55.74** | 52.27 | +3.47 |

Under the feature replay setting, LoDA+CA achieves 66.71 on 10S-ImageNetA, 2.57 points higher than the previous best MACIL (64.14).

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| Full LoDA | Baseline |
| General branch only | Task isolation disappears, new tasks overwrite old, significant drop |
| Isolated branch only | Lacks transfer, underfits on related tasks |
| Isolated branch replaced with null space approximation | Largest degradation on highly correlated tasks like ImageNetA, confirming null space approximation failure |
| w/o GAO | Inter-class gradient conflict increases, old class features more likely to collapse after new tasks arrive |
| w/o closed-form recalibration | Feature drift from general branch is not compensated |

### Key Findings
- The largest improvement (+6 points) is on ImageNetA, which has the highest task correlation and where null space approximation fails most; LoDA's "ratio maximization" isolated subspace is naturally advantageous here.
- LoDA without feature replay already outperforms replay-based SLCA / SSIAT / VQ-Prompt, indicating that subspace design is more effective than extra feature storage.
- The advantage remains stable (+3.47) when increasing the number of tasks from 10 to 20, showing slower degradation for long task sequences.

## Highlights & Insights
- **"Ratio maximization" for isolated direction selection** is superior to traditional "null space approximation": the former directly defines "desired directions," while the latter only defines "directions to avoid," which are not equivalent when tasks are correlated.
- **Freezing down-projection + learning up-projection** makes the LoRA training objective a linearly energy-weighted structure over subspaces (Theorem 3.1), which is key to the method's mathematical foundation and suggests that future PEFT modules can be analyzed via "input space–parameter space" separation.
- **Feature-level closed-form recalibration** bypasses the approximation error of weight-level model merging and is a transferable trick for RLHF / multi-task LoRA fusion.

## Limitations & Future Work
- Each task requires a generalized eigenvalue/SVD decomposition (dimension $D\times D$, e.g., 768 for ViT), so cumulative overhead is nontrivial for many tasks; engineering requires caching and incremental updates.
- The rank $r$ for both general and isolated branches is shared across all layers, with no layer sensitivity analysis; shallow semantic features and deep task features may require different $r$.
- Assumes clearly known task boundaries (task-aware); extension to task-free CL scenarios is needed.
- The isolated subspace assumes $\mathbf{S}^{1:t-1}$ is full rank; with very few tasks or samples, Cholesky may be numerically unstable.

## Related Work & Insights
- **vs InfLoRA / O-LoRA**: Both restrict LoRA to the "null space of previous tasks"; LoDA circumvents null space approximation failure via ratio maximization and restores discarded shared directions.
- **vs Bi-LoRA / PLAN**: Use fixed predefined orthogonal bases (e.g., DCT bases) to freeze down-projection; LoDA uses data-driven second-order spectral decomposition, making it more sensitive to task correlation.
- **vs BECAME / CoMA**: These perform EMA or Fisher-based merging in weight space; LoDA solves for the exact optimum in feature space, avoiding errors from local linear approximations.
- **vs SD-LoRA**: SD-LoRA decouples direction and magnitude for parameter updates along low-loss paths; LoDA directly moves "direction selection" to the down-projection construction stage, making the approaches complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ "Projection energy ratio maximization" is the first to truly separate LoRA-CL isolation/sharing by data, with clear theory
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 datasets × multiple task lengths, with sufficiently recent baselines
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, coherent illustrations and motivation
- Value: ⭐⭐⭐⭐ A clear capability upgrade for the LoRA-CL engineering line, and the method is transferable to other PEFT scenarios

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](../../AAAI2026/model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)
- [\[ACL 2026\] SAMoRA: Semantic-Aware Mixture of LoRA Experts for Task-Adaptive Learning](../../ACL2026/model_compression/samora_semantic-aware_mixture_of_lora_experts_for_task-adaptive_learning.md)
- [\[ICML 2026\] FedRot-LoRA: Mitigating Rotational Misalignment in Federated LoRA](fedrot-lora_mitigating_rotational_misalignment_in_federated_lora.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[ICCV 2025\] PLAN: Proactive Low-Rank Allocation for Continual Learning](../../ICCV2025/model_compression/plan_proactive_low-rank_allocation_for_continual_learning.md)

</div>

<!-- RELATED:END -->
