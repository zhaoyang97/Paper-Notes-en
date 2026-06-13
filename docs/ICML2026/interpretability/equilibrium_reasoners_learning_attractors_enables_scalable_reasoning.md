---
title: >-
  [Paper Note] Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning
description: >-
  [ICML 2026][Interpretability][Fixed-point Dynamical Systems] This paper reinterprets models that "reason via iterative latent updates" as learned attractor dynamical systems. It proposes Equilibrium Reasoners (EqR)…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Fixed-point Dynamical Systems"
  - "Attractors"
  - "Weight-sharing Iteration"
  - "Depth-Breadth Scaling"
  - "Sudoku-Extreme"
date: 2026-05-08
content_hash: f333ec1bcb0e8e26
---

# Equilibrium Reasoners: Learning Attractors Enables Scalable Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.21488](https://arxiv.org/abs/2605.21488)  
**Code**: https://github.com/locuslab/EqR (Available)  
**Area**: LLM Reasoning / Iterative Latent Reasoning / Test-time Compute Scaling  
**Keywords**: Fixed-point Dynamical Systems, Attractors, Weight-sharing Iteration, Depth-Breadth Scaling, Sudoku-Extreme

## TL;DR
This paper reinterprets models that "reason via iterative latent updates" as learned attractor dynamical systems. It proposes Equilibrium Reasoners (EqR), which use two lightweight training interventions—random initialization (RI) and path noise injection (NI)—to shape the attractor landscape. Combined with a two-axis test-time scaling of "depth (iteration steps $D$) + breadth (multiple random restarts $B$)" and a selection rule based on residual convergence, EqR pushes Sudoku-Extreme accuracy from a feedforward baseline of 2.6% to 99.8% (equivalent to 40,000 unfolded layers) despite being trained on only 16 iterations.

## Background & Motivation

**Background**: Modern reasoning models rely increasingly on test-time compute—ranging from search-based AlphaZero to CoT, and more recently, weight-shared iterative models like HRM, TRM, and URM. These models deepen reasoning by repeatedly executing the same update module. Models like HRM/TRM iterate over a latent state and achieve performance on long-range constraint satisfaction tasks (like Sudoku) that far exceeds standard feedforward networks.

**Limitations of Prior Work**: Increasing test-time compute does not always yield benefits; prior literature has reported diminishing or even negative returns for test-time scaling. HRM describes its behavior as "hierarchical convergence," while TRM explicitly notes that latent residuals do not reach zero even after training, thereby rejecting a strict fixed-point interpretation. In other words, a mechanistic explanation of why iterative reasoning works and when it scales effectively remains missing.

**Key Challenge**: The assumption of strict "convergence to a unique fixed point" (DEQ) is too strong, as residuals often do not vanish. However, completely abandoning the convergence perspective fails to explain the empirical phenomenon where "more iterations lead to better performance." An intermediate perspective between "strict fixed points" and "complete black boxes" is needed.

**Goal**: (i) Provide an internal mechanistic explanation for iterative reasoning that is more relaxed than DEQ but still falsifiable; (ii) Translate this explanation into concrete training interventions and test-time scaling strategies; (iii) Verify on controlled benchmarks whether "residual convergence" serves as a reliable scaling signal.

**Key Insight**: The iterative operator $\mathbf{z}_{k+1}=f_\theta(\mathbf{z}_k;\mathbf{x})$ is viewed as a task-conditioned dynamical system, but the objective is relaxed from "finding exact fixed points" to "finding attractors"—stable regions of local entrapment. A "well-aligned" attractor landscape should satisfy a condition where the low-residual basins of the internal landscape coincide with the low-error basins of the task metric. Thus, training involves shaping the internal landscape into a differentiable surrogate of the task metric, and reasoning involves adaptive search on this landscape.

**Core Idea**: "Attractor landscape shaping" provides a unified explanation for both training and test-time scaling. On the training side, Random Initialization (RI) and Noise Injection (NI) make correct attractors both broader and more stable. On the reasoning side, scaling occurs along the "depth $D$ (iterations per trajectory) + breadth $B$ (independent restarts)" axes, using the trajectory with the minimum residual for Top-1 selection. Consequently, performance increases predictably as $D \cdot B$ grows.

## Method

### Overall Architecture
EqR adopts the hierarchical iteration framework of TRM: maintaining a pair of latent states $(\mathbf{z}_H, \mathbf{z}_L)$ (high-level and low-level). The inner loop updates $\mathbf{z}_L$ for $n$ steps conditioned on $\mathbf{z}_H$, followed by one update to $\mathbf{z}_H$ using the refined $\mathbf{z}_L$; the outer loop repeats this for $T$ steps. For $T-1$ steps, `no_grad` and detaching (truncated gradient) are used, corresponding to Segmented Online Training (SOT). An ACT head $\hat q = f_\phi(\mathbf{z}_H)$ is included for difficulty-aware early stopping.

Compared to HRM/TRM, EqR introduces three modifications within the update step: (1) Replacing fixed $\mathbf{z}_0$ with sampling from $\mathcal{N}(0, \sigma_0 I)$; (2) Formulating the update with damping and additive noise as $\mathbf{z}_{k+1}=\mathbf{z}_k+(1-\lambda)r_\theta(\mathbf{z}_k;\mathbf{x})+\beta\varepsilon_k$; (3) Scaling along $D$ (iterations) and $B$ (restarts) during inference, tracked via $\mathrm{NFE}=D \cdot B$, and selecting the Top-1 trajectory based on the minimum average residual of the final steps. These correspond to three attractor-based goals: broad coverage, minor perturbation, and aligned selection signals.

### Key Designs

1.  **Attractor Landscape Perspective + Four-Mode Diagnosis**:
    - **Function**: Provides a unified framework to explain when depth/breadth scaling is effective. It tracks the set of stable long-term states $\mathcal{Z}^*_\theta(\mathbf{x})$ (attractors) reachable for an input $\mathbf{x}$, focusing on task alignment (do attractors decode to correct solutions?) and reachability (are basins easy to enter?).
    - **Mechanism**: Landscapes are categorized into four types: (a) No correct attractor (misjudgment, scaling ineffective); (b) Both correct and incorrect attractors exist (basin selection failure, requires breadth $B$); (c) Correct attractors exist but basins are narrow/weak (reachability failure, $B$ helps select basins while $D$ stabilizes); (d) Broad and stable correct basins (ideal state, $D$ dominates, $B$ has marginal gains). By mapping these to the correlation between "residual vs. task error," $\|f_\theta(\mathbf{z};\mathbf{x})-\mathbf{z}\|$ becomes a task-agnostic diagnostic: strongly correlated in mode (d) and decoupled in mode (a).
    - **Design Motivation**: The binary "convergence vs. non-convergence" of DEQ cannot explain why accuracy increases while residuals stay non-zero; the attractor perspective allows for non-zero residuals and multiple attractors, accurately reflecting real cases like Sudoku with multiple candidates and long-range constraints.

2.  **Training-Side Landscape Shaping: Random Initialization (RI) + Path Noise Injection (NI)**:
    - **Function**: Shapes attractors from "narrow and few" to "broad and aligned" while reducing the train-test distribution gap. RI ensures $\mathbf{z}_0 \sim \mathcal{N}(0, \sigma_0 I)$ instead of zero; NI injects $\beta \varepsilon_k$ per iteration with damping $\lambda$, where $\mathbf{z}_{k+1}=\mathbf{z}_k+(1-\lambda)r_\theta(\mathbf{z}_k;\mathbf{x})+\beta\varepsilon_k$, $\varepsilon_k \sim \mathcal{N}(0, I)$, typically with $\lambda=0.05, \beta=0.01$.
    - **Mechanism**: RI allows the model to "see" more basins during training, and by pairing one $(\mathbf{x}, \mathbf{y})$ with multiple $\mathbf{z}_0$, the loss encourages path independence. NI acts as a stochastic perturbation regulator, allowing the model to escape spurious attractors in modes (b) and (c). During inference, $\beta$ can be increased to enhance exploration (similar to temperature scaling).
    - **Design Motivation**: HRM/TRM training with a single $\mathbf{z}_0$ only shapes the local basin, leading to train-test mismatch during multi-restart inference. NI specifically addresses the "premature entrapment" in modes (b) and (c).

3.  **Two-Axis Test-Time Scaling + Residual Selection**:
    - **Function**: Decomposes compute into independent knobs: Depth $D$ (traversal steps) and Breadth $B$ (independent restarts), using the minimum average residual of the final steps for Top-1 selection.
    - **Mechanism**: After demonstrating that weight-sharing is necessary for generalization, the authors show that models trained on $\le 16$ steps can extrapolate to $> 1024$ steps (equivalent to 40,000 layers) with simultaneous drops in residual and error. For breadth, a Pareto experiment reveals a "cutoff": $B$ is only effective when $D \gtrsim 4$, as trajectories need sufficient steps to meaningfully reach a basin.
    - **Design Motivation**: Splitting scaling into $D$ and $B$ allows for independent diagnosis of "intra-basin refinement" vs. "inter-basin switching." Residual selection eliminates the need for external verifiers or task-specific priors.

### Loss & Training
The main loss follows the TRM style: a CE loss for the LM head $\hat{\mathbf{y}}$ and a BCE loss for the halting head $\hat q$ (fitting $\mathbf{1}[\hat{\mathbf{y}} = \text{gt}]$) at every supervised outer step. Segmented Online Training (SOT) cuts trajectories into segments; the end of each segment involves supervision and an optimizer step, with the next segment starting from a detached carry. This approximates the attractor objective: latent updates find reachable low-residual states, while parameter updates align those states with the ground truth. ACT during training moves confident samples out of the batch early, redistributing compute to harder samples.

## Key Experimental Results

### Main Results

Exact accuracy on **Sudoku-Extreme (9×9)** and **Maze-Unique (30×30)**:

| Method | Sudoku | Maze | Remarks |
| :--- | :--- | :--- | :--- |
| 64-Layer Feedforward | 2.6 | 0.0 | Pure depth is ineffective |
| HRM (Wang 2025) | 55.0† | 0.3 | Hierarchical iteration baseline |
| TRM (Jolicoeur-Martineau 2025) | 84.8† | 44.9 | Prev. SOTA |
| URM (Gao 2025) | 77.6† | 51.4 | — |
| **EqR baseline ($D=16, B=1$)** | 86.4 | 82.2 | +RI+NI on TRM backbone |
| **EqR + depth ($D=64, B=1$)** | 93.0 | 88.9 | Single trajectory extrapolation |
| **EqR + depth+breadth ($D=64, B=128$)** | **99.8** | **93.0** | Residual selection |

The most dramatic improvement is in Maze: pulling performance from TRM's 44.9 to 93.0. RI alone pushed Maze to 68.6, and NI to 82.2, suggesting that the bottleneck was "selecting the wrong basin" rather than model capacity.

### Ablation Study

Evolution of performance (Sudoku-Extreme):

| Configuration | Blocks | Params | NLE | Eval Acc |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla feedforward | 42 | 105.6M | 42 | 2.6 |
| + weight-tied | 2 | 5.03M | 42 | 32.6 |
| + SOT + depth ×16 | 2 | 5.03M | 672 | 74.7 |
| + hierarchical recurrence | 2 | 5.03M | 672 | 76.5 |
| + ACT training | 2 | 5.03M | 672 | 84.8 |

Landscape shaping interventions (Baseline: last row above, $D=16, B=1$):

| Intervention | Sudoku | Maze |
| :--- | :--- | :--- |
| baseline (no RI/NI) | 84.8 | 44.9 |
| + RI | 86.0 | 68.6 |
| + RI + NI (EqR) | 86.4 | 82.2 |

### Key Findings
- **Weight-sharing is necessary for generalization**: Reducing parameters from 105.6M (42-layer feedforward) to 5.03M (2-block weight-tied) increased eval accuracy from 2.6 → 32.6 at constant NLE, proving that repeated application of the same block, rather than distinct layers, is key for OOD.
- **Training for 16 steps extrapolates to 1024+ iterations**: Reasoning over 1024 iterations (equivalent to 40,000 layers) causes residuals to continue falling and accuracy to rise, showing real scaling behavior.
- **Breadth requires an activation threshold of $D \gtrsim 4$**: Pareto heatmaps show $B$ only works when $D$ is sufficiently large, confirming that exploration across the landscape and refinement within a basin are distinct processes.
- **Residual selection $\ge$ majority vote**: Once the landscape is aligned, "Top-1 Converged" (minimum residual) achieves accuracy comparable to or better than majority voting without the overhead of reaching consensus.
- **Major NFE savings at equivalent accuracy**: On Sudoku-Lite, EqR saves $3.76\times$ NFE over the baseline, and EqR+ACT saves $11.34\times$, proving gains come from better reachability of solutions, not just more compute.

## Highlights & Insights
- **Elevating "residual" to a theoretical metric**: While residuals were previously used only for convergence monitoring, this work argues that in an aligned landscape, residuals serve as an internal proxy for task correctness. This leads to the "residual selection" rule that requires no external verifier.
- **The "landscape alignment" metaphor is transferable**: Shaping an internal landscape to align attractors with low-error basins can likely be applied to diffusion sampling, energy-based models, and iterative KV-cache refinement.
- **Minimal cost vs. huge gain of RI/NI**: Both interventions require minimal code and zero additional parameters or compute, yet they nearly doubled Maze performance, suggesting that "exploration injection" is more efficient than increasing model size for constrained tasks.
- **Decoupling train-test compute**: The ability to train on 16 steps and test on 1024 steps decouples the compute upper bounds of training and inference, benefiting deployments with limited training budgets.
- **"Convergence belief" as a first-order ablation**: Switching from majority vote to Top-1 Converged shifts the selection signal from "output space" to "dynamical space," serving as a probe for whether the model has truly learned aligned attractors.

## Limitations & Future Work
- Evaluation is limited to perfectly controlled discrete constraint satisfaction benchmarks (Sudoku, Maze) and has not been tested on natural language, multimodal, or open-ended generation tasks where token-level landscape noise might be prohibitive.
- Hyperparameters $\lambda=0.05, \beta=0.01$ were tuned for Sudoku/Maze; scaling laws for NI intensity across different task difficulties are currently absent.
- $D=64, B=128$ yields $\mathrm{NFE}=8192$; while it scales well, the high per-problem budget may only be economical for tasks with high sensitivity to marginal accuracy gains.
- There is no formal guarantee that residuals will always be a reliable proxy for task accuracy; in fail modes (a) or (b), residual selection will systematically choose incorrect attractors.
- Future directions include replacing fixed Gaussian RI with input-conditioned initialization and upgrading residual selection to a differentiable proxy for "basin volume."

## Related Work & Insights
- **vs DEQ (Bai et al. 2019)**: DEQ strictly solves for one fixed point $\mathbf{z}^*=f_\theta(\mathbf{z}^*;\mathbf{x})$, requiring contractiveness. EqR relaxes this to a set of attractors, allowing multi-modality and non-zero residuals, fitting the behavior of HRM/TRM.
- **vs HRM (Wang et al. 2025) / TRM (Jolicoeur-Martineau 2025)**: While EqR shares their architecture, it replaces their phenomenological "hierarchical convergence" with mechanistic landscape shaping, improving performance from ~84.8 to 99.8.
- **vs URM (Gao et al. 2025)**: URM achieves 51.4 on Maze using a unified iterative framework; EqR achieves 93.0 with the same backbone by simply using better training interventions (RI+NI), proving landscape shaping beats architectural complexity.
- **vs Path-Independence (Anil et al. 2022)**: EqR achieves consistency through RI and joint training on multiple trajectories rather than explicit regularization constraints.
- **vs CoT / search-based reasoning (Wei 2022; Silver 2018)**: While CoT scales in token space, EqR provides a clear counter-example for test-time scaling in the latent space using its own residual as a selection signal.

## Rating
- Novelty: ⭐⭐⭐⭐ The attractor perspective unifies DEQ/HRM/TRM; RI+NI are simple but theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐ Excellent ablation paths, cross-task scaling laws, and extrapolation evidence; lacks natural language verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear conceptual framework followed by empirical proof; consistent terminology and intuitive metaphors.
- Value: ⭐⭐⭐⭐ Provides the first mechanistic, falsifiable explanation for why test-time scaling works in latent reasoning, with a high practical value for constrained tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICLR 2026\] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](../../ICLR2026/interpretability/radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[ICML 2026\] Universal 1/3 Time Scaling in Learning Peaked Distributions](universal_one-third_time_scaling_in_learning_peaked_distributions.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)

</div>

<!-- RELATED:END -->
