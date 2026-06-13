---
title: >-
  [Paper Note] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered
description: >-
  [ICML2026][Interpretability][Zeroth-Order Optimization] This is a position paper in which the authors argue that Zeroth-Order (ZO) optimization in deep learning is "underexplored" rather than "underpowered." They present…
tags:
  - "ICML2026"
  - "Interpretability"
  - "Zeroth-Order Optimization"
  - "Variance Control"
  - "Subspace Optimization"
  - "Distributed Training"
  - "Forward Gradient"
date: 2026-05-08
content_hash: ff05b88567275012
---

# Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered

**Conference**: ICML2026 Spotlight  
**arXiv**: [2605.15622](https://arxiv.org/abs/2605.15622)  
**Code**: To be confirmed  
**Area**: optimization  
**Keywords**: Zeroth-Order Optimization, Variance Control, Subspace Optimization, Distributed Training, Forward Gradient

## TL;DR
This is a position paper in which the authors argue that Zeroth-Order (ZO) optimization in deep learning is "underexplored" rather than "underpowered." They present six propositions (P1–P6) across three dimensions: algorithms, systems, and evaluation. The core position is that by moving beyond the constraints of "full-space element-wise estimators" toward subspace/spectral estimation, system-level dividends of forward-only flows, and deconfounded evaluation protocols, ZO can evolve from a niche tool for memory-efficient fine-tuning into a scalable training paradigm.

## Background & Motivation

**Background**: Zeroth-order optimization estimates gradients using finite differences $\hat{\nabla}_{\mathbf{x}} f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u}) - f(\mathbf{x})}{\mu}\mathbf{u}$, bypassing backpropagation (BP). In 2023, MeZO transitioned ZO from input-level low-dimensional scenarios (like adversarial examples or prompt tuning) to weight-level fine-tuning of large models, sparking a research surge in "memory saving via forward passes" (Figure 1 left: arXiv ZO papers show nearly exponential growth after 2023).

**Limitations of Prior Work**: However, a pessimistic view persists in the community—that ZO estimator variance explodes linearly with parameter dimension $d$ and query costs are prohibitive, making it "impossible to scale." This judgment has relegated ZO to a comfort zone of LLM fine-tuning supported by strong task alignment, preventing it from handling training from scratch or large-scale black-box tasks.

**Key Challenge**: The authors argue this is a "misdiagnosis." Most purported ZO limitations stem not from the essence of gradient-free learning, but from three types of "myopic engineering practices": (i) focusing all efforts on the estimator itself (estimator-centric); (ii) operating exclusively in the original full parameter space (full-space); and (iii) evaluating variance only in element-wise perturbation forms. These choices obscure the true advantages of ZO (forward-only, decomposable into scalar + random seed, and naturally parallelizable).

**Goal**: To redraw the current ZO research map into an "algorithm-system-evaluation" stack and establish six overlooked key points (P1–P6). This aims to encourage the community to move beyond the saturated area of gradient estimators toward extracting system-level, subspace-level, and evaluation-level dividends.

**Key Insight**: Instead of proposing a new ZO algorithm, the authors adopt a diagnostic perspective—deconstructing the RGE formula into three analytical dimensions (variance/query/directional derivative) (P1–P3), and establishing positions for three undervalued directions: "subspace + spectral," "communication efficiency + pipeline parallelism," and "confounding effects of task alignment" (P4–P6). Finally, they conclude with five specific "calls to action."

**Core Idea**: ZO is underexplored, not underpowered—redefining ZO from a "cheap substitute for BP" into an independent optimization paradigm that is forward-inference friendly, naturally distributed, and capable of operating in subspaces.

## Method

As this is a position paper without a new algorithm, the "Method" section details the internal logical chain of the six propositions. The authors use the RGE formula as the unified language, translating each position into a modification or reinterpretation of RGE variables (distribution of $\mathbf{u}$ / batch sizes $m,n$ / limit of $\mu \to 0$ / replacing $\mathbf{u}$ with $\mathbf{Pu}$), thus sharing a mathematical framework across the six claims.

### Overall Architecture

P1–P3 define the feasibility boundaries of the "estimator-centric" paradigm: variance control (P1) $\to$ variance-query tradeoff (P2) $\to$ directional derivative perspective as a baseline (P3). P4–P6 shift focus beyond the estimator: subspace/spectral optimization (P4) $\to$ system-level dividends from forward-only flows (P5) $\to$ stripping the "confounder" of task alignment in evaluation (P6). Section §4 condenses P1–P6 into five calls to action (A1: Evaluation protocols, A2: Moving beyond full-space, A3: Generative estimators, A4: ZO-native system stacks, A5: Broadening application frontiers, particularly quantum computing and inference engine reuse).

In Table 1, the authors evaluate 10 representative ZO works from ICML'25 / NeurIPS'25 / ICLR'26 against P1/P2/P3. The conclusion is that nearly all works satisfy P1, while almost none simultaneously consider P2 query costs and P3 forward gradient baselines—providing quantitative evidence of being "underexplored."

### Key Designs

1.  **From RGE to Subspace RGE: A Geometric View of Variance-Dimension Decoupling (P1+P4)**:
    *   **Function**: Transforming the original ZO gradient estimate $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u})-f(\mathbf{x})}{\mu}\mathbf{u}$ (where $\mathbf{u} \in \mathbb{R}^d$) into S-RGE: $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{Pu})-f(\mathbf{x})}{\mu}\mathbf{Pu}$, where $\mathbf{P} \in \mathbb{R}^{d \times r}$, $r \ll d$, and $\mathbf{u} \in \mathbb{R}^r$.
    *   **Mechanism**: In the directional derivative (DD) limit $\mu \to 0$, $\mathbb{E}_{\mathbf{u}}[\hat{\nabla}_{\mathbf{x}}f] = \mathbf{PP}^\top \nabla_{\mathbf{x}} f$. When the columns of $\mathbf{P}$ are orthogonal, $\mathbf{PP}^\top$ is the projection operator that projects the FO gradient onto the subspace spanned by $\mathbf{P}$. This provide a geometric explanation for S-RGE—it is a "subspace approximation" of the FO gradient, with variance reduced from $O(d)$ to $O(r)$. $\mathbf{P}$ can be obtained randomly via QR decomposition of a Gaussian matrix and only requires "lazy updates," introducing minimal overhead. Given that gradients of deep models are approximately low-rank, the accuracy loss of subspace approximation is significantly smaller than the variance benefit.
    *   **Design Motivation**: Directly addresses the deadlock of P1 (variance proportional to $d$) and P2 (diminishing returns in mini-batch averaging). Since full-space variance cannot be reduced cheaply, estimation should occur in a "low-dimensional but informative" space. P4 emphasizes that this path naturally connects to spectral optimization (e.g., Muon's gradient orthogonalization), utilizing the "low-rank gradient structure" as a prior.

2.  **Forward-Only + Shared Seeds = Communication-Efficient Distributed ZO (P5)**:
    *   **Function**: In distributed/federated ZO training, workers only transmit a scalar instead of a full gradient vector, reducing communication bandwidth from $O(d)$ to $O(1)$.
    *   **Mechanism**: The local S-RGE of each worker is written as $\hat{\nabla}_{\mathbf{x}}f_i(\mathbf{x}) = \Delta_i \cdot \mathbf{Pu}_i$, where $\Delta_i = \frac{f_i(\mathbf{x}+\mu\mathbf{Pu}_i) - f_i(\mathbf{x})}{\mu}$ is a scalar. Worker $i$ only transmits $\Delta_i$ along with the random seed used to generate $\mathbf{u}_i$; the central node reconstructs $\mathbf{u}_i$ (and the projection matrix $\mathbf{P}$) locally using the same seed before aggregation. Internally, structured perturbations allow for feature reuse—only the activations of the perturbed part change, allowing the forward pass to start from the perturbed layer rather than recomputing from the input (as verified by FZOO). Furthermore, because ZO bundles gradient computation and availability into a single forward pass, the 1F1B bubbles characteristic of FO pipeline parallelism are eliminated, allowing for single-directional, near-zero bubble "inference-like" scheduling.
    *   **Design Motivation**: Translates the "perceived disadvantage" of ZO (reliance on random perturbations) into a "system advantage" (scalar synchronization + reconstructible randomness). This also provides a privacy interpretation: ZO estimates are inherently noisy (product of scalar $\times$ Gaussian vector) and can be directly embedded into DP fine-tuning pipelines without requiring additional Gaussian noise injection like FO.

3.  **Deconfounded Evaluation: Stripping Task Alignment to Reveal Real ZO Capabilities (P3+P6)**:
    *   **Function**: Requirement for all ZO evaluations to report results under both (a) with task alignment (aligning downstream tasks to pre-training objectives via prompts) and (b) without task alignment, with forward gradient $f'(\mathbf{x};\mathbf{u})\mathbf{u}$ as a mandatory baseline.
    *   **Mechanism**: As $\mu \to 0$, finite differences converge to the directional derivative $f'(\mathbf{x};\mathbf{u}) = \mathbf{u}^\top \nabla_{\mathbf{x}} f(\mathbf{x})$. The forward gradient estimator retrieves this value directly via one JVP, serving as a "noiseless upper bound" for the ZO estimator's variance structure. If forward gradients cannot solve a task, the bottleneck is task difficulty, not ZO; if forward gradients succeed but ZO fails, only then is the ZO estimator fundamentally inadequate. Task alignment simplifies downstream tasks to resemble the pre-training distribution, making ZO appear strong under these "easy-to-learn" conditions. Experiments using Gemma2-2B on SST-2 / RTE / WiC with four stateful ZO methods (Figure 2) show that removing alignment leads to severe performance drops and ranking reversals, indicating that current protocols confound "ZO optimization capability" with "task simplification degree."
    *   **Design Motivation**: A position paper must provide observable evidence of "current failure." Using forward gradient as a reference frame provides a calibration tool for the ZO community: any purportedly superior ZO method must first prove its distance from this "noiseless upper bound" and its remaining advantages in non-aligned scenarios.

### Loss & Training

No new training strategy; the only training algorithm framework provided is the integration of RGE / S-RGE / CGE / forward gradient into an SGD step: $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \hat{\nabla}_{\mathbf{x}} f(\mathbf{x}_t)$.

## Key Experimental Results

This position paper contains one set of confirmatory experiments (Figure 2) to support P6.

### Main Results: Confounding Effect of Task Alignment on ZO Performance

| Task | Method | w/ alignment | w/o alignment | Trend |
|------|------|--------------|---------------|------|
| SST-2 / RTE / WiC | MeZO | Higher | Significant Drop | Universal Drop |
| SST-2 / RTE / WiC | Sparse-MeZO | Higher | Significant Drop | Universal Drop |
| SST-2 / RTE / WiC | HiZOO | Higher | Significant Drop | Universal Drop |
| SST-2 / RTE / WiC | LOZO | Higher | Significant Drop | Universal Drop |

> Results presented as a bar chart in the paper; qualitative conclusion indicates all four stateful ZO methods on Gemma2-2B show significant performance degradation across three GLUE tasks when alignment is removed, and relative rankings reverse between settings.

### Literature Survey: Coverage of P1/P2/P3 in Existing ZO Works

| Representative Method | Use Case | P1 (Var. Control) | P2 (Query Cost) | P3 (Fwd Grad Baseline) |
|----------|---------|----------------|----------------|---------------------|
| ZO-NP / AdaZO / PaZO / Sparse-MeZO / PseuZO / PAZO / OPZO / HiSo | U1 Fine-tuning or U2 Training from Scratch | ✓ | ✗ | ✗ |
| SharpZO | U1 | ✓ | Partial | ✗ |
| FZOO | U1 | ✗ | ✓ | ✗ |

### Key Findings

*   **Task alignment is a hidden contributor to current "good" ZO results**: After removing prompt alignment, all four stateful ZO methods experienced significant drops and ranking reversals, suggesting current leaderboards reflect task simplification rather than the optimization capability of ZO algorithms.
*   **The community focuses 99% of effort on P1**: Table 1 shows 10 representative works almost exclusively solve variance control (P1), while query cost (P2) and forward gradient baselines (P3) are nearly ignored. Only ZO-Bench (Zhang et al. 2024c) simultaneously considers P3.
*   **U2 (Training from scratch) is the most neglected direction**: Most existing ZO work focuses on U1 (pretrained model fine-tuning). In U2 scenarios, query cost is the true bottleneck, necessitating a prioritization of P2.

## Highlights & Insights

*   **Redefining ZO as an "inference-type workload" is the biggest conceptual leap**: The authors point out that ZO's workload profile is identical to RL rollout or serving stages. Therefore, it should run on inference stacks like vLLM / FlashAttention / PagedAttention rather than DeepSpeed / Megatron / FSDP—shifting the system optimization focus from "compressing backward" to "accelerating forward."
*   **The geometric interpretation of S-RGE welds variance control and subspace learning together**: The formula $\mathbb{E}[\hat{\nabla} f] = \mathbf{PP}^\top \nabla f$ unifies "low-rank gradient priors," "projection bias," and "variance-dimension decoupling," providing a clean interface for future work.
*   **Shared seed + scalar transmission protocols** allow ZO to become communication-optimal in federated/distributed scenarios with minimal implementation effort.
*   **Mandatory forward gradient baselines** serve as the most actionable evaluation suggestion—requiring all future ZO papers to report the forward gradient method (a single line of JVP in PyTorch) as a control group to determine where the method's strength truly lies.
*   **The DP argument is elegant**: Aligning ZO's "inherent noise" with the "noise requirements" of differential privacy allows ZO to effectively gain privacy benefits for free in DP fine-tuning scenarios.

## Limitations & Future Work

*   **Limitations acknowledged by authors**: The effectiveness of S-RGE relies on the empirical observation that model gradients are low-rank, which may not hold for all architectures or tasks. A ZO-native system stack does not yet exist; arguments remain theoretical without end-to-end throughput measurements.
*   **Identified limitations**: As a position paper, the empirical evidence is limited (Gemma2-2B + GLUE subtasks) and lacks quantitative tables. The strength of the arguments relies more on literature review than extensive empirical verification. The scoring in Table 1 for P1/P2/P3 may be subjective. The "subspace + spectral" combination advocated in P4 has only shown early evidence on small models and has yet to be proven at the 7B+ scale.
*   **Future Directions**: First, establishing a unified benchmark for "FO vs. forward-gradient vs. ZO" under a fixed query budget. Second, implementing a ZO-native pipeline schedule in a real inference engine (e.g., vLLM) to verify if near-zero bubbles enable training throughput close to inference. Third, developing generative estimators (A3) using ControlNet-style conditional DMs to denoise RGE gradients toward FO gradients.

## Related Work & Insights

*   **vs MeZO (Malladi et al. 2023)**: MeZO brought ZO to LLM fine-tuning but is highlighted as a representative of the "heavy reliance on task alignment" criticized in P6. The authors view it as a catalyst for ZO's revival rather than the end point.
*   **vs ZO-Bench (Zhang et al. 2024c)**: One of the few works to include forward gradients; this paper elevates that methodology to a "general evaluation protocol" (P3+P6+A1).
*   **vs Forward Gradient (Baydin et al. 2022; Ren et al. 2023)**: Instead of treating forward gradient as a competitor, it is elevated to a calibration tool—a noiseless upper bound for ZO estimators.
*   **vs Distributed FO (DeepSpeed / Megatron / FSDP)**: The authors argue against porting ZO into FO system stacks, as FO designs (recomputation for memory, tensor parallelism) counteract ZO's inherent advantages.
*   **vs Local Learning / Bio-plausible BP-free (Hinton 2022; Nøkland 2016, etc.)**: While these seek biological plausibility, their scalability is often worse than ZO. This paper positions ZO as a middle ground that maintains interpretability with better scalability.

## Rating
*   Novelty: ⭐⭐⭐⭐ Not a new algorithm, but P4 (Subspace+Spectral) and P5 (Inference workload + unidirectional pipeline) offer non-trivial perspective shifts.
*   Experimental Thoroughness: ⭐⭐⭐ Limited to one small-scale empirical verification (4 methods × 3 tasks on Gemma2-2B); lacks quantitative tables but fits a position paper.
*   Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure with a consistent mathematical framework (RGE) derived across all positions.
*   Value: ⭐⭐⭐⭐⭐ A necessary clearing of the air for the ZO community—especially regarding forward gradient baselines, inference stacks, and deconfounded evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ICML 2026\] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)
- [\[ICML 2026\] Expand Neurons, Not Parameters](expand_neurons_not_parameters.md)
- [\[ICML 2026\] LLMs Lean on Priors, Not Programming Language Semantics](llms_lean_on_priors_not_programming_language_semantics.md)

</div>

<!-- RELATED:END -->
