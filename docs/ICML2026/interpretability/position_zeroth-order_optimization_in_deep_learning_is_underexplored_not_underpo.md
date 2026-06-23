---
title: >-
  [Paper Note] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered
description: >-
  [ICML 2026][Interpretability][Paper Note] This is a position paper where the authors argue that Zeroth-Order (ZO) optimization in deep learning is "underexplored" rather than "underpowered." They present six claims (P1–P6) across three main axes: algorithms, systems, and evaluation. Their core stance is that by moving away from the paradigm of "full-space elem
tags:
  - ICML 2026
  - Interpretability
date: 2026-05-08
content_hash: a01b7b0778175b4a
---
# Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered

**Conference**: ICML2026 Spotlight  
**arXiv**: [2605.15622](https://arxiv.org/abs/2605.15622)  
**Code**: To be confirmed  
**Area**: Optimization  
**Keywords**: Zeroth-order optimization, variance control, subspace optimization, distributed training, forward gradient

## TL;DR
This is a position paper where the authors argue that Zeroth-Order (ZO) optimization in deep learning is "underexplored" rather than "underpowered." They present six claims (P1–P6) across three main axes: algorithms, systems, and evaluation. Their core stance is that by moving away from the paradigm of "full-space element-wise estimators" toward subspace/spectral domain estimation, leveraging system-level dividends of forward-only unidirectional flows, and adopting de-confounded evaluation protocols, ZO can evolve from a niche tool for memory-efficient fine-tuning into a scalable training paradigm.

## Background & Motivation

**Background**: Zeroth-order optimization estimates gradients using finite differences $\hat{\nabla}_{\mathbf{x}} f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u}) - f(\mathbf{x})}{\mu}\mathbf{u}$, bypassing backpropagation (BP). In 2023, MeZO transitioned ZO from input-level low-dimensional scenarios (such as adversarial samples or prompt tuning) to weight-level fine-tuning of large models, sparking a research surge in "memory saving via forward passes" (Fig. 1 left: near-exponential growth in arXiv ZO papers after 2023).

**Limitations of Prior Work**: A pessimistic view prevails in the community—contending that ZO estimator variance explodes linearly with parameter dimension $d$ and that query costs are unfavorable, thus making it "fundamentally impossible to scale." This judgment has pigeonholed ZO into a "comfort zone" of LLM fine-tuning supported by strong task alignment, deemed incapable of training from scratch or handling large-scale black-box tasks.

**Key Challenge**: The authors argue this is a "misdiagnosis." Most alleged ZO limitations stem not from the essence of gradient-free learning, but from three types of "shortsighted engineering practices": (i) focusing solely on the estimator itself (estimator-centric); (ii) operating exclusively in the original full-parameter space (full-space); and (iii) evaluating variance only in the form of element-wise perturbations. These choices obscure the true advantages of ZO: its forward-only nature, its decomposability into scalars and random seeds, and its natural parallelism.

**Goal**: To redraw the existing ZO research map across "algorithm–system–evaluation" stacks and establish six ignored key points (P1–P6) as targets, encouraging the community to shift from estimator competition toward mining system-level, subspace-level, and evaluation-level benefits.

**Key Insight**: Instead of proposing a new ZO algorithm, the authors provide a diagnostic perspective—decomposing the Randomized Gradient Estimator (RGE) formula into three analytical dimensions (variance, query, and directional derivative) (P1–P3), while establishing positions for three undervalued directions: "subspace + spectral," "communication efficiency + pipeline parallelism," and "confounding effects of task alignment" (P4–P6). The paper concludes with five specific "calls to action."

**Core Idea**: ZO is underexplored, not underpowered—redefining ZO from a "cheap substitute for BP" into an independent optimization paradigm that is inference-friendly, naturally distributed, and capable of operating in subspaces.

## Method

As a position paper, there is no new algorithm; the "method" refers to the shared argumentative chain behind the six claims (P1–P6). The entire paper employs a single mathematical framework—the finite difference formula of the RGE—and translates each position into a rewriting or reinterpretation of its variables: changing the distribution of $\mathbf{u}$, tuning the batch sizes $m, n$, taking the limit $\mu \to 0$, or replacing $\mathbf{u}$ with $\mathbf{Pu}$. All claims are thus rooted in the same formulaic structure. The training form referenced is the integration of RGE / S-RGE / CGE / forward gradients into a standard SGD step: $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \hat{\nabla}_{\mathbf{x}} f(\mathbf{x}_t)$.

### Overall Architecture

The six positions are divided into two halves. P1–P3 define the feasibility boundaries of the "estimator-centric" paradigm: moving from variance control (P1) to variance–query tradeoffs (P2), and establishing the directional derivative perspective as an indispensable baseline (P3). P4–P6 move beyond the estimator to examine three neglected dividends: subspace and spectral domain optimization (P4), system-level benefits of forward-only flows (P5), and the necessity of stripping the "task alignment" confounder from evaluations (P6). Finally, Section §4 consolidates these into five calls to action (A1 evaluation protocols, A2 moving beyond full-space, A3 generative estimators, A4 ZO-native system stacks, and A5 broadening application frontiers, specifically quantum computing and inference engine stack reuse).

The "underexplored" judgment is quantified in Table 1, where the authors evaluate 10 representative ZO works from ICML'25, NeurIPS'25, and ICLR'26 against P1/P2/P3. Almost all satisfy P1, but none simultaneously address the query costs of P2 and the forward gradient baseline of P3, indicating that community attention is heavily concentrated in one corner.

### Key Designs

**1. From RGE to Subspace RGE: Decoupling Variance from Dimension**

This addresses the deadlock in P1 and P2: the variance of the original ZO estimator $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u})-f(\mathbf{x})}{\mu}\mathbf{u}$ (where $\mathbf{u} \in \mathbb{R}^d$) is of the same order as the parameter dimension $d$. Mini-batch averaging to reduce variance quickly hits diminishing returns, making it expensive in full-space. The authors propose shifting to a low-dimensional subspace, defined as S-RGE: $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{Pu})-f(\mathbf{x})}{\mu}\mathbf{Pu}$, where $\mathbf{P} \in \mathbb{R}^{d \times r}$, $r \ll d$, and $\mathbf{u} \in \mathbb{R}^r$.

This rewrite is supported by a clean geometric interpretation. In the directional derivative limit $\mu \to 0$, $\mathbb{E}_{\mathbf{u}}[\hat{\nabla}_{\mathbf{x}}f] = \mathbf{PP}^\top \nabla_{\mathbf{x}} f$. When the columns of $\mathbf{P}$ are orthogonal, $\mathbf{PP}^\top$ is exactly the projection operator of the first-order (FO) gradient onto the subspace spanned by $\mathbf{P}$. S-RGE is thus a "subspace approximation" of the FO gradient, and variance drops from $O(d)$ to $O(r)$. $\mathbf{P}$ can be generated randomly via a QR decomposition of a Gaussian matrix and can be "lazily updated," making the overhead negligible. As long as the model gradient itself is approximately low-rank (as observed in Zhao et al. 2024), the accuracy loss from approximation is far smaller than the variance saved. P4 further notes that this path naturally connects to spectral domain optimization (e.g., Muon's gradient orthogonalization), utilizing the "low-rank gradient structure" as a prior.

**2. Forward-Only Flow + Shared Seeds: Scalar-Only Distributed ZO**

This translates the perceived weakness that "ZO must rely on random perturbations" into a system advantage. The key observation is that local S-RGE can be decomposed into $\hat{\nabla}_{\mathbf{x}}f_i(\mathbf{x}) = \Delta_i \cdot \mathbf{Pu}_i$, where $\Delta_i = \frac{f_i(\mathbf{x}+\mu\mathbf{Pu}_i) - f_i(\mathbf{x})}{\mu}$ is a scalar. Consequently, worker $i$ does not need to transmit a full gradient vector; it only sends the scalar $\Delta_i$ and the random seed used to generate $\mathbf{u}_i$. The central node reconstructs $\mathbf{u}_i$ (and $\mathbf{P}$, also using shared seeds) locally before aggregation, reducing communication bandwidth from $O(d)$ to $O(1)$.

The benefits extend beyond nodes. Structured perturbations (by layer, block, or coordinate) within a single machine naturally allow for feature reuse—only the activations of the perturbed part change, allowing the forward pass to start from the perturbed layer rather than recomputing from the input (as verified by FZOO). Furthermore, ZO bundles "gradient calculation" and "gradient availability" into a single forward pass. This eliminates the 1F1B bubbles (caused by strong forward/backward coupling) inherent in FO training under pipeline parallelism, allowing for a unidirectional, near-zero-bubble "inference-style" schedule. The scalar-multiplied Gaussian vector structure also provides a privacy interpretation: ZO estimates are inherently noisy and can be directly embedded into DP fine-tuning pipelines without the extra Gaussian noise injection required by FO.

**3. De-confounded Evaluation: Stripping Alignment to Reveal True Capability**

For a position paper to be valid, it must provide observable evidence of "current status failure." The authors mandate that all ZO evaluations report two settings—with task alignment (using prompts to align downstream tasks to pre-training objectives) and without—while using the forward gradient method $f'(\mathbf{x};\mathbf{u})\mathbf{u}$ as a mandatory baseline. The status of the forward gradient is unique: as $\mu \to 0$, finite differences converge to the directional derivative $f'(\mathbf{x};\mathbf{u}) = \mathbf{u}^\top \nabla_{\mathbf{x}} f(\mathbf{x})$. This can be precisely obtained via a single Jacobian-Vector Product (JVP) and serves as the "noise-free upper bound" of the ZO estimator. This allows for clear attribution: if the forward gradient fails, the task difficulty is the bottleneck; if the forward gradient succeeds but ZO fails, the estimator is truly inadequate.

The danger of task alignment is that it simplifies downstream tasks to be close to the pre-training distribution, making ZO appear exceptionally strong in these "easy-to-learn" scenarios. The paper compares MeZO, Sparse-MeZO, HiZOO, and LOZO on Gemma2-2B across SST-2, RTE, and WiC (Fig. 2). Most methods show significant performance degradation without alignment, and relative rankings even reverse. This suggests that current protocols confound "ZO optimization capability" with the degree of task simplification. Any claim of a superior ZO method must first prove its distance from this noise-free upper bound and its remaining advantage in non-aligned scenarios.

## Key Experimental Results

As a position paper, it includes only one set of corroborative experiments (Fig. 2) to support P6.

### Main Results: Confounding Effect of Task Alignment on ZO Performance

| Task | Method | w/ alignment | w/o alignment | Trend |
|------|------|--------------|---------------|------|
| SST-2 / RTE / WiC | MeZO | High | Significant Drop | Universal drop |
| SST-2 / RTE / WiC | Sparse-MeZO | High | Significant Drop | Universal drop |
| SST-2 / RTE / WiC | HiZOO | High | Significant Drop | Universal drop |
| SST-2 / RTE / WiC | LOZO | High | Significant Drop | Universal drop |

> The paper presents this as a bar chart without specific numerical values; the qualitative conclusion is that four stateful ZO methods show significant drops across three GLUE tasks on Gemma2-2B, and relative rankings reverse between the two settings.

### Literature Survey: Coverage of P1/P2/P3 in Existing ZO Work

| Representative Method | Scenario | P1 (Var Control) | P2 (Query Cost) | P3 (Forward Gradient Baseline) |
|----------|---------|----------------|----------------|---------------------|
| ZO-NP / AdaZO / PaZO / Sparse-MeZO / PseuZO / PAZO / OPZO / HiSo | U1 (FT) or U2 (Scratch) | ✓ | ✗ | ✗ |
| SharpZO | U1 | ✓ | Partial | ✗ |
| FZOO | U1 | ✗ | ✓ | ✗ |

### Key Findings

- **Task alignment is the "hidden hero" of current ZO performance**: After removing prompt alignment, all four stateful ZO methods dropped significantly and relative rankings were reshuffled, indicating that current leaderboards largely reflect task simplification rather than the optimization capability of ZO algorithms.
- **The community focuses 99% of its energy on P1**: As shown in Table 1, 10 representative works almost exclusively solve variance control (P1), while query costs (P2) and forward gradient baselines (P3) are nearly universally ignored.
- **U2 (Training from scratch) is the most neglected direction**: Most existing ZO work focuses on U1 (fine-tuning pretrained models). In U2 scenarios, query costs are the true bottleneck, requiring P2 to be the primary focus.

## Highlights & Insights

- **Redefining ZO as an "inference-type workload" is the most significant conceptual shift**: The authors point out that the workload profile of ZO is identical to RL rollout or serving stages. Therefore, it should run on inference stacks like vLLM, FlashAttention, or PagedAttention rather than DeepSpeed, Megatron, or FSDP—flipping the system optimization direction from "compressing backward" to "accelerating forward."
- **Geometric interpretation of S-RGE welds variance control and subspace learning**: The formula $\mathbb{E}[\hat{\nabla} f] = \mathbf{PP}^\top \nabla f$ unifies "low-rank gradient priors," "projection bias," and "variance-dimension decoupling," providing a clean interface for subsequent work (e.g., hybrid FO–ZO, spectral domain ZO).
- **The communication protocol of shared seeds + scalar transmission** makes ZO communication-optimal in federated or distributed scenarios with minimal implementation cost.
- **"Forward gradient as a mandatory baseline"** is the most actionable evaluation suggestion—any future ZO paper should report the forward gradient method (a single-line JVP in PyTorch) as a control group to determine where the method's strength truly lies.
- **The DP argument is elegant**: Aligning ZO's "intrinsic noise" with the requirements of differential privacy allows ZO to "free-ride" on the privacy budget in DP fine-tuning scenarios.

## Limitations & Future Work

- **Limitations acknowledged by the authors**: The effectiveness of S-RGE is built on the empirical observation that model gradients are low-rank, which may not hold for all architectures or tasks. A ZO-native system stack does not yet exist; the arguments remain at the principle level without end-to-end throughput figures.
- **Identified Limitations**: As a position paper, the evidence relies more on literature review than empirical data. The single experiment (Fig. 2) is small-scale (Gemma2-2B + GLUE) and provides only bar charts. The determination of "partially" in Table 1 lacks a quantitative definition. The "subspace + spectral" combination pushed in P4 lacks evidence for scaling to 7B+ models.
- **Future Work**: 1. Supplement with a unified benchmark comparing "FO vs forward-gradient vs ZO" under fixed query budgets. 2. Implement a ZO-native pipeline prototype in real inference engines (e.g., vLLM) to verify if zero-bubble scheduling can achieve inference-level throughput. 3. Develop generative estimators (A3) using Diffusion Model-based gradient denoisers to break the $O(d)$ variance wall.

## Related Work & Insights

- **vs MeZO (Malladi et al. 2023)**: MeZO sparked interest in LLM fine-tuning via ZO, but it is precisely what P6 criticizes for relying on task alignment. This paper views it as a catalyst rather than the terminal goal.
- **vs ZO-Bench (Zhang et al. 2024c)**: One of the few works to include forward gradients; this paper elevates that methodology to a universal evaluation protocol (P3+P6+A1).
- **vs Forward Gradient (Baydin et al. 2022; Ren et al. 2023)**: Rather than viewing forward gradients as rivals, this paper uses them as a "calibration ruler" for the noise-free upper bound of ZO estimators.
- **vs Distributed FO (DeepSpeed/Megatron/FSDP)**: The authors explicitly argue against porting ZO into FO system stacks, as FO design trade-offs (e.g., recomputation for memory) could exacerbate ZO's costs.
- **vs Local Learning / Bio-plausible BP-free (Hinton 2022; Nøkland 2016)**: While those seek biological plausibility, they often lack scalability; this paper positions ZO as a middle ground that retains interpretability while being scalable.

## Rating
- Novelty: ⭐⭐⭐⭐ While not a new algorithm, the repositioning of ZO as an inference/spectral problem (P4, P5) is a non-trivial paradigm shift.
- Experimental Thoroughness: ⭐⭐⭐ Only one set of small-scale empirical proofs (Gemma2-2B), consistent with a position paper but lacking quantitative detail.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure (boundaries in P1–P3, directions in P4–P6, implementation in A1–A5), using a unified mathematical framework for all claims.
- Value: ⭐⭐⭐⭐⭐ Provides a necessary methodological reset for the ZO community—specifically regarding forward gradient baselines and inference stack integration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](../../ICLR2026/interpretability/behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ICML 2026\] A Deep Learning Model of Mental Rotation Informed by Interactive VR Experiments](a_deep_learning_model_of_mental_rotation_informed_by_interactive_vr_experiments.md)
- [\[ICML 2026\] Expand Neurons, Not Parameters](expand_neurons_not_parameters.md)
- [\[ICML 2026\] Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions](physics_from_video_identifiability_of_time-invariant_second-order_odes_under_min.md)

</div>

<!-- RELATED:END -->
