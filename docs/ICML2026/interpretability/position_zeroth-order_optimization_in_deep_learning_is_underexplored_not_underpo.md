---
title: >-
  [Paper Note] Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered
description: >-
  [ICML 2026][Interpretability][Paper Note] This is a position paper where the authors argue that Zeroth-Order (ZO) optimization in deep learning is not "underpowered" but "underexplored." They present six claims (P1–P6) across algorithmic, systematic, and evaluative dimensions. The core stance is to move beyond the "full-space element-wise estimator" paradigm t
tags:
  - ICML 2026
  - Interpretability
date: 2026-05-08
content_hash: 62aa4f8c383a0edf
---
# Position: Zeroth-Order Optimization in Deep Learning Is Underexplored, Not Underpowered

**Conference**: ICML2026 Spotlight  
**arXiv**: [2605.15622](https://arxiv.org/abs/2605.15622)  
**Code**: To be confirmed  
**Area**: Optimization  
**Keywords**: Zeroth-Order Optimization, Variance Control, Subspace Optimization, Distributed Training, Forward Gradient

## TL;DR
This is a position paper where the authors argue that Zeroth-Order (ZO) optimization in deep learning is not "underpowered" but "underexplored." They present six claims (P1–P6) across algorithmic, systematic, and evaluative dimensions. The core stance is to move beyond the "full-space element-wise estimator" paradigm toward subspace/spectral domain estimation, system-level dividends of forward-only flows, and de-confounded evaluation protocols. This shift aims to transition ZO from a niche tool for memory-efficient fine-tuning to a scalable training paradigm.

## Background & Motivation

**Background**: Zeroth-order optimization estimates gradients using finite differences $\hat{\nabla}_{\mathbf{x}} f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u}) - f(\mathbf{x})}{\mu}\mathbf{u}$, bypassing backpropagation (BP). In 2023, MeZO extended ZO from input-level low-dimensional scenarios (like adversarial examples/prompt tuning) to weight-level fine-tuning of large models, sparking a research surge in "memory saving via forward passes" (Fig. 1 left: nearly exponential growth in arXiv ZO papers since 2023).

**Limitations of Prior Work**: However, a pessimistic view prevails in the community—that ZO estimator variance explodes linearly with parameter dimension $d$ and query costs are unfriendly, making it "fundamentally impossible to scale." This judgment has relegated ZO to a comfort zone of LLM fine-tuning supported by strong task alignment, failing to address training from scratch or large-scale black-box tasks.

**Key Challenge**: The authors identify this as a "misdiagnosis." Most alleged ZO limitations stem not from the nature of gradient-free learning, but from "myopic engineering practices": (i) focusing solely on the estimator itself (estimator-centric); (ii) operating exclusively in the original full parameter space (full-space); and (iii) evaluating variance only in element-wise perturbation forms. These choices obscure the true advantages of ZO (forward-only flow, decomposability into scalars + random seeds, and natural parallelism).

**Goal**: To redraw the map of existing ZO research into a three-tier "Algorithm-System-Evaluation" stack, establishing six ignored key points (P1–P6) as targets to encourage the community to shift from estimator competition to harvesting system, subspace, and evaluation dividends.

**Key Insight**: Instead of proposing a new ZO algorithm, the authors adopt a diagnostic perspective—deconstructing the Random Gradient Estimator (RGE) formula into three analytical dimensions: variance, query, and directional derivatives (P1–P3). They then establish positions on three undervalued directions: "subspace + spectral" (P4), "communication efficiency + pipeline parallelism" (P5), and "confounding effects of task alignment" (P6), culminating in five specific "calls to action."

**Core Idea**: ZO is underexplored, not underpowered—redefining ZO from a "cheap substitute for BP" into an independent optimization paradigm that is forward-inference friendly, naturally distributed, and capable of operating in subspaces.

## Method

As a position paper, no new algorithm is introduced. The "Method" refers to the shared argumentative chain behind the six claims (P1–P6). The entire text utilizes a single mathematical skeleton—the finite difference formula of the RGE—reinterpreting or rewriting its variables for each position: changing the distribution of $\mathbf{u}$, adjusting the batch sizes $m, n$, taking the limit $\mu \to 0$, or replacing $\mathbf{u}$ with $\mathbf{Pu}$. All claims are rooted in the same formulaic tree, where the singular training form involves plugging RGE / S-RGE / CGE / Forward Gradient into the same SGD step: $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \hat{\nabla}_{\mathbf{x}} f(\mathbf{x}_t)$.

### Overall Architecture

The six positions are divided into two halves. P1–P3 define the feasibility boundaries of the "estimator-centric" paradigm: moving from variance control (P1) to variance-query trade-offs (P2), and establishing the directional derivative perspective as an indispensable baseline (P3). P4–P6 shift the focus beyond the estimator to neglected benefits: subspace and spectral domain optimization (P4), system-level advantages of forward-only flows (P5), and the removal of the "confounder" of task alignment in evaluation (P6). Finally, Section 4 distills these into five calls to action (A1: Evaluation protocols, A2: Moving beyond full-space, A3: Generative estimators, A4: ZO-native system stacks, A5: Broadening application frontiers, especially quantum computing and inference engine stack reuse).

The "underexplored" status is quantified in Table 1, where 10 representative ZO works from ICML'25/NeurIPS'25/ICLR'26 are audited against P1/P2/P3. Almost all satisfy P1, while none simultaneously address the query cost of P2 and the forward gradient baseline of P3, indicating that community attention is heavily concentrated in one corner.

### Key Designs

**1. From RGE to Subspace RGE: Decoupling Variance from Dimension**

To address the deadlock in P1 and P2, where the variance of the original ZO estimator $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{u})-f(\mathbf{x})}{\mu}\mathbf{u}$ ($\mathbf{u} \in \mathbb{R}^d$) is on the order of parameter dimension $d$, the authors propose operating in a different space. By placing perturbations in a low-dimensional subspace, the S-RGE is formulated as: $\hat{\nabla}_{\mathbf{x}}f(\mathbf{x}) = \frac{f(\mathbf{x}+\mu\mathbf{Pu})-f(\mathbf{x})}{\mu}\mathbf{Pu}$, where $\mathbf{P} \in \mathbb{R}^{d \times r}$, $r \ll d$, and $\mathbf{u} \in \mathbb{R}^r$.

This rewrite is justified by its clean geometric interpretation. In the directional derivative limit $\mu \to 0$, $\mathbb{E}_{\mathbf{u}}[\hat{\nabla}_{\mathbf{x}}f] = \mathbf{PP}^\top \nabla_{\mathbf{x}} f$. When the columns of $\mathbf{P}$ are orthogonal, $\mathbf{PP}^\top$ is exactly the projection operator of the FO gradient into the subspace spanned by $\mathbf{P}$. S-RGE thus serves as a "subspace approximation" of the FO gradient, with variance dropping from $O(d)$ to $O(r)$. $\mathbf{P}$ can be generated randomly via a QR decomposition of a Gaussian matrix and updated "lazily" with negligible overhead. As long as model gradients are approximately low-rank (as observed in Zhao et al. 2024), the accuracy loss from approximation is significantly smaller than the variance reduction. P4 further notes that this naturally connects to spectral domain optimization (e.g., Muon's gradient orthogonalization), utilizing the "low-rank gradient structure" as an exploitable prior.

**2. Forward-Only Flow + Shared Seeds: Enabling Scalar-Only Communication in Distributed ZO**

This design translates the perceived weakness of "relying on random perturbations" into a system advantage. The key observation is that local S-RGE can be decomposed into $\hat{\nabla}_{\mathbf{x}}f_i(\mathbf{x}) = \Delta_i \cdot \mathbf{Pu}_i$, where $\Delta_i = \frac{f_i(\mathbf{x}+\mu\mathbf{Pu}_i) - f_i(\mathbf{x})}{\mu}$ is merely a scalar. Consequently, worker $i$ does not need to transmit the entire gradient vector; it only sends the scalar $\Delta_i$ and the random seed used to generate $\mathbf{u}_i$. The central node reconstructs $\mathbf{u}_i$ (and the projection matrix $\mathbf{P}$) locally using the same seed before aggregation, reducing communication bandwidth from $O(d)$ to $O(1)$.

The dividends extend internally. Within a single machine, structured perturbations (by layer/block/coordinate) naturally allow feature reuse—only the activations of the perturbed part change, allowing the forward pass to start from the perturbed layer rather than recomputing from the input (as verified by FZOO). Furthermore, ZO bundles "gradient calculation" and "immediate gradient availability" into the same forward pass. This eliminates the 1F1B bubbles (caused by strong forward/backward coupling) unique to FO training in pipeline parallelism, allowing for unidirectional, near-zero-bubble "inference-style" scheduling. The scalar × Gaussian vector structure also provides a privacy interpretation: ZO estimates are inherently noisy and can be directly embedded into DP fine-tuning pipelines without the extra Gaussian noise injection required by FO.

**3. De-confounded Evaluation: Peeling Back Task Alignment to Reveal ZO's True Performance**

For the position paper to stand, it must provide observable evidence of "current failures." The authors require all ZO evaluations to report two settings—with and without task alignment (using prompts to align downstream tasks with pre-training objectives)—and mandate the Forward Gradient method $f'(\mathbf{x};\mathbf{u})\mathbf{u}$ as a baseline. The role of the forward gradient is unique: as $\mu \to 0$, finite differences converge to the directional derivative $f'(\mathbf{x};\mathbf{u}) = \mathbf{u}^\top \nabla_{\mathbf{x}} f(\mathbf{x})$, which can be precisely obtained via a single JVP. Structurally, it is the "noise-free upper bound" of ZO estimators. It helps distinguish responsibility: if the forward gradient fails, the task is difficult; if the forward gradient succeeds but ZO fails, the estimator is at fault.

The risk of task alignment is that it simplifies downstream tasks to resemble the pre-training distribution, making ZO appear exceptionally strong in these "learning-friendly" scenarios. The paper uses Gemma2-2B on SST-2 / RTE / WiC to compare four stateful ZO methods: MeZO, Sparse-MeZO, HiZOO, and LOZO (Fig. 2). Without alignment, most methods show a significant performance drop, and the relative rankings even flip. This suggests that current protocols confound "ZO's optimization capability" with the "degree of task simplification." Any claim of a superior ZO method must first demonstrate its distance from this noise-free upper bound and its remaining advantage in non-aligned scenarios.

## Key Experimental Results

As a position paper, it includes one set of confirmatory experiments (Fig. 2) to support P6.

### Main Results: Confounding Effect of Task Alignment on ZO Performance

| Task | Method | w/ alignment | w/o alignment | Trend |
|------|------|--------------|---------------|------|
| SST-2 / RTE / WiC | MeZO | Higher | Significant drop | General decrease |
| SST-2 / RTE / WiC | Sparse-MeZO | Higher | Significant drop | General decrease |
| SST-2 / RTE / WiC | HiZOO | Higher | Significant drop | General decrease |
| SST-2 / RTE / WiC | LOZO | Higher | Significant drop | General decrease |

> Results are presented as bar charts in the paper without specific numerical values; the qualitative conclusion is that all four stateful ZO methods show significant drops across three GLUE tasks on Gemma2-2B, with relative rankings flipping between settings.

### Literature Survey: Coverage of P1/P2/P3 in Existing ZO Work

| Representative Method | Use Case | P1 (Var. Control) | P2 (Query Cost) | P3 (Forward Grad Baseline) |
|----------|---------|----------------|----------------|---------------------|
| ZO-NP / AdaZO / PaZO / Sparse-MeZO / PseuZO / PAZO / OPZO / HiSo | U1 (FT) or U2 (Scratch) | ✓ | ✗ | ✗ |
| SharpZO | U1 | ✓ | Partial | ✗ |
| FZOO | U1 | ✗ | ✓ | ✗ |

### Key Findings

- **Task alignment is a hidden contributor to current ZO "success"**: After removing prompt alignment, all four stateful ZO methods dropped significantly, and relative rankings changed, indicating that current leaderboards largely reflect task simplification rather than the optimization capability of ZO algorithms.
- **The community spends 99% of its effort on P1**: Table 1 shows that nearly all 10 representative works solve only variance control (P1). Query cost (P2) and forward gradient baselines (P3) are almost entirely ignored; only ZO-Bench (Zhang et al. 2024c) considers P3.
- **U2 (Training from scratch) is the most neglected direction**: Most existing ZO works focus on U1 (fine-tuning pretrained models). In U2 scenarios, query cost is the true bottleneck, requiring P2 to be prioritized.

## Highlights & Insights

- **Redefining ZO as an "inference-type workload" is the major conceptual leap**: The authors point out that ZO's workload profile is identical to RL rollout or serving phases. Therefore, it should run on inference stacks like vLLM, FlashAttention, and PagedAttention rather than DeepSpeed, Megatron, or FSDP. This shift reorients the system stack optimization from "compressing backward" to "accelerating forward."
- **Geometric interpretation of S-RGE links variance control with subspace learning**: The formula $\mathbb{E}[\hat{\nabla} f] = \mathbf{PP}^\top \nabla f$ unifies "low-rank gradient priors," "projection bias," and "variance-dimension decoupling," providing a clean interface for subsequent work (Hybrid FO–ZO, Spectral ZO).
- **Communication protocol via shared seeds + scalar transmission** makes ZO communication-optimal in federated or distributed scenarios with minimal implementation cost—a highly transferable trick.
- **"Forward gradient as a mandatory baseline"** is the most actionable evaluation suggestion—future ZO papers should report the forward gradient method (a one-line JVP in PyTorch) as a control group to determine where the method's strength truly lies.
- **Elegant DP argumentation**: Aligning the "intrinsic noise" of ZO with the "noise requirements" of differential privacy allows ZO to "free-ride" on the privacy budget in DP fine-tuning scenarios, unlike FO which requires additional noise injection.

## Limitations & Future Work

- **Acknowledged Limitations**: The effectiveness of S-RGE relies on the empirical observation that "model gradients are low-rank," which may not hold for all architectures and tasks. A ZO-native system stack does not yet exist; the arguments remain theoretical without end-to-end throughput figures.
- **Self-Identified Limitations**: As a position paper, the experimental evidence (Fig. 2) is small-scale (Gemma2-2B + GLUE subtasks) and lacks quantitative tables. The strength of the argument relies more on the literature review than empirical results. The "partially" tick in Table 1 lacks a quantitative definition. The "subspace + spectral" combination pushed in P4 currently only has early evidence on small models (Muon, LOZO, etc.) and hasn't been proven to maintain variance dividends at 7B+ scales.
- **Future Directions**: First, establish a unified benchmark comparing FO vs. forward-gradient vs. ZO under a fixed query budget (implementing P3 directly). Second, implement a prototype ZO-native pipeline schedule in a real inference engine (e.g., vLLM) to verify if near-zero bubbles can bring LLM ZO training close to inference throughput. Third, move generative estimators (A3) from concept to code, using a ControlNet-style conditional DM to learn a "denoiser from RGE noisy gradients to FO gradients" to see if the $O(d)$ variance wall can be broken.

## Related Work & Insights

- **vs MeZO (Malladi et al. 2023)**: MeZO was the breakthrough for ZO in LLM fine-tuning, but it is precisely the representative criticized in P6 for "relying heavily on task alignment." This paper treats it as a catalyst rather than the endgame for the ZO revival.
- **vs ZO-Bench (Zhang et al. 2024c)**: One of the few works to include forward gradients in its baseline; this paper elevates that methodology to a "general evaluation protocol" (P3+P6+A1), calling for forward gradients to be a mandatory control group.
- **vs Forward Gradient (Baydin et al. 2022; Ren et al. 2023)**: Originally a "close rival" to ZO, this paper upgrades it to a calibration tool—the natural upper bound for ZO estimator accuracy.
- **vs Distributed FO (DeepSpeed / Megatron / FSDP)**: The authors explicitly oppose porting ZO into FO system stacks, as FO design trade-offs (recomputation for memory, tensor parallelism for throughput) would conversely amplify ZO's computational costs.
- **vs Local Learning / Bio-plausible BP-free (Hinton 2022; Nøkland 2016, etc.)**: Those lines pursue biological plausibility but have worse scalability than ZO. This paper positions ZO as the middle ground that retains interpretability while being scalable.

## Rating
- Novelty: ⭐⭐⭐⭐ While not a new algorithm, P4 (Subspace+Spectral) and P5 (Inference workload + Unidirectional pipeline) offer non-trivial perspective shifts from optimization to systems/spectral problems.
- Experimental Thoroughness: ⭐⭐⭐ Only one set of small-scale empirical proofs (Gemma2-2B alignment comparison); matches the scope of a position paper but lacks quantitative tables.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure (P1–P3 feasibility boundaries, P4–P6 new directions, A1–A5 implementation); each position is derived from the same RGE mathematical skeleton with logical arguments, rebuttals, and evaluative evidence.
- Value: ⭐⭐⭐⭐⭐ A necessary methodological reckoning for the ZO community—especially the "forward gradient baseline," "ZO on inference stacks," and "de-confounded evaluation" are immediately adoptable improvements.

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
