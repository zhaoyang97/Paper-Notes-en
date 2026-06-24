---
title: >-
  [Paper Note] Task-Restricted Symmetries in Recurrent Weight Space
description: >-
  [ICML 2026][Learning Theory][Weight space symmetry] The paper uses "ordered real Schur coordinates" to decompose the recurrent matrix of a trained single-layer tanh RNN into spectral blocks and non-normal coupling blocks. Through structured ablation by zeroing out blocks, it reveals that certain non-normal couplings can be removed with almost no impact on task behavior (approximate functional invariance), while others are task-critical directions. This profile of "removable/n…
tags:
  - "ICML 2026"
  - "Learning Theory"
  - "RNN Weight Space Analysis"
  - "Weight space symmetry"
  - "Recurrent networks"
  - "Schur decomposition"
  - "Non-normality"
  - "Mechanistic interpretability"
date: 2026-05-08
content_hash: a757bcbe1c9f4308
---

# Task-Restricted Symmetries in Recurrent Weight Space

**Conference**: ICML 2026  
**arXiv**: [2606.18457](https://arxiv.org/abs/2606.18457)  
**Code**: Not disclosed  
**Area**: Learning Theory / RNN Weight Space Analysis  
**Keywords**: Weight space symmetry, Recurrent networks, Schur decomposition, Non-normality, Mechanistic interpretability  

## TL;DR
The paper uses "ordered real Schur coordinates" to decompose the recurrent matrix of a trained single-layer tanh RNN into spectral blocks and non-normal coupling blocks. Through structured ablation by zeroing out blocks, it reveals that certain non-normal couplings can be removed with almost no impact on task behavior (approximate functional invariance), while others are task-critical directions. This profile of "removable/non-removable" components varies across tasks and training solutions, rather than being a universal symmetry of the recurrent weight space.

## Background & Motivation
**Background**: Exact weight space symmetries (e.g., permutation symmetry, orthogonal transformations) have become practical tools for comparing and aligning neural networks, or even directly learning in parameter space (meta-networks taking trained networks as input). These symmetries are characterized by the network implementing a **completely identical** function before and after the transformation.

**Limitations of Prior Work**: Recurrent networks exhibit another type of "soft" redundancy—large structured changes to the recurrent matrix $W_{hh}$ might only keep the behavior **approximately** invariant for a specific task distribution; conversely, changes of the same scale can sometimes destroy behavior. These directions fall outside exact group-theoretic symmetries but significantly shape the functional geometry of the weight space. In original recurrent coordinates, these non-normal structures are difficult to compare across training instances because tanh RNNs do not allow arbitrary orthogonal basis transformations as exact symmetries, unlike linear networks.

**Key Challenge**: There is a disconnect between "distance in parameter space" and "functional equivalence"—large structured changes may preserve functionality, while small directional changes may alter it. Using original weight coordinates makes it impossible to determine which changes are safe.

**Goal**: To find a **reproducible, ablative, and cross-instance comparable** coordinate system to explicitly isolate non-normal couplings in recurrent matrices and test block-by-block: which coupling blocks can be removed with almost no change in behavior (candidate approximate invariances) and which are fragile task-critical directions.

**Key Insight**: Real orthogonal Schur decomposition. It provides an orthogonal basis, a set of (quasi-)diagonal spectral blocks, and strictly upper-triangular non-normal couplings for any real matrix. Even for highly non-normal matrices, the Schur basis remains orthogonal and well-conditioned (unlike eigen-coordinates, which become ill-conditioned during strong transient amplification). Neuroscience has long noted that non-normal couplings shape transient computation in recurrent networks, but a coordinate system to order, align, and ablate them was missing.

**Core Idea**: Use "real Schur coordinates ordered by eigenvalue magnitude" as a diagnostic basis to divide non-normal couplings into structured perturbation directions. With fixed input/readout weights, use block-wise zeroing and evaluate changes in rollout behavior to determine if each coupling block is an "approximate stabilizer" or a "fragile direction."

## Method

### Overall Architecture
The method does not train a new model but applies a "coordinate transformation + structured ablation + behavioral criterion" diagnostic workflow to **already trained** single-layer tanh RNNs. The forward pass of the single-layer tanh RNN is:

$$h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1}), \quad \hat{y}_t = W_{hy}h_t, \quad h_0 = 0,$$

with biases set to zero. The diagnosis only modifies the recurrent matrix $W = W_{hh}$, while the input mapping $W_{xh}$ and readout mapping $W_{hy}$ remain constant. The process consists of three steps: (1) Transform $W$ to ordered real Schur coordinates, separating spectral blocks and non-normal couplings; (2) Zero-out a selected set of coupling blocks, reconstruct $W$, and rerun the network under the original readout coordinates; (3) Use the rollout behavior difference to determine whether the ablation is an "approximate stabilizer" or falls in a fragile direction. This determination is tied to a specific task distribution $\mathcal{D}$, thus yielding **task-restricted, approximate** equivalence classes rather than global group actions in parameter space.

### Key Designs

**1. Ordered Real Schur Coordinates: Turning non-normal couplings into sortable, comparable perturbation directions**

The pain point is that both original recurrent coordinates and eigen-coordinates are difficult to use—the latter become ill-conditioned during strong transient amplification, making stable comparisons across training instances impossible and making ablation basis-sensitive. The paper uses real Schur decomposition $W = QTQ^\top$ ($Q$ is orthogonal, $T$ is real quasi-upper triangular) and splits $T$ into $T = B + N$, where $B$ contains block-diagonal $1\times1$ and $2\times2$ real eigenvalue blocks, and $N$ represents strictly block-upper-triangular **non-normal couplings**. Schur blocks are sorted by non-increasing eigenvalue magnitude. A relative threshold $\alpha$ divides the leading spectral blocks into a reference sector $R=\{i:|\lambda_i|\ge\alpha\rho(W)\}$ (where $\rho(W)$ is the spectral radius, and $R$ corresponds to the rotation-like leading subspace) and a complementary sector $C$. Under this ordered partition, non-normal couplings naturally split into three blocks: intra-sector $T_{RR}$, directional cross-block $T_{C\to R}$ (flowing from the complementary to the leading sector), and intra-complementary $T_{CC}$. The orthogonal Schur basis remains well-conditioned even for strongly non-normal matrices, turning these couplings into reproducible structured perturbation directions that can be aligned across instances. The main experiment fixes $\alpha=0.9$ as a prior selection.

**2. Structured Ablation with Fixed Encoding/Decoding: Causal intervention without retraining readouts**

Once coordinates are established, an intervention means setting corresponding entries in $N$ for a set of Schur coupling blocks $S$ to **zero**, reconstructing $\widetilde{W}_{hh}(S) = Q\widetilde{T}(S)Q^\top$, and re-evaluating the network **without modifying input or readout weights**. This is critical: by fixing the encoder/decoder, the test measures whether the "original input-output mapping is preserved under original readout coordinates." If a linear/ridge regression readout were re-fitted after ablation, it would answer a different question—whether the perturbed latent dynamics still contain task information—which must not be conflated. All perturbations are applied post-training as a purely mechanistic intervention on the trained controller.

**3. Criteria for Approximate Functional Invariance and Two Metrics: Distinguishing "Stabilizers" from "Fragile Directions"**

To determine if a coupling block is removable, the paper defines an $\epsilon$-stabilizer: let $f_W$ be the network's rollout function on task distribution $\mathcal{D}$, $d_\mathcal{D}$ be the rollout difference, and $\epsilon$ be the tolerance. An intervention $S$ is an $\epsilon$-stabilizer on $\mathcal{D}$ when $d_\mathcal{D}(f_W, f_{\widetilde{W}_{hh}(S)})\le\epsilon$. A coupling block that shows minimal difference after zeroing while removing non-negligible Schur mass is a candidate approximate functional invariance; a sharp performance drop indicates a fragile functional direction. For neuroscience-style tasks, the "Fraction of Variance Unexplained" $\mathrm{FVU}=\mathbb{E}\|\hat{y}-y\|^2/\mathbb{E}\|y-\bar{y}\|^2$ is used for the held-out set error, reporting two summary quantities: $\Delta\mathrm{FVU}=\mathrm{FVU}(\widetilde{W}_{hh})-\mathrm{FVU}(W_{hh})$ captures raw degradation, and normalized sensitivity

$$S_{\Delta T} = \frac{\Delta\mathrm{FVU}}{\|\Delta T\|_F/\|T\|_F}$$

measures degradation **per unit of removed Schur mass**, identifying small sectors with disproportionate impact. $\Delta\mathrm{FVU}$ is the primary behavioral effect, while $S_{\Delta T}$ acts as an auxiliary magnifying glass.

### Loss & Training
Ours does not introduce new training objectives. RNNs are trained standardly: Copy task with $N_h\in\{56,64,72\}$ under four recurrent architectures (dense default / dense orthogonal / dense normal / Cayley parameterized orthogonal); Neuroscience-style tasks use $N_h=64$, orthogonal initialization, Adam (LR $10^{-3}$), batch 64, 30 epochs. All ablations are applied after training is completed.

## Key Experimental Results

### Main Results
On a dense orthogonal, $N_h=72$ solution for the copy task, the impact of block-wise ablation on autonomous replay accuracy (first 128 generated symbols) is shown below. Removing $T_{CC}$ alone causes almost no drop (approximate stabilizer), while removing $T_{C\to R}$ pushes the model into a significantly lower accuracy functional class.

| Ablation Configuration | Autonomous Replay Accuracy | Interpretation |
|----------|----------------|------|
| Full Model | 1.00 | Baseline |
| $-T_{CC}$ | 1.00 | Performance preserved, candidate approximate invariance |
| $-T_{RR}$ | 0.876 | Intermediate functional class |
| $-T_{RR}, -T_{CC}$ | 0.875 | Nearly identical to $-T_{RR}$ alone |
| $-T_{C\to R}$ | 0.639 | Pushed into low-accuracy functional class |
| $-T_{C\to R}, -T_{CC}$ | 0.639 | Identical to $-T_{C\to R}$ alone |
| $-T_{RR}, -T_{C\to R}$ | 0.624 | Close to total ablation |
| Full Ablation | 0.624 | Lowest |

Threshold sensitivity checks (Table 1) show the qualitative profile is invariant for $\alpha\in\{0.85, 0.90, 0.95\}$.

### Ablation Study
Raw degradation $\Delta\mathrm{FVU}$ for single-block ablations across three neuroscience-style tasks (Mean values; full model FVU: flip-flop 0.0048, sine 0.0036, context-integration 0.0104):

| Task | $\Delta\mathrm{FVU}$ of $-T_{C\to R}$ | $\Delta\mathrm{FVU}$ of $-T_{CC}$ | Dominant Fragile Block |
|------|------|------|------|
| 3-bit flip-flop | $9.45\times10^{-2}$ | $4.96\times10^{-2}$ | $T_{C\to R}$ ($T_{RR}$ negligible) |
| Sine Generation | 1.73 | 2.08 | $T_{CC}$ for raw $\Delta\mathrm{FVU}$, $T_{C\to R}$ for sensitivity (21.1) |
| Context-Integration | 0.37 | 0.94 | $T_{CC}$ dominates (consistent with slow integration) |

### Key Findings
- **No Schur coupling block is universally safe**: In the copy task, $T_{CC}$ is approximately removable while $T_{C\to R}$ is not; however, in sine generation and context integration, $T_{CC}$ becomes the dominant fragile block. Removability depends entirely on the task and training solution.
- **Normalized sensitivity identifies small critical sectors**: In sine generation, $T_{C\to R}$ does not have the largest raw degradation, but its $S_{\Delta T}=21.1$ is the highest—it has high impact despite small mass.
- **Parameterization dictates redundancy structures**: Cayley orthogonal constructions have almost zero non-normal complementary coupling, so these ablations have almost no effect; dense orthogonal solutions exhibit the removable $T_{CC}$ redundancy.

## Highlights & Insights
- **Using Schur coordinates instead of eigen-coordinates** is the masterstroke: being orthogonal, well-conditioned, and alignable across instances, it converts "hard-to-compare non-normal structures" into "sortable, ablative perturbation directions," avoiding the pitfalls of ill-conditioned eigen-coordinates under strong transient amplification.
- **Fixed readout vs. Re-fitted readout** distinction is disciplined and honest: only fixed encoding/decoding measures if the "original function is preserved," as opposed to whether "latent dynamics still contain task information"—the authors explicitly separate these two questions.
- **Framing conclusions as "task-restricted approximate invariance" rather than "universal symmetry"**: The authors emphasize that ablation profiles vary by task and solution, avoiding the misinterpretation that non-normal components are generally negligible. This restraint adds credibility.

## Limitations & Future Work
- **Low-dimensional task confusion**: The authors acknowledge that since these tasks are low-dimensional, the trained network might only utilize a low-dimensional latent subspace. Thus, Schur ablation might preserve performance simply by avoiding activity directions aligned with the readout, rather than the coupling having no computational role. The experiment does not separate the "subspace explanation" from the "Schur coordinate explanation."
- **Limited scope**: Only tested basic single-layer tanh RNNs on simple tasks with narrow width ranges. Evidence supports "Schur ablation as a diagnostic tool for trained recurrent controllers" rather than a universal claim about non-normal structures.
- Future Work: Extending diagnosis to gated architectures and state-space models (SSMs), combining it with activity manifold analysis to separate subspace vs. coupling explanations.

## Related Work & Insights
- **vs. Exact Weight Space Symmetries (Entezari et al. / Ainsworth et al. / Navon et al.)**: They characterize global group actions that keep functions **identical**; this paper provides **task-restricted, approximate** equivalence classes defined by rollout behavior. These are complementary: Schur ablation captures "large functional-preserving / small functional-altering" changes that exact symmetries miss.
- **vs. Non-normal Recurrent Dynamics (Murphy & Miller / Hennequin et al. / Bondanelli & Ostojic)**: These works show non-normal couplings shape transient computation; this paper places them in a sortable, ablative coordinate system, moving from descriptive phenomena to block-wise causal intervention.

## Rating
- Novelty: ⭐⭐⭐⭐ Using real Schur coordinates as a diagnostic basis for recurrent weight space and defining task-restricted approximate invariance is a novel perspective.
- Experimental Thoroughness: ⭐⭐⭐ Covers four recurrent constructions across tasks with sensitivity checks, but the architecture and scale are narrow.
- Writing Quality: ⭐⭐⭐⭐ Clear conceptual definitions and honest constraints on the scope of conclusions.
- Value: ⭐⭐⭐⭐ Provides a reproducible mechanistic diagnostic tool for functional redundancy/fragility in recurrent networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Catastrophic Forgetting is Low-Rank: A Function-Space Theory for Continual Adaptation](catastrophic_forgetting_is_low-rank_a_function-space_theory_for_continual_adapta.md)
- [\[ICLR 2026\] On the Benefits of Weight Normalization for Overparameterized Matrix Sensing](../../ICLR2026/learning_theory/on_the_benefits_of_weight_normalization_for_overparameterized_matrix_sensing.md)
- [\[ICLR 2026\] In-Context Algorithm Emulation in Fixed-Weight Transformers](../../ICLR2026/learning_theory/in-context_algorithm_emulation_in_fixed-weight_transformers.md)
- [\[ICLR 2026\] Neural Collapse in Multi-Task Learning](../../ICLR2026/learning_theory/neural_collapse_in_multi-task_learning.md)
- [\[ICML 2026\] Understanding the Parameter Space Geometry of Transformers Encoding Boolean Functions](understanding_the_parameter_space_geometry_of_transformers_encoding_boolean_func.md)

</div>

<!-- RELATED:END -->
