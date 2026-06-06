---
title: >-
  [Paper Note] PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials
description: >-
  [ICML 2026][LLM Safety][MLIP] This paper proposes PFT (Phonon Fine-tuning), which randomly samples force constant columns via Hessian-vector products and directly supervises the alignment of the energy Hessian with DFT f…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "MLIP"
  - "Phonon"
  - "Hessian"
  - "Force Constants"
  - "Fine-tuning"
date: 2026-05-08
content_hash: b579080ba4c90af1
---

# PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials

**Conference**: ICML 2026  
**arXiv**: [2601.07742](https://arxiv.org/abs/2601.07742)  
**Code**: None  
**Area**: Scientific Computing / Material Simulation / Machine Learning Interatomic Potentials  
**Keywords**: MLIP, Phonon, Hessian, Force Constants, Fine-tuning

## TL;DR
This paper proposes PFT (Phonon Fine-tuning), which randomly samples force constant columns via Hessian-vector products and directly supervises the alignment of the energy Hessian with DFT force constants during MLIP fine-tuning. Combined with co-training to mitigate catastrophic forgetting, PFT reduces the average phonon thermodynamic error of Nequix MP on the MDR Phonon benchmark by 55% and lowers the thermal conductivity $\kappa_{\text{SRME}}$ from 0.446 to 0.307, achieving SOTA among models trained on MPtrj.

## Background & Motivation

**Background**: Machine Learning Interatomic Potentials (MLIPs) have become cost-effective surrogates for DFT in large-scale material screening. Mainstream universal MLIPs (MACE-MP-0, SevenNet, Nequix, etc.) learn the Born-Oppenheimer Potential Energy Surface (PES) by regressing energy $E$, force $\mathbf{F}=-\nabla E$, and stress $\sigma$ (the EFS loss) on relaxation trajectory datasets like MPtrj and OMat24.

**Limitations of Prior Work**: Many key physical properties—phonon dispersion, vibrational entropy $S$, Helmholtz free energy $F$, constant-volume heat capacity $C_V$, and thermal conductivity $\kappa$—depend not on zero-order or first-order quantities of the PES, but on second-order force constants $\Phi_{aibj}=\partial^{2}E/\partial r_{a,i}\partial r_{b,j}$ or even higher-order derivatives. EFS loss only indirectly constrains second-order derivatives, leading many MPtrj-trained MLIPs to have "over-softened" PES curvatures near equilibrium configurations. This results in systematically low phonon frequencies, or even imaginary frequencies, causing significant distortion in predicted phase stability and thermal conductivity.

**Key Challenge**: Direct supervision of force constants requires calculating the Hessian. However, crystal phonon calculations must be performed on sufficiently large supercells using finite displacement to avoid self-interaction. These supercells often contain thousands of atoms, causing the $3N\times 3N$ full Hessian to explode in both memory and computation at $O(N^2)$, making full training infeasible. Furthermore, phonon data consists entirely of equilibrium configurations, and direct fine-tuning can easily destroy the pre-trained capabilities on non-equilibrium configurations.

**Goal**: (1) Introduce PES second-order curvature as a differentiable training signal into MLIPs; (2) Ensure this signal remains trainable on supercells with thousands of atoms; (3) Complete fine-tuning without losing the original MPtrj capabilities.

**Key Insight**: The authors first plotted a scatter diagram of "Hessian error vs. phonon thermodynamic error" (Fig. 2) across multiple MPtrj-trained base models. They observed a very strong positive correlation—this implies that as long as the Hessian error is reduced, downstream phonon properties will improve accordingly. This reduces the task of "improving phonon properties" to "aligning the Hessian."

**Core Idea**: Add a Hessian alignment loss $\mathcal{L}_\Phi$ to the EFS loss. For each structure, randomly sample one column of the Hessian and calculate the gradient using a single Hessian-vector product (HVP), reducing the single-step training complexity from $O(N^2)$ to $O(N)$. Use co-training with upstream EFS data to prevent forgetting.

## Method

### Overall Architecture
Input: A pre-trained MLIP $\hat{E}_\theta(\mathbf{r})$ (primarily Nequix MP, validated on MACE-MP-0 and Nequix OAM) trained on large-scale trajectory data like MPtrj, plus a phonon dataset (MDR Phonon, ~8.5k training materials, 300,000 finite displacement DFT calculations) and upstream MPtrj data.

Training Pipeline: For each phonon supercell structure $\to$ randomly pick an atom $b$ and Cartesian direction $j$ $\to$ construct a unit vector $\mathbf{v}$ which is 1 only at $(b,j)$ $\to$ calculate the corresponding Hessian column $\nabla^2_\mathbf{r}\hat{E}\,\mathbf{v}$ using HVP $\to$ compute MAE against the DFT force constant column, and combine with the EFS terms to form $\mathcal{L}_\text{PFT}$. For every 1 step of PFT, perform $K=4$ steps of upstream MPtrj EFS fine-tuning (Algorithm 1).

Output: The same $\hat{E}_\theta$, but with PES curvature aligned with DFT. Downstream inference can use either finite displacement or analytical AD to obtain the full force constants; the results are nearly identical.

### Key Designs

1.  **Hessian Column Alignment Loss $\mathcal{L}_\Phi$ + Random Sampling**:
    - **Function**: Explicitly incorporate PES second-order curvature into the training objective, enabling the model to learn the "derivative of force with respect to position" rather than just the "force itself."
    - **Mechanism**: $\mathcal{L}_\Phi = \frac{1}{3N_a}\sum_{a,i}\mathbb{E}_{b,j}\,|\partial^2\hat{E}/\partial r_{a,i}\partial r_{b,j} - \Phi_{aibj}|$. Only one $(b,j)$ is uniformly sampled for each structure within each batch. This is equivalent to comparing only one column of the Hessian. Since the expectation is equivalent to computing the MAE of the full Hessian, it remains an unbiased gradient. The E(3)-equivariant architecture ensures that force constants have strong symmetric redundancy; sampling a single column covers most degrees of freedom in a statistical sense.
    - **Design Motivation**: Authors found that "EFS fine-tuning directly on phonon displacement data" performed worse than the base model (Table 1, Nequix MP fine-tune on disp. $\omega_\text{max}$ error 182 vs. base model 24). This indicates that the EFS signal at displacement points is insufficient to correct curvature; the Hessian must be treated as a first-class supervision target.

2.  **Hessian-Vector Product (HVP) for $O(N)$ Training**:
    - **Function**: Obtain the gradient of any Hessian column within a single backpropagation without explicitly constructing the $3N\times 3N$ Hessian.
    - **Mechanism**: Utilizing the technique from Pearlmutter (1994), $\nabla^2_\mathbf{r}\hat{E}\,\mathbf{v} = \nabla_\mathbf{r}((\nabla_\mathbf{r}\hat{E})^\top \mathbf{v})$. In JAX, this is `jax.jvp(jax.grad(energy), (pos,), (v,))[1]`—one reverse-mode for force, followed by a forward-mode JVP for the directional derivative of the force. The entire batch concatenates multiple structures into a single large disjoint graph and concatenates the sampled $\mathbf{v}$ for each structure into a single vector. A single HVP on the total energy calculates the loss for all structures simultaneously. The optimizer update requires another gradient of the HVP, constituting a "triple-backward."
    - **Design Motivation**: The full Hessian cannot fit in memory for large supercells. HVP reduces each step from $O(N^2)$ to $O(N)$, making Hessian-supervised training on supercells with hundreds of atoms feasible on an A100. The entire PFT process requires only 35 A100 hours (with co-training) / 15 A100 hours (without), significantly lower than the 100 A100 hours of pre-training.

3.  **Upstream EFS Co-training to Prevent Catastrophic Forgetting**:
    - **Function**: Interleave PFT with upstream MPtrj EFS training steps to preserve the model's capabilities on non-equilibrium configurations (relaxation trajectories, energy/stability prediction).
    - **Mechanism**: Every 1 step of PFT is followed by $K=4$ standard EFS update steps on $\mathcal{D}_\text{up}$ (MPtrj) (Algorithm 1, lines 4-7). $K$ is selected by monitoring both validation sets. Phonon data consists naturally of equilibrium configurations; training on it alone causes the PES to drift in non-equilibrium regions, evidenced by a significant increase in energy/force/stress errors on the MPtrj validation set (Fig. 3).
    - **Design Motivation**: Experimental results showed that PFT without co-training significantly worsened EFS errors on MPtrj. Introducing $K=4$ co-training nearly eliminated this degradation while the Hessian MAE only increased slightly. In Matbench Discovery stability classification, co-training limited performance drops to within 1%.

### Loss & Training
$\mathcal{L}_\text{PFT} = \lambda_E\mathcal{L}_E + \lambda_F\mathcal{L}_F + \lambda_\sigma\mathcal{L}_\sigma + \lambda_\Phi\mathcal{L}_\Phi$ (The first three terms follow EFS: MAE for energy/stress, $\ell_2$ for force). For phonon structures, force and stress are approximated as 0 (assumed relaxed to equilibrium). Supercell energy is the unit cell energy multiplied by the number of repetitions. Co-training ratio $K=4$; PFT for 200 epochs. The same recipe was reused for MACE-MP-0 and Nequix OAM without hyperparameter tuning.

## Key Experimental Results

### Main Results

| Dataset / Metric | Nequix MP base | Nequix MP PFT | Nequix MP PFT (w/o co-train) | Prev. SOTA (MPtrj) |
|---|---|---|---|---|
| MDR Phonon $\omega_\text{max}$ (K) MAE | 24 | **12** | 10 | eSEN-MP 24 |
| MDR Phonon $S$ (J/K/mol) MAE | 32 | 14 | **11** | eSEN-MP 14 |
| MDR Phonon $F$ (kJ/mol) MAE | 12 | 5 | **4** | eSEN-MP 4 |
| MDR Phonon $C_V$ (J/K/mol) MAE | 6 | 3 | **2** | eSEN-MP 5 |
| 3rd-order Force Constants $\Phi^{(3)}$ MAE (meV/Å³) | 10.52 | 8.35 | **7.46** | — |
| Matbench Disc. Thermal Cond. $\kappa_\text{SRME}$ ↓ | 0.446 | 0.307 | **0.281** | eSEN-30M 0.340 |

Average reduction in four phonon thermodynamic errors is 55%. The same recipe on MACE-MP-0 reduced $\omega_\text{max}$ from 61 to 19 and $S$ from 60 to 14. PFT still achieved a 50% reduction on the stronger Nequix OAM base model. Furthermore, Nequix MP PFT with 708K parameters outperformed the base OAM model, suggesting Hessian supervision is more efficient than simply stacking upstream data.

### Ablation Study

| Configuration | Hessian MAE | MPtrj EFS Degradation | Description |
|---|---|---|---|
| Nequix MP base | High | 0 | EFS training only |
| EFS fine-tune on phonon displacement | Higher than base | — | $\omega_\text{max}$ jumped from 24 to 182; EFS signal cannot replace Hessian supervision |
| PFT (w/o co-training) | Lowest | Significant worsening on MPtrj (Fig. 3) | Strongest phonons but catastrophic forgetting |
| PFT (co-training, $K=4$) | Slightly higher than w/o co-train | Almost no degradation | Best overall |
| Force Constant Inference: FD vs. Analytical AD | Nearly identical (Table 1) | — | HVP analytical method eliminates the displacement distance hyperparameter |

### Key Findings
- "Hessian error vs. phonon property error" shows strong positive correlation across multiple models (Fig. 2 / Fig. 5). Correcting curvature automatically improves downstream properties—redefining "how to improve phonon prediction" as a regression problem.
- Even though only second-order derivatives are supervised, the model achieves a 20–30% improvement in third-order force constants and thermal conductivity (which depends on third-order derivatives), indicating that Hessian supervision implicitly constrains higher-order smoothness of the PES.
- Direct EFS training on rattled/perturbed structures cannot replace Hessian supervision—this is a significant warning for training paradigms like OMat24 that use "noisy equilibrium configurations."

## Highlights & Insights
- **HVP as a First-Class Training Primitive**: The authors brought HVPs, commonly used in PINNs/implicit differentiation, directly into MLIP training, making $O(N)$ complexity second-order derivative supervision a reality on GPUs. The core implementation is just a few lines of JAX and can be added to any energy model that supports `grad`.
- **"Inverse" Training Targets via Correlation Analysis**: Proving that "Hessian error = proxy for phonon error" first before supervising the Hessian is a robust "find the right loss" paradigm. This can migrate to other physical tasks sensitive to high-order derivatives (electron-phonon coupling, thermoelectricity, elastic moduli, etc.).
- **Minimal Cost Solution for Co-training**: Compared to methods like LoRA or EWC for retaining knowledge, this paper simply mixes upstream data at a $1:K$ ratio. This is engineering-simple yet almost completely suppresses catastrophic forgetting.
- **Small Model + Good Loss > Large Model + More Data**: Nequix MP PFT with 708K parameters outperformed the 30M parameter eSEN-MP and Nequix OAM base models, suggesting that "inductive bias in the loss" may be more valuable than data scale in physical ML.

## Limitations & Future Work
- Validated only on energy-conserving, $E(3)$-equivariant MLIPS. If applied to non-equivariant models, single-column sampling might introduce bias, requiring additional data augmentation as noted by the authors.
- Phonon data is still biased towards systems that are already "dynamically stable." Its effectiveness on systems with strong anharmonicity, polar polarization, or electron-phonon coupling (e.g., ferroelectrics, superconductors) has not been fully verified.
- The source of force constant labels (MDR Phonon) is finite displacement with the PBE functional. Performance on other functionals (SCAN, meta-GGA, hybrid) or distributions with vdW corrections needs further evaluation.
- No public code available (as of v4); reproduction requires rebuilding the training pipeline from Nequix/JAX.
- Potential improvements: Upgrade single-column sampling to block-Hessian sampling, add the acoustic sum rule (ASR) as a hard constraint, or make $K$ adaptive based on the gradient magnitudes of both validation sets.

## Related Work & Insights
- **vs. Pure EFS Large Models (eSEN-MP, SevenNet, MACE-MP-0)**: These rely on stacking data and model capacity to implicitly approximate curvature. PFT chooses to explicitly supervise it. Table 1 shows that the 708K PFT model beats the 30M eSEN-MP, proving "better loss > larger model."
- **vs. EFS Augmentation with Rattled/Perturbed Configurations (OMat24 style)**: Authors experimentally proved this path provides little help or is even harmful to Hessian accuracy, so "noisy samples" alone cannot replace true second-order supervision.
- **vs. Hessian-aware Optimization / Physics-Informed Neural Networks (PINNs)**: Follows the same philosophy of bringing higher-order derivatives into the loss. The contribution here is scaling this mechanism to real material settings with hundreds of atoms in supercells and combining it with co-training to solve forgetting.
- **Insight**: Any task where the "output is a high-order derivative of some scalar function"—molecular vibration spectra, lattice elastic constants, fluid Navier-Stokes residuals, stability analysis of Neural ODEs—can adopt this "HVP column sampling + co-training" recipe.

## Rating
- Novelty: ⭐⭐⭐⭐ — Hessian supervision and HVP are existing tools, but this is the first time they have been combined into a scalable MLIP training paradigm with systematically proven downstream gains.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 3 base models × 2 upstream datasets × 4 benchmark sets (MDR Phonon / 3rd-order FCs / thermal conductivity / Matbench Discovery). The ablations clearly explain "why EFS is not enough" and "why co-training is needed."
- Writing Quality: ⭐⭐⭐⭐ — Equations and algorithms are clear. The correlation analysis in Fig. 2 is very persuasive. Some equation layouts are slightly cramped.
- Value: ⭐⭐⭐⭐⭐ — Achieved double SOTA for phonons and thermal conductivity on MPtrj-trained MLIPs with a training cost less than 1/3 of pre-training. This is a plug-and-play upgrade for the material ML community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning](tcap_tri-component_attention_profiling_for_unsupervised_backdoor_detection_in_ml.md)
- [\[ICML 2026\] From Parameter Dynamics to Risk Scoring: Quantifying Sample-Level Safety Degradation in LLM Fine-tuning](from_parameter_dynamics_to_risk_scoring_quantifying_sample-level_safety_degradat.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)

</div>

<!-- RELATED:END -->
