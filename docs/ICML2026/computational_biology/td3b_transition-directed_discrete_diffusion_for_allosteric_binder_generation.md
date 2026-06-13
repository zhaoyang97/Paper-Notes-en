---
title: >-
  [Paper Note] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation
description: >-
  [ICML 2026][Computational Biology][Allosteric regulation] TD3B formalizes the design of agonists and antagonists as a "directional transition operator" generation task. It employs a framework combining a direction Oracle…
tags:
  - "ICML 2026"
  - "Computational Biology"
  - "Allosteric regulation"
  - "agonist/antagonist"
  - "masked discrete diffusion"
  - "direction Oracle"
  - "gated reward"
date: 2026-05-08
content_hash: 5f95ece2986a23ec
---

# TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2605.09810](https://arxiv.org/abs/2605.09810)  
**Code**: https://huggingface.co/ChatterjeeLab/TD3B (Available)  
**Area**: Medicine & Drugs / Discrete Diffusion / Protein Generation  
**Keywords**: Allosteric regulation, agonist/antagonist, masked discrete diffusion, direction Oracle, gated reward  

## TL;DR
TD3B formalizes the design of agonists and antagonists as a "directional transition operator" generation task. It employs a framework combining a direction Oracle, affinity gating, and tree-search amortized fine-tuning for masked discrete diffusion (MDLM), enabling a pre-trained peptide generator to produce sequences that directionally bias the active/inactive conformational transitions of proteins.

## Background & Motivation
**Background**: Current mainstream binder design methods (such as RFdiffusion, BindCraft, BoltzGen, and RareFoldGPCR) treat proteins as static 3D structures. They define the task as "stabilizing a specific target conformation or interface," which is essentially equilibrium structural matching.

**Limitations of Prior Work**: Allosteric regulation (especially the clinical efficacy of GPCRs) depends on the binder's capacity to shift the direction of the "active ↔ inactive" transition, rather than merely stabilizing one conformation. The distinction between agonists and antagonists lies in asymmetric perturbations along the kinetic path. Purely structural methods cannot systematically distinguish between them, often relying on post-hoc filtering or empirical biases with limited effectiveness.

**Key Challenge**: Allosteric function is inherently a kinetic/non-equilibrium phenomenon (non-reversible, directional), while structural generative models only encode equilibrium priors. These representation spaces are fundamentally mismatched—structure-centric approaches lack the capacity to express the concept that "this binder biases the transition toward activation."

**Goal**: To design a generative framework capable of (i) explicitly modeling the directionality of agonism vs. antagonism, (ii) decoupling from affinity while remaining effective only for true binders, and (iii) leveraging powerful existing discrete diffusion priors for peptides.

**Key Insight**: The authors leverage Markov State Models to abstract allosteric kinetics as sequence-conditioned transition operators $Q(y)=Q_0+\Delta Q(y)$. The critical quantity is the directed asymmetry $\Delta_{ij}(y)=Q(y)(s_i,s_j)-Q(y)(s_j,s_i)$. Since continuous values are unobservable in practice, only discrete labels $\mathrm{sign}(\Delta(y))\in\{+1,-1\}$ are utilized. This provides an honest supervision criterion: instead of regressing kinetic rates, only directional signals are used.

**Core Idea**: Directional control is implemented as amortized guidance layered on top of a pre-trained MDLM. A direction Oracle provides directional gradients, while an affinity model serves as a soft gate. These are combined into a gated reward, which is then used to fine-tune the model via TR2-D2-style importance-weighted denoising.

## Method

### Overall Architecture
TD3B consists of three stages: (1) Training a target-aware direction Oracle $f_\phi(y,x)\to[-1,1]$ that takes target protein $x$ and candidate peptide $y$ to output agonistic/antagonistic tendency; (2) Treating a pre-trained affinity predictor $g_\psi(y,x)$ as a soft gate, multiplied by the directional signal $\sigma(d^\star f_\phi(y,x)/\tau)$ to form a gated reward $R(y;d^\star,x)$; (3) Fine-tuning the pre-trained MDLM (PepTune) using this gated reward through Importance Weighted Denoising Cross-Entropy (WDCE), contrastive loss, and KL regularization. During sampling, a tree search is employed to explore the trajectory space under directional conditions. The entire pipeline operates solely in sequence space without involving 3D structures.

### Key Designs

1.  **Target-Aware Direction Oracle $f_\phi$**:
    - **Function**: Given a target protein sequence $x$ and a candidate peptide $y$, it predicts whether the tendency is agonistic ($+1$) or antagonistic ($-1$).
    - **Mechanism**: Pre-trained encoders are used to pool $h_x, h_y$, followed by gated fusion $z=g\odot h_x+(1-g)\odot h_y$ where $g=\sigma(W_g[h_x;h_y]+b_g)$. Finally, an MLP outputs a scalar score. Supervision uses binary classification with confidence weights: $\mathcal{L}_{\text{dir}}=\mathbb{E}[\kappa(y)\log(1+\exp(-d\cdot f_\phi(y,x)))]$, where partial agonists are assigned lower confidence $\kappa_{\text{part}}\in(0,1)$.
    - **Design Motivation**: Directional information is only available as coarse $\{+1,-1\}$ labels; forcing a regression of continuous kinetic rates would mislead the model. Gated fusion allows the Oracle to utilize both target context and binder structure more flexibly than simple concatenation.

2.  **Affinity Soft-Gated Reward**:
    - **Function**: Merges directional signals and affinity into a single reward to avoid post-hoc Pareto trade-offs.
    - **Mechanism**: $R(y;d^\star,x)=g_\psi(y,x)\cdot\sigma(d^\star\cdot f_\phi(y,x)/\tau)$, where the pre-trained affinity model $g_\psi\in[0,1]$ acts as a multiplicative gate and the directional term acts as an additive offset. Only sequences that are "true binders + directionally correct" receive high rewards; non-binders are gated to zero, and incorrect directions are penalized.
    - **Design Motivation**: Using direction directly as a loss would lead the model to generate directional but non-binding sequences. Multiplicative gating treats "being a binder" as a prerequisite, allowing direction to be optimized within the binding space.

3.  **TR2-D2 Style Amortized Fine-Tuning + Directional Contrastive Loss**:
    - **Function**: Integrates the gated reward into the MDLM sampling distribution and explicitly separates agonists/antagonists in the representation space.
    - **Mechanism**: The target distribution $p^\star(y)\propto p_{\theta_0}(y)\exp(S(y)/\alpha)$ is optimized via WDCE. Trajectory-level importance weights $w(y_{0:1})\propto\exp(S(y_1)/\alpha)\prod_n p_{\theta_0}/p_{\bar\theta}$ correct proposal bias. Simultaneously, a margin-based contrastive loss $\mathcal{L}_{\text{ctr}}=\sum_P\|h_\theta(y_i)-h_\theta(y_j)\|^2+\sum_N\max(0,m-\|\cdot\|)^2$ pulls same-direction samples closer and pushes opposite directions apart. A KL term keeps $\theta$ near $\theta_0$. Sampling utilizes PepTune-style trajectory-aware tree search guided by the gated reward.
    - **Design Motivation**: Pure RL on discrete diffusion suffers from high variance. The combination of amortization and tree search balances exploration (tree search) and internal adaptation (WDCE). Contrastive loss adds pressure in the representation space to prevent the Oracle from learning differences only at the classification head.

### Loss & Training
The total loss is $\mathcal{L}=\mathcal{L}_{\text{WDCE}}+\lambda_{\text{ctr}}\mathcal{L}_{\text{ctr}}+\lambda_{\text{reg}}\mathcal{L}_{\text{KL}}$. Training data $\{(x,y,a)\}$ is derived from peptide-target pairs with functional labels (full/partial agonist, antagonist, negative). Negative samples contribute to affinity gate training but not directional loss.

## Key Experimental Results

### Main Results
The paper validates TD3B on clinically relevant targets such as GPCRs to determine if it outperforms structural baselines and inference-time guidance in "directional selectivity." The core evaluation metrics are the separability in functional space and affinity maintenance for generated agonist vs. antagonist sequences.

| Setup | Metric | TD3B | Structural Baselines (RFdiffusion, etc.) | Key Difference |
|-------|--------|------|-----------------------------------------|----------------|
| Agonist Generation | Directional Selectivity | Significantly Positive | Near Random | Structural methods cannot encode direction |
| Antagonist Generation | Directional Selectivity | Significantly Negative | Near Random | Same as above |
| Affinity Maintenance | Predicted Affinity | Comparable to Baselines | Baseline | Gating ensures no degradation |
| Inference-time Guidance | Post-filter Direction | Inferior to TD3B | — | Post-filtering sacrifices throughput |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Full TD3B | Achieves both directionality and affinity |
| w/o Affinity Gate | Generates "correct direction but non-binding" sequences |
| w/o Contrastive Loss | Oracle separation in representation space decreases |
| Replace Gate with Pareto Weighting | Hard to tune; trade-off between direction and affinity |
| Inference-time Guidance (no FT) | Both diversity and directional accuracy decrease |

### Key Findings
- Amortized fine-tuning is more reliable than pure inference-time guidance: gradient guidance is limited in discrete spaces, so embedding rewards into the model distribution is more robust.
- Treating affinity as a soft gate rather than a Pareto term is a critical engineering decision; the latter causes the model to oscillate between objectives.
- Even with coarse binary directional labels, contrastive loss can amplify separability in the representation space.

## Highlights & Insights
- **"Direction as a Generative Goal"**: This is the first work to explicitly optimize directional allosteric function during sequence generation rather than via post-hoc filtering, providing a new interface for function-oriented protein design.
- **Gated Reward Philosophy**: Treating "necessary conditions (binding)" as a soft gate and "preferences (direction)" as an additive/multiplicative term is a cleaner multi-objective fusion paradigm than Pareto weighting, applicable to many "X given Y" generation tasks.
- **Honest Supervision Granularity**: The authors consciously avoid regressing continuous kinetic rates, using only $\mathrm{sign}(\Delta)$. This approach of having a theoretical framework ahead of supervision density without over-extrapolation is a valuable model for biological ML.

## Limitations & Future Work
- Supervision is limited to coarse directional labels and cannot quantify "intensity"—clinical classification of partial agonists requires finer labels or active learning.
- The method is based entirely on sequence space and does not explicitly model 3D interfaces; communication paths between complex conformations might lose structural specificity.
- The Oracle training data (peptide-target pairs with functional labels) is limited in scale; generalization beyond GPCRs remains to be fully verified.
- The computational cost of tree search + WDCE is significant, resulting in slower throughput compared to inference-only guidance.
- The affinity gate $g_\psi$ is a pre-trained model; its biases will be inherited by TD3B.

## Related Work & Insights
- **vs RFdiffusion / BindCraft / BoltzGen**: These are structure-centric methods focused on stabilizing interfaces; TD3B shifts the objective to biasing transition directions, acting as a complement rather than a replacement.
- **vs PepTune / TR2-D2**: Also based on MDLM guided fine-tuning, but their objectives are typically affinity or multi-objective Pareto; TD3B extends this to individual kinetic levels.
- **vs DRAKES / GLID2E**: These use RL-style updates for discrete diffusion; TD3B uses more stable amortized paths + tree search and structures the reward in a gated format.
- **vs Classifier Guidance / SMC**: Gradient guidance is limited in the discrete domain; this paper resolves this through amortization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Explicitly targeting directional allosteric control in diffusion generation is a significant gap-filler)
- Experimental Thoroughness: ⭐⭐⭐ (GPCR validation is a strong start, but cross-family generalization and wet-lab validation are needed)
- Writing Quality: ⭐⭐⭐⭐ (Mathematical framework from transition operators to gated rewards is logically sound and clear)
- Value: ⭐⭐⭐⭐ (Provides a new paradigm for designing functional binders for high-value clinical targets)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct](../../ICLR2026/computational_biology/ultra-fast_language_generation_via_discrete_diffusion_divergence_instruct.md)
- [\[ICML 2026\] Scalable Single-Cell Gene Expression Generation with Latent Diffusion Models](scalable_single-cell_gene_expression_generation_with_latent_diffusion_models.md)
- [\[NeurIPS 2025\] Constrained Discrete Diffusion](../../NeurIPS2025/computational_biology/constrained_discrete_diffusion.md)
- [\[ICLR 2026\] Discrete Diffusion Trajectory Alignment via Stepwise Decomposition](../../ICLR2026/computational_biology/discrete_diffusion_trajectory_alignment_via_stepwise_decomposition.md)
- [\[ICML 2026\] Protein Autoregressive Modeling via Multiscale Structure Generation](protein_autoregressive_modeling_via_multiscale_structure_generation.md)

</div>

<!-- RELATED:END -->
