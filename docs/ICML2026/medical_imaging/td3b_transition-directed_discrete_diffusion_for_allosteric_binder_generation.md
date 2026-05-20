---
title: >-
  [Paper Note] TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation
description: >-
  [ICML 2026][Medical Imaging][Allosteric regulation] TD3B frames the design of agonists/antagonists as a "directional transition operator" generation task…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Allosteric regulation"
  - "agonist/antagonist"
  - "masked discrete diffusion"
  - "directional Oracle"
  - "gated reward"
date: 2026-05-08
content_hash: 452d1f28b9cc2a47
---

# TD3B: Transition-Directed Discrete Diffusion for Allosteric Binder Generation

**Conference**: ICML 2026  
**arXiv**: [2605.09810](https://arxiv.org/abs/2605.09810)  
**Code**: https://huggingface.co/ChatterjeeLab/TD3B (available)  
**Area**: Medicine & Drug Discovery / Discrete Diffusion / Protein Generation  
**Keywords**: Allosteric regulation, agonist/antagonist, masked discrete diffusion, directional Oracle, gated reward

## TL;DR
TD3B frames the design of agonists/antagonists as a "directional transition operator" generation task, using a directional Oracle + affinity gating + tree search amortized fine-tuning within a masked discrete diffusion framework. This enables a pretrained peptide generator to produce peptide sequences that can specifically bias protein conformational transitions toward activation or inactivation.

## Background & Motivation
**Background**: Current mainstream binder design methods (RFdiffusion, BindCraft, BoltzGen, RareFoldGPCR, etc.) treat the protein as a fixed 3D structure and define the task as "stabilizing a target conformation/interface," essentially a thermodynamic structure-matching problem.

**Limitations of Prior Work**: Allosteric regulation (especially clinical efficacy of GPCRs) depends on a binder's ability to bias the "active ↔ inactive" transition direction, not just stabilize a single conformation. The difference between agonists and antagonists is an asymmetric perturbation along the kinetic pathway; purely structural methods cannot systematically distinguish them, relying only on post hoc filtering or empirical biases, which are limited in effectiveness.

**Key Challenge**: Allosteric function is fundamentally a kinetic/non-equilibrium phenomenon (irreversible, directional), while structure generation models only encode equilibrium priors—their representational spaces are fundamentally mismatched. Structure-centric methods cannot express "this binder biases the transition toward activation."

**Goal**: Design a generative framework that can (i) explicitly model agonist vs antagonist directionality, (ii) decouple from affinity but only act on true binders, and (iii) leverage strong pretrained peptide discrete diffusion priors.

**Key Insight**: The authors borrow the Markov state model to abstract allosteric kinetics as a sequence-conditional transition operator $Q(y)=Q_0+\Delta Q(y)$, where the key quantity is the directed asymmetry $\Delta_{ij}(y)=Q(y)(s_i,s_j)-Q(y)(s_j,s_i)$. In practice, only the discrete label $\mathrm{sign}(\Delta(y))\in\{+1,-1\}$ is observable, not the continuous value. This provides an honest supervision signal: do not regress kinetic rates, use only the direction signal.

**Core Idea**: Treat direction control as amortized target guidance layered on top of a pretrained MDLM: the directional Oracle provides a direction gradient, the affinity model acts as a soft gate, and the combined gated reward is used for TR2-D2 style importance-weighted denoising fine-tuning.

## Method

### Overall Architecture
TD3B consists of three stages: (1) Train a target-aware directional Oracle $f_\phi(y,x)\to[-1,1]$, which takes target protein $x$ and candidate peptide $y$ as input and outputs agonist/antagonist tendency; (2) Use a pretrained affinity predictor $g_\psi(y,x)$ as a soft gate, multiply it with the directional signal $\sigma(d^\star f_\phi(y,x)/\tau)$ to form the gated reward $R(y;d^\star,x)$; (3) Use the gated reward for importance-weighted denoising cross-entropy (WDCE) + contrastive loss + KL regularization to fine-tune the pretrained masked discrete diffusion language model PepTune, and during sampling, incorporate tree search to explore trajectory space under directional conditions. The entire process operates in sequence space, never entering 3D structure.

### Key Designs

1. **Target-Aware Directional Oracle $f_\phi$**:

    - **Function**: Given target protein sequence $x$ and candidate peptide $y$, determine whether it is agonist-leaning ($+1$) or antagonist-leaning ($-1$).
    - **Mechanism**: Use pretrained encoders to pool $h_x,h_y$, then apply gated fusion $z=g\odot h_x+(1-g)\odot h_y$ where $g=\sigma(W_g[h_x;h_y]+b_g)$, followed by an MLP to output a scalar score. Supervision uses confidence-weighted binary classification $\mathcal{L}_{\text{dir}}=\mathbb{E}[\kappa(y)\log(1+\exp(-d\cdot f_\phi(y,x)))]$, with partial agonists assigned lower confidence $\kappa_{\text{part}}\in(0,1)$.
    - **Design Motivation**: Directional information is only coarse-grained $\{+1,-1\}$; regressing continuous kinetic rates would mislead the model. Gated fusion allows the Oracle to utilize both target context and binder structure, more flexibly than simple concatenation.

2. **Affinity Soft-Gated Reward**:

    - **Function**: Combine direction signal and affinity into a single reward, avoiding post hoc Pareto trade-off.
    - **Mechanism**: $R(y;d^\star,x)=g_\psi(y,x)\cdot\sigma(d^\star\cdot f_\phi(y,x)/\tau)$, where the pretrained affinity model $g_\psi\in[0,1]$ acts as a multiplicative gate and the direction term provides an additive bias. Only sequences that "truly bind + have correct direction" receive high reward; non-binders are zeroed out, and wrong-direction sequences are suppressed.
    - **Design Motivation**: Directly using direction as loss would generate "directionally correct but non-binding" junk sequences; explicit Pareto weighting is hard to tune. The multiplicative gate enforces "must be a binder" as a hard condition, with direction only selected within the binding space.

3. **TR2-D2 Style Amortized Fine-Tuning + Directional Contrastive Loss**:

    - **Function**: Bake the gated reward into the MDLM sampling distribution and explicitly separate agonist/antagonist in representation space.
    - **Mechanism**: The training target $p^\star(y)\propto p_{\theta_0}(y)\exp(S(y)/\alpha)$ is optimized with WDCE, trajectory-level importance weights $w(y_{0:1})\propto\exp(S(y_1)/\alpha)\prod_n p_{\theta_0}/p_{\bar\theta}$ correct proposal bias; a margin-based contrastive loss $\mathcal{L}_{\text{ctr}}=\sum_P\|h_\theta(y_i)-h_\theta(y_j)\|^2+\sum_N\max(0,m-\|\cdot\|)^2$ pulls together same-direction samples and pushes apart opposite-direction samples; finally, a KL term tethers $\theta$ near $\theta_0$ to prevent drift. On the sampling side, PepTune-style trajectory-aware tree search is used, with gated reward guiding importance-weighted branch selection.
    - **Design Motivation**: Pure RL has high variance in discrete diffusion; amortization + tree search combines exploration (tree search) and internalization (WDCE). Contrastive loss adds pressure in representation space, preventing the Oracle from learning direction differences only at the classification head.

### Loss & Training
The total loss is $\mathcal{L}=\mathcal{L}_{\text{WDCE}}+\lambda_{\text{ctr}}\mathcal{L}_{\text{ctr}}+\lambda_{\text{reg}}\mathcal{L}_{\text{KL}}$. Training data $\{(x,y,a)\}$ consists of peptide-target pairs with functional labels (full/partial agonist, antagonist, negative); negatives do not participate in direction loss but contribute to affinity gate training.

## Key Experimental Results

### Main Results
The paper validates TD3B on clinically relevant targets such as GPCRs, testing whether it can surpass structure-based and inference-time guided baselines in "directional selectivity." The core evaluation is the separability of generated agonist vs antagonist sequences for the same target in functional space, while maintaining affinity.

| Setting | Metric | TD3B | Structure Baseline (RFdiffusion, etc.) | Key Difference |
|---------|--------|------|----------------------------------------|---------------|
| Agonist-directed generation | Directional selectivity | Significantly positive | Near random | Structure methods cannot encode direction |
| Antagonist-directed generation | Directional selectivity | Significantly negative | Near random | Same as above |
| Affinity maintenance | Predicted affinity | Comparable to structure baseline | Baseline | Gating ensures no degradation |
| Inference-time guidance baseline | Post-filtered direction | Worse than TD3B | — | Post-filtering loses throughput |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| Full TD3B | Achieves both direction & affinity |
| w/o affinity gate | Tends to generate "directionally correct but non-binding" junk sequences |
| w/o contrastive loss | Oracle's representation space direction separability decreases |
| Replace gate with Pareto weighting | Hard to tune weights, direction-affinity trade-off |
| Inference-time guidance instead of fine-tuning | Diversity and direction accuracy both decrease |

### Key Findings
- Amortized fine-tuning is more reliable than inference-time guidance: in discrete space, gradient guidance is inherently limited, so baking rewards into the model distribution is more stable.
- Implementing affinity as a soft gate rather than a Pareto term is a key engineering decision; the latter causes the model to "oscillate" between the two objectives.
- Even with coarse-grained binary direction supervision, contrastive loss can amplify its separability in representation space.

## Highlights & Insights
- **"Direction as a Generative Objective"**: The first to explicitly use allosteric directionality as a sequence generation optimization target, rather than post hoc filtering; this opens a new interface for "function-oriented protein design."
- **Design Philosophy of Gated Reward**: Making "necessary condition (binding)" a soft gate and "directional preference" an additive term is a cleaner multi-objective fusion paradigm than Pareto weighting, transferable to any "first X, then optimize Y" generation task.
- **Honest Supervision Granularity**: The authors deliberately avoid regressing continuous kinetic rates, using only $\mathrm{sign}(\Delta)$; this approach, where the theoretical framework exceeds supervision granularity but avoids overextension, is worth emulating in bio-ML.

## Limitations & Future Work
- Supervision is only at the coarse direction level, unable to directly quantify "strength"—clinical partial agonist subtypes require finer labels or active learning.
- The entire method operates in sequence space, without explicit 3D interface modeling; complex conformational communication pathways may lose structural specificity.
- The Oracle training dataset (peptide-target pairs with functional labels) is limited in scale, and generalization beyond GPCRs is not fully validated.
- Tree search + WDCE is computationally expensive, with lower throughput than inference-only guidance.
- The affinity gate $g_\psi$ itself is a pretrained model, and its biases are seamlessly propagated to TD3B.

## Related Work & Insights
- **vs RFdiffusion / BindCraft / BoltzGen**: These are structure-centric methods aiming to stabilize contact interfaces; TD3B shifts the objective to biasing transition direction, making it complementary rather than a replacement.
- **vs PepTune / TR2-D2**: Also based on MDLM-guided fine-tuning, but their targets are affinity or multi-objective Pareto; TD3B introduces direction supervision, extending the target to the kinetic level.
- **vs DRAKES / GLID2E**: Both use RL-style updates for discrete diffusion policies; TD3B adopts a more stable amortized + tree search path and structures the reward as a gate.
- **vs Classifier Guidance / SMC**: In the discrete domain, gradient guidance is limited; this work uses amortization to address it.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Makes directional allosteric control a diffusion generation objective, a previously unexplored area
- Experimental Thoroughness: ⭐⭐⭐ GPCR validation is a good starting point, but cross-family generalization and real wet-lab validation are still needed
- Writing Quality: ⭐⭐⭐⭐ The mathematical framework (transition operator → direction supervision → gated reward) is logically progressive, with very clear motivation
- Value: ⭐⭐⭐⭐ Provides a new paradigm for functional binder design for clinically valuable targets like GPCRs

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OT-Bridge Editor: Geometrically Constrained Stenosis Editing in Coronary Angiography via Entropic Optimal Transport](geometrically_constrained_stenosis_editing_in_coronary_angiography_via_entropic_.md)
- [\[ICML 2026\] Turning Drift into Constraint: Robust Reasoning Alignment in Non-Stationary Multi-Stream Environments](turning_drift_into_constraint_robust_reasoning_alignment_in_non-stationary_envir.md)
- [\[ICML 2026\] Probing RLVR Training Instability through the Lens of Objective-Level Hacking](probing_rlvr_training_instability_through_the_lens_of_objective-level_hacking.md)
- [\[ICML 2026\] From Holo Pockets to Electron Density: GPT-style Drug Design with Density](from_holo_pockets_to_electron_density_gpt-style_drug_design_with_density.md)
- [\[ICML 2026\] Federated Distillation for Whole Slide Image via Gaussian-Mixture Feature Alignment and Curriculum Integration](federated_distillation_for_whole_slide_image_via_gaussian-mixture_feature_alignm.md)

</div>

<!-- RELATED:END -->
