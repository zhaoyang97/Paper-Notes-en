---
title: >-
  [Paper Note] Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor
description: >-
  [ICML 2026][Scientific Computing][Ionic transport] This work treats expensive atomic trajectories as a "privileged auxiliary modality" during training. A bimodal trainer first learns dynamics from trajectories…
tags:
  - "ICML 2026"
  - "Scientific Computing"
  - "Ionic transport"
  - "molecular dynamics"
  - "auxiliary modality learning"
  - "closed-form ridge regression initialization"
  - "privileged information"
date: 2026-05-08
content_hash: 248665cab7220197
---

# Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor

**Conference**: ICML 2026  
**arXiv**: [2605.09311](https://arxiv.org/abs/2605.09311)  
**Code**: https://github.com/jykim-git/MD.git (available)  
**Area**: AI for Science / Materials Prediction / Auxiliary Modality Learning  
**Keywords**: Ionic transport, molecular dynamics, auxiliary modality learning, closed-form ridge regression initialization, privileged information

## TL;DR
This work treats expensive atomic trajectories as a "privileged auxiliary modality" during training. A bimodal trainer first learns dynamics from trajectories, then distills its hidden representations via closed-form ridge regression into a non-autoregressive predictor that only sees equilibrium structures. On lithium ion mean squared displacement prediction, it is 200× faster and more accurate than autoregressive SOTA.

## Background & Motivation
**Background**: Predicting ionic transport properties (MSD, diffusivity, conductivity) of battery materials currently relies on molecular dynamics (MD) simulations: starting from equilibrium structures, numerically integrating Newton's equations to obtain atomic trajectories, then computing transport quantities. Even with MLIP acceleration, a single material still takes hours. The ML community has two main alternatives—autoregressive MD acceleration (e.g., LiFlow, stepwise trajectory generation) and non-autoregressive property prediction (e.g., MatFormer, ComFormer, DenseGNN, structure → property in one forward pass).

**Limitations of Prior Work**:
- Autoregressive approaches are slow at inference and accumulate errors, causing trajectory divergence;
- Non-autoregressive approaches are fast but sacrifice accuracy, as they lack access to dynamics information;
- Existing methods can only utilize either "with trajectory" or "structure-only" datasets, but in reality both are scarce and cannot support each other.

**Key Challenge**: Ionic transport is inherently a long-timescale dynamical process (rare jumps + vibrational background), but fast inference requires starting from static structures; wanting both dynamics and fast inference is a fundamental "input modality vs inference cost" contradiction. Moreover, in small-sample regimes, traditional KD's iterative optimization has high variance, making knowledge transfer unreliable.

**Goal**: (i) Enable a non-autoregressive predictor to learn dynamics priors without requiring trajectories at inference; (ii) Simultaneously utilize both "with trajectory" and "structure-only" datasets; (iii) Achieve stable knowledge transfer even in few-shot scenarios with scarce trajectory data.

**Key Insight**: Treat atomic trajectories as "privileged information"/"auxiliary modality" (auxiliary modality learning, AML), present only during training; use pretrained scientific foundation models (SevenNet for structure embeddings, MOMENT for temporal embeddings) to provide strong priors, avoiding learning from scratch on scarce data; use closed-form ridge regression instead of iterative optimization for modality alignment, avoiding SGD variance explosion in small-sample settings.

**Core Idea**: The combination of privileged dynamics modality + closed-form distillation + cross-dataset representation initialization enables the structure-only predictor to implicitly inherit dynamics representations learned from trajectories.

## Method

### Overall Architecture
Two-level auxiliary modality learning: (1) **Model-level** — First, train a bimodal trainer $g$ on the trajectory dataset $\mathcal{D}^{trj}$ (consuming trajectory embeddings $\mathbf{E}_\mathbf{p}$, structure embeddings $\mathbf{E}_\mathbf{x}$, and temperature embeddings $\mathbf{E}_T$), then use closed-form ridge regression to distill $g$'s merged hidden representation $\mathbf{H}=\mathbf{H}_\mathbf{p}+\mathbf{H}_{\mathbf{x},T}$ into the encoder of predictor $f_1$, followed by finetuning; (2) **Data-level** (optional) — When training predictor $f_2$ on the structure-only dataset $\mathcal{D}^{str}$, initialize the encoder from $g$'s structure encoder and the decoder from $f_1$'s decoder, transferring dynamics knowledge learned from the trajectory dataset across datasets.

### Key Designs

1. **Bimodal Trainer $g$ + Structure-only Regularization**:

    - **Function**: Forces the structure encoder to learn useful dynamics-related representations even when trajectories are available, preventing the strong trajectory encoder from dominating.
    - **Mechanism**: $g$ contains two linear layers $\mathbf{W}_\mathbf{p}$ (trajectory) and $\mathbf{W}_{\mathbf{x},T}$ (structure+temperature), summed to obtain $\mathbf{H}$, then passed through an MLP decoder. The loss includes an auxiliary term requiring accurate prediction using only structure embeddings: $\mathcal{L}(\hat y^i,y_s^i)+\lambda_b\mathcal{L}(\hat y^i_{\mathbf{x},T},y_s^i)$. Structure embeddings use SevenNet's node+edge feature aggregation, then third-order polynomial expansion $\mathbf{E}_\mathbf{x}=[\mathbf{E}_{a,s};\mathbf{E}_{a,s}\odot\mathbf{E}_{a,s};\mathbf{E}_{a,s}^{\odot 3}]$ to enhance linear layer expressiveness.
    - **Design Motivation**: Trajectory signals are too strong; without regularization, the structure encoder becomes a "placeholder," leaving nothing to inherit during closed-form distillation to the structure-only model.

2. **Closed-form Ridge Regression Distillation Initialization**:

    - **Function**: Directly transfers $g$'s hidden representations $\mathbf{H}^i$ to the structure-only predictor $f_1$'s encoder $\mathbf{W}^{trj}$.
    - **Mechanism**: Solve $\min_{\mathbf{W}^{trj}}\sum_i\|\mathbf{X}^i\mathbf{W}^{trj}-\mathbf{H}^i\|_F^2+\lambda_r\|\mathbf{W}^{trj}\|_F^2$, with closed-form solution $\mathbf{W}^{trj}=(\sum_i(\mathbf{X}^i)^\top\mathbf{X}^i+\lambda_r\mathbf{I})^{-1}(\sum_i(\mathbf{X}^i)^\top\mathbf{H}^i)$, solved to floating-point precision via Cholesky, etc. The decoder directly reuses $g_{\text{dec}}$. Subsequent finetuning on $\mathcal{D}^{trj}$ uses only structure-temperature embeddings, without feeding trajectories.
    - **Design Motivation**: Traditional KD uses iterative gradient optimization, which has high variance in small datasets (tens to hundreds of samples) for ionic transport; closed-form solution is one-shot, requires no learning rate/early stopping tuning, and is naturally suited for data-scarce scenarios.

3. **Cross-dataset Initialization (data-level AML)**:

    - **Function**: Transfers dynamics priors learned from the trajectory dataset to the structure-only dataset $\mathcal{D}^{str}$.
    - **Mechanism**: The encoder $\mathbf{W}^{str}$ of $f_2$ is initialized from $g$'s structure encoder $\mathbf{W}_{\mathbf{x},T}$ (which, under structure-only regularization, learns more general representations), while the decoder is initialized from $f_1$'s $f_{\text{dec}}^{trj}$ (which captures robust mappings from hidden representations to transport properties). This "encoder via structure path, decoder via trajectory path" cross-initialization cleverly avoids the issue of $\mathbf{W}^{trj}$ being tied to the trajectory distribution.
    - **Design Motivation**: $\mathbf{W}^{trj}$, due to closed-form fitting to $\mathbf{H}$, is biased toward the trajectory distribution and generalizes poorly if reused for structure-only data; $\mathbf{W}_{\mathbf{x},T}$, pushed by the regularization term to learn structure representations, is more suitable as a starting point for new datasets.

### Loss & Training
$L_1$ loss is used throughout to predict transport quantities on a $\log_{10}$ scale; the bimodal trainer adds a structure-only auxiliary term weighted by $\lambda_b$; closed-form initialization uses $\lambda_r$ to control fit vs overfit. The three datasets are: Dataset 1 (MD-computed Li-MSD, trajectory-based), Dataset 2 (MD-computed multi-element diffusivity, structure-only, Na held out for unseen species test), Dataset 3 (experimental Li conductivity, structure-only).

## Key Experimental Results

### Main Results

| Method | Type | Dataset 1 Inference Time (s) | MAE@600K | MAE@800K | MAE@1000K | MAE@1200K |
|--------|------|-----------------------------|----------|----------|-----------|-----------|
| LiFlow (Nam 2025) | Autoregressive | 2910 | 0.378 | 0.392 | 0.457 | 0.407 |
| MatFormer | Non-autoregressive | 22 | 0.604 | 0.685 | 0.894 | 1.207 |
| ComFormer | Non-autoregressive | 14 | 0.451 | 0.531 | 0.642 | 0.760 |
| DenseGNN | Non-autoregressive | 29 | 0.412 | 0.472 | 0.531 | 0.523 |
| **Ours** | Non-autoregressive | **14** | **0.344** | **0.367** | **0.402** | **0.390** |

Approximately 200× faster than LiFlow, with lower MAE at all temperatures (dynamics knowledge is retained).

Cross-dataset results:

| Method | Dataset 2 MAE($\log_{10}D_{Na}$)@2500K | Dataset 3 MAE($\log_{10}\sigma_{Li}$)@300K |
|--------|----------------------------------------|--------------------------------------------|
| MatFormer | 0.651 | 2.090 |
| ComFormer | 0.517 | 2.150 |
| DenseGNN | 0.312 | 2.048 |
| **Ours** | **0.064** | **1.388** |

5× improvement on Dataset 2; on real experimental Dataset 3, MAE drops by 0.66.

### Ablation Study

| Configuration | Dataset 1 MAE@600K |
|---------------|-------------------|
| Full | 0.344 |
| w/o model-level AML | 0.395 |

The appendix further verifies: removing the structure-only regularization renders the structure encoder useless after closed-form distillation; removing cross-dataset data-level AML greatly reduces improvements on Datasets 2/3; replacing closed-form solution with iterative SGD reduces accuracy in data-scarce settings.

### Key Findings
- **Dynamics priors are distillable**: Even without trajectories at inference, the model inherits vibrational + jump patterns learned from trajectories, thanks to Fourier transforming trajectories to the frequency domain and MOMENT's temporal foundation model for compact representations.
- **Cross-dataset and cross-ion transfer works**: Na ions are excluded during training, yet still benefit from representations learned from Li data.
- **Small data + strong priors**: On datasets with only hundreds of samples, closed-form solution + pretrained foundation model embeddings far outperform training deep networks from scratch.

## Highlights & Insights
- **Practical combination of "privileged information + closed-form distillation"**: Implements the LUPI framework in materials science, and clearly shows that closed-form solutions are more stable than SGD distillation in small data, making it a valuable recipe for "few-shot knowledge transfer."
- **Polynomial embedding enhances linear layers**: Using $[\mathbf{E}; \mathbf{E}^{\odot 2}; \mathbf{E}^{\odot 3}]$ turns linear mapping into finite-order nonlinearity; combined with SevenNet's strong priors, this boosts expressiveness without introducing more parameters.
- **Cross-dataset encoder/decoder cross-initialization**: Avoids the pitfall of "encoder tied to source domain after closed-form distillation," and the idea is transferable to any "pretrain in rich domain, transfer to poor domain" scenario.

## Limitations & Future Work
- Closed-form solution requires inversion of a $D\times D$ matrix; caution is needed for large embedding dimensions. The current use of linear layers + polynomial expansion cleverly avoids this.
- Only validated on a few ion species (Li/Na); more complex multi-element co-diffusion scenarios require further validation.
- On real experimental Dataset 3, MAE is still 1.388 (order-of-magnitude error!), indicating sim-to-real remains an open problem, requiring more experimental data and domain adaptation.
- Assumes trajectory length $L$ is fixed; more complex MD protocols (variable-length trajectories, temperature ramps) require redesign of Fourier representations.

## Related Work & Insights
- **vs LiFlow (autoregressive)**: LiFlow uses generative models to sample atomic trajectories stepwise and then compute transport, which is slow and accumulates errors; this work directly predicts NAR and is more accurate.
- **vs MatFormer / ComFormer / DenseGNN**: These are also non-autoregressive structure → property models, but pure structure input cannot learn dynamics; this work injects AML.
- **vs Traditional KD (Hinton 2015)**: Uses iterative gradient-based logit distillation; this work uses closed-form representation distillation for data-scarce adaptation.
- **vs LUPI / Generalized Distillation**: First to treat atomic trajectories as a privileged modality for materials prediction.
- **Insights**: In fields like biomedicine and chemistry, where data is scarce but "expensive oracles" (e.g., wet experiments, quantum simulations) exist, the paradigm of "pretrain on rich modality, distill via closed form, transfer across datasets" is highly reusable.

## Rating
- Novelty: ⭐⭐⭐⭐ First to treat atomic trajectories as privileged modality + closed-form distillation; approach is novel in AI4Science
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets covering sim+real, compared with autoregressive and multiple NAR baselines, complete ablation
- Writing Quality: ⭐⭐⭐⭐ Motivation (cost/accuracy/data scarcity) is clear, method diagrams are intuitive
- Value: ⭐⭐⭐⭐ 200× acceleration + cross-dataset transfer, with direct engineering value for battery material screening

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[ICML 2026\] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves](polynomial_neural_sheaf_diffusion_a_spectral_filtering_approach_on_cellular_shea.md)
- [\[ICML 2026\] Investigating Advanced Reasoning of Large Language Models via Black-Box Environment Interaction](investigating_advanced_reasoning_of_large_language_models_via_black-box_environm.md)
- [\[ICML 2026\] Top-W: Geometry-Aware Decoding with Wasserstein-Regularized Truncation and Mass Penalties for LLMs](geometry-aware_decoding_with_wasserstein-regularized_truncation_and_mass_penalti.md)
- [\[ICML 2026\] Dual-branch Robust Unlearnable Examples](dual-branch_robust_unlearnable_examples.md)

</div>

<!-- RELATED:END -->
