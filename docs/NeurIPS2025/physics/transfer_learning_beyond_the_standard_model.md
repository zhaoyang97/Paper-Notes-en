---
title: >-
  [Paper Note] Transfer Learning Beyond the Standard Model
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][Transfer Learning] This work investigates whether neural networks pre-trained on the standard cosmological model (ΛCDM) can transfer to beyond-standard-model scenarios (mass…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "Transfer Learning"
  - "Cosmological Inference"
  - "ΛCDM"
  - "Negative Transfer"
  - "Foundation Models"
date: 2026-05-08
content_hash: ecc298a9384011c7
---

# Transfer Learning Beyond the Standard Model

**Conference**: NeurIPS 2025
**arXiv**: [2510.19168](https://arxiv.org/abs/2510.19168)  
**Code**: None (uses the Quijote simulation dataset, publicly available)  
**Area**: Physics
**Keywords**: Transfer Learning, Cosmological Inference, ΛCDM, Negative Transfer, Foundation Models

## TL;DR
This work investigates whether neural networks pre-trained on the standard cosmological model (ΛCDM) can transfer to beyond-standard-model scenarios (massive neutrinos, modified gravity, primordial non-Gaussianity). The study finds that a dummy node architecture can reduce simulation requirements by an order of magnitude, but negative transfer emerges when parameters exhibit strong physical degeneracies (e.g., $\sigma_8$–$M_\nu$).

## Background & Motivation

**Background**: Simulation-based inference (SBI) has been successfully applied to ΛCDM cosmological parameter inference. A central goal of Stage-IV surveys (e.g., DESI) is to detect new physics beyond the standard model—massive neutrinos, modified gravity, and primordial non-Gaussianity.

**Limitations of Prior Work**: Simulations for beyond-ΛCDM scenarios are far more computationally expensive than ΛCDM simulations and must cover a substantially larger parameter space, forming the primary bottleneck for inference.

**Key Challenge**: Training inference models requires large numbers of costly beyond-ΛCDM simulations, yet computational budgets are limited.

**Goal**: To assess whether transfer learning via ΛCDM pre-training followed by beyond-ΛCDM fine-tuning can reduce the number of required beyond-ΛCDM simulations.

**Key Insight**: Drawing an analogy to the foundation model paradigm—ΛCDM as the "foundation model" and beyond-ΛCDM tasks as "downstream tasks."

**Core Idea**: A dummy node is appended to the output layer of the pre-trained network, providing unsupervised latent capacity that can be repurposed during fine-tuning to learn new-physics parameters, while simultaneously exposing the phenomenon of negative transfer induced by physical parameter degeneracies.

## Method

### Overall Architecture
Pre-training on Quijote ΛCDM simulations (32,768 samples) → frozen/fine-tuned weight transfer → fine-tuning with a small number of beyond-ΛCDM simulations (50–2,000) → evaluation of parameter inference MSE.

### Key Designs

1. **Dummy Node Architecture**:

    - **Function**: Additional "dummy" nodes are appended to the output layer during pre-training.
    - **Mechanism**: During pre-training, the network outputs the 5 ΛCDM parameters plus $N$ dummy nodes, with MSE computed only over the ΛCDM parameters; during fine-tuning, the dummy nodes are repurposed to output new-physics parameters (e.g., $M_\nu$, $f_{R0}$).
    - **Design Motivation**: Dummy nodes develop extra representational capacity during pre-training that can be reused to learn new-physics signals during fine-tuning, analogous to the modular head design in foundation models.

2. **Comparison of Three Transfer Architectures**:

    - Dummy node: best-performing; provides additional representational capacity.
    - No-dummy (weight initialization only): second-best; new parameters are initialized from scratch.
    - Attach head (frozen pre-training + appended inference head): worst; pre-trained representations are excessively rigid.

3. **Three Beyond-ΛCDM Scenarios**:

    - Massive neutrinos $M_\nu \in [0.01, 1.0]$ eV: strongly degenerate with $\sigma_8$.
    - Modified gravity f(R): $f_{R0} \in [-3\times10^{-4}, 0]$.
    - Primordial non-Gaussianity: equilateral $f_{NL} \in [-600, 600]$, local $f_{NL} \in [-300, 300]$.

### Loss & Training
- MSE loss, AdamW optimizer ($\beta_1=0.5$, $\beta_2=0.999$).
- Pre-training learning rate: $[10^{-5}, 10^{-1}]$; fine-tuning learning rate: $[10^{-6}, 10^{-3}]$ (more conservative).
- Optuna hyperparameter search (100 trials).
- Input: 79-bin matter power spectrum $P(k)$, $k \in [0.0089,\, 0.5]\ h/\mathrm{Mpc}$.

## Key Experimental Results

### Main Results — Simulation Efficiency

| Beyond-ΛCDM Scenario | Transfer Learning Effect | Simulation Savings |
|---|---|---|
| Massive neutrinos $P(k)$ | Total MSE substantially improved | ~10× |
| Massive neutrinos $MP(k)$ | Negative transfer on $\sigma_8$ and $M_\nu$ | Uncertain |
| Modified gravity f(R) | Substantially improved | ~10× |
| Equilateral $f_{NL}$ | Consistent improvement | Significant |
| Local $f_{NL}$ | No improvement (prior mismatch) | 0 |

### Ablation Study — Architecture Comparison

| Architecture | Total MSE Performance | Degree of Negative Transfer |
|---|---|---|
| Dummy node | Best | Mild (only under $\sigma_8$–$M_\nu$ degeneracy) |
| No-dummy | Second-best | Moderate |
| Attach head | Worst | Severe (negative transfer also in total MSE) |

### Key Findings
- **Dummy node consistently optimal**: outperforms the no-transfer baseline in total MSE across all scenarios.
- **Negative transfer driven by physical degeneracy**: the signals of $\sigma_8$ and $M_\nu$ in the marked power spectrum overlap substantially, requiring the pre-trained $\sigma_8$ mapping to be "unlearned" before $M_\nu$ can be learned.
- **SHAP analysis reveals the mechanism**: small-scale power spectrum information is used to infer $\sigma_8$ during pre-training; after fine-tuning, the same information is reassigned to $M_\nu$, with SHAP value signs for $\sigma_8$ reversing.
- **Benefits emerge with as few as 2,000 pre-training simulations**: the full 32K simulation set is not required; a small pre-training corpus already confers transfer advantages.

## Highlights & Insights
- **The double-edged nature of the foundation model paradigm in physics**: pre-training can accelerate inference but may also bias representations—"pre-training on large standard-model datasets can dramatically reduce costs, but may also bias representations in ways that hinder the discovery of new physics."
- **Negative transfer as a physical signal**: the emergence of negative transfer itself reflects the physical degeneracy structure of the parameter space and can serve as a diagnostic tool.
- **Elegance of the dummy node design**: conceptually simple yet highly effective, offering an architectural pattern applicable to a broad range of transfer learning tasks.

## Limitations & Future Work
- Only simple fully connected networks are used; more expressive architectures (e.g., normalizing flows) are not evaluated.
- Only the matter power spectrum is considered; realistic observables (galaxy clustering, weak lensing) are not validated.
- The failure on local $f_{NL}$ is attributed to prior mismatch rather than the method itself.
- Systematic errors and observational noise are not considered.

## Related Work & Insights
- **vs. Multi-fidelity SBI (Thiele2025, Saoulis2025)**: those works transfer between different fidelity levels of the same physics; this paper transfers between different physical models—a more challenging setting.
- **vs. Foundation models (BERT, CLIP)**: the dummy node is analogous to a modular head design; this paper demonstrates that such a paradigm is also effective for physical inference.
- **Implication**: any application of foundation models to scientific inference should be alert to negative transfer—particularly when new parameters are degenerate with parameters encountered during pre-training.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of transfer from the standard cosmological model to beyond-standard-model scenarios; the negative transfer finding is of independent value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four beyond-ΛCDM scenarios + three architectures + SHAP analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure with a well-balanced presentation of physical intuition and machine learning methodology.
- Value: ⭐⭐⭐⭐⭐ Carries broad cautionary implications for the application of foundation models to physical inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] One-Shot Transfer Learning for Nonlinear PDEs with Perturbative PINNs](oneshot_transfer_learning_nonlinear_pdes_perturbative_pinns.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] Encoding and Understanding Astrophysical Information in Large Language Model-Generated Summaries](encoding_and_understanding_astrophysical_information_in_large_language_model-gen.md)
- [\[NeurIPS 2025\] Integration Matters for Learning PDEs with Backward SDEs](integration_matters_for_learning_pdes_with_backward_sdes.md)
- [\[NeurIPS 2025\] F-Adapter: Frequency-Adaptive Parameter-Efficient Fine-Tuning in Scientific Machine Learning](f-adapter_frequency-adaptive_parameter-efficient_fine-tuning_in_scientific_machi.md)

</div>

<!-- RELATED:END -->
