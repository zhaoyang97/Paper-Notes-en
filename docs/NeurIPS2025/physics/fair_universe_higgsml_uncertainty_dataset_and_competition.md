---
title: >-
  [Paper Note] FAIR Universe HiggsML Uncertainty Dataset and Competition
description: >-
  [NeurIPS 2025][Physics][Higgs boson] This work provides a standardized dataset of 280 million simulated LHC collision events and a competition platform featuring six parameterized systematic biases (detector calibration…
tags:
  - "NeurIPS 2025"
  - "Physics"
  - "Higgs boson"
  - "systematic uncertainties"
  - "confidence intervals"
  - "competition dataset"
  - "bias parameterization"
  - "nuisance parameters"
  - "profile likelihood"
date: 2026-05-08
content_hash: 42a2a8ae66de726b
---

# FAIR Universe HiggsML Uncertainty Dataset and Competition

**Conference**: NeurIPS 2025
**arXiv**: [2410.02867](https://arxiv.org/abs/2410.02867)
**Code**: [FAIR-Universe/HEP-Challenge](https://github.com/FAIR-Universe/HEP-Challenge) (competition platform + baseline code)
**Area**: Physics / High-Energy Physics ML / Uncertainty Quantification
**Keywords**: Higgs boson, systematic uncertainties, confidence intervals, competition dataset, bias parameterization, nuisance parameters, profile likelihood

## TL;DR

This work provides a standardized dataset of 280 million simulated LHC collision events and a competition platform featuring six parameterized systematic biases (detector calibration + background composition) alongside an asymmetric coverage penalty metric. Participants are required to construct robust 68.27% confidence intervals for the Higgs signal strength $\mu$. The winning solutions, based on profile-free surrogate modeling, achieve confidence intervals approximately 20% narrower than conventional binned methods.

## Background & Motivation

**Background**: High-energy physics (HEP) requires rigorous quantification of systematic uncertainties ("known unknowns") to support statistical significance claims in new particle discoveries. The 2014 HiggsML competition focused solely on event classification (signal vs. background), whereas the central challenge in modern physics measurements is constructing robust confidence intervals under multiple correlated systematic biases.

**Limitations of Prior Work**: (a) There is no standardized ML benchmark that incorporates parameterized systematic uncertainties — existing datasets address classification tasks and do not involve the profiling of nuisance parameters; (b) the mature profile likelihood methods of the physics community lack effective integration with ML uncertainty quantification techniques (conformal prediction, Bayesian methods, etc.); (c) traditional binned histogram analyses discretize continuous features, discarding event-level information.

**Key Challenge**: AI-for-Physics must simultaneously pursue **precision** (narrow confidence intervals → stronger discovery power) and **reliability** (correct coverage → no spurious discoveries), yet these two objectives are inherently in tension under systematic biases — ignoring biases yields narrow intervals with insufficient coverage, while excessive conservatism sacrifices discovery power.

**Goal**: To create a standardized uncertainty-aware benchmark — 280M events, six bias parameters, and well-defined evaluation metrics — enabling the ML community to develop and compare uncertainty-aware methods in a controlled environment.

**Key Insight**: Systematics scripts are provided so that the dataset can be regenerated for any $(\mu, \vec{\alpha})$ combination, advancing the problem from "is the classification correct?" to "is the confidence interval robust?" The dataset is effectively a **function** rather than a static table.

**Core Idea**: By combining a regenerable dataset with parameterized systematic biases and an asymmetric coverage penalty metric, the work establishes a standardized bridge connecting ML uncertainty quantification methods to the precision requirements of physical measurements.

## Method

### Overall Architecture

The data generation pipeline proceeds as follows: Pythia 8.2 Monte Carlo event generation → Delphes 3.5 fast detector simulation → 28 tabular features (14 primary features: energy and 3-momenta of four particles; 14 derived features: invariant mass $m_{inv}$, transverse mass $m_T$, missing transverse energy $E_T^{miss}$, etc.). The signal process is $H\to\tau\tau$ decay (~52M events). Background processes include $Z\to\tau\tau$, $t\bar{t}$, and diboson (~208M events combined, weighted by physical cross-sections). Participants are required to output 68.27% confidence intervals $[\mu_{16}, \mu_{84}]$ for the signal strength $\mu$.

### Key Designs

1. **Systematics Parameterization**:

    - **Function**: Encodes six "known unknowns" into the data generation process so that they can be explicitly modeled by submitted methods.
    - **Mechanism**: Three detector calibration biases — $\alpha_{tes}$ (tau energy scale), $\alpha_{jes}$ (jet energy scale), $\alpha_{soft\_met}$ (soft missing energy resolution) — propagate through cascaded scaling/smearing to affect event-level feature values; three background normalization biases — $\alpha_{t\bar{t}}$, $\alpha_{diboson}$, $\alpha_{bkg}$ — adjust the event weights of respective background processes. The systematics scripts support resampling the dataset for any parameter combination.
    - **Design Motivation**: In real ATLAS/CMS analyses, systematic uncertainties are the dominant contributor to confidence interval width (often exceeding 50%). ML methods must explicitly handle these biases to be credibly deployed in physics analyses.

2. **Asymmetric Coverage Penalty Metric**:

    - **Function**: Designs a unified score that jointly evaluates interval width $w$ and coverage rate $c$.
    - **Mechanism**: The penalty function is $f(c) = 1 + \max(0, (p-c)/p)^4 + \max(0, (c-p)/p)^3$, where $p=0.6827$ is the target coverage. The final score is $S = -\ln((w + 10^{-2}) \cdot f(c))$. Under-coverage is penalized more severely (fourth power) than over-coverage (third power), forcing methods to err on the side of conservatism.
    - **Design Motivation**: In high-energy physics, spurious precision (claiming narrow intervals with insufficient coverage) is far more dangerous than excessive conservatism — the former may lead to false discovery claims, while the latter merely reduces discovery efficiency.

3. **Winning Solution I: HEPHY Profile-Free Alternative Analysis**:

    - **Function**: Avoids the explicit optimization over nuisance parameters required by traditional profile likelihood methods.
    - **Mechanism**: Defines six disjoint event-selection regions (two signal-enriched regions + four background control regions). Within each region, exponential parameterization (rather than linear) captures the nonlinear dependence of event yields on systematic biases. A joint likelihood then simultaneously constrains $\mu$ and the six bias parameters.
    - **Design Motivation**: Traditional binned analyses discretize continuous distributions and lose information; exponential parameterization better captures the nonlinear relationship between biases and yields.

4. **Winning Solution II: Ibrahim Contrastive Normalizing Flow (CNF)**:

    - **Function**: Learns a neural network approximation to the likelihood ratio, bypassing explicit density estimation.
    - **Mechanism**: Normalizing flow models are trained to contrast event distributions under different values of $\mu$, directly outputting likelihood ratio statistics with only ~10 GPU-hours of computation.
    - **Design Motivation**: Explicit density estimation is unreliable in high-dimensional spaces; likelihood ratio methods need only learn the **difference** between distributions, reducing modeling complexity.

### Loss & Training

- **HEPHY**: Profile-free negative log-likelihood with multi-region joint fitting; background control regions automatically profile nuisance parameters.
- **Ibrahim CNF**: Contrastive loss with hyperparameter $c \in \{0.5, 2.0\}$ controlling the trade-off between coverage and interval width; ensemble averaging improves robustness.
- **Competition baseline**: XGBoost classifier + simple binned template fit, serving as the reference baseline.

## Key Experimental Results

### Competition Leaderboard (Main Results)

| Method | Overall Score $S$ | Interval Width | Coverage | Category |
|--------|-------------------|----------------|----------|----------|
| HEPHY (profile-free likelihood) | **-0.582** | Narrow | ≈68.27% | Parameterized alternative analysis |
| Ibrahim (CNF) | -0.576 | Narrow | ≈68.27% | Neural likelihood ratio |
| Hzume (decision tree ensemble) | -2.16 | Moderate | ≈68% | Boosted decision trees |
| Baseline XGBoost | Lower | Wider | Below target | Classification + template fit |

### Ablation Study

| Analysis Method | Improvement over Binned | Bias Constraint Strength | Compute Cost |
|-----------------|------------------------|--------------------------|--------------|
| Traditional binned template fit | Baseline | Weak (biases poorly constrained) | Low |
| Profile-free surrogate (HEPHY) | **~20% narrower intervals** | $\nu_{t\bar{t}}$, $\nu_{jes}$ impact reduced ~65% | Medium |
| Contrastive normalizing flow (Ibrahim) | ~18% narrower intervals | Strong bias constraints | 10 GPU-hours |
| Purely classification-trained (no bias awareness) | Unstable interval width | No constraint capability | Low |

### Key Findings

- **Profile-free > Profile**: Continuous parameterization preserves full information gradients in feature–bias space, outperforming traditional discretized binned templates by ~20%.
- **Bias constraint capability**: Top solutions reduce the contribution of $\nu_{tes}$ and $\nu_{jes}$ to interval width by ~65%, demonstrating significant potential for ML methods to simultaneously constrain parameters of interest and nuisance parameters.
- **Method diversity not saturated**: HEPHY and Ibrahim achieve nearly identical scores yet produce uncorrelated predictions (ensemble combination provides no mutual benefit), suggesting the optimal frontier remains far from fully explored.
- **Data scale effect**: Scaling from 10M to 280M events yields measurable improvements in interval width, confirming that large-scale datasets are essential for advancing uncertainty quantification methods.

## Highlights & Insights

- **Dataset as a function**: The systematics scripts make the dataset a function evaluable at any $(\mu, \vec{\alpha})$, rather than a static table, greatly expanding the experimental design space — researchers can probe extreme bias scenarios.
- **Physics–ML metric alignment**: The asymmetric coverage penalty precisely reflects the high-energy physics statistical philosophy of "err on the side of conservatism," avoiding the common ML pitfall of optimizing against the wrong objective.
- **Implications of complementary solutions**: Two uncorrelated top solutions suggest that further interval narrowing is achievable through method ensembling, and that the solution space for this problem remains broad.
- **Transferable evaluation paradigm**: The joint framework of asymmetric coverage + interval width can be directly extended to other scientific domains requiring uncertainty quantification, such as medical imaging and climate prediction.

## Limitations & Future Work

- **"Known unknowns" assumption**: All systematic biases are perfectly parameterized, whereas real physics analyses involve "unknown unknowns" (e.g., unmodeled detector effects, theoretical uncertainties) not covered by the current benchmark.
- **Simulation fidelity**: Pythia 8.2 + Delphes 3.5 are fast simulation tools, far simpler than full GEANT4 ATLAS/CMS simulations; the generated event feature distributions differ from real data.
- **Limited feature dimensionality**: Only 28 tabular features are provided, whereas real physics analyses typically involve hundreds of features or even raw detector-level inputs.
- **Data volume vs. real scenarios**: 280M events correspond roughly to two weeks of LHC running; actual long-run analyses use data accumulated over multiple years.
- **Single signal process**: Only the $H\to\tau\tau$ channel is considered, without the complexity of multi-channel combined analyses.

## Related Work & Insights

- **vs. Original HiggsML Competition (2014)**: Advances from event classification (AMS metric) to measurement + uncertainty quantification (confidence interval metric), more closely reflecting the practical demands of real physics analyses.
- **vs. INFERNO/neos**: Those works explore differentiable analysis pipelines; this paper instead provides a standardized evaluation platform and dataset.
- **vs. Conformal Prediction**: Competition results demonstrate that physics-inspired parameterization methods (profile-free analysis) outperform general distribution-free methods on this task.
- **Insights**: The design paradigm of datasets with parameterized biases has broad cross-domain applicability — any scientific field requiring inference under systematic uncertainties can benefit from this framework.

## Rating

- Novelty: ⭐⭐⭐⭐ First standardized ML uncertainty quantification benchmark with parameterized systematic biases
- Experimental Thoroughness: ⭐⭐⭐⭐ 280M-event scale, multi-method comparison, detailed competition analysis and method decomposition
- Writing Quality: ⭐⭐⭐⭐ Clear connection between physics motivation and ML methodology, accessible to readers from both communities
- Value: ⭐⭐⭐⭐⭐ A standardized benchmark with significant impact for the physics-ML community, filling an important gap

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Platonic Universe: Do Foundation Models See the Same Sky?](the_platonic_universe_do_foundation_models_see_the_same_sky.md)
- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)
- [\[NeurIPS 2025\] Transfer Learning Beyond the Standard Model](transfer_learning_beyond_the_standard_model.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)
- [\[NeurIPS 2025\] Neural Deprojection of Galaxy Stellar Mass Profiles](neural_deprojection_of_galaxy_stellar_mass_profiles.md)

</div>

<!-- RELATED:END -->
