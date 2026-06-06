---
title: >-
  [Paper Note] How 'Neural' is a Neural Foundation Model?
description: >-
  [ICML 2026][Self-Supervised Learning][Neural Foundation Model] The authors treat a "SOTA foundation model for mouse visual cortex (FNN)" as a physiological experimental subject. By analyzing its encoder, recurrent…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Neural Foundation Model"
  - "Decoding Manifold"
  - "Encoding Manifold"
  - "tubularity metric"
  - "Digital Twin"
date: 2026-05-08
content_hash: 898253df3630e0c1
---

# How 'Neural' is a Neural Foundation Model?

**Conference**: ICML 2026  
**arXiv**: [2601.21508](https://arxiv.org/abs/2601.21508)  
**Code**: None (Based on public FNN + reuse of public manifolds pipelines)  
**Area**: Neuroscience Foundation Models / Interpretability / Representation Learning  
**Keywords**: Neural Foundation Model, Decoding Manifold, Encoding Manifold, tubularity metric, Digital Twin

## TL;DR
The authors treat a "SOTA foundation model for mouse visual cortex (FNN)" as a physiological experimental subject. By analyzing its encoder, recurrent, and readout modules using a toolkit consisting of decoding manifolds, encoding manifolds, and decoding trajectories, they find that FNN's fitting accuracy is primarily sustained by a set of homogeneous feature maps in the readout, while only the recurrent module is truly "brain-like." Using a newly proposed "tubularity" metric, they quantitatively show that "early encoding layers lack biological-grade temporal structures," providing explicit recommendations for future neural foundation models to "add recurrence early and reduce feature dimensions in the readout."

## Background & Motivation

**Background**: In the era of digital twins, a series of "neural foundation models" have emerged in neuroscience, capable of directly predicting spike sequences in regions like the mouse primary visual cortex (V1) from input videos. FNN has achieved SOTA performance on the largest-scale functional connectomics data (MICrONS), with normalized response correlations approaching 70%, and is frequently used as a "silicon twin" for interventional brain science experiments.

**Limitations of Prior Work**: Response correlation is a "forward prediction" metric that ignores the "inverse problem"—how many different inputs can correspond to the same output. Furthermore, FNN contains millions of units and can typically only be analyzed via pairwise RSA-like methods. Current alignment evaluations do not guarantee that the model works like the brain on OOD (Out-of-Distribution) data. In other words, "good fitting" does not equal "correct mechanism."

**Key Challenge**: There is a need to treat the model as a black box to calculate alignment scores while simultaneously "investigating the black box" to verify mechanisms. However, existing interpretability tools (RSA / CCA / Linear Predictivity / DSA) are either pairwise or single-layered, failing to capture population-level temporal dynamics.

**Goal**: (a) Perform module-wise physiological-style population analysis without retraining FNN; (b) introduce quantitative metrics to compare "model temporal structure vs. real retinal/V1 temporal structure"; (c) propose feasible architectural improvement suggestions.

**Key Insight**: The authors start from the perspective of "identifiability" in control theory—when a perfect forward model is unavailable, one must open the box. They borrow a toolkit from neuroscientists: decoding manifolds (how stimuli cluster in the population activity space), encoding manifolds (how neurons cluster in the stimulus-response space), and decoding trajectories (how population activity evolves over time), applying all three to a single foundation model for the first time.

**Core Idea**: Use a four-part suite—"decoding manifold + encoding manifold + decoding trajectory + tubularity metric"—to inspect each FNN module as a candidate brain region and check if its population-level dynamics are consistent with the real retina / V1.

## Method

### Overall Architecture
Unit activities were sampled from each of FNN's three modules: the encoder (10 convolutional layers, including 3D convolutions capturing 12 temporal steps), the recurrent module (Conv-LSTM with attention), and the readout (Gaussian readout + one linear map per mouse). PSTHs were stimulated using a set of parameterized stimuli (8 directions of drifting square-wave gratings + naturalistic optical flow, totaling 88 sequences). On each module, the following were performed: ① PCA on time-averaged population activity to obtain the decoding manifold; ② time-step expansion of population activity to obtain decoding trajectories; ③ tensor decomposition (Williams et al., 2018) to embed neurons into 2D based on "spatiotemporal response patterns to 88 stimuli" to obtain the encoding manifold; ④ quantification of the above comparisons using tubularity (tightness + crossings), cross-validated with existing RSA / CCA / LP / DSA.

### Key Designs

1.  **Triple Population-Level Manifold Analysis**:
    - **Function**: Replaces pairwise RSA by separating "how the population encodes stimuli" from "how neurons are driven by stimuli."
    - **Mechanism**: In the decoding manifold, each point represents a stimulus trial, with coordinates being PCA-reduced population activity; the same stimulus should form clusters (readability). In the encoding manifold, each point is a unit, with coordinates formed by "stimulus-response" features from tensor decomposition; functionally similar units should be proximate. Decoding trajectories expand each trial over time into a curve; integrating activity along the trajectory returns to the decoding manifold.
    - **Design Motivation**: Traditional RSA only calculates one-to-one similarity and misses "population geometry." Manifolds visualize "global topology + local similarity + temporal evolution" simultaneously, corresponding exactly to the "encoding-decoding-dynamics" questions of greatest interest to neuroscientists.

2.  **Tubularity Metric (tightness + crossings)**:
    - **Function**: Converts the difference between "biological-grade temporal structure vs. model temporal structure" into comparable numbers.
    - **Mechanism**: For each stimulus class bundle, $S_{\text{tight}}$ measures whether trajectories of the same stimulus cluster tightly into a "tube" (biological retina $S_{\text{tight}} \approx 1.99$, FNN encoder L8 only $\approx 0.07$, indicating a failure to form tubes). $S_{\text{cross}}$ measures the number of crossings between different stimulus trajectories (biology shows significantly more crossings than FNN, $p < 0.005$). Together, they answer whether the activity expands into stable but interacting bundles based on stimuli.
    - **Design Motivation**: Existing dynamic similarity metrics like DSA may judge two trajectories as aligned if they have similar shapes but different causes. The authors found that L1 naturally forms loops due to convolutional translation equivariance, leading to false high DSA scores. Tubularity evaluates "shape pairs" and "semantic pairs" separately, thus exposing DSA's false alarms.

3.  **Module-wise Comparison (Retina vs. V1)**:
    - **Function**: Provides explicit biological counterparts as anchors for evaluation at each stage.
    - **Mechanism**: Real retinal data serves as an example of "early + strong discrete clusters" (highly clustered encoding manifold), while V1 serves as a "late + smooth continuous" example (continuous transition in encoding manifold). FNN layers are then checked: early encoder should resemble the retina, recurrent should resemble V1, and readout should maintain V1-style. Results: the encoder resembles neither (possessing a "non-selective intensity arm" $\gamma$ not found in biology); the recurrent module finally shows directional selectivity and tubular trajectories, most resembling V1; the readout collapses into numerous highly homogeneous discrete clusters (deviating from V1's continuity); the output is a linear combination of readouts, appearing smooth but with transient PSTHs, still unlike V1.
    - **Design Motivation**: Foundation models are often praised for "end-to-end fitting," but module-wise auditing—comparing "what should look like what"—clearly delineates which layers contribute biological relevance versus those merely fitting individual variance.

### Loss & Training
No new models were trained; all analyses were conducted on the FNN checkpoints published by Wang et al. 2025. A new tubularity calculation pipeline was added, consisting of descriptive geometric statistics requiring no training.

## Key Experimental Results

### Main Results

| Region | Enc L1 | Enc L2 | Enc L4 | Enc L5 | Enc L7 | Enc L8 | Rec | Readout | Output |
|---|---|---|---|---|---|---|---|---|---|
| Avg. Alignment (Retina) | 0.26 | 0.26 | 0.30 | 0.33 | 0.28 | 0.28 | **0.40** | 0.34 | 0.34 |
| Avg. Alignment (V1) | 0.29 | 0.21 | 0.32 | 0.30 | 0.30 | 0.32 | **0.53** | 0.38 | 0.48 |

| Stage | Decoding Acc | $S_{\text{tight}}$ (Higher = Tube-like) | $S_{\text{cross}}$ (Significant in Bio) |
|---|---|---|---|
| Retina (Bio) | — | 1.99 | $1.8\times 10^{-6}$ |
| V1 (Bio) | — | 0.33 | $4.0\times 10^{-6}$ |
| FNN Encoder L8 | 0.74 | 0.07 | $1.3\times 10^{-5}$ |
| FNN Recurrent | **0.89** | 0.12 | $2.7\times 10^{-7}$ |
| FNN Readout | 0.88 | 0.15 | $3.5\times 10^{-6}$ |
| FNN Output | 0.77 | 0.14 | $4.1\times 10^{-5}$ |

### Ablation Study

| Removal Item | Phenomenon |
|---|---|
| Intensity arm $\gamma$ in Enc L8 | Decoding trajectories immediately become highly steady-state, proving "pseudo-temporal structure" comes from intensity rise rather than true temporal encoding. |
| Single Manifold Only | Neither perspective alone reveals the contradiction of the readout being "highly clustered while output resembles V1"; the triple-set is required. |
| DSA vs. Tubularity | DSA mislabels L1 as "highly aligned" (due to translation equivariance); tubularity exposes this false alignment. |

### Key Findings
- FNN's classification accuracy peaks at the recurrent module (0.89) then decreases—counterintuitive to the "deeper is better" heuristic, implying readout/output primarily perform personalized spike fitting rather than higher-order encoding.
- The retinal encoding manifold is "distinctly clustered" while V1 is "smoothly continuous." FNN's readout goes the opposite way—large numbers of highly homogeneous discrete clusters—representing its largest mechanistic mismatch with biology.
- Biological trajectories show significantly higher $S_{\text{cross}}$ than FNN: even within tubes, biological populations exhibit more population-level interactions (likely from traveling waves or clique interactions) that FNN lacks.
- The early encoder lacks tubular temporal structure, meaning even with 3D convolutions, early processing merely "extracts intensity features" rather than "forming temporal codes."

## Highlights & Insights
- Migrating the "slice experiment" logic of physiologists to foundation models: instead of asking "what is the alignment score," asking "which layer does what, and which brain region does it resemble"—this diagnostic interpretability is far more meaningful than a single number.
- Tubularity is a simple yet sharp geometric metric designed to detect "correct shape, incorrect semantics" pseudo-alignment; revealing DSA's blind spots is a substantial methodological contribution.
- Exposing the readout as an "appendage" module—it carries most of the fitting accuracy but uses mechanisms unlike V1, suggesting future models should stop stacking feature maps and instead place inductive biases for neural diversity in earlier layers.
- Actionable advice: "add early recurrence to simulate amacrine connectivity" and "reduce feature counts in readout" are direct data-driven observations.

## Limitations & Future Work
- Only a single FNN model was analyzed; cross-model consistency remains unverified.
- Stimulus sets were restricted to 88 sequences for biological comparison, which is narrower than the natural videos used for FNN training; OOD behavior cannot be fully inferred.
- Tubularity is a new metric without established baselines or robustness testing on synthetic data; like RSA/DSA, it may have inherent biases.
- No empirical evidence was provided showing how much alignment scores would increase if the architecture were modified as suggested.

## Related Work & Insights
- **vs RSA / CCA / Linear Predictivity / DSA**: Traditional metrics are pairwise or single-layer summaries. This work adds a population geometry perspective and finds that DSA can be deceived by convolutional recurrent structures.
- **vs Doerig et al. 2023, etc.**: While surveys emphasize that "good end-to-end fitting proves DNNs are brain models," this work provides a "mechanistic reconciliation" reality check, noting that high prediction accuracy $\neq$ brain-like internal representation.
- **vs Klindt et al. / Lurz et al. on readout**: Gaussian readout designs have been considered efficient and interpretable, but this work proves they produce readout representations far from V1 manifold structures, challenging this convention.
- **Insight**: Foundation model interpretability can be more "biologized"—using population manifolds of real brain data as ground-truth to "audit" each model layer. This path could be inversely applied to LLMs using human fMRI as an anchor.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying the triple-set to foundation models + tubulartiy is a rare methodological contribution; individual components exist.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Crosses multiple layers and metrics with standard alignment controls, but limited to one model.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent readability with manifolds/trajectories presented alongside biological ground-truth.
- **Value**: ⭐⭐⭐⭐ Provides actionable architectural improvements for the neural digital twin field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](../../NeurIPS2025/self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[ICML 2026\] InfoAtlas: A Foundation Model for Zero-Shot Statistical Dependence Estimation](infoatlas_a_foundation_model_for_zero-shot_statistical_dependence_estimate.md)
- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)
- [\[ICML 2026\] FLAG: Foundation Model Representation with Latent Diffusion Alignment via Graph for Spatial Gene Expression Prediction](flag_foundation_model_representation_with_latent_diffusion_alignment_via_graph_f.md)

</div>

<!-- RELATED:END -->
