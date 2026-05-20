---
title: >-
  [Paper Note] How 'Neural' is a Neural Foundation Model?
description: >-
  [ICML 2026][Self-Supervised Learning][Neural Foundation Model] The authors treat a "state-of-the-art foundation model (FNN) of mouse visual cortex" as a physiological experimental subject, analyzing its encoder…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Neural Foundation Model"
  - "decoding manifold"
  - "encoding manifold"
  - "tubularity metric"
  - "digital twin"
date: 2026-05-08
content_hash: a7e2923fb7e72b47
---

# How 'Neural' is a Neural Foundation Model?

**Conference**: ICML 2026  
**arXiv**: [2601.21508](https://arxiv.org/abs/2601.21508)  
**Code**: None (reuses public FNN + public manifolds pipeline)  
**Area**: Neural Foundation Models / Interpretability / Representation Learning  
**Keywords**: Neural Foundation Model, decoding manifold, encoding manifold, tubularity metric, digital twin

## TL;DR
The authors treat a "state-of-the-art foundation model (FNN) of mouse visual cortex" as a physiological experimental subject, analyzing its encoder, recurrent, and readout modules using the trio of decoding manifold, encoding manifold, and decoding trajectory. They find that FNN's fitting accuracy mainly relies on the readout's homogeneous feature maps, while only the recurrent module is truly "brain-like." Using a newly proposed tubularity metric, they quantitatively show that early encoding layers lack biological temporal structure, and provide clear recommendations for future neural foundation models: "add recurrence early, reduce feature dimensions in readout."

## Background & Motivation

**Background**: In the era of digital twins, neuroscience has seen the emergence of "neural foundation models" capable of directly predicting spike sequences in mouse primary visual cortex (V1) from input videos. FNN achieves state-of-the-art performance (normalized response correlation close to 70%) on large-scale functional connectomics data such as MICrONS, and is often used as a "silicon twin" for interventional brain science experiments.

**Limitations of Prior Work**: Response correlation is a "forward prediction" metric that ignores the "inverse problem"—how many different inputs can produce the same output. With over a million units in FNN and only pairwise RSA-type analyses possible, current alignment evaluations cannot guarantee brain-like behavior on OOD data. In other words, "good fit" does not equal "correct mechanism."

**Key Challenge**: One must treat the model as a black box to compute alignment scores, yet also "look inside" to verify mechanisms. Existing interpretability tools (RSA / CCA / Linear Predictivity / DSA) are pairwise or single-layer, unable to capture population-level temporal dynamics.

**Goal**: (a) Perform physiology-style population analysis on each FNN module without retraining; (b) introduce quantitative metrics to compare "model temporal structure vs. real retina/V1 temporal structure"; (c) propose feasible architectural improvements.

**Key Insight**: Drawing from control theory's "identifiability"—when a perfect forward model is lacking, one must open the box. The authors borrow the neuroscientist's toolkit: decoding manifold (how stimuli cluster in population activity space), encoding manifold (how neurons cluster in stimulus-response space), and decoding trajectory (how population activity evolves over time), applying all three to the same foundation model for the first time.

**Core Idea**: Using the "decoding manifold + encoding manifold + decoding trajectory + tubularity metric" quartet, treat each FNN module as a candidate brain region and examine whether its population-level dynamics match those of real retina/V1.

## Method

### Overall Architecture
The FNN's encoder (10 convolutional layers, including 3D convolutions for 12 time steps), recurrent (attention-based Conv-LSTM), and readout (Gaussian readout + per-mouse linear mapping) modules are each sampled for unit activity. A set of parameterized stimuli (8 directions of drifting square-wave gratings + naturalistic optical flow, totaling 88 sequences) is used to elicit PSTH. For each module: ① PCA on population activity averaged over all time yields the decoding manifold; ② population activity unfolded over time yields decoding trajectories; ③ tensor decomposition (Williams et al., 2018) embeds neurons into 2D by their spatiotemporal response patterns to 88 stimuli, yielding the encoding manifold; ④ the above are quantified into tubularity (tightness + crossings), cross-validated with existing RSA / CCA / LP / DSA.

### Key Designs

1. **Trio of Population-Level Manifold Analyses**:

    - **Function**: Replaces pairwise RSA, separating "how the population encodes stimuli" from "how neurons are driven by stimuli."
    - **Mechanism**: In the decoding manifold, each point is a stimulus trial, coordinates are PCA-reduced population activity; same stimuli should cluster (readable). In the encoding manifold, each point is a unit, coordinates are tensor-decomposed "stimulus-response" features; functionally similar units should be neighbors. Decoding trajectories unfold each stimulus trial over time as a curve, whose activity integral returns to the decoding manifold.
    - **Design Motivation**: Traditional RSA only computes pairwise similarity, missing "population geometry"; manifolds visualize "global topology + local similarity + temporal evolution" at once, matching neuroscientists' core questions of "encoding—decoding—dynamics."

2. **Tubularity Metric (tightness + crossings)**:

    - **Function**: Quantifies differences between "biological vs. model temporal structure."
    - **Mechanism**: For each stimulus class's trajectory bundle, $S_{\text{tight}}$ measures whether trajectories tightly form a "tube" (biological retina $S_{\text{tight}} \approx 1.99$, FNN encoder L8 only $\approx 0.07$ indicates no tube), $S_{\text{cross}}$ measures the number of crossings between different stimulus trajectories (biological cross is significantly higher than FNN, $p < 0.005$). Together, they answer whether the system forms stable yet interacting bundles like neural populations.
    - **Design Motivation**: Existing dynamic similarity metrics like DSA may align "different causes but similar shapes"; the authors find L1 naturally forms loops due to convolutional translation invariance, causing DSA to give false positives. Tubularity separates "shape alignment" from "semantic alignment," exposing DSA's false positives.

3. **Module-by-Module Comparison: Retina vs V1**:

    - **Function**: Anchors each stage with a clear biological counterpart for evaluation.
    - **Mechanism**: Real retina data serves as the "early + strongly clustered" example (encoding manifold highly clustered), V1 as "late + smoothly continuous" (encoding manifold smoothly transitions). Each FNN layer is checked: encoder should resemble retina, recurrent should resemble V1, readout should maintain V1 style. Results: encoder resembles neither retina nor V1 (shows a "non-selective intensity arm" $\gamma$ absent in biology); recurrent finally shows direction selectivity and tubular trajectories, most V1-like; readout collapses into many highly homogeneous discrete clusters (far from V1's continuity); output is a linear combination of readout, appearing smooth but PSTH is mostly transient, still unlike V1.
    - **Design Motivation**: Foundation models are often praised for "good end-to-end fit," but module-by-module comparison with corresponding brain regions clarifies "which layer contributes biological relevance, which just fits individual differences."

### Loss & Training
No new models are trained; all analyses are performed on the public FNN checkpoint from Wang et al. 2025. Only a new tubularity computation pipeline is added, with all parameters being descriptive geometric statistics, requiring no training.

## Key Experimental Results

### Main Results

| Region | Enc L1 | Enc L2 | Enc L4 | Enc L5 | Enc L7 | Enc L8 | Rec | Readout | Output |
|---|---|---|---|---|---|---|---|---|---|
| Mean Alignment with Retina (RSA/CCA/LP/DSA) | 0.26 | 0.26 | 0.30 | 0.33 | 0.28 | 0.28 | **0.40** | 0.34 | 0.34 |
| Mean Alignment with V1 | 0.29 | 0.21 | 0.32 | 0.30 | 0.30 | 0.32 | **0.53** | 0.38 | 0.48 |

| Stage | Decoding Acc | $S_{\text{tight}}$ (higher = more tubular) | $S_{\text{cross}}$ (biological significantly higher) |
|---|---|---|---|
| Retina (biological) | — | 1.99 | $1.8\times 10^{-6}$ |
| V1 (biological) | — | 0.33 | $4.0\times 10^{-6}$ |
| FNN Encoder L8 | 0.74 | 0.07 | $1.3\times 10^{-5}$ |
| FNN Recurrent | **0.89** | 0.12 | $2.7\times 10^{-7}$ |
| FNN Readout | 0.88 | 0.15 | $3.5\times 10^{-6}$ |
| FNN Output | 0.77 | 0.14 | $4.1\times 10^{-5}$ |

### Ablation Study

| Removed Component | Phenomenon |
|---|---|
| "Non-selective intensity arm" $\gamma$ in Encoder L8 | Decoding trajectories immediately become highly stationary, barely changing over time, proving that the previous "pseudo-temporal structure" was entirely due to intensity increase, not true temporal coding |
| Encoding manifold only / Decoding manifold only | Neither single perspective can reveal the contradiction that readout is "highly clustered but output resembles V1"; only the trio together shows that "output fakes continuity by linearly combining readout's rich PSTH" |
| DSA only vs tubularity | DSA falsely rates L1 as "highly aligned" (due to convolutional translation invariance causing natural stimulus loops), tubularity exposes this false alignment |

### Key Findings
- FNN's classification accuracy peaks at the recurrent module (0.89), then declines—contradicting the intuition that "deeper is better," indicating that readout and output mainly perform "per-mouse spike fitting" rather than higher-order encoding.
- The retina's encoding manifold is "clearly clustered," V1 is "smoothly continuous"; FNN's readout does the opposite—many highly homogeneous discrete clusters, marking its biggest mechanistic mismatch with biology; output appears continuous but is achieved by linearly combining many feature maps' transients, not true population dynamics.
- Biological trajectories' $S_{\text{cross}}$ is significantly higher than FNN: even when both form tubes, biological populations show more group-level interactions (possibly from traveling waves, clique interactions), which FNN lacks in dynamical complexity.
- Early encoder lacks any tubular temporal structure, meaning that even with 3D convolutions, FNN's early processing only "extracts intensity features" rather than "forms temporal codes"—a strong hint for future architectural improvements.

## Highlights & Insights
- Transplants the physiologist's "slice experiment" mindset to foundation models: not asking "what is the alignment score," but "which layer does what, and which brain region does it resemble"—this diagnostic interpretability is far more meaningful than a single alignment number.
- Tubularity is a simple yet sharp geometric metric, specifically exposing "shape-aligned but semantically misaligned" false alignments; revealing DSA's blind spot is the paper's most practical methodological contribution.
- Exposes the readout as an "appendage module"—it carries most of the fitting accuracy but uses mechanisms unlike V1, suggesting future neural foundation models should not keep stacking feature maps, but instead inject neural diversity inductive bias into earlier layers.
- The two improvement suggestions—"add recurrence early to mimic amacrine connectivity, reduce feature count in readout"—are directly data-driven and easily testable in future work.

## Limitations & Future Work
- Only one FNN model is analyzed; cross-model consistency is unverified. If other video-based neural foundation models also show "only recurrent is V1-like," the conclusion would be stronger.
- The stimulus set is limited to 88 parameterized sequences to match biological controls, still narrower than the natural videos used in FNN training; OOD behavior cannot be fully inferred.
- Tubularity is a new metric, with no established baseline or robustness tests on synthetic data; like RSA/DSA, it may have its own biases.
- No empirical evidence is provided for "if the architecture is modified as suggested, alignment scores will improve"—the next step of most interest to engineering.

## Related Work & Insights
- **vs RSA / CCA / Linear Predictivity / DSA**: Traditional alignment metrics are pairwise or single-layer summaries, unable to capture population dynamics; this work adds "manifold + tubularity" for a population geometry perspective, and finds DSA can be fooled by convolutional loops.
- **vs Doerig et al. 2023 and other "DNN as brain model" reviews**: Reviews emphasize "good end-to-end fit proves DNN is a brain model," while this work takes a "mechanism reconciliation" stance, cautioning that high predictive accuracy ≠ brain-like internal representations.
- **vs Klindt et al. / Lurz et al. on readout**: Gaussian readout has long been considered "efficient and interpretable," but this work shows its readout representations are far from V1 manifold structure, challenging this popular approach.
- **Insight**: Foundation model interpretability can be more "biological"—using real brain data's population manifolds as ground-truth, reconciling each model layer with a brain region is more robust than LLM-as-judge or custom scoring; this approach could be applied to LLMs (using human fMRI as anchor).

## Rating
- Novelty: ⭐⭐⭐⭐ Combining the trio on foundation models + proposing tubularity is a rare methodological contribution; but each component has precedent.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple layers + metrics + standard alignment comparisons, but only one model covered.
- Writing Quality: ⭐⭐⭐⭐⭐ Each manifold/trajectory plot is presented side-by-side with biological ground-truth, highly readable.
- Value: ⭐⭐⭐⭐ Provides actionable architectural improvement suggestions, directly advancing neural digital twin research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Manifolds and Modules: How Function Develops in a Neural Foundation Model](../../NeurIPS2025/self_supervised/manifolds_and_modules_how_function_develops_in_a_neural_foundation_model.md)
- [\[AAAI 2026\] Spikingformer: A Key Foundation Model for Spiking Neural Networks](../../AAAI2026/self_supervised/spikingformer_a_key_foundation_model_for_spiking_neural_networks.md)
- [\[ICLR 2026\] Maximizing Asynchronicity in Event-based Neural Networks](../../ICLR2026/self_supervised/maximizing_asynchronicity_in_event-based_neural_networks.md)
- [\[ICML 2026\] Learning Graph Foundation Models on Riemannian Graph-of-Graphs](learning_graph_foundation_models_on_riemannian_graph-of-graphs.md)
- [\[CVPR 2026\] SpHOR: A Representation Learning Perspective on Open-set Recognition for Identifying Unknown Classes in Deep Neural Networks](../../CVPR2026/self_supervised/sphor_a_representation_learning_perspective_on_open-set_recognition_for_identify.md)

</div>

<!-- RELATED:END -->
