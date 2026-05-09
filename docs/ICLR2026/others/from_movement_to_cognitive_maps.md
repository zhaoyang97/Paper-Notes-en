---
title: >-
  [Paper Note] From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding
description: >-
  [ICLR 2026 Oral][hippocampus] By combining cluster analysis of infant rodent locomotor development with a shallow RNN predictive learning model, this work provides the first computational demonstration that developmental changes in movement statistics (crawling → walking → running → adult) drive the sequential emergence of spatially tuned hippocampal neurons (place cells, head direction cells, and conjunctive coding cells). The model quantitatively reproduces the developmental timeline observed in rat hippocampal recordings and predicts a progressive increase in conjunctive place-HD coding cells during development — a prediction subsequently validated in experimental data.
tags:
  - ICLR 2026 Oral
  - hippocampus
  - spatial coding
  - locomotor development
  - RNN
  - place cells
  - head direction cells
  - cognitive maps
date: 2026-05-08
content_hash: b1defcc1392b0a9c
---

# From Movement to Cognitive Maps: RNNs Reveal How Locomotor Development Shapes Hippocampal Spatial Coding

**Conference**: ICLR 2026 Oral
**OpenReview**: [8bM7MkxJee](https://openreview.net/forum?id=8bM7MkxJee)
**Code**: Available
**Area**: Computational Neuroscience
**Keywords**: hippocampus, spatial coding, locomotor development, RNN, place cells, head direction cells, cognitive maps

## TL;DR
By combining cluster analysis of infant rodent locomotor development with a shallow RNN predictive learning model, this work provides the first computational demonstration that developmental changes in movement statistics (crawling → walking → running → adult) drive the sequential emergence of spatially tuned hippocampal neurons (place cells, head direction cells, and conjunctive coding cells). The model quantitatively reproduces the developmental timeline observed in rat hippocampal recordings and predicts a progressive increase in conjunctive place-HD coding cells during development — a prediction subsequently validated in experimental data.

## Background & Motivation
**State of the Field**: The hippocampus contains spatially tuned neurons including place cells, head direction (HD) cells, border cells, and grid cells. These emerge sequentially during ontogeny along a specific developmental timeline (HD cells earliest at ~P12, place cells at ~P16, grid cells at ~P20), yet the computational mechanisms driving their emergence remain unknown.

**Limitations of Prior Work**: Existing models (e.g., Cueva & Wei 2018, TEM) can produce spatial representations during training, but they use constant locomotor patterns and provide spatial coordinates directly as supervisory signals, without considering the influence of real locomotor developmental statistics. Two competing hypotheses — "intrinsic circuit maturation" vs. "experience-dependent development" — have not been directly tested by any computational model.

**Root Cause**: Locomotor experience is hypothesized to be critical for spatial cognition, yet no computational model has explained why the movement statistics at different developmental stages (speed, acceleration, turning frequency, etc.) cause different types of spatial neurons to emerge at specific time points.

**Paper Goals**: To establish a causal computational link between locomotor developmental statistics and the emergence of hippocampal spatial coding.

**Starting Point**: A data-driven approach is used to extract developmental stage statistics from real infant rodent locomotion, which then drive a predictive learning RNN, to test whether it spontaneously produces a spatial coding developmental timeline matching biological data.

**Core Idea**: Developmental changes in the statistics of embodied sensorimotor experience are sufficient to drive the ontogeny of hippocampal spatial coding.

## Method

### Overall Architecture
1. Analyze real infant rodent open-field behavioral data, extract locomotor statistics, and identify developmental stages via clustering; 2. Use each stage's locomotor statistics to drive an agent in a simulated environment to generate trajectories; 3. Train a shallow RNN on these trajectories with a one-step visual prediction task; 4. After sequential training through each developmental stage, analyze the spatial tuning properties of RNN hidden states and compare against hippocampal recording data.

### Key Designs
1. **Locomotor Developmental Stage Extraction**:
    - Function: Extract locomotor statistical features (speed, acceleration, angular velocity, etc.) from published infant rodent open-field data and apply clustering to identify developmental stages.
    - Mechanism: K-means clustering automatically partitions locomotor data from P12–P60 into three developmental stages — crawling (~P12–P15), walking (~P16–P19), and running (~P20+) — plus an adult stage.
    - Design Motivation: To avoid manually defining developmental stages and allow data-driven clustering to reveal natural transition points in locomotor patterns, ensuring that the movement statistics fed to the RNN faithfully reflect the underlying biology.

2. **Predictive Learning RNN Model**:
    - Function: A shallow RNN receives panoramic visual input $\mathbf{v}_t \in \mathbb{R}^{80}$ and vestibular signals (angular velocity $\omega_t$) at each timestep, and uses its hidden state $\mathbf{h}_t$ to predict the next visual input $\hat{\mathbf{v}}_{t+1}$.
    - Mechanism: $\mathbf{h}_t = f(\mathbf{W}_{vh}\mathbf{v}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{W}_{\omega h}\omega_t + \mathbf{b})$, with loss $\mathcal{L} = \|\hat{\mathbf{v}}_{t+1} - \mathbf{v}_{t+1}\|^2$.
    - Design Motivation: The predictive learning framework is broadly supported (Eichenbaum et al. 2004; Levy 1989), with the hippocampus viewed as a system that compares incoming sensory signals against memory-based predictions. Egocentric visual input is used rather than positional coordinates, thereby avoiding the provision of privileged spatial information.

3. **Progressive Developmental Exposure Training Strategy**:
    - Function: The RNN is sequentially exposed to locomotor trajectories from each developmental stage — first trained to convergence on crawling-mode trajectories, then switched to walking, running, and adult stages in order.
    - Mechanism: Trajectories at each stage are generated by a simulated agent moving in a 0.625×0.625 m environment with locomotor statistics (speed distribution, turning frequency, etc.) characteristic of that stage. The adult stage additionally incorporates grid cell input $g(\mathbf{x}) = \sum_k \cos(\mathbf{k}_i \cdot \mathbf{x})$ with scale parameters $\lambda \in \{0.2, 0.4, 0.6\}$ m.
    - Design Motivation: To simulate the real developmental process of animals — infant rodents exhibit qualitatively distinct locomotor patterns at different ages, and these patterns provide sensory experiences with different statistical properties.

### Spatial Tuning Quantification
Standard spatial information $SI = \sum_i p_i \frac{r_i}{\bar{r}} \log_2 \frac{r_i}{\bar{r}}$ is used to quantify place coding, Rayleigh vector length (RVL) to quantify directional selectivity, and thresholding to identify place cells and HD cells. Control experiments include: reversing the developmental order, controlling inter-frame temporal intervals, and controlling cumulative training volume.

## Key Experimental Results

### Main Results: Developmental Timeline Matching

| Developmental Stage | Corresponding Age | Place Cells | HD Cells | Conjunctive Coding | Match to Biological Data |
|---|---|---|---|---|---|
| Crawling | ~P12–15 | Few/absent | Weak | Absent | ✓ |
| Walking | ~P16–19 | Emerge | Adult-like | Emerge | ✓ |
| Running | ~P20+ | Increase | Stable | Increase | ✓ |
| Adult | >P30 | Mature | Mature | Mature | ✓ |

### Ablation Study

| Configuration | Key Metric | Description |
|---|---|---|
| Normal developmental order | SI and RVL increase across stages | Baseline: matches biological data |
| Reversed developmental order | Abnormal spatial coding emergence pattern | Demonstrates importance of developmental order |
| Accelerated sensory change only | Place-centered representations do not emerge | Changing frame rate alone is not equivalent to locomotor development |
| Controlled cumulative training volume | No effect on emergence timeline | Rules out training volume as confound |
| Controlled inter-frame interval | Core conclusions unchanged | Rules out temporal resolution as confound |

### Key Findings
- The model predicts that directional selectivity emerges primarily through conjunctive place-HD coding rather than through the prior emergence of pure HD cells — a prediction validated in hippocampal recording data.
- Cross-trial spatial coding correlations exceed 0.8, demonstrating that the learned representations are stable and reliable.
- Pure HD cells develop adult-like tuning during the walking stage (~P16), consistent with experimental data showing HD cell maturation in MEC at ~P15.

## Highlights & Insights
- **First computational bridge between locomotor development and spatial coding emergence**: Prior work offered only descriptive observations; this paper provides the first mechanistic explanation.
- **Computational evidence for embodied cognition**: Movement is not merely a motor output but a critical input shaping cognitive representations — with direct implications for embodied AI.
- **Prediction–validation closed loop**: The model generates novel predictions (developmental dynamics of conjunctive coding cells) that are subsequently confirmed in experimental data, representing the gold standard in computational neuroscience.
- **Inspiration for developmental curriculum learning**: Analogous to curriculum learning, the "simple-before-complex" developmental exposure order proves critical for representation learning.

## Limitations & Future Work
- The single-layer RNN cannot spontaneously produce grid cells, which require more complex attractor dynamics.
- The environment is a simple 2D square arena; complex multi-room environments have not been tested.
- Training is conducted on short segments of ~5.9 s; long-timescale spatial memory has not been evaluated.
- Linear decoding may underestimate the complexity of the learned representations.
- No direct manipulation of locomotor development in real animals has been performed for causal validation.

## Related Work & Insights
- **vs. Cueva & Wei 2018**: Both use RNNs to produce spatial representations, but Cueva employs a path integration task (with x-y coordinates provided directly as supervision) and does not consider developmental changes in locomotion.
- **vs. TEM (Whittington et al. 2020)**: TEM uses a multi-component architecture (attractor + path integration + multiple loss functions) to address structured knowledge problems, a concern orthogonal to this paper's focus on developmental mechanisms.
- **vs. Levenstein et al. 2024**: Also uses predictive learning but focuses on adult representations without addressing development.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First computational mechanism linking locomotor development to the emergence of spatial coding; the problem is both novel and important.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Computational model with 5 control experiments and validation against real hippocampal recordings; the only limitation is the absence of direct animal intervention experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ — Cross-disciplinary writing is clear and precise; reviewers awarded a top score of 10 ("should be highlighted as oral").
- Value: ⭐⭐⭐⭐⭐ — Broad implications for computational neuroscience and embodied AI; establishes a new research direction.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Characterizing and Optimizing the Spatial Kernel of Multi Resolution Hash Encodings](characterizing_and_optimizing_the_spatial_kernel_of_multi_resolution_hash_encodi.md)
- [\[AAAI 2026\] How Wide and How Deep? Mitigating Over-Squashing of GNNs via Channel Capacity Constrained Estimation](../../AAAI2026/others/how_wide_and_how_deep_mitigating_over-squashing_of_gnns_via_channel_capacity_con.md)
- [\[AAAI 2026\] How to Marginalize in Causal Structure Learning?](../../AAAI2026/others/how_to_marginalize_in_causal_structure_learning.md)
- [\[NeurIPS 2025\] RNNs Perform Task Computations by Dynamically Warping Neural Representations](../../NeurIPS2025/others/rnns_perform_task_computations_by_dynamically_warping_neural_representations.md)
- [\[NeurIPS 2025\] Sheaf Cohomology of Linear Predictive Coding Networks](../../NeurIPS2025/others/sheaf_cohomology_of_linear_predictive_coding_networks.md)

<!-- RELATED:END -->
