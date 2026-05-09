---
title: >-
  [Paper Note] Envisioning the Future, One Step at a Time
description: >-
  [CVPR 2026][Video Understanding][Open-set motion prediction] This paper formulates open-set future scene dynamics prediction as stepwise reasoning over sparse point trajectories, enabling rapid generation of thousands of diverse future hypotheses from a single image via an autoregressive diffusion model — orders of magnitude faster than dense prediction models.
tags:
  - CVPR 2026
  - Video Understanding
  - Open-set motion prediction
  - sparse trajectories
  - autoregressive diffusion model
  - future prediction
  - world model
date: 2026-05-08
content_hash: 83679085eab2e636
---

# Envisioning the Future, One Step at a Time

**Conference**: CVPR 2026
**arXiv**: [2604.09527](https://arxiv.org/abs/2604.09527)
**Code**: [http://compvis.github.io/myriad](http://compvis.github.io/myriad)
**Area**: Video Understanding / Motion Prediction
**Keywords**: Open-set motion prediction, sparse trajectories, autoregressive diffusion model, future prediction, world model

## TL;DR

This paper formulates open-set future scene dynamics prediction as stepwise reasoning over sparse point trajectories, enabling rapid generation of thousands of diverse future hypotheses from a single image via an autoregressive diffusion model — orders of magnitude faster than dense prediction models.

## Background & Motivation

**Background**: Most future prediction methods rely on dense video or latent-space prediction, expending substantial model capacity on appearance rather than underlying motion trajectories, making large-scale exploration of future hypotheses computationally prohibitive.

**Limitations of Prior Work**: (1) Dense video generation methods incur a "visual tax" — every pixel must be rendered before motion can be reasoned about; (2) single-step prediction methods fail in long-horizon scenarios involving multiple contacts; (3) physics engine methods cannot generalize to open-set motion.

**Key Challenge**: Real-world dynamics are highly complex and stochastic — a large number of possible futures must be considered, yet dense prediction renders such exploration computationally infeasible.

**Goal**: Achieve open-set, stepwise, and massively parallelizable motion prediction without incurring the visual tax.

**Key Insight**: Analogous to human cognition — we do not "paint" pictures of the future but instead track meaningful changes. Sparsity is leveraged to make future foresight tractable.

**Core Idea**: Motion prediction is modeled as a stepwise autoregressive diffusion process over user-defined sparse point trajectories.

## Method

### Overall Architecture

Given a single reference frame and $K$ visible query points, incremental motion at each timestep is generated autoregressively. Each step is a conditional diffusion model predicting locally predictable short-range transitions. The model factorizes the joint distribution causally across both the temporal and trajectory dimensions:

$$p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0, \mathcal{I}_0) = \prod_t \prod_i p_\theta(x_t^{(i)} | \mathbf{x}_t^{(<i)}, \mathbf{x}_{<t}, \mathcal{I}_0)$$

### Key Designs

1. **Motion Token Design**:

    - Function: Construct informative representations for each (time, trajectory) pair.
    - Mechanism: Three sources of information are fused — (1) appearance features sampled at the original position $x_0^{(i)}$ ("what"); (2) local context features sampled at the current position $x_t^{(i)}$ ("where"); (3) Fourier-encoded current motion $\Delta x_t^{(i)}$. A trajectory identifier $id_{traj}^{(i)} \sim \mathcal{U}(\mathbb{S}^{d-1})$ sampled uniformly from the unit hypersphere is additionally appended.
    - Design Motivation: Random IDs prevent the model from over-relying on fixed indices and allow scaling to arbitrary $K$; dual-position sampling of appearance and context enables each token to simultaneously encode what the target is and where it currently resides.

2. **Fast Reasoning Blocks**:

    - Function: Substantially accelerate sampling speed during autoregressive inference.
    - Mechanism: Parallel Transformer blocks are adopted, merging self-attention, cross-attention, and FFN into a single residual update: $\mathbf{h} \leftarrow \mathbf{h} + SA(\mathbf{h}) + CA(\mathbf{h}, \mathbf{h}_{cross}) + FFN(\mathbf{h})$. Shared pre-normalization and fused projections are applied; image tokens remain frozen (serving solely as keys and values for cross-attention), while motion tokens causally attend to both streams.
    - Design Motivation: Multiple kernel launches in conventional Transformer layers constitute the primary bottleneck for autoregressive inference; the fused design significantly reduces the number of launches.

3. **Flow Matching Posterior Parameterization**:

    - Function: Model the distribution of per-step motion with high fidelity.
    - Mechanism: Conditional flow matching is applied to model the distribution of incremental motion $\Delta x_t^{(i)}$ at each step, naturally accommodating uncertainty in multimodal motion. Independent denoising at each step allows uncertainty to grow naturally over time in long-horizon prediction.
    - Design Motivation: Compared to deterministic regression, flow matching inherently models multimodality and critically avoids the mode-averaging problem.

### Loss & Training

The model is trained on diverse in-the-wild videos using the standard conditional probability flow matching loss from flow matching. KV caching is employed to accelerate autoregressive inference.

## Key Experimental Results

### Main Results

| Method Type | Prediction Accuracy | Sampling Speed | Diversity |
|-------------|-------------------|----------------|-----------|
| Dense video models | High | Extremely slow | Low (cost-limited) |
| Physics engine methods | High (in-domain) | Moderate | Low (domain-limited) |
| Ours | Comparable / superior | Orders of magnitude faster | High (thousands of hypotheses) |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| w/o trajectory ID | Significant performance drop | Essential for multi-trajectory setting |
| w/o Fast Reasoning | Substantial speed degradation | Fused blocks are critical |
| Single-step prediction | Degradation in long-horizon | Stepwise reasoning is necessary |
| Full model | Best | All components synergize |

### Key Findings

- Accuracy matches or exceeds dense models on the OWM benchmark while sampling at speeds orders of magnitude faster.
- Random trajectory IDs are critical for multi-trajectory modeling — fixed IDs cause the model to memorize indices rather than learn dynamics.
- Stepwise reasoning allows uncertainty to grow naturally in long-horizon prediction, consistent with physical intuition.

## Highlights & Insights

- **"Track motion, not paint the world" philosophy**: The visual tax is entirely avoided, concentrating computation on motion dynamics that truly matter.
- **Engineering innovation in Fast Reasoning Blocks**: The combination of fused projections, frozen image tokens, and prefix attention substantially improves throughput.
- **Introduction of the OWM benchmark**: Provides a standardized evaluation framework for open-set motion prediction.

## Limitations & Future Work

- Sparse point trajectories cannot capture continuum motion such as deformation and rotation.
- Autoregressive generation still accumulates errors over very long horizons.
- The gap between motion prediction and scene understanding remains to be bridged.

## Related Work & Insights

- **vs. Video world models**: These methods incur a substantial visual tax to predict every pixel; this work demonstrates that sparse trajectories suffice to capture the essence of motion.
- **vs. Physics engine methods**: Physics engines are restricted to closed-set domains, whereas the proposed approach achieves generalization in open-set settings through data-driven learning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A paradigm shift combining sparse trajectories with stepwise autoregressive diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ OWM benchmark with multi-scenario validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is compellingly articulated with insightful analogies.
- Value: ⭐⭐⭐⭐⭐ Opens an efficient and scalable new paradigm for future prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TCEI: Dual-level Adaptation for Multi-Object Tracking via Test-Time Calibration](tcei_dual_level_adaptation_multi_object_tracking.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[NeurIPS 2025\] Token Bottleneck: One Token to Remember Dynamics](../../NeurIPS2025/video_understanding/token_bottleneck_one_token_to_remember_dynamics.md)
- [\[CVPR 2026\] Dual-level Adaptation for Multi-Object Tracking: Building Test-Time Calibration from Experience and Intuition](dual-level_adaptation_for_multiobject_tracking_building_testtime_calibration_from.md)
- [\[NeurIPS 2025\] PreFM: Online Audio-Visual Event Parsing via Predictive Future Modeling](../../NeurIPS2025/video_understanding/prefm_online_audio-visual_event_parsing_via_predictive_future_modeling.md)

</div>

<!-- RELATED:END -->
