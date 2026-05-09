---
title: >-
  [Paper Note] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI
description: >-
  [ICLR 2026][Robotics][embodied AI] This work proposes the D2E framework, demonstrating that desktop gaming interaction data can serve as an effective pretraining substrate for embodied AI. Through the OWA toolkit, 335 hours of human demonstrations are collected; a Generalist-IDM pseudo-annotates 1,000+ hours of YouTube gameplay videos; and VAPT transfer training yields a 1B-parameter model that achieves 96.6% on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models 7× larger.
tags:
  - ICLR 2026
  - Robotics
  - embodied AI
  - desktop pretraining
  - inverse dynamics model
  - vision-action pretraining
  - robotics transfer
date: 2026-05-08
content_hash: a6e785253839bb6c
---

# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI

**Conference**: ICLR 2026
**arXiv**: [2510.05684](https://arxiv.org/abs/2510.05684)
**Code**: [Project Page](https://worv-ai.github.io/d2e/)
**Area**: Robotics
**Keywords**: embodied AI, desktop pretraining, inverse dynamics model, vision-action pretraining, robotics transfer

## TL;DR
This work proposes the D2E framework, demonstrating that desktop gaming interaction data can serve as an effective pretraining substrate for embodied AI. Through the OWA toolkit, 335 hours of human demonstrations are collected; a Generalist-IDM pseudo-annotates 1,000+ hours of YouTube gameplay videos; and VAPT transfer training yields a 1B-parameter model that achieves 96.6% on LIBERO manipulation and 83.3% on CANVAS navigation, matching or surpassing models 7× larger.

## Background & Motivation
**State of the Field**: LLMs have achieved cross-task generalization by leveraging internet-scale text data, yet collecting physical trajectory data for embodied AI is prohibitively expensive—requiring specialized hardware, human operation, and complex annotation—leaving data scale far insufficient to drive comparable scaling.

**Limitations of Prior Work**: Existing robot datasets (e.g., DROID) are small-scale, domain-specific, and format-incompatible. VPT is confined to the single domain of Minecraft, and SIMA spans multiple games but relies on proprietary data.

**Root Cause**: Embodied AI demands large-scale action-annotated data, yet physical data collection is not scalable; desktop interactions (keyboard and mouse) are abundant and standardized, but their transferability to physical robots remains an open question.

**Paper Goals**: To establish a complete pipeline spanning desktop data collection, pseudo-annotation, and embodied task transfer validation.

**Starting Point**: Gaming interactions exhibit complex sensorimotor patterns—navigation, manipulation, and planning—closely analogous to embodied AI challenges, and can be acquired at scale via YouTube.

**Core Idea**: Desktop data constitutes a low-cost pretraining source for embodied AI; OWA collection + Generalist-IDM pseudo-annotation + VAPT transfer forms a complete pipeline.

## Method

### Overall Architecture
A three-component pipeline: (1) OWA Toolkit for data collection and formatting → (2) Generalist-IDM for pseudo-annotating YouTube videos → (3) VAPT pretraining on desktop data followed by transfer to robot tasks.

### Key Designs
1. **OWA Toolkit**:

    - **ocap Recorder**: Synchronously records screen (60 Hz), keyboard, and mouse events via Windows API + GStreamer with precise temporal alignment.
    - **OWAMcap Format**: Extends the MCAP standard with H.265 encoding (217× compression vs. raw) and MediaRef external references for efficient random access.
    - **Data Pipeline Optimization**: FSLDataset (fixed-sequence-length packing) + adaptive batch decoding achieve a throughput of 119 img/s (10.2× improvement), with an average disk read of only 18.73 KB/img (41× lower than TorchCodec).
    - 14 annotators collected 335 hours of data across 31 games in one month (compared to DROID's 50 collectors × 13 institutions × 12 months).

2. **Generalist-IDM**:

    - **Timestamped Event Tokenization**: Each event is serialized as `<EVENT_START>{TYPE}{TIMESTAMP}{DETAIL}<EVENT_END>`, without reliance on fixed tick intervals.
    - **NEP-τ (Next-Event Prediction with Temporal Offset)**: The observation window is shifted forward by $\tau$ steps to provide future context:
    $$\mathcal{L}_{\text{NEP-}\tau} = -\mathbb{E}\left[\sum_t \log P_\theta(a_t | o_{1:\min(t+\tau,T)}, a_{1:t-1})\right]$$
      $\tau = 100$ ms is found to be optimal; at $\tau = 0$, Pearson correlation is nearly zero.
    - Built on the InternVL3-1B architecture, requiring only ~192 H100 hours (~$800) to train.
    - Zero-shot generalization to unseen games, achieving 63% keyboard accuracy on Battlefield 6 (matching specialist IDMs).

3. **VAPT (Vision-Action PreTraining)**:

    - Pretrains InternVL3-1B on 1,300+ hours of desktop data (259 hours human-annotated + 1,000+ hours pseudo-annotated).
    - Hypothesized transfer mechanisms: (i) action modality alignment, (ii) goal-directed sequential decision-making, and (iii) high diversity across 20+ games.
    - Training loss curves show that VAPT-initialized models converge immediately, whereas baselines exhibit an initial plateau.

### Loss & Training
- Generalist-IDM: NEP-τ objective (autoregressive next-event prediction).
- VAPT: Standard vision-action pretraining objective with downstream task fine-tuning.

## Key Experimental Results

### Main Results
LIBERO manipulation benchmark (success rate %):

| Method | Params | Spatial | Object | Goal | 10 (long) | Total |
|--------|--------|---------|--------|------|-----------|-------|
| OpenVLA | 7B | 84.7 | 88.4 | 79.2 | 53.7 | 76.5 |
| π₀ | 3.3B | 90.0 | 86.0 | 95.0 | 73.0 | 86.0 |
| SmolVLA | 2.25B | 93.0 | 94.0 | 91.0 | 77.0 | 88.7 |
| **VAPT w/o pseudo** | **1B** | 95.8 | 98.4 | **98.6** | **93.6** | **96.6** |

CANVAS navigation benchmark (Baseline 75.3% → VAPT w/ pseudo **83.3%**).

### Ablation Study
Generalist-IDM vs. Specialist-IDM (6 in-domain games):

| Game | Model | Pearson-X | Keyboard Accuracy |
|------|-------|-----------|-------------------|
| Brotato | Specialist | 65.92 | 28.80 |
| Brotato | **Generalist** | **73.65** | **86.36** |
| Apex Legends | Specialist | 65.16 | 67.47 |
| Apex Legends | **Generalist** | **83.90** | **76.55** |

The Generalist-IDM outperforms the Specialist-IDM on all games, with keyboard accuracy improvements of up to 57.6%.

### Key Findings
- The 1B-parameter VAPT model surpasses 3.3B π₀ and 7B OpenVLA on LIBERO, with a particularly pronounced advantage on long-horizon tasks (93.6% vs. 73.0%).
- Pseudo-annotated data benefits **navigation** (+8%) but is detrimental to **manipulation**, which requires precise human supervision.
- On a real robot (SO101 pick-and-place), success rate improves from 70% to 80%, validating physical transfer.
- OWA data collection is far more efficient than robot data collection: 14 annotators × 1 month vs. DROID's 50 collectors × 13 institutions × 12 months.

## Highlights & Insights
- **Paradigm Innovation**: The first work to systematically demonstrate the feasibility of desktop gaming → physical robot transfer.
- **Complete Pipeline**: An end-to-end contribution spanning tooling, data, modeling, transfer, and validation.
- **Engineering Value of OWA Toolkit**: 152× compression, 41× I/O optimization, and a standardized format represent practical contributions to the community.
- **NEP-τ Design**: Timestamp-driven event prediction is more efficient than tick-based approaches, skipping idle intervals.
- **Zero-Shot Generalization of Generalist-IDM**: Outperforming specialist models on unseen games makes internet-scale pseudo-annotation feasible.

## Limitations & Future Work
- The transfer mechanism from desktop to robot remains at the hypothesis level, lacking rigorous theoretical analysis.
- Only the InternVL3-1B backbone is evaluated.
- Real-robot experiments are limited to a single pick-and-place task.
- The OWA Toolkit currently supports Windows only.
- Gaming data may introduce visual biases inconsistent with the real world.
- Absolute success rates on Meta-World remain low (only 20–24% on Very Hard).

## Related Work & Insights
- VPT is a pioneering work but limited to the single domain of Minecraft; SIMA spans multiple games but is proprietary; D2E is the first open-source, multi-game framework with validated transfer.
- Complementary to RT-X/Open X-Embodiment: D2E provides a low-cost pretraining data source.
- The latent action pretraining approach of LAPA is conceptually similar, but D2E operates directly on explicit keyboard-and-mouse actions.
- Key insight: Internet video + Generalist-IDM pseudo-annotation is a viable path for scaling embodied AI data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The desktop-to-embodied transfer paradigm is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ LIBERO/Meta-World/CANVAS + real robot validation, though real-robot experiments are relatively simple.
- Writing Quality: ⭐⭐⭐⭐ Systematically organized, though the broad scope requires consulting the appendix for certain details.
- Value: ⭐⭐⭐⭐⭐ Open-sourced tooling, data, and models open a new direction for scaling embodied AI data.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Grounding Generative Planners in Verifiable Logic: A Hybrid Architecture for Trustworthy Embodied AI](grounding_generative_planners_in_verifiable_logic_a_hybrid_architecture_for_trus.md)
- [\[ICLR 2026\] TwinVLA: Data-Efficient Bimanual Manipulation with Twin Single-Arm Vision-Language-Action Models](twinvla_data-efficient_bimanual_manipulation_with_twin_single-arm_vision-languag.md)
- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[ICLR 2026\] Capability-Based Scaling Trends for LLM-Based Red-Teaming](capability-based_scaling_trends_for_llm-based_red-teaming.md)
- [\[CVPR 2026\] PULSE: Privileged Knowledge Transfer from Rich to Deployable Sensors for Embodied Multi-Sensory Learning](../../CVPR2026/robotics/pulse_privileged_knowledge_transfer_from_rich_to_deployable_sensors_for_embodied.md)

<!-- RELATED:END -->
