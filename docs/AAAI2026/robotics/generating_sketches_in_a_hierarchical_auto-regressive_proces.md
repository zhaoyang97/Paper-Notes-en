---
title: >-
  [Paper Note] Sketch-HARP: Hierarchical Autoregressive Sketch Generation for Flexible Stroke-Level Drawing Manipulation
description: >-
  [AAAI 2026][Robotics][Sketch Generation] This paper proposes Sketch-HARP, a hierarchical autoregressive sketch generation framework that achieves, for the first time…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Sketch Generation"
  - "Hierarchical Autoregressive"
  - "Stroke-Level Manipulation"
  - "Sketch-HARP"
  - "Sketch Editing"
date: 2026-05-08
content_hash: 82057af744f42fd1
---

# Sketch-HARP: Hierarchical Autoregressive Sketch Generation for Flexible Stroke-Level Drawing Manipulation

**Conference**: AAAI 2026
**arXiv**: [2511.07889](https://arxiv.org/abs/2511.07889)  
**Code**: [https://github.com/SCZang/Sketch-HARP](https://github.com/SCZang/Sketch-HARP)  
**Area**: Sketch Generation / Image Generation
**Keywords**: Sketch Generation, Hierarchical Autoregressive, Stroke-Level Manipulation, Sketch-HARP, Sketch Editing

## TL;DR

This paper proposes Sketch-HARP, a hierarchical autoregressive sketch generation framework that achieves, for the first time, flexible stroke-level manipulation during the drawing process through a three-stage hierarchical pipeline (predicting stroke embeddings → determining canvas positions → generating drawing action sequences). The method significantly outperforms SketchEdit on tasks including stroke replacement, erasure, and extension.

## Background & Motivation

Controllable sketch manipulation is an important generative task. Existing methods (e.g., SketchEdit) require all edited stroke embeddings to be collected and fed into the generator simultaneously before generation begins, **prohibiting further manipulation during the generation process**. This limits precise control over specific local content such as individual stroke shapes.

The core challenges are:

1. Exposing editable intermediate representations that users can modify at any point during generation
2. Maintaining global consistency in autoregressive generation — the shape and position of the current stroke must coordinate with previously drawn strokes
3. Simultaneously controlling stroke features (what to draw) and position (where to draw), which are tightly coupled
4. Existing instance-level manipulation methods (controllable synthesis, sketch inpainting, sketch analogy) lack the granularity for reliable single-stroke control
5. Diffusion-based methods, while high quality, are difficult to interface at the stroke level
6. The model must emulate the human drawing process — deciding what to draw, then where, then executing the stroke

## Method

### Overall Architecture

**Encoding stage**: Input sketches are stroke-separated → a Stroke Encoder (BiLSTM) extracts stroke embeddings $e_k \in \mathbb{R}^{128}$ and a Position Encoder (FC layers) extracts position embeddings $p_k \in \mathbb{R}^{128}$ → a Relationship Encoder (gMLP) learns inter-stroke relationships $r_k$ → embeddings are fused as $\tilde{e}_k = e_k + r_k$ → a Sketch Encoder (LSTM) produces the sketch code $y \in \mathbb{R}^{128}$.

**Generation stage (three-stage hierarchical)**: Sketch code $y$ → (I) Stroke Decoder autoregressively predicts stroke embeddings $\hat{e}_k$ and stop tokens $\hat{\eta}_k$ → (II) Position Decoder predicts 2D starting coordinates via a bivariate Gaussian distribution → (III) Sequence Decoder translates embeddings into drawing action sequences using a GMM. All three decoders receive preceding outputs as conditional input in an autoregressive manner.

### Key Designs

1. **Relationship Embedding**: A gMLP (2 layers, $d_\text{model}=128$, $d_\text{ffn}=512$) serves as the relationship encoder, integrating three types of inter-stroke relationships — spatial (relative positions, e.g., nose above mouth), contextual (drawing-order proximity, e.g., two eyes of a pig drawn consecutively), and semantic (component-level dependencies, e.g., a pig requiring exactly two ears). The relationship vector $r_k$ is added to the stroke embedding $e_k$, endowing each stroke with a global view.

2. **Hierarchical Autoregressive Generation**: The framework emulates the human drawing process — (I) *what to draw* (predicting stroke embeddings): an LSTM stroke decoder (hidden=1024) takes the sketch code $y$ and the previous step's $\tilde{e}_{k-1} + p_{k-1}$ to output $\hat{e}_k$ and stop token $\hat{\eta}_k$; (II) *where to place the stroke* (determining position): an LSTM position decoder models a bivariate Gaussian distribution $\mathcal{N}(\mu_{px}, \mu_{py}, \sigma_{px}, \sigma_{py}, \rho_p)$ and samples the starting coordinate; (III) *executing the stroke* (generating actions): an LSTM sequence decoder models offsets $(\Delta x, \Delta y)$ with $M=20$ bivariate Gaussian mixtures and pen states with a categorical distribution.

3. **Flexible Manipulation Interface**: The exposed intermediate stroke embeddings $\{\hat{e}_k\}$ can be edited at any point during generation, supporting stroke replacement (substituting $\hat{e}_1$ with another stroke embedding), stroke erasure (skipping position determination and action generation for a given stroke), stroke extension (injecting an external embedding $\varepsilon$ to add a new stroke with automatically generated position and actions), and combined operations (erasure followed by extension equals replacement). Error accumulation effects are mild due to the short sequence length at each level (average stroke count ~7, average actions per stroke ~10).

### Loss & Training

Five loss terms are combined with weights: $\mathcal{L} = \mathcal{L}_\text{seq} + \mathcal{L}_\text{pos} + \mathcal{L}_\text{stp} + 5 \cdot \mathcal{L}_\text{sok} + 0.5 \cdot \mathcal{L}_\text{img}$.

| Loss | Meaning | Weight | Form |
|------|---------|--------|------|
| $\mathcal{L}_\text{seq}$ | Drawing action reconstruction | 1.0 | GMM negative log-likelihood + pen state cross-entropy |
| $\mathcal{L}_\text{pos}$ | Starting position reconstruction | 1.0 | Bivariate Gaussian negative log-likelihood |
| $\mathcal{L}_\text{stp}$ | Stop token prediction | 1.0 | Categorical cross-entropy |
| $\mathcal{L}_\text{sok}$ | Stroke embedding regularization | **5.0** | L2 distance; highest weight to ensure embedding quality |
| $\mathcal{L}_\text{img}$ | Canvas image reconstruction | 0.5 | CNN decoder reconstructing $128\times128$ rasterized image |

## Key Experimental Results

### Main Results: Sketch Generation and Retrieval Quality

| Dataset | Metric | Sketch-HARP | DC-gra2seq | SketchEdit | SketchKnitter | SP-gra2seq |
|---------|--------|-------------|------------|------------|---------------|------------|
| DS1 (17 classes) | FID↓ | **9.96** | 12.83 | 25.40 | 11.32 | 14.64 |
| DS1 | LPIPS↓ | **0.28** | 0.30 | 0.37 | 0.31 | 0.33 |
| DS1 | Rec@1↑ | **89.90%** | 85.45% | 70.91% | 78.45% | 80.18% |
| DS2 (5 classes) | FID↓ | **6.97** | 11.01 | 28.64 | 8.10 | 13.42 |
| DS2 | Ret@1↑ | **96.00%** | 95.27% | 82.00% | 95.07% | 92.13% |

### Ablation Study & Human Evaluation

| Configuration / Evaluation Type | FID↓ / Human Score↑ | Rec@1↑ | Key Note |
|---------------------------------|----------------------|--------|----------|
| Full model | **9.96** | **89.90%** | All three stages active |
| w/o relationship embedding | 25.45 | 74.55% | Most critical component (FID +155%) |
| w/o $\mathcal{L}_\text{img}$ | 10.30 | 70.18% | Image regularization strongly affects retrieval |
| w/o $\mathcal{L}_\text{sok}$ | 52.31 | 65.91% | Necessity of stroke embedding regularization at $w=5$ |
| w/o position encoder | 14.51 | 82.91% | Position information is important |
| Stroke replacement (human score) | **2.09±1.06** | — | vs. SketchEdit 1.45±1.27 (+44%) |
| Stroke erasure (human score) | **2.30±0.96** | — | vs. SketchEdit 1.04±1.14 (+121%) |

### Key Findings

- Human evaluation achieves extremely high statistical significance: paired t-test $p = 6.04 \times 10^{-13}$, demonstrating statistically significant superiority over all baselines
- t-SNE visualization shows that relationship embeddings cause strokes to cluster automatically by semantic function (e.g., left wheel / right wheel / window position)
- Without visual features (pure sequence modeling) the method still surpasses DC-gra2seq, which uses visual inputs, validating the effectiveness of the hierarchical design
- The weight of $\mathcal{L}_\text{sok}$ at 5 is far larger than other terms, reflecting the cascading impact of stroke embedding accuracy on downstream position and sequence generation
- The stop token $\hat{\eta}_k$ enables the model to adaptively determine the number of strokes without hard-coded upper limits

## Highlights & Insights

- The generation process precisely emulates the human drawing cognitive pipeline: deciding what to draw → where to place the stroke → executing it, with the three stages corresponding to distinct cognitive steps
- Exposing intermediate stroke embeddings as an editable interface enables genuine manipulation *during* generation rather than only *prior* to it
- Cross-category stroke replacement still produces coherent sketches (e.g., angel wings replaced with different shapes but size-matched)
- A special case of stroke extension — continuing from an incomplete sketch — can generate creative cross-category sketches (e.g., a pig-head clock)

## Limitations & Future Work

- Processes only sequential-format sketches (QuickDraw dataset); rasterized or natural photo sketches are not supported
- Autoregressive generation still carries the risk of error accumulation (though practical impact is limited)
- Integration with stronger generators such as diffusion models has not been explored
- Stroke count and action length are constrained by fixed upper bounds ($N_\text{max}^\text{num}=25$, $N_\text{max}^\text{len}=32$)
- The application domain is relatively niche and the industrial deployment pathway is unclear

## Related Work & Insights

- vs. SketchEdit: manipulation is possible *during* generation vs. only *before* generation; human evaluation scores are substantially higher
- vs. DC-gra2seq: superior FID is achieved without visual features, validating the sufficiency of sequence modeling
- vs. SketchKnitter (diffusion-based): finer stroke-level control is provided compared to holistic generation
- The hierarchical autoregressive paradigm is transferable to structured generation tasks such as vector graphics generation, CAD design, and architectural sketching

## Rating

⭐⭐⭐⭐ (4/5)
The method is elegantly designed with a natural and well-motivated hierarchical process, and the human evaluation results demonstrate strong statistical significance. The application domain is relatively niche, but the methodology offers broad inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] H-GAR: A Hierarchical Interaction Framework via Goal-Driven Observation-Action Refinement for Robotic Manipulation](h-gar_a_hierarchical_interaction_framework_via_goal-driven_observation-action_re.md)
- [\[AAAI 2026\] Realistic Synthetic Household Data Generation at Scale](realistic_synthetic_household_data_generation_at_scale.md)
- [\[NeurIPS 2025\] HiMaCon: Discovering Hierarchical Manipulation Concepts from Unlabeled Multi-Modal Data](../../NeurIPS2025/robotics/himacon_discovering_hierarchical_manipulation_concepts_from_unlabeled_multi-moda.md)
- [\[AAAI 2026\] LaF-GRPO: In-Situ Navigation Instruction Generation for the Visually Impaired via GRPO with LLM-as-Follower Reward](laf-grpo_in-situ_navigation_instruction_generation_for_the_visually_impaired_via.md)
- [\[NeurIPS 2025\] DexFlyWheel: A Scalable Self-Improving Data Generation Framework for Dexterous Manipulation](../../NeurIPS2025/robotics/dexflywheel_a_scalable_and_self-improving_data_generation_framework_for_dexterou.md)

</div>

<!-- RELATED:END -->
