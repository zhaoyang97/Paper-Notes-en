---
title: >-
  [Paper Note] Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT
description: >-
  [AAAI2026][Interpretability][mechanistic interpretability] This work applies mechanistic interpretability to reverse-engineer the internal circuits of a Video Vision Transformer (ViViT), revealing a functional division of labor in which attention heads are responsible for "gathering evidence" and MLP modules for "composing concepts." The analysis demonstrates that the model develops semantic knowledge beyond its training objective even on simple classification tasks.
tags:
  - "AAAI2026"
  - "Interpretability"
  - "mechanistic interpretability"
  - "Video Vision Transformer"
  - "Activation Patching"
  - "Circuit Analysis"
  - "Trustworthy AI"
date: 2026-05-08
content_hash: 8f96fcc296736df3
---

# Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT

**Conference**: AAAI2026  
**arXiv**: [2603.11142](https://arxiv.org/abs/2603.11142)  
**Code**: To be confirmed  
**Area**: Interpretability  
**Keywords**: mechanistic interpretability, Video Vision Transformer, Activation Patching, Circuit Analysis, Trustworthy AI

## TL;DR

This work applies mechanistic interpretability to reverse-engineer the internal circuits of a Video Vision Transformer (ViViT), revealing a functional division of labor in which attention heads are responsible for "gathering evidence" and MLP modules for "composing concepts." The analysis demonstrates that the model develops semantic knowledge beyond its training objective even on simple classification tasks.

## Background & Motivation

Video Vision Transformers (ViViT) have achieved strong performance on video classification benchmarks, yet they share the "black-box" problem common to deep learning models. For video AI systems deployed in high-stakes scenarios such as autonomous driving and medical applications, understanding the internal reasoning process is a critical prerequisite for establishing trust.

Existing interpretability research has focused predominantly on language and image models; the video domain has received comparatively less attention owing to its higher spatiotemporal dimensionality. Mechanistic Interpretability aims to reverse-engineer internal computations into human-understandable algorithms, but its application to video Transformers remains largely unexplored.

The central motivation of this paper is to ask: does a ViViT model trained solely to classify human actions (e.g., "bowling") develop fine-grained semantic understanding of action outcomes (e.g., success or failure) in its internal representations? What are the implications of such "hidden cognition" for AI safety and trustworthy deployment?

## Core Problem

1. When a pretrained ViViT produces identical classification outputs (both labeled "bowling"), does it generate distinct internal representations for a strike versus a gutter ball?
2. If such an internal signal exists, what roles do the respective architectural components (Attention vs. MLP) play?
3. How robust is this internal representational circuit, and can it be disrupted by simple ablation?

## Method

### Experimental Setup

- **Model**: `google/vivit-b-16x2-kinetics400`, a 12-layer ViViT-B using 16×16 spatial and 2-frame temporal tubelet embeddings.
- **Data**: A minimal contrastive pair constructed from the Kinetics-400 "bowling" category — one strike video and one gutter-ball video — both correctly classified as bowling (Label 31) by the model.
- **Fixed random seed**: 42, to ensure reproducibility.

### Observational Analysis

1. **Direct Logit Attribution (DLA)**: Analyzes the contribution of the [CLS] token at each layer to the final classification logit; model confidence increases markedly from Layer 9 onward.
2. **Token-wise Heatmap**: Visualizes the contribution of spatiotemporal tokens to the output class; contributions concentrate in the region of ball–pin interaction.
3. **CLS Token Attention Visualization**: Layer 10 Head 8 functions as a semantic "outcome detector," tracking the ball trajectory and moment of impact in the strike video, and attending to the lane gutter and standing pins in the gutter video.
4. **Linear Probe**: Logistic regression classifiers trained on [CLS] token activations at all 12 layers to distinguish strike from gutter achieve 100% accuracy from Layer 0 onward, suggesting that the probe captures only surface-level differences ("fingerprint scanning") rather than semantic concepts.

### Signal Localization: Delta Analysis

The internal signal is localized by computing the activation difference between the two videos:

$$\Delta = act_{strike} - act_{gutter}$$

The L2 norm of the per-layer delta is used as a measure of "signal strength." Results show that from Layer 5 to Layer 11 the L2 norm grows by more than 300% (approximately 75 → 250+), exhibiting a clear "amplification cascade." Unlike the linear probe, which detects differences from Layer 0, delta analysis reveals that the semantic signal only gradually emerges in the middle-to-deep layers, indicating that the model computes high-level semantic abstractions rather than low-level feature differences.

### Causal Analysis

1. **Component Ablation**: The top 10% of tokens (313 patches) identified by DLA as the highest-contributing are set to zero. Result: the bowling logit for the strike video drops by only 0.34 (16.99→16.66), and for the gutter video by only 0.02 (16.52→16.50), leaving classification essentially unaffected. This demonstrates that the classification circuit is highly distributed and that the "outcome signal" circuit operates independently of the classification circuit.

2. **Activation Patching**: Individual component activations (Attention or MLP) from the strike forward pass are transplanted into the gutter forward pass, and the recovered fraction of the "success vs. failure" signal at Layer 11 is measured. Signal recovery is computed as:

$$\text{Recovery}(\%) = \frac{\|\Delta_{patch}\|}{\|\Delta_{strike}\|} \cdot \text{sign}(\Delta_{patch} \cdot \Delta_{strike}) \times 100$$

## Key Experimental Results

### Activation Patching Results (Layers 4–10)

| Layer | Component | Signal Recovery |
|---|---|---|
| Layer 4 | Attention | 54.41% |
| Layer 4 | MLP | **60.17%** |
| Layer 5 | Attention | 50.22% |
| Layer 5 | MLP | **57.49%** |
| Layer 6 | Attention | 43.62% |
| Layer 6 | MLP | **49.11%** |
| Layer 7 | Attention | 40.38% |
| Layer 7 | MLP | **42.55%** |
| Layer 8 | Attention | 37.72% |
| Layer 8 | MLP | **42.10%** |
| Layer 9 | Attention | 44.43% |
| Layer 9 | MLP | **58.66%** |
| Layer 10 | Attention | 47.61% |
| Layer 10 | MLP | 43.39% |

### Key Findings

- **Attention heads** recover 37–54% of the signal, playing the role of "Evidence Gatherers."
- **MLP modules** recover 42–60% of the signal, playing the role of "Concept Composers" and serving as the primary driver in generating the "success" signal.
- No single component recovers 100% of the signal, confirming that the circuit is **distributed and redundant**.
- Classification is nearly unaffected by ablation (logit change < 0.34), validating the robustness of the circuit.

## Highlights & Insights

1. **First systematic mechanistic interpretability analysis on a video Transformer**, extending MechInterp from language and image models to the video domain.
2. **Reveals a clear functional division of labor**: "Attention Gathers, MLPs Compose" — attention aggregates spatiotemporal evidence while MLPs compose semantic concepts, supporting the hypothesis of functional differentiation within Transformers.
3. **Discovers "hidden cognition"**: a model trained solely to classify "bowling" spontaneously develops internal representations that distinguish action outcomes, carrying important implications for AI safety.
4. **Methodological contribution**: demonstrates a complete analytical pipeline combining delta analysis and activation patching, from signal localization to causal attribution.
5. **The failure-case analysis of linear probes** is instructive — 100% accuracy paradoxically indicates that the probe is capturing surface-level features, underscoring the necessity of causal intervention methods.

## Limitations & Future Work

- **Extremely small sample size**: only a single contrastive video pair (strike vs. gutter) is used; it is unclear whether the identified circuit generalizes to more samples or additional action categories.
- **Single architecture**: validation is limited to ViViT-B; other video Transformers such as TimeSformer have not been tested.
- **Feature specificity cannot be ruled out**: the identified circuit may partly depend on low-level features such as background texture specific to the video pair, rather than purely semantic concepts.
- **No quantitative comparison with standard interpretability methods**: systematic baselines such as Integrated Gradients and CAV are not included.
- Future directions include large-scale validation using Automated Circuit Discovery (ACDC) and cross-architecture generalization experiments.

## Related Work & Insights

| Aspect | Ours | Traditional Interpretability Methods |
|---|---|---|
| Analysis granularity | Component-level causal analysis (Attention vs. MLP) | Input feature attribution (gradient heatmaps) |
| Method type | Causal intervention (activation patching) | Observational (saliency maps, IG) |
| Domain | Video Transformer | Primarily language / image models |
| Discovery capability | Can distinguish functional roles (gather vs. compose) | Can only identify "which inputs matter" |

This work resonates with research in the Eliciting Latent Knowledge (ELK) direction (Burns et al. 2022; Mallen et al. 2023): it provides empirical evidence of hidden knowledge in the video domain, whereas ELK primarily investigates language models. The failure of the linear probe to convey meaningful information despite achieving 100% accuracy from Layer 0 is also consistent with findings by Mallen et al. — simple probes may capture shallow features rather than genuinely hidden knowledge.

The findings carry important implications for **AI safety**: even models trained on simple tasks may develop internal representations that exceed their training objectives, and standard output-level monitoring cannot detect such "hidden cognition," necessitating mechanistic interpretability tools for deeper inspection. The **redundant cascade mechanism** of MLPs implies that naive safety interventions (e.g., removing a single "harmful" component) may be ineffective, requiring more targeted intervention strategies. The **analytical framework is transferable**: the delta analysis + activation patching methodology can be applied to other video understanding tasks and architectures. This work also provides additional supporting evidence in the video domain for findings already established in the Transformer interpretability literature (attention performs information routing; MLPs perform knowledge storage and composition).

## Rating

- Novelty: ⭐⭐⭐⭐ (First application of a complete MechInterp pipeline to a video Transformer; the research question is original.)
- Experimental Thoroughness: ⭐⭐⭐ (Methodology is complete, but the sample size is too small — only a single contrastive video pair — limiting generalizability.)
- Writing Quality: ⭐⭐⭐⭐ (Logic is clear; the narrative progresses coherently from observation to causal analysis; figures and tables are well-designed.)
- Value: ⭐⭐⭐⭐ (Significant implications for AI safety and trustworthy deployment; the methodology has transfer value, though large-scale validation is needed.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Where Computation Lives Inside TabPFN: Causal Localisation of Attention Head Function](../../ICML2026/interpretability/where_computation_lives_inside_tabpfn_causal_localisation_of_attention_head_func.md)
- [\[AAAI 2026\] LLM Circuit Analyses Are Consistent Across Training and Scale](llm_circuit_analyses_consistent_across_training_and_scale.md)
- [\[ICLR 2026\] Why Low-Precision Transformer Training Fails: An Analysis on Flash Attention](../../ICLR2026/interpretability/why_low-precision_transformer_training_fails_an_analysis_on_flash_attention.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](../../NeurIPS2025/interpretability/causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[CVPR 2026\] CIGMA: Causal Information-Gain Mechanistic Attribution of Attention Heads in Vision Transformers](../../CVPR2026/interpretability/cigma_causal_information-gain_mechanistic_attribution_of_attention_heads_in_visi.md)

</div>

<!-- RELATED:END -->
