---
title: >-
  [Paper Note] Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning
description: >-
  [ICLR 2026][Video representation alignment] This work is the first to extend the Platonic Representation Hypothesis (PRH) to the temporal domain, systematically studying video–text representation alignment. It finds that increasing the number of frames and captions at test time can substantially improve alignment scores (up to doubling), and proposes a precise parameterized test-time scaling law.
tags:
  - ICLR 2026
  - Video representation alignment
  - Platonic representation hypothesis
  - Test-time scaling laws
  - Cross-modal alignment
  - Self-supervised video models
date: 2026-05-08
content_hash: a8519701bd0c0ef4
---

# Dynamic Reflections: Probing Video Representations with Text-Driven Reasoning

**Conference**: ICLR 2026
**arXiv**: [2511.02767](https://arxiv.org/abs/2511.02767)
**Code**: [Project Page](https://video-prh.github.io)
**Area**: Interpretability
**Keywords**: Video representation alignment, Platonic representation hypothesis, Test-time scaling laws, Cross-modal alignment, Self-supervised video models

## TL;DR

This work is the first to extend the Platonic Representation Hypothesis (PRH) to the temporal domain, systematically studying video–text representation alignment. It finds that increasing the number of frames and captions at test time can substantially improve alignment scores (up to doubling), and proposes a precise parameterized test-time scaling law.

## Background & Motivation

The Platonic Representation Hypothesis (PRH) posits that as neural networks scale, representation spaces across different modalities converge toward a shared statistical model. However:

1. **Static limitations**: Existing PRH studies are confined to static image–text modality alignment; the temporal dynamics of video (motion, causality, temporal dependencies) remain entirely unexplored.
2. **Uncertainty about alignment ceilings**: Huh et al. (2024) raised the question of whether an alignment score of 0.16 constitutes strong alignment, but this has not previously been answered.
3. **Neglect of test-time factors**: Prior work focuses on training-time resources (model size, training data volume); the effect of test-time data richness on alignment has not been studied.
4. **Costly evaluation of video models**: Current evaluation of self-supervised video models requires training task-specific decoders for each downstream task, which is expensive.

The paper's core insight is that increasing the number of frames and diversity of captions at test time can substantially boost alignment—in some settings nearly doubling it from ~0.16 to ~0.4—without any retraining.

## Method

### Overall Architecture

Building on the Mutual k-NN methodology of Huh et al. (2024), the framework is extended to multi-frame, multi-caption video–text alignment evaluation. A total of 121 vision and language models are evaluated.

### Key Designs

**Mutual k-NN Alignment Metric:**

Given a video embedding matrix $\mathbf{X} \in \mathbb{R}^{N \times p}$ and a text embedding matrix $\mathbf{Y} \in \mathbb{R}^{N \times q}$:

$$\mathcal{A}^{\text{MkNN}}(\mathbf{X}, \mathbf{Y}) = \frac{1}{kN} \sum_{i=1}^{N} \sum_{j=1}^{N} (\mathbf{M}_{\mathbf{X}} \odot \mathbf{M}_{\mathbf{Y}})_{ij}$$

where $\mathbf{M}_{\mathbf{X}}$ and $\mathbf{M}_{\mathbf{Y}}$ are $k$-nearest-neighbor indicator matrices and $\odot$ denotes the Hadamard product. The core operation computes the average overlap of $k$-nearest-neighbor sets across the two embedding spaces.

**Multi-Frame Video Encoding Strategy:**
- For video encoders natively supporting $n_o$ frames: $n_f$ frames are extracted via uniform linear interpolation.
- When $n_f > n_o$: nearest-neighbor interpolation extracts an integer multiple of $n_o$ frames; sub-clips are passed through the encoder and their representations are averaged.
- For image encoders: a simple extension averages representations temporally across 8 frames.

**Multi-Caption Text Encoding Strategy:**
- Selected captions are concatenated into a single string.
- Intermediate-layer features are extracted via text encoders (including LLM-based ones).
- Per-token embeddings are averaged along the token dimension, yielding a [layer, hidden-dim] feature vector.

**Parameterized Test-Time Scaling Law:**

$$\text{score}(n_f, n_c) = S_\infty - (C_f n_f^{-\alpha} + C_c n_c^{-\beta})$$

- $S_\infty$: theoretical saturation alignment score (ideal upper bound)
- $C_f, C_c, \alpha, \beta$: fitted scalar parameters
- $n_f, n_c$: number of frames and captions

Analogous to training-time compute-optimal scaling laws (e.g., Chinchilla), this law measures alignment score (higher is better), with the subtracted terms representing the "penalty" imposed by limited test-time data.

### Loss & Training

No model training is involved—all models are pretrained and frozen. The core contribution lies in the evaluation methodology and test-time data utilization strategy. Optimization is limited to selecting the optimal pair of intermediate layers (from the visual and text encoders) to maximize alignment score.

## Key Experimental Results

### Main Results

**Video–text alignment on VATEX (single caption, against Gemma 2 9B-it):**

| Visual Model Category | Best Alignment Score | Notes |
|----------------------|---------------------|-------|
| Image-only models (single frame) | ~0.18 | Reproduces Huh et al. results |
| Image models (multi-frame average) | ~0.223 | Simple temporal averaging is effective |
| VideoMAEv2 (self-supervised video) | Highest | Native video model outperforms DINOv2 |
| Gemma text encoders | Best | Raises best image–text score to ~0.206 |

**Test-time scaling effect (1→10 captions):** Alignment score improves by an average of 60%.

**Scaling law fitting results:**

| Model | $S_\infty$ | $C_f$ | $C_c$ | $\alpha$ | $\beta$ | $R^2$ |
|-------|-----------|-------|-------|---------|--------|-------|
| VideoMAEv2 | 0.41 | 0.15 | 0.13 | 0.75 | 1.30 | 0.9791 |
| DINOv2 | 0.37 | 0.05 | 0.13 | 1.76 | 1.40 | 0.9964 |

Key difference: VideoMAEv2's frame coefficient $C_f = 0.15$ is nearly three times that of DINOv2, indicating that video models are better at leveraging temporal information.

### Ablation Study

**Correlation between cross-modal alignment and downstream tasks (self-supervised video models, Figure 4):**

| Downstream Task | Correlation with Text Alignment | Task Type |
|----------------|--------------------------------|-----------|
| SSv2 action recognition | Strong positive | Semantic |
| Kinetics action recognition | Strong positive | Semantic |
| Camera pose estimation | Significant | Non-semantic perception |
| Depth prediction | Significant | Non-semantic perception |
| Object tracking | Significant | Non-semantic perception |
| Point tracking | Weak | Non-semantic (highly local) |

**Temporal awareness analysis (Test of Time dataset):**
- At $k=3$, all models achieve near-perfect alignment (each sample has 3 unambiguous neighbors).
- At $k=1,2$, notable differences emerge: text models exhibit bag-of-words behavior and are insensitive to temporal order; video models also differ in their temporal encoding strategies.

**VideoComp temporal reordering experiment:**
- Positive captions vs. temporally reordered negative captions: alignment drops but not significantly.
- Models with higher alignment scores are more affected by reordering, suggesting these models may have learned temporally aware structure.

### Key Findings

1. **Self-supervised video models surpass image models**: VideoMAEv2 achieves stronger text alignment than DINOv2, despite never being exposed to text supervision.
2. **Test-time data richness is critical**: Multi-frame + multi-caption inputs can double alignment scores (0.16 → 0.4).
3. **Video–text alignment serves as a zero-shot metric**: It strongly correlates with both semantic and non-semantic downstream tasks.
4. **Temporal reasoning in current video and language models remains limited**, especially under hard negative examples.

## Highlights & Insights

1. **Test-time scaling laws**: Analogous to training-time scaling laws, this work is the first to reveal the systematic effect of test-time data (frame count + caption count) on alignment quality; the fitting accuracy of $R^2 > 0.97$ is impressive.
2. **Answering an open question in PRH**: The low alignment score of 0.16 is attributed primarily to test-time data poverty rather than fundamental differences in representation spaces.
3. **Large experimental scale**: 121 models (85 visual + 36 language), covering self-supervised, contrastive, and generative paradigms across multiple datasets.
4. **Practical utility of zero-shot evaluation**: Expensive task-specific decoder training is no longer required to assess video representation quality.
5. **Counterintuitive finding on multi-caption strategies**: Even synthetic multi-captions decomposed from a single caption by an LLM improve alignment.

## Limitations & Future Work

1. **Insufficient causal inference**: Correlation between alignment scores and downstream performance does not imply causation.
2. **Temporal reasoning challenges remain unresolved**: The bag-of-words behavior of text models and the limited temporal awareness of video models suggest that PRH has not yet fully held in the temporal domain.
3. **Generalizability of scaling laws**: Validated only on VATEX and PVD; parameters may differ under different data distributions.
4. **Weak alignment of generative video model representations**: How to leverage their latent representations remains an open problem.
5. **Impact of caption quality underexplored**: Differences in annotator style and LLM-generated caption quality warrant further investigation.

## Related Work & Insights

- **Platonic Representation Hypothesis** (Huh et al., 2024) is the direct foundation; this paper's extension to the temporal dimension is a natural and important contribution.
- **VideoMAEv2** (Wang et al., 2023): This work reveals its strong semantic alignment capability in the absence of text supervision.
- **DINOv2** (Oquab et al., 2023): Serves as an upper-bound reference for image encoders.
- Inspiration: The discovery of test-time scaling laws points to an entirely new model evaluation paradigm—characterizing the upper bound of a model's representational capacity by varying the amount of test-time data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to extend PRH to the video temporal domain; test-time scaling laws are an original contribution.
- **Technical Depth**: ⭐⭐⭐⭐ — Mathematical modeling of scaling laws is rigorous; experimental design is systematic and comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Large-scale evaluation of 121 models across multiple datasets and downstream tasks.
- **Value**: ⭐⭐⭐⭐ — Zero-shot evaluation metrics offer practical value to the community.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Fluent narrative, polished figures, and clear structure.

**Overall**: ⭐⭐⭐⭐⭐ (4.5/5) — A highly insightful study in representation learning; both the test-time scaling laws and the temporal extension of PRH constitute important contributions, with excellent experimental scale and quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Reflections: Probing Video Representations with Text Alignment](dynamic_reflections_probing_video_representations_with_text_alignment.md)
- [\[ICLR 2026\] One Language, Two Scripts: Probing Script-Invariance in LLM Concept Representations](one_language_two_scripts_probing_script-invariance_in_llm_concept_representation.md)
- [\[ICLR 2026\] Beyond Linear Probes: Dynamic Safety Monitoring for Language Models](beyond_linear_probes_dynamic_safety_monitoring_for_language_models.md)
- [\[ICLR 2026\] Uncovering Grounding IDs: How External Cues Shape Multimodal Binding](uncovering_grounding_ids_how_external_cues_shape_multimodal_binding.md)
- [\[ICLR 2026\] The Reasoning Trap — Logical Reasoning as a Mechanistic Pathway to Situational Awareness](the_reasoning_trap_--_logical_reasoning_as_a_mechanistic_pathway_to_situational_.md)

</div>

<!-- RELATED:END -->
