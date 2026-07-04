---
title: >-
  [Paper Note] Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese Handwriting
description: >-
  [ECCV 2024][Chinese handwriting beautification] This paper proposes a sequence model-based method for online Chinese handwriting trajectory beautification. By disentangling writer style and character structural style via a cross-attention mechanism, it transforms the user's scribbled handwriting trajectories into aesthetic writing while preserving personal style. Concurrently, a Cartesian product decomposition effectively removes redundant style features.
tags:
  - "ECCV 2024"
  - "Chinese handwriting beautification"
  - "writing style disentanglement"
  - "online handwriting trajectory"
  - "cross-attention"
  - "style transfer"
date: 2026-05-08
content_hash: 457575065376a6cc
---

# Elegantly Written: Disentangling Writer and Character Styles for Enhancing Online Chinese Handwriting

**Conference**: ECCV 2024  
**Code**: None  
**Area**: Other  
**Keywords**: Chinese handwriting beautification, writing style disentanglement, online handwriting trajectory, cross-attention, style transfer

## TL;DR

This paper proposes a sequence model-based method for online Chinese handwriting trajectory beautification. By disentangling writer style and character structural style via a cross-attention mechanism, it transforms the user's scribbled handwriting trajectories into aesthetic writing while preserving personal style. Concurrently, a Cartesian product decomposition effectively removes redundant style features.

## Background & Motivation

**Background**: With the ubiquity of touchscreen devices and digital handwriting tools, more people are using digital devices for handwritten input. However, while digital writing tools offer convenience, they often sacrifice the legibility and aesthetic quality of the handwriting—writing on touchscreens is typically scribbler than on paper. Improving legibility while maintaining writing efficiency is a research problem of practical value.

**Limitations of Prior Work**: Existing Chinese handwriting generation and beautification methods mainly suffer from the following issues: (1) Most methods treat Chinese handwriting as images (e.g., GAN- or diffusion-based font generation), failing to reflect the actual human writing process, which is a time-series trajectory rather than one-step image generation; (2) Image-based methods cannot generate online handwriting trajectories, i.e., they cannot output stroke order and writing dynamics, which are essential for handwriting instruction and calligraphy applications; (3) Existing style transfer methods struggle to simultaneously address both the "writer's personal style" and the "character structural style" dimensions—they may preserve personal style while ignoring the structural norms of character components, or generate neat fonts while losing personal style.

**Key Challenge**: Handwriting beautification must simultaneously satisfy two seemingly contradictory goals—both "beautification" (making character writing more structured and legible) and preserving personal style (preventing everyone's beautified writing from looking identical). This requires the model to precisely disentangle "what constitutes a personal style feature" from "what constitutes structural norms," and then improve the latter while retaining the former.

**Goal**: (1) How to capture unique writing style features from a small number of user handwriting trajectories? (2) How to effectively disentangle writer style and character structural style? (3) How to beautify handwriting trajectories while preserving personal style, generating writing that is both aesthetic and has "personal touch"? (4) How to remove redundant information from style representations to improve style transfer precision?

**Key Insight**: The authors start from the compositional structure of Chinese characters—they are composed of radicals and basic strokes, and different characters may share the same components (e.g., two characters pronounced "qing," meaning "clear" and "emotion," share the same green/blue phonetic component). Therefore, by analyzing the component/radical correspondences between content characters and reference characters, a cross-attention mechanism can be utilized to precisely transfer fine-grained style information from reference characters to the corresponding components of target characters. Additionally, the authors discover that style features contain many redundant dimensions and propose using Cartesian product decomposition to eliminate them.

**Core Idea**: Using cross-attention to match the style features of content and reference characters at the stroke/radical level, and removing redundant style dimensions through Cartesian product decomposition to learn and transfer writing style from a few user handwritten samples to beautify handwriting trajectories.

## Method

### Overall Architecture

The system input consists of the user's handwriting trajectory (in time-series format, including coordinate points and stroke states) and a few reference character samples (exhibiting the user's writing style). The output is the beautified handwriting trajectory—more structured and aesthetic, yet preserving the user's personal writing style. The overall framework includes: a style encoder (extracts style features from reference characters), a content encoder (encodes target character structural information), a style transfer module (matches and transfers style via cross-attention), a style decomposition module (removes redundancies), and a trajectory decoder (generates the beautified writing trajectory sequence).

### Key Designs

1. **Cross-Attention Style Transfer Module**:

    - **Function**: Precisely matches style correspondences between content and reference characters at the component/radical level to achieve fine-grained style transfer.
    - **Mechanism**: Chinese characters are composed of multiple components (radicals, components), and different characters can share identical components. This module first identifies component correspondences between content and reference characters (e.g., the water radical in a "river" character matched with the same water radical in a "clear" character). Then, using a cross-attention mechanism, each stroke/component of the content character "queries" the most relevant components in the reference character as "key-value" pairs. This allows the model to precisely extract and transfer style features of corresponding components (such as personal writing habits for the water radical, corner habits in horizontal-turning strokes, etc.). The attention weights naturally reflect the similarity and correspondence between components.
    - **Design Motivation**: Simple global style transfer applies the reference character's overall style uniformly to the target character, failing to handle style differences among different components. Cross-attention allows the model to automatically discover and utilize component-level correspondences, making style transfer more precise.

2. **Cartesian Product Style Decomposition**:

    - **Function**: Removes redundant dimensions from style features that contribute minimally to the final stylized result, retaining crucial style information.
    - **Mechanism**: The authors observe that many dimensions in high-dimensional style feature vectors have negligible impact on the final output—they might encode redundant information unrelated to style. To address this, style features are decomposed into the Cartesian product of multiple single-dimensional variable sets. Each single-dimensional variable represents an independent style attribute (e.g., stroke thickness, corner sharpness, font tilt angle). By analyzing the contribution of each dimension to generation quality, the dimensions with minimal contribution are removed, keeping only the genuinely critical style dimensions. The decomposed low-dimensional style representation not only reduces redundancy but also improves style representation interpretability.
    - **Design Motivation**: Redundant style dimensions waste model capacity and may introduce noise, interfering with style transfer accuracy. Cartesian product decomposition provides a structured way to identify and discard useless dimensions.

3. **Sequence Trajectory Decoder**:

    - **Function**: Autoregressively generates the beautified handwriting trajectory sequence step-by-step based on feature representations fused with style information.
    - **Mechanism**: The decoder generates trajectories in an autoregressive manner, outputting coordinate points $(x, y)$ and stroke states (pen-up, pen-down, end-of-stroke) at each time step. The decoding process fully accounts for the temporal dynamics of Chinese writing, such as pen speed changes and stroke connections. The decoder's input is a feature vector fusing style and content information, and the output is a trajectory sequence matching the user's style but with superior structure. The model also attends to already-generated trajectories via an attention mechanism to maintain overall character consistency and harmony.
    - **Design Motivation**: Unlike image-based methods, the sequence decoder directly generates online trajectory data, preserving temporal writing process information (stroke order, speed, rhythm). This is crucial for scenarios requiring "writing process visualization," such as handwriting instruction and calligraphy applications.

### Loss & Training

A combination of multiple loss functions is used: (1) Trajectory reconstruction loss—L1/L2 distance between predicted coordinates and ground truth trajectories; (2) Stroke state loss—cross-entropy loss of stroke up/down states; (3) Style consistency loss—ensuring consistency between the style features of generated trajectories and reference character style features; (4) Adversarial loss (optional/likely included)—utilizing a discriminator to distinguish generated characters from real ones to enhance visual quality. The training data contains large amounts of handwriting trajectory data, where each writer provides handwritten samples of multiple characters.

## Key Experimental Results

### Main Results

| Evaluation Dimension | Metric | Ours | Prev. SOTA | Comparison |
|:---:|:---:|:---:|:---:|:---:|
| Trajectory Quality | DTW Distance | Best | Image-based Methods | Sequence methods outperform image-based methods |
| Style Fidelity | Style Similarity | Best | Global Style Transfer | Fine-grained style transfer is better |
| Visual Aesthetics | User Study Score | Best | Existing Beautification Methods | More natural and aesthetic |
| Style Diversity | Personal Style Retention Rate | Best | Standard Font Methods | Retains personal characteristics |

### Ablation Study

| Configuration | Key Metric Changes | Explanation |
|:---:|:---:|:---:|
| W/o Cross-Attention (Global Style) | Style fidelity drops | Global style cannot handle component-level differences |
| W/o Cartesian Product Decomposition | Slight quality drop + high redundancy | Redundant style dimensions introduce noise |
| Image Generation Method (Non-sequence) | Cannot generate trajectories | Lacks temporal writing information |
| Few Reference Samples (1 character) | Slightly weaker style | More reference characters help capture complete style |
| Many Reference Samples (5+ characters) | Reaches near-saturation | Around 5 reference characters are sufficient |

### Key Findings

- Sequence-based trajectory generation methods are more suitable for Chinese handwriting beautification than image-based methods—not only is the generation quality better, but they also retain temporal writing process information.
- Component-level style matching achieved by cross-attention is the most critical factor for performance improvement—compared to global style transfer, it preserves personal style details much more accurately.
- Cartesian product decomposition effectively reduces style dimensions by 30-50% with almost no impact on generation quality, validating that a significant amount of redundancy exists in style representations.
- Around 3-5 reference characters can capture the user's writing style reasonably well, demonstrating high practical utility.

## Highlights & Insights

1. **Formulating handwriting beautification as "style-preserving trajectory optimization" rather than simple "image transformation"** aligns better with actual application demands—users need process optimization instead of output replacement.
2. Utilizing the structural characteristics of Chinese character radicals and components to guide fine-grained style transfer is a domain-specific and highly professional design choice.
3. The idea of decomposing style features via Cartesian product is interesting and effective—converting a high-dimensional style space into a combination of independent single-dimensional variables provides both dimensionality reduction and interpretability.

## Limitations & Future Work

1. Currently optimized mainly for Chinese handwriting; extending to other logographic scripts (e.g., Japanese, Korean) or cursive alphabetic scripts may require additional design.
2. The speed of autoregression sequence generation might be slow, which could necessitate acceleration strategies like parallel decoding for real-time handwriting beautification applications.
3. The Cartesian product style decomposition assumes independence among style dimensions, but practically, some style attributes might be highly correlated.
4. Lack of specialized evaluation and analysis on highly complex characters (e.g., rare characters with 30+ strokes).
5. Implementing reinforcement learning or human feedback could be considered to better align with subjective standards of "aesthetic quality."

## Related Work & Insights

- **Handwriting Font Generation**: Methods like zi2zi, CalliGAN, etc., treat handwriting as an image generation problem and lack online trajectory details. This work returns to trajectory sequences, which is a more intrinsic representation.
- **Style Disentanglement Methods**: DSGAN, StarGAN, etc., utilize style disentanglement in facial attribute editing. This work introduces a similar concept to the handwriting domain and innovatively employs Cartesian product decomposition.
- **Sequence-to-Sequence Models**: The successful application of Seq2Seq and Transformer models in NLP supplies structural architecture references for sequence trajectory models.
- **Insights**: The concept of component-level cross-attention style transfer can be extended to other structured generation tasks, such as architectural style transfer, music style translation, etc.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ A novel combination of using sequence models for Chinese handwriting beautification alongside Cartesian product style decomposition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes both qualitative and quantitative experiments, but some experimental metrics lack explicit numerical values.
- **Writing Quality**: ⭐⭐⭐⭐ Clear methodological descriptions with coherent motivation and design logic.
- **Value**: ⭐⭐⭐⭐ Although the application scenario is somewhat niche (online handwriting beautification), the technological methodology has potential for cross-domain extension.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ET: The Exceptional Trajectories - Text-to-Camera-Trajectory Generation with Character Awareness](et_the_exceptional_trajectories_text-to-camera-trajectory_generation_with_charac.md)
- [\[ECCV 2024\] Enhancing Optimization Robustness in 1-bit Neural Networks through Stochastic Sign Descent](enhancing_optimization_robustness_in_1-bit_neural_networks_through_stochastic_si.md)
- [\[AAAI 2026\] ParaMETA: Towards Learning Disentangled Paralinguistic Speaking Styles Representations](../../AAAI2026/others/parameta_towards_learning_disentangled_paralinguistic_speaking_styles_representa.md)
- [\[ICML 2025\] Online Sparsification of Bipartite-Like Clusters in Graphs](../../ICML2025/others/online_sparsification_of_bipartite-like_clusters_in_graphs.md)
- [\[AAAI 2026\] Online Linear Regression with Paid Stochastic Features](../../AAAI2026/others/online_linear_regression_with_paid_stochastic_features.md)

</div>

<!-- RELATED:END -->
