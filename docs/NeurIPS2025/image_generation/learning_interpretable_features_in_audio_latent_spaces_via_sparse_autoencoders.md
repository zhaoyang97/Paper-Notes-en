---
title: >-
  [Paper Note] Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders
description: >-
  [NeurIPS 2025][Image Generation][Sparse Autoencoders] This paper proposes a framework for extracting interpretable features from the latent spaces of audio generative models via sparse autoencoders (SAEs). Linear probes…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Sparse Autoencoders"
  - "Audio Latent Space"
  - "Interpretability"
  - "Music Generation"
  - "Control Vectors"
date: 2026-05-08
content_hash: 35467ab21973336d
---

# Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders

**Conference**: NeurIPS 2025
**arXiv**: [2510.23802](https://arxiv.org/abs/2510.23802)
**Code**: None (anonymous audio samples available: [https://anonymous.4open.science/r/audio_samples-A301/](https://anonymous.4open.science/r/audio_samples-A301/))
**Area**: Image Generation
**Keywords**: Sparse Autoencoders, Audio Latent Space, Interpretability, Music Generation, Control Vectors

## TL;DR

This paper proposes a framework for extracting interpretable features from the latent spaces of audio generative models via sparse autoencoders (SAEs). Linear probes are used to map SAE features to human-understandable acoustic concepts (pitch, amplitude, timbre), enabling controllable manipulation and visualization of the audio generation process.

## Background & Motivation

As neural networks become increasingly embedded in society, their lack of interpretability has emerged as a critical concern. SAEs have become a key tool in mechanistic interpretability research, achieving notable success in large language models (LLMs)—SAEs can identify sparse directions in activation spaces, isolate underlying disentangled features, and automatically characterize monosemantic features.

However, extending SAE methods to audio generative networks faces fundamental challenges:

**Density of audio signals**: Unlike text, audio is intrinsically a dense signal that typically requires compression via autoencoders (VAE/VQ) before processing, and this compression step obscures the semantic meaning of individual "tokens."

**Difficulty of automatic characterization**: In LLMs, SAE features can be summarized using the model itself (via token-level perturbations), whereas current audio understanding models are not powerful enough to provide equivalent automatic feature characterization.

**Lack of mature frameworks**: Interpretability research for audio generative models lags far behind the text domain.

These limitations necessitate new approaches for interpretable feature discovery in audio generation systems.

## Method

### Overall Architecture

A three-stage framework:
1. **SAE Training**: Train SAEs on latent representations from audio autoencoders to extract sparse features.
2. **Linear Mapping**: Learn linear probes from SAE features to discretized acoustic attributes.
3. **Generation Process Decomposition**: Track the evolution of specific acoustic attributes throughout the synthesis process.

### Key Designs

#### 1. Modified SAE Architecture

An RMS normalization layer is added to the standard SAE:

$$\mathbf{h} = \text{ReLU}(W_{\text{enc}} \mathbf{x} + \mathbf{b}_{\text{enc}}), \quad \mathbf{f} = \text{RMSNorm}(\mathbf{h})$$

**Design Motivation**: RMS normalization maintains consistent activation magnitudes, and is empirically found to prevent out-of-distribution (OOD) artifacts during feature manipulation. This is particularly important in the audio domain given the wide dynamic range of audio signals.

Training loss: $\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|_2^2 + \lambda \|\mathbf{h}\|_1$

This combines a reconstruction fidelity term with an L1 sparsity constraint. A systematic grid search is conducted over hidden layer dimensionality (4× to 256× the input dimension) and sparsity coefficient $\lambda$ (0.005 to 0.15).

#### 2. Linear Mapping of Acoustic Concepts

Continuous acoustic attributes are discretized into interpretable "units":
- **Pitch**: Discretized according to the Western tuning system (e.g., C4, C#4), extracted using CREPE, with 66 bins.
- **Amplitude**: Computed via windowed RMS energy (librosa), with 20 equally spaced bins.
- **Timbre**: Approximated via windowed spectral centroid, with 20 equally spaced bins.

A linear classifier is trained for each acoustic attribute:

$$p^{(a)} = \text{softmax}(W^{(a)} \mathbf{f} + \mathbf{b}^{(a)})$$

**Significance of Linear Mapping**: If linear probes can effectively predict acoustic attributes, this indicates that SAE features encode these attributes in a near-linear manner, validating the alignment between learned representations and human-understandable concepts.

Bidirectional interpretability: The contribution of SAE feature $j$ to acoustic class $k$ is defined as $c_{j \to k}^{(a)} = W_{kj}^{(a)} \cdot f_j$.

#### 3. Control Vectors

The invertibility of the linear mapping is exploited to enable controllable audio manipulation:
- A scaled probe weight vector $\alpha \cdot \mathbf{w}_k^{(a)}$ is added to the SAE features.
- RMS re-normalization is applied to maintain valid activation magnitudes.
- Modified audio is decoded through the SAE and audio decoder.

Control strength $\alpha \in \{1, 10, 20, 30\}$; increasing $\alpha$ induces isolated changes in the target attribute while leaving non-target attributes largely unchanged.

### Generation Process Visualization

The generation process of DiffRhythm (a rectified flow model) is analyzed:
- At each of the 32 inference steps, the latent representation $\mathbf{X}_t$ is extracted, and acoustic concept activations are obtained via the SAE and linear probes.
- A normalized L1 distance metric is defined to track attribute evolution:

$$s_t^{(a)} = \frac{1}{K_a} \sum_{k=1}^{K_a} \frac{|p_{t,k}^{(a)} - p_{0,k}^{(a)}|}{|p_{T,k}^{(a)} - p_{0,k}^{(a)}|}$$

$s_t^{(a)} \in [0, 1]$ measures the progression from initial noise ($t=0$) to the final audio structure ($t=T=31$).

### Loss & Training

- SAE training: composite loss (reconstruction MSE + L1 sparsity), with grid search over hidden dimensionality and $\lambda$.
- Linear probes: standard multi-class cross-entropy.
- Dataset: ~31 hours of mixed audio (CocoChorales 11.2h, DAMP-VSEP 11.7h, Groove MIDI 7.8h, GuitarSet 0.4h, MAESTRO 0.55h).
- Audio encoders: DiffRhythm-VAE (continuous), EnCodec (discrete), WavTokenizer (discrete).

## Key Experimental Results

### Main Results: Acoustic Concept Mapping

Sparsity and linear probe accuracy of SAEs across different encoders:

| Encoder | Sparsity Range | Pitch Acc. | Amplitude Acc. | Timbre Acc. |
|--------|----------|----------|----------|----------|
| DiffRhythm-VAE | 0.65–0.98 | 0.75–0.87 | 0.17–0.40 | 0.17–0.35 |
| WavTokenizer | 0.993–0.999 | 0.75–0.82 | 0.17–0.30 | 0.17–0.25 |
| EnCodec | 0.55–0.95 | 0.78–0.87 | **0.56–0.63** | 0.30–0.46 |

**Key Observations**:
- Pitch is the most linearly separable attribute (0.75–0.87), remaining stable across all sparsity levels → fundamental frequency information is encoded highly linearly in the latent space.
- EnCodec significantly outperforms other models on amplitude prediction (0.56–0.63 vs. 0.17–0.49).
- Timbre is most difficult for all models (0.17–0.46), suggesting that timbre encoding is more distributed.

### Ablation Study: Attribute Evolution During Generation

Analysis of DiffRhythm's generation process over 500 MusicCaps prompts (32 inference steps):

| Acoustic Attribute | Convergence Step (approx.) | Convergence Order | Notes |
|---------|------------|------------|------|
| Pitch | ~Step 21 | First | Fundamental frequency established earliest |
| Timbre | ~Step 25 | Second | Textural features refined subsequently |
| Amplitude | Not converged | Last | Dynamic details processed last |

**Coarse-to-Fine Generation Hierarchy**: The model first establishes fundamental frequency structure (pitch), then refines texture (timbre), and finally processes dynamics (amplitude), forming a coarse-to-fine generation progression.

### Key Findings

1. **SAE features naturally align with acoustic attributes**: The effectiveness of linear mapping demonstrates that SAE features encode acoustic concepts in a near-linear manner.
2. **Different encoders exhibit distinct characteristics**: WavTokenizer produces the sparsest representations (0.993–0.999), indicating that its discrete tokens already encode highly disentangled features; EnCodec is optimal for amplitude encoding.
3. **Control vectors effectively isolate attributes**: Increasing $\alpha$ produces changes in pitch/timbre/amplitude that do not interfere with one another.
4. **Generation follows a coarse-to-fine hierarchy**: The convergence order of pitch → timbre → amplitude aligns with the hierarchical structure of human music perception.
5. **Larger hidden dimensionality consistently improves reconstruction quality**: This holds across all models.

## Highlights & Insights

- **Framework generality**: Although experiments focus on audio, the framework is theoretically extensible to other latent space-based generative models such as image and video generation.
- **Practical value of RMS normalization**: This simple modification resolves the OOD artifact problem in audio SAEs and represents an important engineering contribution.
- **Theoretical implications of linear separability**: The ability of linear probes to accurately predict acoustic attributes from SAE features indicates that these concepts are organized in a near-linear manner within the latent space, consistent with the linear representation hypothesis observed in LLMs.
- **Generation process visualization reveals the model's "reasoning"**: The coarse-to-fine hierarchy has direct implications for understanding and improving music generative models.

## Limitations & Future Work

1. **Limited coverage of acoustic attributes**: Only three basic attributes—pitch, amplitude, and timbre—are explored; richer musical features such as rhythm, harmony, and instrument identity are not addressed.
2. **Overly simplified proxy for timbre**: Representing timbre via spectral centroid is a coarse approximation, as timbre is intrinsically a multi-dimensional concept.
3. **Limited dataset scale**: ~31 hours of training data may lack sufficient diversity to cover all musical styles.
4. **Only three encoders analyzed**: Other popular architectures such as RAVE and AudioLDM are not examined.
5. **Limited granularity of control vectors**: Control operates globally; fine-grained local control along the temporal dimension is not achievable.
6. **Absence of generation quality evaluation**: Objective quality metrics for manipulated audio (e.g., FD, FAD) are not reported.

## Related Work & Insights

- **Methodological transfer from LLMs to audio**: The success of SAEs in LLM interpretability provides a methodological template for the audio domain, though challenges specific to audio—density and automatic characterization—must be addressed.
- **Connection to Concept Bottleneck methods**: The mapping from linear probes to acoustic concepts is analogous to the concept bottleneck model paradigm, while retaining end-to-end flexibility.
- **Contribution to understanding generative models**: The discovery of a coarse-to-fine generation hierarchy offers insights for diffusion model sampling strategy design—if pitch is established first, different inference steps may benefit from different guidance strengths.

## Rating
- Novelty: ⭐⭐⭐⭐ — First systematic application of the SAE interpretability framework to audio latent spaces; the RMSNorm modification and control vector design are creative contributions.
- Experimental Thoroughness: ⭐⭐⭐ — Covers three encoders and one generative model, but acoustic attributes are limited and quality evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rich visualizations, and concise framework description.
- Value: ⭐⭐⭐⭐ — Opens a new direction for interpretability in audio generation with a highly extensible framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Interpretable and Steerable Concept Bottleneck Sparse Autoencoders](../../CVPR2026/image_generation/interpretable_and_steerable_concept_bottleneck_sparse_autoencoders.md)
- [\[NeurIPS 2025\] Guided Diffusion Sampling on Function Spaces with Applications to PDEs](guided_diffusion_sampling_on_function_spaces_with_applications_to_pdes.md)
- [\[NeurIPS 2025\] MGE-LDM: Joint Latent Diffusion for Simultaneous Music Generation and Source Extraction](mge-ldm_joint_latent_diffusion_for_simultaneous_music_generation_and_source_extr.md)
- [\[NeurIPS 2025\] Exploring Variational Graph Autoencoders for Distribution Grid Data Generation](exploring_variational_graph_autoencoders_for_distribution_grid_data_generation.md)
- [\[NeurIPS 2025\] Latent Zoning Network: A Unified Principle for Generative Modeling, Representation Learning, and Classification](latent_zoning_network_a_unified_principle_for_generative_modeling_representation.md)

</div>

<!-- RELATED:END -->
