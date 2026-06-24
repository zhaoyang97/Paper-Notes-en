---
title: >-
  [Paper Note] Sounding that Object: Interactive Object-Aware Image to Audio Generation
description: >-
  [ICML 2025][Audio & Speech][Object-aware audio generation] An interactive object-aware image-to-audio generation model is proposed, which learns the correlation between image regions and sounds during training using multimodal dot-product attention, and replaces the attention weights with SAM segmentation masks during testing, allowing users to generate corresponding sounds by clicking on visual objects in the image.
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Object-aware audio generation"
  - "conditional latent diffusion model"
  - "segmentation mask"
  - "visual-audio learning"
  - "object-centric learning"
date: 2026-05-08
content_hash: 77f4c21c84ecd06a
---

# Sounding that Object: Interactive Object-Aware Image to Audio Generation

**Conference**: ICML 2025  
**arXiv**: [2506.04214](https://arxiv.org/abs/2506.04214)  
**Code**: [Project Page](https://tinglok.netlify.app/files/avobject/)  
**Area**: Segmentation/Multimodal  
**Keywords**: Object-aware audio generation, conditional latent diffusion model, segmentation mask, visual-audio learning, object-centric learning

## TL;DR

An interactive object-aware image-to-audio generation model is proposed, which learns the correlation between image regions and sounds during training using multimodal dot-product attention, and replaces the attention weights with SAM segmentation masks during testing, allowing users to generate corresponding sounds by clicking on visual objects in the image.

## Background & Motivation

Humans naturally perceive the world as a collection of distinct objects and their associated sounds—on a busy street, we can distinguish car horns, footsteps, and crowd chatter. However, enabling computational models to replicate this object-level audio specificity remains highly challenging.

**Limitations of Prior Work**:

**Global scene generation**: Vision-based methods (such as Im2Wav) analyze the entire visual scene to produce a single audio track, often overlooking subtle but important sound sources (such as a small airplane in the background) and failing to precisely control the sound of specific objects;

**Multi-event omission/binding**: Text-based methods (such as AudioLDM) tend to either omit certain sounds (such as footsteps) or incorrectly bind co-occurring events (such as mixing crowd noise with wind) when facing prompts containing multiple sound events, due to the entangled correlations among features;

**Impractical manual re-weighting**: Hand-tuning individual sound events in the diffusion latent space can alleviate these issues, but it is labor-intensive and impractical for large-scale applications.

**Key Challenge**: Real-world sounds in complex scenes are often unbalanced and mixed, making it extremely difficult to disentangle different sound sources.

**Ours**: Inspired by how humans parse complex soundscapes, this paper proposes anchoring audio generation to user-selected visual objects—allowing the model to process the overall scene context while disentangling individual sound events. The core innovations are: (1) introducing multimodal dot-product attention to learn sound-object associations in a self-supervised manner; (2) replacing attention weights with SAM segmentation masks at test time to achieve fine-grained user interaction control.

## Method

### Overall Architecture

The system comprises three core components:

1. **Conditional Latent Diffusion Model**: Based on the pre-trained AudioLDM, it performs audio generation in latent space, using a VAE to encode mel-spectrograms + a HiFi-GAN vocoder to reconstruct waveforms;
2. **Text-guided Visual Object Localization Model**: Extracts features via a CLIP image encoder + a CLAP text encoder, and then fuses text and image patch information using scaled dot-product attention to learn sound-object associations;
3. **Test-time Segmentation Mask Substitution**: After training is complete, the attention weights are replaced by SAM-generated segmentation masks, allowing users to select objects of interest by clicking with the mouse.

### Key Designs

#### 1. Conditional Latent Diffusion Model

Operating in latent space improves computational efficiency. Given a text prompt $\boldsymbol{t}_q$ and a noise vector $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$, the model generates audio through $N$ steps of iterative denoising. The training objective is to minimize the difference between the predicted and ground-truth noise:

$$\mathcal{L}_\theta = \mathbb{E}_{\boldsymbol{z}_0, \boldsymbol{t}_q, \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}), n} \|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\boldsymbol{z}_n, n, \boldsymbol{t}_q)\|_2^2$$

where $\boldsymbol{z}_0$ is the latent representation of the real audio, and $\boldsymbol{z}_n$ is the noisy latent variable at step $n$. The VAE compresses the mel-spectrogram $\boldsymbol{a} \in \mathbb{R}^{T \times F}$ into a low-dimensional latent representation $\boldsymbol{z} \in \mathbb{R}^{T' \times F' \times d}$ ($d=8$ channels).

**Design Motivation**: Operating directly in the mel-spectrogram space is computationally expensive. Operating in the latent space significantly enhances efficiency while retaining critical semantic information.

#### 2. Scaled Dot-Product Attention Fusion

This is the most core design of the paper—leveraging multimodal attention to achieve sound-object association learning:

- Use CLAP to encode the text $\mathcal{E}_t(\boldsymbol{t}_q) \in \mathbb{R}^L$ as Query
- Use CLIP to encode the image patches $\mathcal{E}_v(\boldsymbol{i}_q) \in \mathbb{R}^{P \times L}$ as Key and Value

Calculate attention weights:

$$\text{Attention}(\boldsymbol{Q}, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{Q}\boldsymbol{K}^\top}{\sqrt{d_k}}\right) \boldsymbol{V}$$

where $\boldsymbol{Q} = \mathcal{E}_t(\boldsymbol{t}_q)\boldsymbol{W}^Q$, $\boldsymbol{K} = \mathcal{E}_v(\boldsymbol{i}_q)\boldsymbol{W}^K$, $\boldsymbol{V} = \mathcal{E}_v(\boldsymbol{i}_q)\boldsymbol{W}^V$.

**Design Motivation**: Dot-product attention matches the text description with corresponding regions in the image. High attention weights naturally fall on image patches that match the text description—this is functionally equivalent to a segmentation mask. Additive attention completely fails, as confirmed by ablation studies, due to its incompatibility with the comparative learning InfoNCE loss (addition vs. multiplicative dot product).

#### 3. Segmentation Mask Substitution

After training, the softmax attention weights are replaced by normalized segmentation masks $\boldsymbol{m}_q \in \mathbb{R}^P$ generated by SAM. The masks are rescaled so that their mean and variance match the attention weights.

**Design Motivation**: The paper theoretically proves the rationality of this substitution. The key insight is that the InfoNCE contrastive loss can be viewed as the maximum likelihood estimation of the softmax attention weights. Therefore, a trained encoder will assign high attention to image patches matching the text and low attention to irrelevant patches, which is functionally equivalent to a segmentation mask.

**Theoretical Guarantee**: Theorem 3.1 gives the upper bound of the test error:

$$\text{err}_{\text{test}} \leq L_v \cdot L_f \cdot (\epsilon_{\boldsymbol{V}} + B_v \cdot (\epsilon_{\text{sam}} + 2\sqrt{2\epsilon_{\text{contrast}}})) + L_v \cdot \epsilon_{\text{sam}} + \epsilon_f$$

Various errors (segmentation model error $\epsilon_{\text{sam}}$, contrastive learning error $\epsilon_{\text{contrast}}$, model fitting error $\epsilon_f$) are minimized by large-scale pre-training, ensuring the reliability of the substitution.

#### 4. Other Design Details

- **Learnable Position Encoding**: Added to key and value embeddings to provide spatial information, helping the model distinguish objects at different positions;
- **Classifier-Free Guidance (CFG)**: During training, conditional inputs are randomly dropped with a 10% probability, and a guidance scale of $\lambda=2.0$ is used during testing;
- **Single-Head Attention**: Experiments found that while multi-head attention enhances text-audio alignment, it hurts the controllability of the segmentation mask (each head attends to different regions, reducing interpretability).

### Loss & Training

- **Main Loss**: Standard diffusion model noise prediction MSE loss (Equation 1)
- **Training Configuration**: AdamW optimizer, batch size 64, learning rate $10^{-4}$, $\beta_1=0.95$, $\beta_2=0.999$, weight decay $10^{-3}$, trained for 300 epochs
- **Data Processing**: Audio truncated/zero-padded to 10 seconds, sampling rate of 16kHz, 512-point DFT, window length of 64ms, hop length of 10ms
- **Diffusion Configuration**: Linear noise schedule with $N=1000$ steps ($\beta_1=0.0015$ to $\beta_N=0.0195$), and 200 DDIM sampling steps
- **Datasets**: Preprocessed AudioSet yielding 748 hours of video for training, and AudioCaps for evaluation

## Key Experimental Results

### Main Results

Comparison with 12 baseline methods on the AudioCaps dataset (objective metrics):

| Method | ACC ↑ | FAD ↓ | KL ↓ | IS ↑ | AVC ↑ |
|------|-------|-------|------|------|-------|
| Retrieve & Separate | 0.276 | 4.051 | 1.572 | 1.550 | 0.764 |
| AudioLDM 1 | 0.336 | 3.576 | 1.537 | 1.545 | 0.724 |
| AudioLDM 2 | 0.513 | 2.976 | 1.162 | 1.779 | 0.743 |
| Diff-Foley | 0.683 | 1.908 | 0.783 | 2.010 | 0.842 |
| FoleyCrafter | 0.732 | 1.760 | 0.665 | 2.007 | 0.811 |
| SSV2A | 0.806 | 1.265 | 0.525 | 2.100 | 0.893 |
| **Ours** | **0.859** | **1.271** | **0.517** | **2.102** | **0.891** |

Subjective evaluation (50 participants, rating 1-5):

| Method | OVL ↑ | RET ↑ | REI ↑ | REO ↑ |
|------|-------|-------|-------|-------|
| SSV2A | 3.22±0.02 | 3.50±0.03 | 3.35±0.02 | 3.48±0.06 |
| **Ours** | **3.31±0.04** | **3.62±0.05** | **3.48±0.04** | **3.74±0.07** |

### Ablation Study

| Configuration | ACC ↑ | FAD ↓ | KL ↓ | IS ↑ | AVC ↑ | Description |
|------|-------|-------|------|------|-------|------|
| (i) Freeze diffusion weights | 0.692 | 1.543 | 1.047 | 1.943 | 0.733 | Finetuning omission leads to performance drop |
| (ii) Multi-head attention | 0.415 | 2.238 | 1.903 | 2.115 | 0.887 | High AVC but sharp drop in ACC |
| (iii) Additive attention | 0.103 | 15.747 | 7.425 | 1.343 | 0.137 | Complete collapse |
| (iv) Text-image attention (at inference) | 0.856 | 1.270 | 0.520 | 2.097 | 0.890 | Performance comparable to using masks |
| (v) Audio-image attention | 0.634 | 1.761 | 1.232 | 1.731 | 0.692 | Significant performance drop |
| (vi) Train with masks | 0.763 | 1.446 | 0.742 | 1.947 | 0.797 | Hard masks hurt performance |
| **Full model** | **0.859** | **1.271** | **0.517** | **2.102** | **0.891** | — |

### Interactive Satisfaction Evaluation

| Method | Avg Time ↓ | Attempts ↓ | Satisfaction ↑ |
|------|-----------|-----------|---------|
| AudioLDM 1 | 7.34 min | 3.20 | 2.00±0.88 |
| AudioLDM 2 | 5.10 min | 2.40 | 2.80±1.04 |
| FoleyCrafter | 3.00 min | 2.80 | 3.00±1.96 |
| SSV2A | 2.95 min | 1.80 | 3.40±1.42 |
| **Ours** | **2.67 min** | **1.60** | **3.60±0.68** |

### Key Findings

1. **Additive attention completely fails**: FAD skyrocketed to 15.747, validating the theoretical analysis—additive operations are incompatible with CLAP/CLIP's contrastive loss and cannot produce attention maps equivalent to segmentation masks;
2. **Trade-off between single-head and multi-head attention**: Multi-head attention enhances text-audio alignment (AVC 0.887), but severely compromises mask controllability (ACC only 0.415), as each head attends to different regions, weakening overall interpretability;
3. **Soft attention outperforms hard mask training**: Using segmentation masks during training (configuration vi) actually degrades performance. This is because hard masks impose an overly rigid prior over the entire object region, whereas sounds typically emanate from specific parts of an object (e.g., a dog's head rather than its tail);
4. **Text-image attention is equivalent to segmentation masks**: Configuration (iv) yields almost identical performance to the full model, empirically validating the theoretical analysis;
5. **Audio-image attention is inferior to text-image**: The CLAP model has inherent limitations in representing overlapping audio, introducing noise that weakens the audio-visual correlation.

## Highlights & Insights

1. **Elegant training-testing paradigm shift**: Training with dot-product attention (without segmentation annotations) and seamlessly replacing it with a segmentation mask during testing reduces the training data requirement while providing fine-grained user control;
2. **Deep theoretical analysis**: Starting from the equivalence between contrastive learning loss (InfoNCE) and softmax attention weights, the paper rigorously proves that attention weights can be replaced by segmentation masks, offering a rare and valuable theoretical contribution in the audio generation domain;
3. **Natural fusion of multi-object interactions**: When multiple objects are selected, the model dynamically considers the context to naturally blend sounds (e.g., a loud siren from a vehicle will drown out ambient noise), rather than simply overlaying independent audio clips;
4. **Adaptability to visual texture changes**: The model can generate distinct soundscapes based on changes in visual textures (e.g., sunny → rainy days, water → grass surfaces), demonstrating a deep understanding of visual semantics;
5. **Interaction-aware object sound generation**: It captures interactions between objects (e.g., a stick hitting the water surface to produce splashing sounds), rather than generating generic water sounds.

## Limitations & Future Work

1. **Static image limitations**: Relying on static images makes it difficult to produce non-stationary audio (such as impact sounds) synchronized with dynamic events, failing to handle sound events requiring temporal information;
2. **Ambiguity in sound types for identical categories**: Lacks precise sound-type control for similar objects (e.g., a car might generate a siren or an engine sound), requiring additional semantic constraints;
3. **Dataset constraints**: Training data is derived from AudioSet (748 hours), and the quality of audio-visual correspondence depends on preprocessing filters, which may contain noise;
4. **Extension to video**: A natural extension is to scale the method to video inputs, generating temporally-aligned object-level sounds;
5. **More fine-grained control**: Current methods control "which objects produce sound"; future work can further control specific attributes of the sound (such as volume, pitch, and tempo).

## Related Work & Insights

- **Object-Centric Learning** (Greff et al., 2019; Locatello et al., 2020): Decomposing visual scenes into discrete object representations provides the paradigm foundation for the object-level audio generation in this work;
- **Audio-Visual Separation** (Zhao et al., 2018; Afouras et al., 2020): Utilizing the correspondence between visual objects and audio for source separation inspired the sound-object association learning in this work;
- **SSV2A** (Guo et al., 2024): A concurrent work that uses bounding boxes from an external object detector to generate multi-source audio; this work is more elegant as it does not require explicit segmentation annotations during training;
- **SAM** (Kirillov et al., 2023): Acquiring segmentation masks via text prompts or clicks provides the interface for test-time interaction in this work;
- **CLAP & CLIP Alignment**: Leveraging the contrastive pre-trained text-image/audio alignment spaces is the crucial foundation for the theoretical analysis (attention $\approx$ segmentation mask) in this work.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm shift of training with attention and testing with segmentation masks is novel, though the conditional diffusion model framework itself is a mature technology.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, including comparisons with 12 baselines, 6 ablation groups, subjective evaluation (50 participants), interactive satisfaction, cross-dataset generalization, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear, and theoretical analysis and experiments support each other with intuitive diagrams; however, some theoretical derivations are notation-heavy.
- Value: ⭐⭐⭐⭐ Provides a new interactive paradigm for controllable audio generation, offering inspiring value for multimodal generation and audio-visual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Object-aware Sound Source Localization via Audio-Visual Scene Understanding](../../CVPR2025/audio_speech/object-aware_sound_source_localization_via_audio-visual_scene_understanding.md)
- [\[NeurIPS 2025\] DeepASA: An Object-Oriented Multi-Purpose Network for Auditory Scene Analysis](../../NeurIPS2025/audio_speech/deepasa_an_object-oriented_multi-purpose_network_for_auditory_scene_analysis.md)
- [\[NeurIPS 2025\] Node-Based Editing for Multimodal Generation of Text, Audio, Image, and Video](../../NeurIPS2025/audio_speech/node-based_editing_for_multimodal_generation_of_text_audio_image_and_video.md)
- [\[ICML 2025\] MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners](musecontrollite_multifunctional_music_generation_with_lightweight_conditioners.md)
- [\[CVPR 2025\] Improving Sound Source Localization with Joint Slot Attention on Image and Audio](../../CVPR2025/audio_speech/improving_sound_source_localization_with_joint_slot_attention_on_image_and_audio.md)

</div>

<!-- RELATED:END -->
