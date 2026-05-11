---
title: >-
  [Paper Note] BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment
description: >-
  [CVPR 2026][Image Generation][Visual-textual-acoustic alignment] This paper proposes the BioVITA framework, comprising a million-scale tri-modal (image–text–audio) biological dataset, a two-stage alignment model, and a six-direction cross-modal species-level retrieval benchmark, achieving for the first time unified visual-textual-acoustic representation learning in the biological domain.
tags:
  - CVPR 2026
  - Image Generation
  - Visual-textual-acoustic alignment
  - cross-modal retrieval
  - bioacoustics
  - species recognition
  - multimodal representation learning
date: 2026-05-08
content_hash: df7c795bcf7bb58a
---

# BioVITA: Biological Dataset, Model, and Benchmark for Visual-Textual-Acoustic Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.23883](https://arxiv.org/abs/2603.23883)
**Code**: [Project Page](https://dahlian00.github.io/BioVITA_Page/)
**Area**: Image Generation
**Keywords**: Visual-textual-acoustic alignment, cross-modal retrieval, bioacoustics, species recognition, multimodal representation learning

## TL;DR
This paper proposes the BioVITA framework, comprising a million-scale tri-modal (image–text–audio) biological dataset, a two-stage alignment model, and a six-direction cross-modal species-level retrieval benchmark, achieving for the first time unified visual-textual-acoustic representation learning in the biological domain.

## Background & Motivation
**Background**: Biodiversity research relies on multiple sensory modalities (images for appearance, audio for vocalizations, text for taxonomic descriptions). Models such as BioCLIP have achieved success in image–text alignment, and CLAP has made progress on audio–text alignment.

**Limitations of Prior Work**: Existing multimodal datasets focus only on paired modalities (image–text or audio–text), lacking a unified tri-modal training and evaluation framework. Pioneer efforts such as SSW60 cover only 60 species, making them severely insufficient in scale.

**Key Challenge**: Biodiversity research requires comprehensive perception of species, yet visual-textual-acoustic (VITA) alignment remains an open challenge—different datasets employ inconsistent taxonomic systems and vary greatly in scale.

**Goal**: To construct a complete VITA alignment framework enabling free species-level cross-modal retrieval among images, audio, and text.

**Key Insight**: Beginning with dataset construction, the paper collects million-scale tri-modal data with ecological attribute annotations, and aligns audio representations to an established visual-textual representation space via a two-stage training strategy.

**Core Idea**: Leveraging the powerful image–text representations pretrained by BioCLIP 2, the paper efficiently achieves unified tri-modal representation through a two-stage strategy of audio–text contrastive alignment followed by joint tri-modal contrastive alignment.

## Method

### Overall Architecture
BioVITA consists of three components: (1) BioVITA Train: a million-scale tri-modal training dataset; (2) BioVITA Model: a unified representation model with audio, image, and text encoders; and (3) BioVITA Bench: a six-direction cross-modal species-level retrieval benchmark.

### Key Designs

1. **BioVITA Train Dataset Construction**:

    - **Three-step pipeline**: audio data curation → fine-grained annotation → visual data integration
    - Collects 1.3 million audio recordings from iNaturalist, Xeno-Canto, and the Animal Sound Archive, paired with 2.3 million images from a subset of ToL-200M
    - Covers 14,133 species with 34 ecological attribute labels (diet type, activity pattern, habitat, etc.)
    - **Design Motivation**: Existing datasets are limited to either audio or images alone, precluding joint tri-modal training; the 34 attribute labels support fine-grained ecological analysis

2. **Two-Stage Training Strategy**:

    - **Stage 1 (Audio–Text Alignment)**: Trains only the ATC loss to align audio encoder representations with text
    $\mathcal{L}_{\text{ATC}} = \frac{1}{2}(\ell(\mathbf{S}_{\text{AT}}) + \ell(\mathbf{S}_{\text{AT}}^\top))$
      Trained for 30 epochs, learning rate $10^{-4}$, batch size 64
    - **Stage 2 (Tri-modal Alignment)**: Activates AIC and ITC losses to achieve full VITA alignment
    $\mathcal{L} = \mathcal{L}_{\text{ATC}} + \lambda(\mathcal{L}_{\text{AIC}} + \mathcal{L}_{\text{ITC}})$
      Trained for 10 epochs, with $\lambda$ linearly warmed up from 0 to 0.1 over the first 2 epochs
    - **Design Motivation**: Direct joint tri-modal training is unstable due to the difficulty of fine-grained visual and acoustic discrimination; aligning audio–text first and then gradually introducing images exploits the strong image–text representation space of pretrained BioCLIP 2

3. **Encoder Architecture**:

    - **Audio Encoder**: HTS-AT (hierarchical Transformer with 4 SwinT groups), extracting 768-dimensional representations from mel spectrograms
    - **Image–Text Encoder**: Pretrained BioCLIP 2 (ViT-L/14 + 12-layer Transformer), 768-dimensional
    - **Design Motivation**: Reusing a mature biological image–text encoder requires training only the audio encoder for alignment

### Loss & Training
- Contrastive learning employs standard InfoNCE-style cross-entropy loss; the temperature hyperparameter $\tau$ controls the sharpness of the similarity distribution
- A linear schedule for $\lambda$ in Stage 2 prevents the ATC loss from increasing again
- At most 20 recordings per species per epoch; audio is randomly cropped to 10-second segments to increase diversity

## Key Experimental Results

### Main Results

| Retrieval Direction | Metric | BioVITA | ImageBind | CLAP | Gain |
|---|---|---|---|---|---|
| Audio→Text (Top-1) | Species | **Best** | — | 2nd | Significant |
| Text→Audio (Top-1) | Species | **Best** | — | 2nd | Significant |
| Audio→Image (Top-1) | Species | **Best** | 2nd | — | First achieved |
| Image→Audio (Top-1) | Species | **Best** | 2nd | — | First achieved |
| Image→Text (Top-1) | Species | **Best** | — | — | Matches BioCLIP 2 |
| Text→Image (Top-1) | Species | **Best** | — | — | Matches BioCLIP 2 |

### Ablation Study

| Configuration | Audio→Text | Text→Audio | Note |
|---|---|---|---|
| Stage 1 only | High | High | Audio–text alignment is effective |
| Stage 1+2 (full) | **Best** | **Best** | Joint tri-modal training yields further gains |
| Single-stage joint training | Lower | Lower | Validates the necessity of the two-stage strategy |

### Key Findings
- BioVITA is the first to achieve species-level retrieval across all six directions, with substantial gains on audio-related directions
- Two-stage training outperforms single-stage joint training, as audio–text alignment constitutes the foundation of tri-modal alignment
- Ecological attribute labels reveal interesting associations between acoustic and visual features (e.g., nocturnal animals have more distinctive vocalizations)
- The model generalizes well to unseen species (325 species) in zero-shot settings

## Highlights & Insights
- The dataset scale and coverage far surpasses prior work (1.3M audio + 2.3M images + 14K species + 34 ecological attributes)
- The two-stage training strategy cleverly leverages pretrained models, avoiding tri-modal alignment from scratch
- The systematic six-direction retrieval benchmark provides a standardized evaluation protocol for biological multimodal research
- Ecological attribute annotations add an entirely new dimension to cross-modal biological understanding

## Limitations & Future Work
- Audio and images are not strictly paired (different individuals of the same species), precluding individual-level correspondence learning
- The video modality (temporal information of animal behavior) is not considered
- Bird species constitute the vast majority of data; coverage of other taxonomic classes may be uneven
- End-to-end fine-tuning of the image–text encoder, rather than freezing it, is worth exploring

## Related Work & Insights
- BioCLIP/BioCLIP 2 demonstrated the effectiveness of structured taxonomic text prompts for biological image–text alignment
- CLAP's success in general audio–language pretraining provides a foundation for bioacoustic alignment
- ImageBind's cross-modal alignment approach via a shared embedding space is an important reference, though it lacks sufficient biological domain data
- This work suggests that for new modality alignment, a two-stage "align first, then jointly train" approach is more robust than a one-step end-to-end strategy

## Rating
- Novelty: ⭐⭐⭐⭐ First million-scale biological tri-modal dataset and benchmark, though the methodology itself is based on established contrastive learning
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive six-direction retrieval, multi-granularity analysis, and ecological perspective, but lacks downstream task evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed description of the dataset construction process
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both biodiversity research and multimodal learning; the dataset alone constitutes a major contribution

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[CVPR 2026\] Learnability-Guided Diffusion for Dataset Distillation](learnability-guided_diffusion_for_dataset_distillation.md)
- [\[CVPR 2026\] CognitionCapturerPro: Towards High-Fidelity Visual Decoding from EEG/MEG via Multi-modal Information and Asymmetric Alignment](cognitioncapturerpro_towards_highfidelity_visual_d.md)
- [\[CVPR 2026\] ViStoryBench: Comprehensive Benchmark Suite for Story Visualization](vistorybench_comprehensive_benchmark_suite_for_story_visualization.md)
- [\[CVPR 2026\] PosterIQ: A Design Perspective Benchmark for Poster Understanding and Generation](posteriq_a_design_perspective_benchmark_for_poster_understanding_and_generation.md)

</div>

<!-- RELATED:END -->
