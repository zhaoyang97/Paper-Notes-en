---
title: >-
  [Paper Note] HandX: Scaling Bimanual Motion and Interaction Generation
description: >-
  [CVPR 2026][Human Understanding][Bimanual Motion Generation] This work introduces HandX—a unified bimanual motion generation infrastructure comprising 54.2 hours of motion data and 485K fine-grained text annotations. It proposes a decoupled automatic annotation strategy (kinematic feature extraction + LLM-based description generation) and benchmarks two generation paradigms—diffusion and autoregressive—demonstrating clear data and model scaling trends.
tags:
  - CVPR 2026
  - Human Understanding
  - Bimanual Motion Generation
  - Dexterous Hand Interaction
  - Motion Capture Dataset
  - Text-to-Motion
  - Scaling Law
date: 2026-05-08
content_hash: 153e8888fe3836a4
---

# HandX: Scaling Bimanual Motion and Interaction Generation

**Conference**: CVPR 2026
**arXiv**: [2603.28766](https://arxiv.org/abs/2603.28766)
**Code**: [https://handx-project.github.io](https://handx-project.github.io)
**Area**: Human Understanding / Motion Generation
**Keywords**: Bimanual Motion Generation, Dexterous Hand Interaction, Motion Capture Dataset, Text-to-Motion, Scaling Law

## TL;DR
This work introduces HandX—a unified bimanual motion generation infrastructure comprising 54.2 hours of motion data and 485K fine-grained text annotations. It proposes a decoupled automatic annotation strategy (kinematic feature extraction + LLM-based description generation) and benchmarks two generation paradigms—diffusion and autoregressive—demonstrating clear data and model scaling trends.

## Background & Motivation

**Background**: Whole-body human motion generation has achieved significant progress (e.g., MDM, MotionDiffuse), yet nearly all methods treat the hands as rigid end-effectors, lacking fine-grained finger joint representations. Hand-specific datasets and evaluation metrics are equally scarce—existing data either omit hand details (HumanML3D, InterAct) or are confined to object manipulation scenarios (ARCTIC, H2O), with coarse annotation granularity.

**Limitations of Prior Work**: (1) Absence of high-fidelity motion data capturing fine finger dynamics and bimanual coordination; (2) heterogeneous skeleton definitions, frame rates, and annotation protocols across data sources that preclude straightforward merging; (3) prohibitive cost of large-scale manual annotation; (4) lack of evaluation metrics that measure hand motion fidelity and bimanual coordination quality.

**Key Challenge**: Generating realistic bimanual motion demands large volumes of high-quality data with fine-grained annotations, yet high-quality data capture is expensive, manual annotation does not scale, and no unified evaluation framework exists.

**Goal**: Establish a unified infrastructure spanning data, annotation, and evaluation to support high-quality bimanual motion generation research.

**Key Insight**: A three-pronged strategy of integration, self-capture, and automatic annotation to address the data problem, complemented by benchmarking two generation paradigms to study scaling behavior.

**Core Idea**: Build a large-scale bimanual motion infrastructure by (1) consolidating existing datasets, (2) collecting new motion capture data, and (3) applying a decoupled LLM-based automatic annotation strategy—while empirically validating clear scaling trends.

## Method

### Overall Architecture
HandX contributes on three levels: (1) **Data**—integrating five existing datasets (GigaHands, HOT3D, ARCTIC, H2O, HoloAssist) with newly captured motion capture data, unified under a shared skeleton representation, yielding 54.2 hours of motion data after quality filtering; (2) **Annotation**—a two-stage automatic annotation pipeline that first extracts structured kinematic features (contact events, finger curl, etc.) and then employs an LLM to generate multi-granularity text descriptions (485K annotations); (3) **Generation**—benchmarking diffusion-based and autoregressive generation paradigms under multiple conditioning modes.

### Key Designs

1. **Unified Data Integration and Quality Filtering**:

    - **Function**: Consolidate heterogeneous data sources into a consistent, high-quality training set.
    - **Mechanism**: All sequences are converted to a unified skeleton representation and coordinate frame. An intensity-aware filter based on joint angular velocity removes dominant static or near-static segments, retaining only meaningful interaction motions. Self-captured data are recorded with a 36-camera OptiTrack optical motion capture system; each subject wears 25 retroreflective markers to capture fine finger joint motion, and hand skeletons are reconstructed via joint-center estimation combined with anatomical-constraint optimization.
    - **Design Motivation**: Inconsistent motion representations across datasets are the primary obstacle to merging them. The self-captured sequences specifically target bimanual interaction scenarios (e.g., bimanual coordination, inter-finger contact) to fill critical gaps in existing data.

2. **Decoupled Automatic Annotation Strategy**:

    - **Function**: Generate fine-grained, semantically rich motion-text descriptions in a scalable manner.
    - **Mechanism**: "Motion understanding" and "language generation" are decoupled into two stages. The first stage extracts structured kinematic descriptors (finger curl, finger-palm distance, spatial relationships between hands, etc.) and detects temporal events (contact, separation, hyperextension, etc.), organized as structured JSON. The second stage uses crafted prompts to guide an LLM in generating text descriptions at five granularity levels (brief summary → moderate detail → comprehensive description), covering the left hand, right hand, and bimanual relationship dimensions while preserving temporal ordering.
    - **Design Motivation**: LLMs excel at linguistic reasoning and generation but cannot directly process high-dimensional continuous motion data. By first converting motion into structured event descriptions that LLMs can interpret, and then having the LLM produce natural language, the pipeline leverages LLM language capability while ensuring motion-aligned annotations. Multi-granularity design increases annotation diversity.

3. **Dual-Paradigm Generation Model Benchmark**:

    - **Function**: Compare diffusion-based and autoregressive generation paradigms on bimanual motion tasks.
    - **Mechanism**: The diffusion model uses a joint representation of coordinates and rotation scalars, and applies triple cross-attention to process left-hand, right-hand, and bimanual interaction text descriptions separately (avoiding the left-right confusion caused by naive concatenation), predicting clean motion sequences. The autoregressive model employs FSQ (Finite Scalar Quantization) to discretize motion into tokens and performs autoregressive next-token prediction with text prefixes. The diffusion model additionally supports multiple inference-time conditioning modes (motion in-betweening, keyframe generation, wrist trajectory following, single-hand reaction generation, long-sequence generation).
    - **Design Motivation**: The triple cross-attention design resolves the issue of models assigning right-hand actions to the left hand when text descriptions are naively concatenated. FSQ exhibits better codebook utilization and scaling behavior compared to VQ-VAE.

### Loss & Training
The diffusion model is trained with an x-prediction objective using a standard denoising MSE loss. The autoregressive tokenizer is trained with the reconstruction loss $\|\mathbf{x} - \mathcal{D}(\hat{\mathbf{y}})\|_2^2$, and the autoregressive component uses standard cross-entropy loss. Hand interaction-specific metrics including contact precision, recall, and F1 are introduced, with a contact distance threshold of 2 cm.

## Key Experimental Results

### Main Results (Diffusion Model Scaling)

| Data Ratio | Decoder Layers | R-Prec Top1↑ | FID↓ | CF1↑ |
|-----------|---------------|-------------|------|------|
| 5% | 4 | 0.142 | 2.574 | 0.523 |
| 5% | 12 | 0.343 | 1.837 | 0.618 |
| 20% | 12 | 0.357 | 1.140 | 0.606 |
| 100% | 12 | **0.427** | 1.349 | **0.641** |
| 100% | 16 | 0.382 | 1.675 | 0.624 |
| Ground Truth | - | 0.854 | 0.000 | 0.984 |

### Ablation Study (Autoregressive Model Scaling)

| Model Size (M) | Codebook | R-Prec Top1↑ | FID↓ |
|---------------|----------|-------------|------|
| 4.63 | 512 | 0.366 | 8.377 |
| 26.33 | 1024 | 0.322 | 2.750 |
| 38.95 | 2048 | 0.305 | 3.245 |
| 215.31 | 4096 | 0.281 | **1.721** |

### Key Findings
- The diffusion model exhibits clear scaling trends: scaling from 5% to 100% data and from 4 to 12 decoder layers improves R-Precision Top1 from 0.142 to 0.427 (3×) and contact F1 from 0.523 to 0.641.
- A 16-layer decoder underperforms relative to 12 layers, indicating overfitting or optimization difficulties.
- In the autoregressive model, codebook size and model capacity must scale jointly: enlarging the codebook without increasing model capacity degrades performance.
- FID is best at the largest model and largest data configuration (diffusion: 1.140; autoregressive: 1.721), yet a substantial gap from Ground Truth remains.

## Highlights & Insights
- The decoupled annotation strategy is the most valuable contribution of this work—separating kinematic feature extraction from language generation so that the LLM handles only the linguistic reasoning it excels at. This paradigm is transferable to any large-scale annotation task in action understanding.
- The triple cross-attention design that resolves left-right confusion is a simple yet effective engineering detail critical to bimanual motion generation.
- This work is the first to systematically demonstrate scaling behavior in bimanual motion generation, consistent with scaling law trends observed in NLP and CV.
- The transfer of generated dexterous motions to a real humanoid robot demonstrates practical application potential.

## Limitations & Future Work
- R-Precision Top1 reaches only 0.427 (GT: 0.854), indicating a substantial quality gap between generated and real motions.
- The volume of self-captured data is relatively limited, and quality and consistency issues may arise from the integrated external datasets.
- The motion representation uses 3D coordinates rather than rotation parameters, potentially limiting physical plausibility.
- Contact detection relies on a simple distance threshold (2 cm) without modeling contact mechanics.
- Although contact F1 is introduced, evaluation metrics still lack measures for the temporal coordination of bimanual motion.

## Related Work & Insights
- **vs. BOTH2Hands**: BOTH2Hands provides 8.31 hours of bimanual motion with coarse annotations. HandX is 6.5× larger with multi-level fine-grained annotations.
- **vs. CLUTCH**: CLUTCH reconstructs hand motion from in-the-wild videos with action-level annotations. HandX uses optical motion capture for high-precision data with annotations covering finger-level details.
- **vs. Motion-X**: Motion-X is a whole-body motion dataset with coarse hand annotations. HandX focuses specifically on the hands, filling the data gap in hand motion generation.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Unified infrastructure and decoupled annotation strategy are novel in design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual-paradigm comparison, multi-scale scaling analysis, diverse conditioning modes, and robot transfer experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with thorough data statistics.
- **Value**: ⭐⭐⭐⭐⭐ — Fills a critical infrastructure gap in bimanual motion generation with significant community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MoLingo: Motion-Language Alignment for Text-to-Human Motion Generation](molingo_motion-language_alignment_for_text-to-motion_generation.md)
- [\[CVPR 2026\] ReMoGen: Real-time Human Interaction-to-Reaction Generation via Modular Learning from Diverse Data](remogen_real-time_human_interaction-to-reaction_generation_via_modular_learning_.md)
- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[CVPR 2026\] ParTY: Part-Guidance for Expressive Text-to-Motion Synthesis](party_part-guidance_for_expressive_text-to-motion_synthesis.md)
- [\[NeurIPS 2025\] HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion](../../NeurIPS2025/human_understanding/hoi-dyn_learning_interaction_dynamics_for_human-object_motion_diffusion.md)

</div>

<!-- RELATED:END -->
