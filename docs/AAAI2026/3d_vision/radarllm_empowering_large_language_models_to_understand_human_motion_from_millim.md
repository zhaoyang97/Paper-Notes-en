---
title: >-
  [Paper Note] RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence
description: >-
  [AAAI 2026][3D Vision][millimeter-wave radar] This paper proposes RadarLLM, the first end-to-end framework leveraging large language models for semantic-level human motion understanding from millimeter-wave radar point cloud sequences. The framework comprises a motion-guided radar tokenizer based on Aggregate VQ-VAE and a radar-aware language model, along with a physics-aware simulation pipeline for generating large-scale paired radar-text data.
tags:
  - AAAI 2026
  - 3D Vision
  - millimeter-wave radar
  - large language models
  - human motion understanding
  - vector quantization
  - privacy preservation
date: 2026-05-08
content_hash: d6edb09e64b6843b
---

# RadarLLM: Empowering Large Language Models to Understand Human Motion from Millimeter-Wave Point Cloud Sequence

**Conference**: AAAI 2026
**arXiv**: [2504.09862](https://arxiv.org/abs/2504.09862)
**Code**: [inowlzy.github.io/RadarLLM](https://inowlzy.github.io/RadarLLM/)
**Area**: 3D Vision
**Keywords**: millimeter-wave radar, large language models, human motion understanding, vector quantization, privacy preservation

## TL;DR

This paper proposes RadarLLM, the first end-to-end framework leveraging large language models for semantic-level human motion understanding from millimeter-wave radar point cloud sequences. The framework comprises a motion-guided radar tokenizer based on Aggregate VQ-VAE and a radar-aware language model, along with a physics-aware simulation pipeline for generating large-scale paired radar-text data.

## Background & Motivation

Human motion understanding is critical in scenarios such as elderly care, smart home, and health monitoring. Traditional vision-based systems are constrained by **illumination variation, occlusion, and privacy concerns**, rendering them unsuitable for long-term real-world deployment.

**Millimeter-wave (mmWave) radar** offers a privacy-preserving and environmentally robust alternative, operating reliably under low-light conditions, occlusion, rain, and fog without capturing visual identity information. However, existing radar-based methods primarily focus on **classification or regression tasks** (e.g., activity recognition, pose estimation), which are confined to predefined label sets and lack the ability to generate fine-grained motion descriptions.

Applying LLMs to radar data presents **two core challenges**:

**Spatiotemporal modeling of sparse, noisy point clouds**: Radar point clouds are far sparser than LiDAR or RGB-D data (only 128 points per frame), with low signal-to-noise ratios that hinder meaningful semantic feature extraction.

**Semantic gap between radar signals and language**: A substantial gap exists between the low-level physical properties of radar signals (Doppler shifts, range-velocity information) and the high-level semantics of natural language.

Furthermore, **paired radar-text data are extremely scarce**: existing radar datasets are small in scale (<9 hours, <40 subjects, <27 categories) and entirely lack natural language annotations.

The authors address these challenges through three innovations: a motion-guided radar tokenizer, a radar-aware language model, and a physics-aware data synthesis pipeline.

## Method

### Overall Architecture

RadarLLM consists of three core components (Figure 3):
1. **Radar-text dataset preparation**: A physics-aware simulation pipeline generates paired data.
2. **Motion-guided radar tokenizer**: Encodes sparse radar point cloud sequences into discrete semantic tokens.
3. **Radar-aware language model**: Performs cross-modal alignment and text generation.

### Key Designs

#### 1. **Physics-Aware Virtual Data Synthesis**

**Function**: Synthesizes realistic paired radar point cloud–text data from motion-text datasets (HumanML3D, 13,308 SMPL-X sequences with text annotations).

Synthesis pipeline (Figure 2):
- **IF signal simulation**: Ray tracing is performed between rendered human body meshes and virtual radar antennas. RF adaptive sampling focuses on the human body region, and Physical Optics Integration (POI) accumulates ray information. Gaussian noise is added to simulate realistic noise floor:

$$R'_{IF}(t) = R_{IF}(t) + \sqrt{P_{\text{signal}} / 10^{\text{SNR}/10}} \cdot \epsilon(t)$$

- **Point cloud generation**: Range-FFT → Doppler-FFT → static clutter removal → selection of the 128 highest-intensity points (ensuring consistent per-frame point count).

Each point is represented by a 6D feature vector: $\mathbf{p}_m = [x, y, z, r, v, 10\log_{10}(|D_m|)]^T$, comprising 3D coordinates, radial distance, velocity, and log Doppler intensity.

**Design Motivation**: The lack of large-scale paired data is the fundamental bottleneck for LLM training; physics-aware simulation avoids the high cost of manual annotation.

#### 2. **Motion-Guided Radar Tokenizer**

**Function**: Compresses sparse, noisy radar point cloud sequences into discrete semantic tokens suitable for LLM processing. Built upon a novel Aggregate VQ-VAE architecture (Figure 4).

**Three stages**:

**(1) Template-Prior Grouping**:
- Initializes $N_g$ anchor points on a deterministic $N_x \times N_y \times N_z$ grid within a human bounding box template.
- Performs temporal aggregation of neighboring points around each anchor.
- Extracts grouped features $\mathbf{F}_{group} \in \mathbb{R}^{L \times N_g \times C}$ using a P4Conv encoder $\mathbf{E}$.

**Design Motivation**: Addresses inconsistencies in inter-frame point positions and counts by establishing stable temporal body-region associations through a deterministic template.

**(2) Masked Context Aggregation**:
- Randomly masks 50% of anchor trajectories to obtain visible features $\mathbf{F}_{vis}$.
- A Transformer decoder reconstructs the masked features via cross-attention: $\mathbf{F}_{msk} = D(\mathbf{F}_{vis})$.
- Merges into $\mathbf{F}_{all} = [\mathbf{F}_{vis}, \mathbf{F}_{msk}]$.
- Aligns radar features with paired motion semantic features $\mathbf{F}_{mot}$ via an embedding loss.

**Design Motivation**: The masking strategy compels the model to learn inter-part dependencies across body regions; motion semantic guidance accelerates feature learning.

**(3) Aggregated Quantization**:
- Maps each timestep's $\mathbf{F}^t_{all}$ to the nearest codeword in a learnable codebook $\mathcal{Z} = \{\mathbf{z}_k\}_{k=1}^K \subset \mathbb{R}^{512 \times 512}$:

$$\mathbf{z}_t = \arg\min_{\mathbf{z}_k \in \mathcal{Z}} \|\mathbf{F}^t_{all} - \mathbf{z}_k\|_2$$

Total tokenizer loss:

$$\mathcal{L}_{VQ} = \mathcal{L}_{rec} + \mathcal{L}_{emb} + \mathcal{L}_{commit}$$

- $\mathcal{L}_{rec}$: Chamfer Distance reconstruction loss (reconstructing masked point cloud tubes).
- $\mathcal{L}_{emb} = \|\mathbf{F}_{all} - \mathbf{F}_{mot}\|_2^2$: Motion-guided embedding loss.
- $\mathcal{L}_{commit}$: Codebook commitment loss (with stop-gradient).

#### 3. **Radar-Aware Language Model**

**Function**: Aligns radar tokens and text tokens in a unified space for autoregressive motion description generation.

Built upon the T5 architecture with a unified vocabulary $\mathcal{V} = \mathcal{V}_{\text{text}} \cup \mathcal{V}_{\text{radar}}$ (32,768 WordPieces + $K$ radar tokens + special markers).

**Two-stage training**:

**(1) Pre-training**—joint training on three tasks:
- **Radar prediction**: Masks 15% of radar tokens → predicts original tokens ($\mathcal{L}_{\text{pred}}$).
- **Radar→Text**: Encodes radar tokens → decodes to generate text ($\mathcal{L}_{\text{r2t}}$).
- **Text→Radar**: Encodes text → autoregressively generates radar tokens ($\mathcal{L}_{\text{t2r}}$).

$$\mathcal{L}_{\text{pretrain}} = \lambda_1 \mathcal{L}_{\text{pred}} + \lambda_2 \mathcal{L}_{\text{r2t}} + \lambda_3 \mathcal{L}_{\text{t2r}}$$

**(2) Instruction fine-tuning**: Uses instruction-aware prompts (e.g., "Describe the motion <Motion_Placeholder>…") with similarity loss for fine-grained alignment.

### Loss & Training

- Tokenizer: 100 epochs, lr = 3.5×10⁻⁴
- Language model pre-training: 300 epochs, lr = 2×10⁻⁴
- Instruction fine-tuning: 100 epochs
- Unified batch size = 16, trained on a single RTX 3090 GPU

## Key Experimental Results

### Main Results

#### Radar-to-Text Generation Performance

| Model | Data Domain | ROUGE-L | BLEU-1 | BLEU-4 | METEOR | CIDEr | BERTScore | SimCSE |
|-------|-------------|---------|--------|--------|--------|-------|-----------|--------|
| AvatarGPT* | Virtual | 30.0 | 36.3 | 5.0 | 28.3 | 6.8 | 82.4 | 88.7 |
| Video-LLaMA2* | Virtual | 26.7 | 35.2 | 3.6 | 30.4 | 4.2 | 81.0 | 88.4 |
| MotionGPT* | Virtual | 29.4 | 37.6 | 5.0 | 26.1 | 6.5 | 82.6 | 88.9 |
| **RadarLLM** | **Virtual** | **36.0** | **48.0** | **11.4** | **33.7** | **8.3** | **83.3** | **89.6** |
| AvatarGPT* | Real | 28.8 | 38.1 | 4.2 | 25.6 | 5.6 | 81.4 | 88.1 |
| **RadarLLM** | **Real** | **28.8** | **44.2** | **5.0** | **25.7** | **4.0** | **81.4** | **88.1** |

RadarLLM achieves comprehensive superiority on virtual data: ROUGE-L +20.0%, BLEU-4 +128%, CIDEr +22.1%. It remains competitive on real data.

### Ablation Study

| Configuration | ROUGE-1 | ROUGE-L | BLEU-4 | CIDEr | Note |
|---------------|---------|---------|--------|-------|------|
| w/o template anchors | 27.9 | 25.7 | 3.8 | 3.2 | −27.3% ROUGE-1 |
| w/o masked training | 35.0 | 32.4 | 8.7 | 11.3 | −23.7% BLEU-4 |
| w/o embedding loss | 28.6 | 26.5 | 4.2 | 3.8 | −54.2% CIDEr |
| **RadarLLM (full)** | **38.4** | **36.0** | **11.4** | **8.3** | Best |

| LLM Architecture | Parameters | FPS↑ | ROUGE-L↑ | SimCSE↑ | Note |
|------------------|------------|------|---------|---------|------|
| T5-small | 60M | **97.0** | 36.0 | 89.6 | Most balanced |
| GPT2-M | 355M | 72.7 | 35.4 | 89.5 | Speed-quality trade-off |
| Deepseek-R1 | 1.8B | 53.6 | **37.4** | **89.9** | Highest quality but slowest |

| Training Strategy | ROUGE-L | BLEU-1 | METEOR | BERTScore |
|-------------------|---------|--------|--------|-----------|
| R→T only | 33.0 | 42.8 | 31.2 | 82.5 |
| R→T & T→R | 33.0 | 43.1 | 31.2 | 82.5 |
| R→T & R-Pred | 33.9 | 43.2 | 32.4 | 82.9 |
| **All tasks** | **36.0** | **48.0** | **33.7** | **83.3** |

Joint three-task training improves over R→T only: ROUGE-L +9.1%, BLEU-1 +12.2%, METEOR +8.0%.

### Key Findings

1. **End-to-end outperforms two-stage**: RadarLLM's end-to-end approach outperforms two-stage pipelines that first perform HPE and then feed results into vision/motion LLMs.
2. **Template priors are critical**: Removing template anchors causes a 27.3% drop in ROUGE-1, demonstrating that sparse radar point clouds require structural priors.
3. **Multi-task training is substantially effective**: Joint training with bidirectional translation and masked prediction consistently improves all metrics.
4. **Semantic capability is preserved under adverse conditions**: Under rain, smoke, low-light, and occlusion conditions, ROUGE-L drops by only 14.2% (28.8→24.7) and SimCSE by only 1.4%.

## Highlights & Insights

1. **Establishes a new paradigm**: RadarLLM fundamentally shifts radar-based motion understanding from "classification over predefined labels" to "natural language description."
2. **Physics-aware data synthesis**: Effectively resolves the paired data bottleneck by synthesizing radar-text data for 13K+ motion sequences.
3. **Elegant Aggregate VQ-VAE design**: The progressive design of template grouping → masked aggregation → motion-guided quantization provides clear technical motivation at each stage.
4. **Privacy-preserving and environmentally robust**: mmWave radar inherently satisfies both of these increasingly important requirements.

## Limitations & Future Work

1. Simulation parameters are calibrated for specific radar hardware (TI AWR1843BOOST); cross-hardware generalization remains to be validated.
2. Synthetic data excludes environmental context and human-object interactions, limiting the richness of scene understanding.
3. The real-world dataset is limited in scale (375 sequences); large-scale real-world evaluation is still lacking.
4. The current framework supports only single-person motion description; multi-person scenarios represent an important direction for extension.
5. T5-small performs well under resource constraints but with slightly lower semantic accuracy; full fine-tuning of larger models warrants further exploration.

## Related Work & Insights

- **Distinction from MotionGPT and AvatarGPT**: These methods generate text from SMPL-X skeletons or video, whereas RadarLLM operates directly on raw radar point clouds, avoiding error accumulation from intermediate HPE steps.
- **Analogy with PointLLM**: PointLLM connects 3D object point clouds to LLMs; RadarLLM follows a similar approach but processes temporally sequential, sparse radar point clouds, posing greater challenges.
- The data augmentation strategy via physics-aware simulation is generalizable to other sensor modalities (ultrasound, WiFi, etc.).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — A pioneering work that opens the radar+LLM direction; the VQ-VAE and data pipeline designs are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual evaluation on virtual and real data, comprehensive ablations, and notable adverse-environment testing.
- **Writing Quality**: ⭐⭐⭐⭐ — Architecture diagrams are clear, module motivations are well-articulated, and supplementary materials are thorough.
- **Value**: ⭐⭐⭐⭐ — Semantic motion understanding under privacy-preserving and adverse-environment conditions addresses genuine practical needs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Scalable Object Relation Encoding for Better 3D Spatial Reasoning in Large Language Models](../../CVPR2026/3d_vision/scalable_object_relation_encoding_for_better_3d_spatial_reasoning_in_large_langu.md)
- [\[AAAI 2026\] Point Cloud Quantization through Multimodal Prompting for 3D Understanding](point_cloud_quantization_through_multimodal_prompting_for_3d_understanding.md)
- [\[ICCV 2025\] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding](../../ICCV2025/3d_vision/3dgraphllm_combining_semantic_graphs_and_large_language_models_for_3d_scene_unde.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[CVPR 2026\] Back to Point: Exploring Point-Language Models for Zero-Shot 3D Anomaly Detection](../../CVPR2026/3d_vision/back_to_point_exploring_point-language_models_for_zero-shot_3d_anomaly_detection.md)

<!-- RELATED:END -->
