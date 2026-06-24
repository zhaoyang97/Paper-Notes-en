---
title: >-
  [Paper Note] SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning
description: >-
  [Human Understanding] > SemGes proposes a two-stage framework that integrates semantic information at both global and fine-grained levels through semantic coherence and semantic relevance learning, generating co-speech gestures aligned with speech semantics. The method surpasses existing approaches on two benchmarks: BEAT and TED-Expressive.
tags:
  - "Human Understanding"
date: 2026-05-08
content_hash: 710695c2f6a58a07
---

# SemGes: Semantics-aware Co-Speech Gesture Generation using Semantic Coherence and Relevance Learning

- **Conference**: ICCV 2025
- **arXiv**: 2507.19359
- **Code**: https://semgesture.github.io/
- **Area**: Human Understanding
- **Keywords**: co-speech gesture generation, semantic coherence, VQ-VAE, cross-modal fusion, semantic relevance

## TL;DR

> SemGes proposes a two-stage framework that integrates semantic information at both global and fine-grained levels through semantic coherence and semantic relevance learning, generating co-speech gestures aligned with speech semantics. The method surpasses existing approaches on two benchmarks: BEAT and TED-Expressive.

## Background & Motivation

Human communication is inherently multimodal, with gestures and speech complementing each other to convey pragmatic and semantic information. Co-speech gesture generation aims to synthesize nonverbal cues synchronized with speech, with important applications in digital humans and AI agents.

Core problems faced by existing methods:

**Overemphasis on beat gestures**: Most methods focus on generating beat gestures aligned with speech rhythm, neglecting richer expressive forms such as iconic gestures that convey semantic content.

**Disconnect between global and local semantics**: Existing methods either focus solely on global semantics (e.g., aligning text and motion via CLIP) or on keyword-level local semantics, failing to address both within a unified framework.

**Insufficient exploitation of semantic relevance**: Different gesture types (beat, iconic, metaphoric) carry different semantic importance, a distinction that existing methods fail to leverage effectively for generation guidance.

These issues result in generated gestures that appear natural but lack strong semantic association with speech content.

## Method

### Overall Architecture

SemGes adopts a two-stage design:

- **Stage 1**: Trains a VQ-VAE to learn motion priors and establish an efficient discrete motion codebook.
- **Stage 2**: Fuses speech, textual semantics, and speaker identity to generate gestures through semantic coherence and relevance learning.

### Stage 1: VQ-VAE Motion Prior Learning

Independent VQ-VAEs with dedicated codebooks are trained separately for hand and body motions. The encoder $\mathcal{E}_m$ maps motion into a latent vector $\hat{z}$, which is vector-quantized via nearest-neighbor lookup:

$$\mathbf{q}(\hat{\boldsymbol{z}}) = \arg\min_{z^i \in \mathcal{Z}} \|\hat{z}^j - z^i\|$$

The training loss includes a reconstruction loss (over position, velocity, and acceleration) and the VQ-VAE commitment loss:

$$\mathcal{L}_{\text{VQ-VAE}} = \|\mathbf{g}-\hat{\mathbf{g}}\|^2 + \|\dot{\mathbf{g}}-\hat{\dot{\mathbf{g}}}\|^2 + \|\ddot{\mathbf{g}}-\hat{\ddot{\mathbf{g}}}\|^2 + \|\text{sg}[\mathbf{E(g)}]-\mathbf{q}(\hat{\boldsymbol{z}})\|^2 + \|\mathbf{E(g)}-\text{sg}[\mathbf{q}(\hat{\boldsymbol{z}})]\|^2$$

### Stage 2: Semantics-Driven Gesture Generation

#### Semantic Coherence Embedding Learning

Textual semantics and motion representations are aligned in a shared embedding space. A pretrained FastText model encodes text as $\mathcal{Z}^S = \mathcal{E}_s(S)$, while the frozen Stage 1 motion encoder produces motion embeddings $\mathcal{Z}^h, \mathcal{Z}^b$. Alignment is enforced via cosine similarity loss:

$$\mathcal{L}_{\text{semantic-coherence}} = 1 - \cos(\mathcal{Z}^h, \mathcal{Z}^s) + 1 - \cos(\mathcal{Z}^b, \mathcal{Z}^s)$$

#### Cross-Modal Fusion

A Transformer encoder fuses three modalities:
- HuBERT features $\mathcal{Z}^a$ extracted from raw speech
- Speaker identity embedding $\mathcal{Z}^i$
- Textual semantic features $\mathcal{Z}^s$

The audio and identity streams are processed through a self-attention layer, then interact with textual semantics via a cross-attention layer to produce the fused representation $\mathcal{Z}^f$.

A **multimodal quantization consistency loss** ensures that the fused representation aligns with ground-truth motion codes:

$$\mathcal{L}_{\text{quantization}} = \|Quant^h(\mathcal{Z}^f) - Quant^h(\mathcal{Z}^h)\|^2 + \|Quant^b(\mathcal{Z}^f) - Quant^b(\mathcal{Z}^b)\|^2$$

#### Semantic Relevance Loss

A Smooth L1 loss (a variant of Huber loss) is used to prioritize semantic gestures (iconic, metaphoric, etc.) while preventing excessive penalization of small errors:

$$\mathcal{L}_{\text{semantic-relevance}} = \mathbb{E}[\lambda \Psi(\mathbf{G} - \hat{\mathbf{G}})]$$

where $\Psi$ applies quadratic penalization for small errors and linear penalization for large errors, and $\lambda$ is an annotated relevance factor.

### Total Loss

$$\mathcal{L}_{\text{SemGes}} = \mathcal{L}_{\text{semantic-coherence}} + \mathcal{L}_{\text{semantic-relevance}} + \mathcal{L}_{\text{quantization}}$$

### Long-Sequence Inference

Long-sequence generation is achieved via an overlap-stitch algorithm: the input is segmented into clips, with 4-frame overlaps between adjacent clips to ensure smooth transitions.

## Key Experimental Results

### Main Results (BEAT Dataset)

| Method | FGD ↓ | BC ↑ | Diversity ↑ | SRGR ↑ |
|------|-------|------|-------------|--------|
| CaMN | 8.510 | 0.797 | 206.789 | 0.231 |
| DiffGesture | 9.632 | 0.876 | 210.678 | 0.106 |
| LivelySpeaker | 13.378 | 0.891 | 214.946 | 0.229 |
| DiffSheg | 6.623 | 0.922 | 257.674 | 0.250 |
| **SemGes** | **4.467** | 0.453 | **305.706** | **0.256** |

SemGes achieves state-of-the-art performance on FGD (32.5% reduction), Diversity (18.6% improvement), and SRGR.

### Ablation Study

| Model Variant | FGD ↓ | BC ↑ | Diversity ↑ | SRGR ↑ |
|---------|-------|------|-------------|--------|
| Baseline (VQ-VAE only) | 10.348 | 0.564 | 198.568 | 0.176 |
| w/o Semantic Coherence | 8.053 | 0.556 | 249.550 | 0.180 |
| w/o Semantic Relevance | 7.549 | 0.573 | 245.319 | 0.195 |
| w/ SpeechCLIP Encoder | 6.787 | 0.468 | 289.621 | 0.245 |
| **SemGes (Full)** | **4.467** | 0.453 | **305.706** | **0.256** |

### Key Findings

1. **Necessity of the two-stage design**: The standalone VQ-VAE (Stage 1) yields generation quality well below SOTA baselines; the two-stage design substantially improves quality by decoupling motion prior learning from semantics-driven generation.
2. **Significant contribution of the semantic coherence module**: Its removal degrades FGD from 4.467 to 8.053, with a notable drop in diversity.
3. **Semantic relevance module improves semantic gesture recall**: Its removal reduces SRGR from 0.256 to 0.195.
4. **Trade-off in BC metric**: The lower BC of SemGes reflects its focus on semantic alignment rather than strict rhythmic synchronization; BC reaches 0.689 on segments containing only beat gestures, indicating that rhythmic coherence is preserved.
5. **User study validation**: In an evaluation with 30 participants, SemGes significantly outperforms CaMN and DiffShEG in naturalness, diversity, and speech alignment ($p < 0.05$).

## Highlights & Insights

- The unified modeling of global and local semantics is novel: semantic coherence targets global text–motion alignment, while semantic relevance emphasizes key semantic frames.
- The separate codebook design for hands and body is well-motivated, allowing each body part to independently learn its motion patterns.
- The alignment strategy of freezing the Stage 1 encoder and training only the text encoder avoids catastrophic forgetting of motion representations.
- The long-sequence inference strategy is simple yet effective (4-frame overlap stitching).

## Limitations & Future Work

- The lower BC metric suggests that semantic enhancement may come at the cost of rhythmic precision.
- The method relies on frame-level semantic annotations (e.g., gesture type labels in the BEAT dataset); the semantic relevance loss cannot be applied to datasets lacking such annotations (e.g., TED Expressive).
- FastText as the text encoder may limit the depth of semantic understanding.
- Facial expressions are not modeled; the method focuses solely on body and hand motions.

## Related Work & Insights

- **Co-speech gesture generation**: CaMN (LSTM + multimodal), DiffShEG (diffusion model), LivelySpeaker (CLIP global semantics)
- **Semantics-aware generation**: SEEG (hierarchical semantic alignment), HA2G (hierarchical audio-to-gesture)
- **Two-stage latent space methods**: VQ-VAE discretization followed by conditional generation has become a prevailing paradigm in recent work

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Effectiveness | ⭐⭐⭐⭐ |
| Clarity | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall | 8.0/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SemTalk: Holistic Co-speech Motion Generation with Frame-level Semantic Emphasis](semtalk_holistic_co-speech_motion_generation_with_frame-level_semantic_emphasis.md)
- [\[ICCV 2025\] OpenAnimals: Revisiting Person Re-Identification for Animals Towards Better Generalization](openanimals_revisiting_person_re-identification_for_animals_towards_better_gener.md)
- [\[ICCV 2025\] What's Making That Sound Right Now? Video-centric Audio-Visual Localization](whats_making_that_sound_right_now_video-centric_audio-visual_localization.md)
- [\[ICCV 2025\] Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection](sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_v.md)
- [\[ICCV 2025\] RayPose: Ray Bundling Diffusion for Template Views in Unseen 6D Object Pose Estimation](raypose_ray_bundling_diffusion_for_template_views_in_unseen_6d_object_pose_estim.md)

</div>

<!-- RELATED:END -->
