---
title: >-
  [Paper Note] Implicit Counterfactual Learning for Audio-Visual Segmentation
description: >-
  [ICCV 2025][Segmentation][audio-visual segmentation] This paper proposes the Implicit Counterfactual Framework (ICF), which employs multi-granularity implicit text as a modality bridge to reduce the audio-visual representation gap, and leverages semantic counterfactuals to generate orthogonal counterfactual samples that mitigate modality preference. Combined with Collaborative Distribution-Aware Contrastive Learning (CDCL), ICF achieves unbiased cross-modal understanding and…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "audio-visual segmentation"
  - "counterfactual learning"
  - "cross-modal alignment"
  - "implicit text bridging"
  - "contrastive learning"
date: 2026-05-08
content_hash: 54b84dc930327ff2
---

# Implicit Counterfactual Learning for Audio-Visual Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.20740](https://arxiv.org/abs/2507.20740)  
**Code**: N/A  
**Area**: Semantic Segmentation / Audio-Visual Segmentation
**Keywords**: audio-visual segmentation, counterfactual learning, cross-modal alignment, implicit text bridging, contrastive learning

## TL;DR

This paper proposes the Implicit Counterfactual Framework (ICF), which employs multi-granularity implicit text as a modality bridge to reduce the audio-visual representation gap, and leverages semantic counterfactuals to generate orthogonal counterfactual samples that mitigate modality preference. Combined with Collaborative Distribution-Aware Contrastive Learning (CDCL), ICF achieves unbiased cross-modal understanding and state-of-the-art performance on three AVS benchmarks.

## Background & Motivation

**Background**: Audio-Visual Segmentation (AVS) aims to segment sounding objects in video frames guided by audio cues. It is an emerging cross-modal understanding task requiring models to jointly comprehend audio content (what is sounding) and visual content (where it is sounding), producing pixel-level segmentation masks. Existing methods primarily focus on improving cross-modal fusion, e.g., via attention mechanisms or audio-guided query designs.

**Limitations of Prior Work**: Existing AVS methods emphasize interaction efficiency while overlooking two deeper issues. First, the **modality gap**: audio and vision are highly heterogeneous modalities (one is a spectral signal, the other a pixel array), whose representation spaces are inherently misaligned, making direct interaction prone to erroneous matching—especially in complex scenes with ambiguous visual content or multiple audio sources. Second, **modality preference**: the visual modality typically carries richer information, leading models to over-rely on visual features and marginalize audio cues, diminishing the effective role of audio in decision-making.

**Key Challenge**: AVS inherently requires equal collaboration between audio and visual modalities, yet the representation gap caused by modality heterogeneity and the resulting modality preference give rise to a "nominally cross-modal, effectively unimodal" problem—models may learn to exploit visual features alone rather than genuinely utilizing audio information.

**Goal**: To achieve truly unbiased cross-modal understanding—simultaneously narrowing the audio-visual representation gap and eliminating modality preference, so that audio cues play a substantive role in segmentation decisions.

**Key Insight**: The authors propose two core strategies. (1) Using text as a bridge—text can describe both visual and audio content, making it a natural shared semantic space for aligning heterogeneous modalities. (2) Using counterfactual learning to eliminate preference—by generating counterfactual samples in latent space (e.g., "visual present but semantically irrelevant audio"), models are forced to avoid relying on any single modality.

**Core Idea**: ICF = Multi-granularity Implicit Text (MIT) + Semantic Counterfactual (SC) + Collaborative Distribution-Aware Contrastive Learning (CDCL), improving AVS performance from two dimensions: representation alignment and preference elimination.

## Method

### Overall Architecture

ICF builds upon a standard AVS backbone (visual encoder + audio encoder + decoder), augmented with three core modules. Given a video frame sequence and its corresponding audio clip, the pipeline proceeds as follows: (1) visual and audio encoders extract respective features; (2) the MIT module generates multi-granularity implicit text representations as a modality bridge; (3) the SC module generates counterfactual samples in latent space; (4) the CDCL module leverages factual-counterfactual and cross-modal contrastive signals to align representations and eliminate preference; (5) the fused features are passed to the decoder to produce segmentation masks.

### Key Designs

1. **Multi-granularity Implicit Text (MIT)**:

    - **Function**: Constructs a modality-shared semantic space to reduce the representation gap between audio and visual modalities.
    - **Mechanism**: Leverages the text encoder of a pretrained vision-language model (e.g., CLIP) to generate implicit text representations at three granularities—video-level (global semantics of the entire video), clip-level (semantics of each temporal segment), and frame-level (fine-grained content of individual frames). These representations are not natural language sentences but rather "pseudo-text" embeddings generated via learnable prompts in the output space of the CLIP text encoder. Since CLIP's text space is naturally aligned with its visual space, these implicit text representations serve as "translators" that map audio features into this shared space, thereby bridging heterogeneous modalities. Multi-granularity design ensures semantic alignment from global to local levels.
    - **Design Motivation**: Direct alignment of audio and visual features is difficult due to large modality discrepancy; however, both can be aligned with text (as both have corresponding linguistic descriptions). Using text as a bridge is an indirect strategy—leveraging CLIP's shared space to achieve audio-visual alignment implicitly. Multi-granularity design prevents information loss that would arise from using a single granularity alone.

2. **Semantic Counterfactual (SC)**:

    - **Function**: Generates counterfactual samples in latent space to eliminate decision bias caused by modality preference.
    - **Mechanism**: In the joint representation space, orthogonal transformations are applied to factual samples (genuine audio-visual pairs) to produce counterfactual samples—i.e., replacing one modality's feature with a semantically orthogonal representation while keeping the other modality unchanged. Specifically, the audio feature $a$ is decomposed along the direction of the visual feature $v$: $a = a_{\parallel v} + a_{\perp v}$, and the counterfactual audio feature is the orthogonal component $a_{\perp v}$. The resulting counterfactual samples preserve the statistical properties of the original modality while being semantically uncorrelated with the paired modality. A key advantage is that this counterfactual generation is implicit—operating directly in latent space without requiring actual counterfactual image or audio generation, avoiding the complexity and artifacts of generative models.
    - **Design Motivation**: If a model can produce correct segmentation relying solely on visual features, its predictions remain unchanged regardless of the audio—this is modality preference. Contrastive training between factual and counterfactual samples compels the model to attend to audio information. Orthogonal decomposition ensures diversity and semantic validity of counterfactual samples, outperforming naive random replacement or dropout.

3. **Collaborative Distribution-Aware Contrastive Learning (CDCL)**:

    - **Function**: Jointly optimizes representation alignment and modality debiasing using factual-counterfactual pairs and cross-modal pairs.
    - **Mechanism**: CDCL comprises two contrastive losses. (1) *Factual-counterfactual contrastive loss*: treats factual samples (correct audio-visual pairs) as positives and counterfactual samples (mismatched pairs) as negatives, maximizing their distance in representation space. This trains the model to distinguish "genuinely correlated" from "semantically unrelated" cross-modal pairings. (2) *Cross-modal cohesion contrastive loss*: pulls audio and visual representations of factual pairs closer together, ensuring tightly clustered alignment of correlated modalities. The two losses are collaborative—the former handles debiasing (eliminating modality preference) and the latter handles alignment (narrowing the modality gap). "Distribution-aware" refers to weighting the loss computation by sample distribution characteristics, assigning higher weights to hard samples (complex scenes with weak audio-visual correlation).
    - **Design Motivation**: Pure cross-modal alignment cannot resolve preference (models may collapse everything into a vision-dominated subspace), and pure counterfactual contrastive learning cannot resolve the gap (a gap may persist even after debiasing). Joint application is necessary to simultaneously achieve both alignment and balance.

### Loss & Training

The total loss comprises four components: (1) BCE + Dice segmentation loss—standard pixel-level supervision; (2) factual-counterfactual contrastive loss—InfoNCE formulation with correct pairs as positives and counterfactual pairs as negatives; (3) cross-modal cohesion contrastive loss—pulling correlated audio and visual representations closer; (4) distribution-aware weighting—adaptively adjusting contrastive loss weights according to per-sample alignment difficulty. Training strategy: the backbone is first warmed up under the standard AVS setting (CLIP frozen), then all modules are jointly fine-tuned.

## Key Experimental Results

### Main Results

Performance comparison on three public AVS benchmarks:

| Method | AVSBench-S4 mIoU↑ | AVSBench-MS3 mIoU↑ | AVSS mIoU↑ | Params |
|------|-------------------|-------------------|------------|-------|
| AVSBench (baseline) | 72.8 | 45.7 | 29.8 | 42M |
| AVSA | 76.4 | 48.9 | 33.2 | 48M |
| CATR | 78.2 | 50.3 | 35.4 | 53M |
| AVSegFormer | 79.9 | 52.1 | 36.5 | 56M |
| GAVS | 80.6 | 53.4 | 37.1 | 58M |
| **ICF (Ours)** | **82.3** | **55.2** | **39.0** | **55M** |

Detailed comparison under different settings (S4: single sound source; MS3: multiple sound sources):

| Method | S4 mIoU↑ | S4 F-score↑ | MS3 mIoU↑ | MS3 F-score↑ |
|------|---------|------------|----------|-------------|
| AVSegFormer | 79.9 | 87.2 | 52.1 | 62.8 |
| GAVS | 80.6 | 87.9 | 53.4 | 64.1 |
| ICF (Ours) | 82.3 | 89.1 | 55.2 | 66.3 |

### Ablation Study

| Configuration | S4 mIoU↑ | MS3 mIoU↑ | Note |
|------|---------|----------|------|
| Baseline (w/o ICF modules) | 78.5 | 50.8 | Standard AVS backbone |
| + MIT only | 80.4 | 52.9 | Implicit text bridge only |
| + SC only | 79.8 | 52.1 | Semantic counterfactual only |
| + CDCL only | 79.5 | 51.6 | Contrastive learning only |
| + MIT + SC | 81.2 | 53.8 | Bridge + counterfactual |
| + MIT + SC + CDCL (Full ICF) | 82.3 | 55.2 | Full method |
| SC replaced by random dropout | 80.0 | 52.3 | Random substitute for orthogonal counterfactual |
| MIT with frame-level only (w/o multi-granularity) | 80.8 | 53.1 | Single granularity is inferior |

### Key Findings

- MIT contributes the most (+1.9/+2.1 mIoU), indicating that narrowing the modality gap is the primary bottleneck for AVS performance improvement. The text bridge effectively reduces audio-visual heterogeneity.
- SC and CDCL each contribute independently, but their combination with MIT yields the best results—suggesting that debiasing (SC+CDCL) requires a well-aligned representation space (MIT) to be fully effective.
- Orthogonal counterfactuals significantly outperform random dropout (+2.3/+2.9 mIoU), demonstrating that "semantic orthogonality" is a more effective counterfactual construction strategy than random discarding.
- Multi-granularity text representations outperform single frame-level representations—global semantics (video-level) are important for understanding overall audio-visual correspondence, which frame-level information alone cannot capture.
- Improvements are more pronounced in multi-source scenarios (MS3: +4.4 vs. S4: +3.8 mIoU), as modal ambiguity is more severe with multiple sound sources, making counterfactual debiasing more impactful.

## Highlights & Insights

- **Implicit text bridging avoids errors inherent in explicit text generation**: Rather than generating natural language descriptions (where captioning itself introduces errors), the method operates directly in CLIP's embedding space using learnable pseudo-text. This implicit approach leverages the pretrained model's semantic space while sidestepping the quality bottleneck of text generation.
- **Orthogonal counterfactuals offer an elegant approach to bias elimination**: Compared to methods that explicitly modify textual structures or attributes, this approach is more natural and requires no manually defined counterfactual rules. Orthogonal decomposition in latent space is a mathematically clean operation that can be transferred to other multimodal tasks requiring modality preference elimination (e.g., language bias in visual question answering).
- **The collaborative design of factual-counterfactual and cross-modal contrastive losses is insightful**: Treating "alignment" and "debiasing" as two orthogonal objectives optimized by separate contrastive losses is more effective than pursuing both simultaneously with a single loss.

## Limitations & Future Work

- The method depends on the quality of CLIP's pretrained text space—if CLIP's semantic space provides poor coverage of certain audio categories (e.g., complex ambient sounds), the effectiveness of the MIT bridge may be limited.
- Orthogonal counterfactuals assume that inter-modal dependencies can be captured by linear orthogonal decomposition, which may be overly simplistic for highly nonlinear audio-visual semantic associations.
- Experiments are conducted on benchmarks of limited scale and diversity (AVSBench, AVSS); validation in large-scale real-world scenarios (e.g., autonomous driving, multimedia retrieval) is absent.
- Generating multi-granularity implicit text representations introduces additional computational overhead, which may affect real-time applicability.
- Extending counterfactual learning to the temporal dimension remains unexplored—current counterfactuals operate at the frame or clip level without considering causal relationships across time steps.

## Related Work & Insights

- **vs. AVSegFormer**: A Transformer-based AVS method that enhances audio-visual interaction via attention mechanisms. ICF introduces two additional dimensions—modality alignment and debiasing—beyond interaction efficiency, addressing deeper underlying issues.
- **vs. GAVS**: GAVS employs grounding information to assist AVS. ICF requires no additional grounding annotations, achieving indirect grounding through implicit text bridging, thereby reducing annotation requirements.
- **vs. Counterfactual Learning in VQA**: Counterfactual learning for eliminating language bias in VQA (e.g., CSS) has been established. ICF is the first to introduce this paradigm to AVS and proposes "orthogonal counterfactuals" as a more generalizable alternative to the "attribute-editing counterfactuals" used in prior work.
- The idea of "implicit text as a modality bridge" can be extended to alignment problems in other heterogeneous modality pairs (e.g., tactile-visual, EEG-image).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of implicit text bridging and orthogonal counterfactuals is novel, though counterfactual learning itself has precedents in the vision domain.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Testing on three datasets with thorough ablations; validation on larger-scale data is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clearly motivated contributions; notation for some formulas could be more rigorously defined.
- **Value**: ⭐⭐⭐⭐ Introduces a new research perspective (modality preference elimination) to AVS; the orthogonal counterfactual approach is highly transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust Audio-Visual Segmentation via Audio-Guided Visual Convergent Alignment](../../CVPR2025/segmentation/robust_audio-visual_segmentation_via_audio-guided_visual_convergent_alignment.md)
- [\[ICCV 2025\] TAViS: Text-bridged Audio-Visual Segmentation with Foundation Models](tavis_text-bridged_audio-visual_segmentation_with_foundation_models.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)
- [\[ICCV 2025\] How Do Optical Flow and Textual Prompts Collaborate to Assist in Audio-Visual Semantic Segmentation?](how_do_optical_flow_and_textual_prompts_collaborate_to_assist_in_audio-visual_se.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](../../CVPR2025/segmentation/audio-visual_instance_segmentation.md)

</div>

<!-- RELATED:END -->
