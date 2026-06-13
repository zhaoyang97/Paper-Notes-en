---
title: >-
  [Paper Note] Attend to Anything: Foundation Model for Unified Human Attention Modeling
description: >-
  [ICML2026][Audio & Speech][Human Attention] AAM unifies image, video, and audio-visual saliency prediction into an attention foundation model with text conditioning, hyperbolic hierarchical constraints…
tags:
  - "ICML2026"
  - "Audio & Speech"
  - "Human Attention"
  - "Visual Saliency"
  - "Hyperbolic Representation"
  - "Fokker-Planck Dynamics"
  - "Multimodal Foundation Model"
date: 2026-05-08
content_hash: 200604457ceb1064
---

# Attend to Anything: Foundation Model for Unified Human Attention Modeling

**Conference**: ICML2026  
**arXiv**: [2606.03540](https://arxiv.org/abs/2606.03540)  
**Code**: https://github.com/wz-zhao/Attend-to-Anything  
**Area**: Human Understanding / Attention Modeling  
**Keywords**: Human Attention, Visual Saliency, Hyperbolic Representation, Fokker-Planck Dynamics, Multimodal Foundation Model  

## TL;DR
AAM unifies image, video, and audio-visual saliency prediction into an attention foundation model with text conditioning, hyperbolic hierarchical constraints, and Fokker-Planck temporal dynamics. It consistently outperforms specialized models across 16 benchmarks and improves video inference speed to approximately 111 FPS.

## Background & Motivation
**Background**: Human attention modeling is traditionally bifurcated into sub-fields such as image saliency, video saliency, and audio-visual attention. Each branch maintains distinct datasets, model architectures, and training protocols. Image-based methods rely on CNNs or Transformers for static maps; video methods add optical flow, 3D convolutions, or temporal Transformers; and audio-visual methods employ supplemental audio branches to capture speakers or sound sources.

**Limitations of Prior Work**: This fragmentation enables high performance within single datasets but hinders cross-scenario generalization. The paper notes that even when scaling model capacity and data, existing models suffer significant performance degradation in cross-dataset testing. This suggests the bottleneck is not merely lack of samples, but the problem definition itself, which segments a unified human cognitive mechanism into disconnected local tasks.

**Key Challenge**: Human attention is a unified cognitive process, yet current modeling treats scenario differences, task intentions, and modal variations as isolated statistical biases. Models need to express both "universal attention priors" and "task-specific conditions" while integrating static images and dynamic videos into a single inferable framework rather than separate branches for each input form.

**Goal**: The authors aim to construct a reusable attention foundation model for image, video, and audio-visual tasks. It must support text-conditioned control, cross-dataset generalization, frame-by-frame prediction of arbitrary-length video, and reduced redundant computation in video models without sacrificing accuracy.

**Key Insight**: The paper interprets attention differences as a hierarchical entailment relationship from "universal attention" to "task-specific attention," localized within a hyperbolic space. It treats temporal changes in video attention as the transport, diffusion, and correction of probability densities, using the Fokker-Planck equation to bridge static saliency maps and dynamic attention.

**Core Idea**: By unifying scenarios and task conditions via hyperbolic hierarchical semantics and bridging image and video attention through physics-inspired temporal dynamics, the authors reformulate fragmented saliency prediction into a single multimodal conditional foundation model.

## Method
AAM accepts images, video sequences, and audio signals as input and outputs corresponding spatial attention maps. Rather than concatenating specialized models, it maps different datasets, modalities, and tasks into a unified conditional attention space. Visual inputs use a frozen DINOv3 backbone adapted via LoRA; text prompts are encoded by CLIP to describe cognitive conditions; audio is encoded by Wav2CLIP and mapped to visual semantics; videos utilize a Fokker-Planck Dynamics (FPD) module for temporal evolution on top of frame-wise visual representations.

### Overall Architecture
The model extracts visual features, text conditions, and optional audio features. Text and visual representations are lifted into a Lorentz hyperbolic space, where the model learns partial order relations between a universal attention anchor, text condition points, and visual instance points. A hyperbolic decoder regulates multi-scale features and spatial focus weights based on the hierarchical depth of the text condition and the relative direction of the visual instance. For video input, the FPD module performs drift, diffusion, and observation correction on the attention distribution of each frame, ensuring the output preserves current evidence while absorbing temporal consistency. Training utilizes KLD, CC, and SIM saliency losses alongside a hyperbolic entailment loss.

### Key Designs
1. **Hyperbolic Hierarchical Entailment Modeling**:
    - **Function**: Organizes universal attention, task text conditions, and specific visual instances into a hierarchy from general to specific.
    - **Mechanism**: The model learns partial order relations $z_{img} \preceq z_{txt} \preceq z_{anc}$ in a Lorentz hyperbolic space, where $z_{anc}$ is the universal anchor, $z_{txt}$ is the task/dataset text condition, and $z_{img}$ is the visual instance. Through hyperbolic entailment cone constraints, text conditions must fall within the cone permitted by the universal anchor, and visual instances must remain hierarchically consistent with the text condition.
    - **Design Motivation**: Attention conditions are hierarchical cognitive modulations rather than unordered labels. The exponential volume growth of hyperbolic space is better suited for tree-like hierarchical semantics, supporting cross-scenario generalization better than independent parameters for each dataset.

2. **Geometry-Aware Hyperbolic Decoder**:
    - **Function**: Translates hierarchical conditions from hyperbolic space back to pixel-level saliency maps, allowing granular conditions to control different scales and spatial regions.
    - **Mechanism**: The hyperbolic distance from the text point to the origin represents condition specialization depth, used to select multi-scale operator weights $w_k=softmax_k(-d_L(z_{txt},\mu_k))$. The geodesic direction $\Delta$ of the visual instance relative to the text condition is used to calculate spatial focus weights.
    - **Design Motivation**: By mapping hierarchical depth to scale modulation and semantic offset to spatial focus, the geometric decoder better approximates the way human attention narrows or expands based on task objectives compared to simple feature concatenation.

3. **Fokker-Planck Video Attention Dynamics**:
    - **Function**: Extends static frame attention to continuous-time evolution for arbitrary-length video, avoiding the high redundancy of fixed-window models.
    - **Mechanism**: The video attention distribution $u_t$ is treated as a probability density over the spatial domain. Its temporal change consists of drift (aggregating information via bidirectional temporal self-attention), diffusion (smoothing high-frequency noise via second-order central differences), and correction (an adaptive compromise between dynamic prediction and current observation $u_t^{obs}$ via a Kalman-like gain).
    - **Design Motivation**: Visual attention in video shifts with moving targets while requiring temporal continuity and robustness against error propagation. The Fokker-Planck perspective decomposes these requirements into interpretable physical processes that are more efficient than stacked Transformers.

### Loss & Training
The model is trained on Attention-1.75M, comprising 8 image, 4 video, and 6 audio-visual datasets totaling 1.75 million human fixation instances. Training proceeds in stages: starting with image and video data using a free-viewing warm-start for the universal anchor, followed by audio-visual data after 10 epochs. The visual backbone is frozen, with adaptation via LoRA and task-specific heads.

The total loss is $L_{total}=L_{KLD}-L_{CC}-L_{SIM}+L_{HAE}$. $L_{HAE}$ constrains the anchor-to-text and text-to-image entailment relationships. Audio-visual fusion employs correlation-gated cross-attention, strengthening audio contributions only when audio cues align with visual semantics.

## Key Experimental Results

### Main Results
AAM is evaluated across 16 benchmarks. It achieves stable improvements in natural images, webpages, e-commerce, video, and audio-visual scenarios.

| Task/Dataset | Metric | Ours (AAM) | Prev. Strong Baseline | Gain |
| :--- | :--- | :--- | :--- | :--- |
| MIT1003 Image | CC ↑ | 0.831 | SUM 0.768 | +0.063 |
| CAT2000 Image | SIM ↑ | 0.769 | SUM 0.754 | +0.015 |
| SALICON Image | KLD ↓ | 0.163 | SUM 0.192 | -0.029 |
| DIEM Audio-Visual | CC ↑ | 0.710 | TAVDiff 0.670 | +0.040 |
| ETMD Audio-Visual | NSS ↑ | 3.66 | CASP 3.34 | +0.32 |
| DHF1K Video | NSS ↑ | 3.272 | MSFF-Net 3.066 | +0.206 |
| Hollywood2 Video | SIM ↑ | 0.599 | VSSM 0.583 | +0.016 |
| UCF Video | CC ↑ | 0.736 | VSSM 0.705 | +0.031 |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Single-dataset training | Weak cross-dataset generalization | Learns local distributions only; fails to express unified hierarchy. |
| Image joint training | Improved average image results | Shared hierarchical conditions stabilize natural image and e-commerce tasks. |
| Full multimodal training | Stable gains across all tasks | Multimodal data does not disrupt existing tasks, indicating mitigated modal conflict. |
| W/o temporal module | Decreased video results | Frame-wise static prediction lacks temporal transport and smoothing. |
| Standard temporal self-attention | Better than none, worse than FPD | Aggregates information but lacks diffusion/correction structural constraints. |
| FPD temporal module | Best video ablation | Drift, diffusion, and correction jointly improve dynamic stability. |
| W/o hyperbolic learning | Significant drop in complex scenes | Linear features fail to represent general-to-specific cognitive relations. |
| Hyperbolic loss + decoder | Best hyperbolic ablation | Constrains representation structure and injects geometry into decoding. |

### Efficiency Comparison
AAM utilizes frame-by-frame prediction and FPD evolution, avoiding the overhead of multi-frame windowed inputs.

| Method | Backbone | Input Length | FPS | Trainable Parameters |
| :--- | :--- | :--- | :--- | :--- |
| TASED | 3D Conv | Fixed Window | 17 | 82M |
| STSANet | Video Swin | Fixed Window | 28 | 643M |
| TMFI-Net | Video Swin | Fixed Window | 30 | 234M |
| AAM | DINOv3 | Arbitrary | 111 | 21.4M |

### Key Findings
- Average gains are balanced across tasks (~5.2% for image, 5.8% for audio-visual, 6.0% for video), suggesting the unified modeling is not biased toward a specific modality.
- FPD provides benefits in both accuracy and throughput, increasing video inference speed to 111 FPS with only 21.4M trainable parameters.
- Condition generalization tests show that correct task conditions outperform generalized ones, which in turn outperform incorrect ones; low standard deviation across prompt rewordings (CC SD < 0.01) indicates the model responds to semantics rather than specific phrasing.
- As prompt granularity moves from general to specific, some tasks plateau early, reflecting that dominant task contexts are sufficient to determine major fixation patterns.

## Highlights & Insights
- The primary value of this work is re-framing saliency prediction from "dataset-specific regression" to a "unified attention process under cognitive conditioning." This re-formulation is more significant than scaling, as it explains why increased capacity alone fails to solve generalization.
- Hyperbolic space is not merely decorative; it directly maps to the hierarchical hypothesis of attention. It creates an interpretable link between text abstraction, visual instance specialization, and decoding scale.
- The Fokker-Planck module decomposes temporal consistency into drift, diffusion, and correction. This approach is transferable to tasks like video segmentation or dynamic depth estimation requiring continuous temporal smoothing.

## Limitations & Future Work
- The "foundation" aspect is currently limited to the attention modeling family; its direct transferability to broader human-computer interaction or robot policy learning remains to be proven.
- Text conditions are derived from dataset protocols; real-world user intent may be more ambiguous or conflict with visual evidence, necessitating robust testing of open-ended text.
- FPD provides a physical intuition, but its drift and diffusion terms are discrete neural approximations, which may still differ from actual neural eye-movement mechanisms.
- Future work could utilize AAM outputs as attention priors for downstream models, such as region-based VLM prompting or active perception in robotics.

## Related Work & Insights
- **vs UNISAL**: While UNISAL attempted to unify image and video saliency, it relied on task-specific training. AAM extends this to text conditions, hyperbolic hierarchies, and audio-visual data.
- **vs SUM**: AAM interprets cross-dataset differences as a cognitive hierarchy rather than independent statistical domains, avoiding the need for parameter isolation.
- **vs Video Swin/3D Conv models**: Traditional models use fixed windows with heavy computation. FPD formulates temporal change as probability distribution evolution, making arbitrary-length frame-by-wise inference an intrinsic part of the architecture.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Strong problem re-formulation using hyperbolic semantics and Fokker-Planck dynamics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evidence across 16 benchmarks and multiple modalities.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, though some information density in figures is quite high.
- Value: ⭐⭐⭐⭐⭐ Offers a clear unified paradigm for saliency and cognitive-inspired vision models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/audio_speech/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](../../ICLR2026/audio_speech/pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](../../AAAI2026/audio_speech/dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)

</div>

<!-- RELATED:END -->
