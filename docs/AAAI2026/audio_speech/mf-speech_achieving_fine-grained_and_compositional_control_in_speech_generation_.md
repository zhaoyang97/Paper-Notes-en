---
title: >-
  [Paper Note] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement
description: >-
  [AAAI 2026][Speech Generation] This paper proposes MF-Speech, a framework that employs multi-objective optimization to disentangle speech signals into three high-purity, independent factor representations—content, timbre…
tags:
  - "AAAI 2026"
  - "Speech Generation"
  - "Factor Disentanglement"
  - "Controllable Speech Synthesis"
  - "Contrastive Learning"
  - "Adaptive Style Injection"
date: 2026-05-08
content_hash: b5fc456f9072e716
---

# MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement

**Conference**: AAAI 2026
**arXiv**: [2511.12074](https://arxiv.org/abs/2511.12074)  
**Code**: [GitHub (Demo)](https://guoyang25.github.io/mf-speech/)  
**Area**: Others
**Keywords**: Speech Generation, Factor Disentanglement, Controllable Speech Synthesis, Contrastive Learning, Adaptive Style Injection

## TL;DR

This paper proposes MF-Speech, a framework that employs multi-objective optimization to disentangle speech signals into three high-purity, independent factor representations—content, timbre, and emotion—and subsequently leverages dynamic fusion and Hierarchical Style Adaptive Normalization (HSAN) to achieve fine-grained, compositional control in speech generation, significantly outperforming existing methods on multi-factor compositional speech generation tasks (WER=4.67%, SECS=0.5685).

## Background & Motivation

Controllable speech generation is one of the core objectives in generative AI, with applications spanning emotion-aware assistants, personalized voice restoration, and expressive media synthesis. Voice Conversion (VC), as a key enabling technology, allows flexible manipulation of fundamental speech factors such as content, timbre, and emotion. However, existing methods face two fundamental challenges:

**Challenge 1: Factor Entanglement — Difficulty in Pure Factor Separation.** Content, timbre, and emotion are naturally intertwined in speech and difficult to disentangle. Existing methods (e.g., VQMIVC, StyleVC, StableVC) act as coarse filters, suffering from unclear factor definitions, limited architectural capacity, and incomplete training objectives, resulting in timbre leakage and attribute interference that constrain the transferability of factor representations.

**Challenge 2: Control Failure — Lack of Fine-Grained Control.** Even given relatively pure factor representations, precisely controlling them remains a major challenge. Existing control mechanisms can be categorized into two levels: primitive methods relying on static concatenation and implicit global modulation, and more advanced methods using dynamic fusion and explicit modulation—yet none systematically combines dynamic weighting with hierarchical style injection, making it difficult to balance content fidelity and style similarity.

**Core Idea**: Design a two-stage framework of "factor purifier + speech conductor"—first producing high-purity disentangled representations via multi-objective optimization, then achieving precise multi-factor compositional control via dynamic fusion and HSAN.

## Method

### Overall Architecture

MF-Speech consists of three training stages: Stage 1 trains a high-fidelity waveform-feature conversion module (autoencoder); Stage 2 trains the MF-SpeechEncoder (factor disentanglement encoder); Stage 3 trains the MF-SpeechGenerator (multi-factor controllable generator). The entire system comprises two core components.

### Key Designs

1. **MF-SpeechEncoder — Multi-Factor Speech Encoder (Factor Purifier)**:

    - **Function**: Decomposes raw speech signals into three high-purity, mutually independent discrete representations of content, timbre, and emotion.
    - **Mechanism**: Adopts a three-stream architecture, with dedicated sub-modules for each factor. The **content factor module** uses a pretrained Wav2Vec2 as the backbone to extract initial representations, suppresses residual timbre and emotion information via sentence-level content contrastive learning, and discretizes the output using Residual Vector Quantization (RVQ). The **emotion factor module** employs a two-stage design: a lightweight predictor first explicitly generates F0 and energy representations at intermediate layers (supervised by prosody priors), from which emotion representations are derived and further refined by an emotion contrastive loss. The **timbre factor module** uses a SeaNet encoder and multi-head attention to aggregate global timbre representations, purified by a timbre contrastive loss.
    - **Design Motivation**: Each factor has dedicated contrastive learning for internal purity enhancement, complemented by information-theoretic constraints (mutual information minimization using CLUB and MINE estimators) to penalize inter-factor redundancy. MI constraints are introduced gradually via a warm-up schedule to avoid disrupting early-stage representation learning.
    - Encoder total loss: $\mathcal{L}_{\text{Encoder}} = \sum_{f} \lambda_{com}^f \cdot \mathcal{L}_{com}^f + \sum_{f} \lambda_w^f \cdot \mathcal{L}_w^f + \lambda_p \cdot \mathcal{L}_p + \alpha(\text{epoch}) \cdot \sum_{X,Y} \mathcal{L}_{MI}(X,Y)$

2. **MF-SpeechGenerator — Multi-Factor Speech Generator (Speech Conductor)**:

    - **Function**: Realizes fine-grained and compositional speech generation based on the discrete factor representations produced by the encoder.
    - **Mechanism**: Comprises four collaborative modules:
        - **Dynamic Fusion Module**: Generates time-varying weights via a dynamic gating mechanism to adaptively weight and fuse the discrete representations of content, timbre, and emotion, allowing the model to flexibly modulate the influence of each factor at different time steps.
        - **Style Injection Module**: Acts as a style parameter generator, inferring multi-level style parameters from the discrete timbre and emotion representations for subsequent use by HSAN.
        - **Conditional Generation Module**: Uses stacked residual blocks and multi-scale convolutions as the backbone, with **HSAN (Hierarchical Style Adaptive Normalization)** applied at every layer. HSAN first fuses timbre and emotion representations via cross-attention, then projects to obtain affine parameters $\gamma, \beta$ and a residual modulation term $\alpha$, with the transformation formula: $\mathbf{y} = \text{IN}(\mathbf{x})(1 + \tanh(\gamma)) + \beta + \lambda \tanh(\alpha) \odot \mathbf{x}$
        - **Waveform Synthesis Module**: Uses a SeaNet decoder, fine-tuned with a staged unfreezing strategy.
    - **Design Motivation**: Dynamic fusion addresses the non-adaptability of static concatenation, while HSAN injects style information at every layer to ensure fine-grained control. The two are complementary—the former handles multi-factor coordination and content integrity, while the latter is key to powerful style expression and control.

### Loss & Training

- **Stage 1** (Autoencoder): 92K iterations, batch size 24.
- **Stage 2** (Encoder): 27.5K iterations, batch size 12; total loss includes contrastive losses for three factors, RVQ loss, prosody prior loss, and mutual information minimization loss.
- **Stage 3** (Generator): 91.8K iterations, batch size 72; adversarial training is employed. Generator loss: $\mathcal{L}_{\text{Generator}} = \lambda_{\text{gate}}\mathcal{L}_{\text{gate}} + \lambda_g\mathcal{L}_g + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}} + \lambda_t\mathcal{L}_t + \lambda_f\mathcal{L}_f + \lambda_{\text{sim}}\mathcal{L}_{\text{sim}}$; the discriminator uses hinge loss with a multi-scale discriminator.
- All training is conducted on a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results (Multi-Factor Compositional Speech Generation)

| Method | nMOS↑ | sMOS_t↑ | sMOS_e↑ | SECS↑ | Corr↑ | WER↓ |
|--------|-------|---------|---------|-------|-------|------|
| StyleVC | 2.81 | 2.98 | 2.40 | 0.0985 | 0.48 | 24.83% |
| NS2VC | 3.76 | 3.11 | 3.44 | 0.1552 | 0.55 | 23.33% |
| DDDM-VC | 3.58 | 3.50 | 3.13 | 0.3723 | 0.62 | 11.67% |
| FACodec | 2.83 | 2.38 | 3.14 | 0.1866 | 0.58 | 29.17% |
| **MF-Speech** | **3.96** | **3.86** | **3.78** | **0.5685** | **0.68** | **4.67%** |

### Ablation Study

| Configuration | SECS | WER | Corr | Note |
|---------------|------|-----|------|------|
| Full MF-Speech | 0.5685 | 4.76% | 0.68 | Full model |
| w/o Dynamic Fusion (G1) | 0.5551 | 5.17% | - | Timbre similarity drops, WER increases |
| w/o HSAN (G2) | 0.1576 | - | 0.64 | SECS drops drastically, style control severely degraded |

### Key Findings

- In the multi-factor compositional generation task, MF-Speech achieves state-of-the-art performance on nearly all metrics; its WER (4.67%) is far lower than the second-best DDDM-VC (11.67%), and its SECS (0.5685) leads by a large margin.
- MF-SpeechEncoder achieves best-in-class performance on both target task accuracy (content 0.9593, timbre 0.9979, emotion 0.9296) and non-target information leakage (as low as 0.0054).
- t-SNE visualizations show that removing contrastive learning leads to severe information entanglement, while removing prosody priors causes disordered emotion clustering.
- HSAN is the critical component for style control; its removal causes SECS to drop sharply from 0.5685 to 0.1576.

## Highlights & Insights

- Unlike prior work that conflates emotion and timbre under a generic "style" representation, this paper explicitly models emotion as an independent factor, guided by prosody information (F0 and energy)—a notably elegant design choice.
- In the three-stream architecture, each factor benefits from dedicated contrastive learning combined with shared mutual information minimization, forming a dual guarantee of "internal purification + external decoupling."
- HSAN's design integrates the affine transformation of instance normalization with residual modulation, providing greater expressiveness while preserving normalization stability.

## Limitations & Future Work

- Experiments are conducted solely on the ESD dataset, which is limited in scale; generalization to broader settings remains to be validated.
- On the speech reconstruction task, DDDM-VC still slightly outperforms MF-Speech on most metrics, though the gap is small.
- Performance on the UTMOS metric is inferior to StyleVC and DDDM-VC, suggesting room for improvement in perceptual synthesis quality.
- Whether the discretized factor representations generalize well to other downstream tasks (e.g., speech recognition, emotion detection) is not experimentally verified.
- Inference efficiency and real-time applicability are not discussed.

## Related Work & Insights

- VQMIVC separates factors via vector quantization and mutual information minimization but does not explicitly model F0; StableVC applies gradient reversal layers on top of FACodec but does not explicitly model emotion. The three-factor explicit modeling in this paper represents a systematic advancement over both.
- HierVST proposes hierarchical style injection but lacks dynamic weighting; this paper's HSAN, combining hierarchical injection with dynamic weighting, is a direct improvement.
- The overall paradigm of "disentangle then synthesize" offers reference value for other multi-factor controllable generation tasks, such as image editing and video generation.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](../../ACL2026/audio_speech/segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[AAAI 2026\] Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)

</div>

<!-- RELATED:END -->
