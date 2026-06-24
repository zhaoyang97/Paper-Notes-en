---
title: >-
  [Paper Note] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement
description: >-
  [AAAI 2026][Audio & Speech][Speech Generation] The MF-Speech framework is proposed to disentangle speech signals into high-purity independent content, timbre, and emotion factor representations via multi-objective optimization. It utilizes dynamic fusion and Hierarchical Style Adaptive Normalization (HSAN) to achieve fine-grained, compositional speech generation control, significantly outperforming existing methods on multi-factor compositional speech generation tasks (WER=4.…
tags:
  - "AAAI 2026"
  - "Audio & Speech"
  - "Speech Generation"
  - "Factor Disentanglement"
  - "Controllable Speech Synthesis"
  - "Contrastive Learning"
  - "Adaptive Style Injection"
date: 2026-05-08
content_hash: f271d5c12f03c546
---

# MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement

**Conference**: AAAI 2026  
**arXiv**: [2511.12074](https://arxiv.org/abs/2511.12074)  
**Code**: [GitHub (Demo)](https://guoyang25.github.io/mf-speech/)  
**Area**: Others  
**Keywords**: Speech Generation, Factor Disentanglement, Controllable Speech Synthesis, Contrastive Learning, Adaptive Style Injection

## TL;DR

The MF-Speech framework is proposed to disentangle speech signals into high-purity independent content, timbre, and emotion factor representations via multi-objective optimization. It utilizes dynamic fusion and Hierarchical Style Adaptive Normalization (HSAN) to achieve fine-grained, compositional speech generation control, significantly outperforming existing methods on multi-factor compositional speech generation tasks (WER=4.67%, SECS=0.5685).

## Background & Motivation

Controllable speech generation is one of the core objectives in generative AI, covering application scenarios such as emotion-aware assistants, personalized voice restoration, and expressive media synthesis. Voice Conversion (VC), as a key technology to achieve this goal, allows flexible manipulation of fundamental speech factors including content, timbre, and emotion. However, existing methods face two primary challenges:

**Challenge 1: Genetic Hybridization — Difficulty in Pure Factor Disentanglement**. Content, timbre, and emotion in speech are naturally entangled and hard to separate. Existing methods (such as VQMIVC, StyleVC, StableVC, etc.) function like crude filters. Due to design deficiencies such as unclear factor definitions, limited architecture capability, and incomplete training objectives, they suffer from timbre leakage and attribute interference, thereby limiting the transferability of factor representations.

**Challenge 2: Instruction Failure — Lack of Fine-Grained Control**. Even with relatively pure factors obtained, precisely controlling them remains a significant challenge. Existing control mechanisms can be categorized into two levels: naive methods rely on static concatenation and implicit global modulation, while advanced methods use dynamic fusion and explicit modulation. However, neither systematically combines dynamic weights with hierarchical style injection, leading to a difficult trade-off between maintaining content fidelity and style similarity.

**Core Idea**: Design a two-stage framework consisting of a "factor purifier + speech conductor" — first, use multi-objective optimization to generate high-purity disentangled representations; second, utilize dynamic fusion and HSAN to achieve precise multi-factor compositional control.

## Method

### Overall Architecture

MF-Speech is divided into three training stages: Stage 1 trains high-precision conversion between waveform and features (autoencoder), Stage 2 trains MF-SpeechEncoder (factor disentanglement encoder), and Stage 3 trains MF-SpeechGenerator (multi-factor controllable generator). The entire system consists of two core components.

### Key Designs

1. **MF-SpeechEncoder — Multi-Factor Speech Encoder (Factor Purifier)**:

    - **Function**: Decompiles the raw speech signal into high-purity, mutually independent discrete representations of content, timbre, and emotion.
    - **Mechanism**: A three-stream architecture is adopted, where each factor has a dedicated sub-module. The **content factor module** uses a pre-trained Wav2Vec2 backbone to extract initial representations, suppresses residual timbre and emotion information via sentence-level content contrastive learning, and finally discretizes it using Residual Vector Quantization (RVQ). The **emotion factor module** is designed as a two-stage architecture that first explicitly generates F0 and energy representations at the intermediate layer using a lightweight predictor (supervised by a prosodic prior), and then derives the emotion representation from them, enhanced by an emotion contrastive loss to improve discriminability. The **timbre factor module** utilizes a SeaNet encoder and a multi-head attention mechanism to aggregate the global timbre representation, purified with a timbre contrastive loss.
    - **Design Motivation**: Dedicated contrastive learning is applied to each factor to enhance its own purity, and information-theoretic constraints (Mutual Information minimization, estimated using CLUB and MINE) are introduced to penalize redundant information between factors. The mutual information constraint is gradually introduced via a warm-up schedule to avoid affecting representation learning in the early stages.
    - **Encoder Total Loss**: $\mathcal{L}_{\text{Encoder}} = \sum_{f} \lambda_{com}^f \cdot \mathcal{L}_{com}^f + \sum_{f} \lambda_w^f \cdot \mathcal{L}_w^f + \lambda_p \cdot \mathcal{L}_p + \alpha(\text{epoch}) \cdot \sum_{X,Y} \mathcal{L}_{MI}(X,Y)$

2. **MF-SpeechGenerator — Multi-Factor Speech Generator (Speech Conductor)**:

    - **Function**: Achieves fine-grained and compositional speech generation based on the discrete factor representations produced by the encoder.
    - **Mechanism**: Contains four collaborative modules —
        - **Dynamic Fusion Module**: Generates time-varying weights through a dynamic gating mechanism to adaptively fuse discrete representations of content, timbre, and emotion, allowing the model to flexibly adjust the influence of each factor at different time steps.
        - **Style Injection Module**: Essentially a style parameter generator that infers multi-layer style parameters from the discrete representations of timbre and emotion, which are then used by the subsequent HSAN.
        - **Conditional Generation Module**: Employs stacked residual blocks and multi-scale convolutions as the backbone. Crucially, **HSAN (Hierarchical Style Adaptive Normalization)** is applied at each layer. HSAN first fuses timbre and emotion representations through cross-attention, then projects them to obtain the affine parameters $\gamma, \beta$ and the residual modulation term $\alpha$. The transformation formula is $\mathbf{y} = \text{IN}(\mathbf{x})(1 + \tanh(\gamma)) + \beta + \lambda \tanh(\alpha) \odot \mathbf{x}$.
        - **Waveform Synthesis Module**: Uses a SeaNet decoder, fine-tuned using a stage-wise unfreezing strategy.
    - **Design Motivation**: Dynamic fusion solves the inability of static concatenation to adaptively integrate, while HSAN injects style information at each layer to ensure fine-grained control. The two are complementary — the former is responsible for multi-factor coordination and content integrity, while the latter is key to strong style expressiveness and control.

### Loss & Training

- **Stage 1** (Autoencoder): 92K iterations, batch size 24.
- **Stage 2** (Encoder): 27.5K iterations, batch size 12. The total loss includes contrastive losses for the three factors, RVQ loss, prosodic prior loss, and mutual information minimization loss.
- **Stage 3** (Generator): 91.8K iterations, batch size 72, using adversarial learning. The generator loss is $\mathcal{L}_{\text{Generator}} = \lambda_{\text{gate}}\mathcal{L}_{\text{gate}} + \lambda_g\mathcal{L}_g + \lambda_{\text{feat}}\mathcal{L}_{\text{feat}} + \lambda_t\mathcal{L}_t + \lambda_f\mathcal{L}_f + \lambda_{\text{sim}}\mathcal{L}_{\text{sim}}$, and the discriminator uses the hinge loss of multi-scale discriminators.
- All trained on a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results (Multi-factor Compositional Speech Generation)

| Method | nMOS↑ | sMOS_t↑ | sMOS_e↑ | SECS↑ | Corr↑ | WER↓ |
|------|-------|---------|---------|-------|-------|------|
| StyleVC | 2.81 | 2.98 | 2.40 | 0.0985 | 0.48 | 24.83% |
| NS2VC | 3.76 | 3.11 | 3.44 | 0.1552 | 0.55 | 23.33% |
| DDDM-VC | 3.58 | 3.50 | 3.13 | 0.3723 | 0.62 | 11.67% |
| FACodec | 2.83 | 2.38 | 3.14 | 0.1866 | 0.58 | 29.17% |
| **MF-Speech** | **3.96** | **3.86** | **3.78** | **0.5685** | **0.68** | **4.67%** |

### Ablation Study

| Configuration | SECS | WER | Corr | Explanation |
|------|------|-----|------|------|
| Full MF-Speech | 0.5685 | 4.76% | 0.68 | Full model |
| w/o Dynamic Fusion (G1) | 0.5551 | 5.17% | - | Timbre similarity decreases, WER increases |
| w/o HSAN (G2) | 0.1576 | - | 0.64 | SECS drops significantly, style control severely weakened |

### Key Findings

- In the multi-factor compositional generation task, MF-Speech achieves the optimal performance across almost all metrics, with a WER (4.67%) significantly lower than the second-best DDDM-VC (11.67%), and a leading SECS (0.5685).
- MF-SpeechEncoder achieves the best performance in terms of target task accuracy (content 0.9593, timbre 0.9979, emotion 0.9296) and minimizes non-target task information leakage (as low as 0.0054).
- t-SNE visualizations indicate that removing contrastive learning causes severe information entanglement; omitting prosodic priors leads to disordered emotion clustering.
- HSAN is the critical component for style control, without which SECS plummets from 0.5685 to 0.1576.

## Highlights & Insights

- Unlike prior works that bundle emotion and timbre into a single "style" representation, this work explicitly models emotion as an independent factor guided by prosodic information (F0 and energy), which is an ingenious design.
- In the three-stream architecture, each factor has its own dedicated contrastive learning alongside public mutual information minimization, establishing a double guarantee of "internal purification + external decoupling".
- The design of HSAN combines affine transformations of instance normalization with residual modulation, providing greater expressiveness while ensuring normalization stability.

## Limitations & Future Work

- Experimental validation is limited to the ESD dataset. The scale of this dataset is small, leaving external generalization yet to be verified.
- On speech reconstruction tasks, DDDM-VC still slightly outperforms MF-Speech on most metrics (though the margin is narrow).
- The model underperforms StyleVC and DDDM-VC in terms of the UTMOS metric, indicating room for improvement in perceptual synthesis quality.
- There is a lack of experimental verification on whether the discretized factors perform well in other downstream tasks (e.g., speech recognition, emotion detection).
- Inference efficiency and real-time performance have not been discussed.

## Related Work & Insights

- VQMIVC disentangles factors via vector quantization and mutual information minimization but lacks explicit modeling of F0, while StableVC uses gradient reversal layers on top of FaCodec without explicit emotion modeling — the explicit three-factor modeling in this paper represents systematic progress.
- HierVST proposes hierarchical style injection but lacks dynamic weighting — HSAN in this paper combines hierarchical injection and dynamic weighting, offering a direct improvement.
- The overall paradigm ("disentanglement followed by synthesis") holds reference value for other multi-factor controllable generation tasks (e.g., image editing, video generation).

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SegTune: Structured and Fine-Grained Control for Song Generation](../../ACL2026/audio_speech/segtune_structured_and_fine-grained_control_for_song_generation.md)
- [\[ACL 2026\] FIGMA: Towards Fine-Grained Music Retrieval](../../ACL2026/audio_speech/figma_towards_fine-grained_music_retrieval.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[AAAI 2026\] Factor(U,T): Controlling Untrusted AI by Monitoring their Plans](factorut_controlling_untrusted_ai_by_monitoring_their_plans.md)

</div>

<!-- RELATED:END -->
