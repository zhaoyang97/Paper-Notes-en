---
title: >-
  [Paper Note] Siamese Vision Transformers are Scalable Audio-Visual Learners
description: >-
  [ECCV 2024][Audio & Speech][Audio-Visual Learning] The AVSiam framework is proposed, which uses a single weight-shared ViT backbone to simultaneously process both audio and visual inputs. Combined with a multi-ratio random masking strategy and a dual-objective pre-training scheme (contrastive plus reconstruction), AVSiam achieves state-of-the-art (SOTA) performance on audio-visual classification and retrieval at an extremely low cost (28.9 times faster than MAViL).
tags:
  - "ECCV 2024"
  - "Audio & Speech"
  - "Audio-Visual Learning"
  - "Siamese Networks"
  - "Vision Transformer"
  - "Contrastive Learning"
  - "Masked Autoencoder"
date: 2026-05-08
content_hash: 0726f9377ea75385
---

# Siamese Vision Transformers are Scalable Audio-Visual Learners

**Conference**: ECCV 2024  
**arXiv**: [2403.19638](https://arxiv.org/abs/2403.19638)  
**Code**: [https://github.com/GenjiB/AVSiam](https://github.com/GenjiB/AVSiam)  
**Area**: Audio and Speech  
**Keywords**: Audio-Visual Learning, Siamese Networks, Vision Transformer, Contrastive Learning, Masked Autoencoder

## TL;DR

The AVSiam framework is proposed, which uses a single weight-shared ViT backbone to simultaneously process both audio and visual inputs. Combined with a multi-ratio random masking strategy and a dual-objective pre-training scheme (contrastive plus reconstruction), AVSiam achieves state-of-the-art (SOTA) performance on audio-visual classification and retrieval at an extremely low cost (28.9 times faster than MAViL).

## Background & Motivation

**Background**: Significant progress has been made in audio-visual representation learning recently, with methods like MAEs performing exceptionally well in joint audio-visual modeling. However, existing methods generally rely on independent audio and visual backbones.

**Limitations of Prior Work**: Independent backbones incur high computational and memory costs. For instance, MAViL requires 5120 V100 GPU hours for pre-training, and MBT requires over 48GB of GPU memory, which is unaffordable for many researchers.

**Key Challenge**: Although independent backbones offer good performance, they suffer from low parameter efficiency, lack of scalability, and an inability to flexibly handle missing modalities (audio-only or video-only). Furthermore, modality-specific designs introduce manual priors and inductive biases, which may hinder data-driven representation learning.

**Goal**: To replace independent audio and visual encoders with a single weight-shared ViT backbone to achieve efficient and scalable audio-visual pre-training.

**Key Insight**: Audio spectrograms are 2D signals with structures similar to images; thus, ViTs are fully capable of processing both modalities simultaneously. While existing studies have demonstrated that ViTs can generalize across modalities, the audio-visual field still persistently uses independent backbones.

**Core Idea**: A weight-shared Siamese ViT combined with multi-ratio random masking is sufficient to learn audio-visual representations efficiently, without requiring modality-specific designs.

## Method

### Overall Architecture

AVSiam consists of three components: (1) a weight-shared ViT encoder that processes both audio spectrograms and visual frames; (2) a multimodal fusion layer (2-layer self-attention) that jointly processes audio-visual tokens; and (3) a 6-layer self-attention decoder for reconstructing masked tokens. Pre-training employs a dual-objective approach of contrastive matching and reconstruction, while fine-tuning adopts a mixed modality input strategy.

### Key Designs

1. **Shared-Weight Encoder**:

    - Function: Simultaneously encode visual frames and audio spectrograms using a standard ViT.
    - Mechanism: The visual frame $I \in \mathbb{R}^{H_v \times W_v \times 3}$ is partitioned into $n$ patches to obtain visual embeddings $\mathbf{X}_v \in \mathbb{R}^{n \times d}$; the audio spectrogram $A \in \mathbb{R}^{H_a \times W_a}$ is similarly partitioned into $k$ patches to obtain audio embeddings $\mathbf{X}_a \in \mathbb{R}^{k \times d}$. For the single-channel input of audio, the three-channel projection layer weights of the pre-trained ViT are averaged to obtain the single-channel weights. After encoding, mean pooling is applied to obtain $\mathbf{F}_a \in \mathbb{R}^d$ and $\mathbf{F}_v \in \mathbb{R}^d$.
    - Design Motivation: Audio spectrograms possess a 2D spatial structure similar to images. Weight sharing significantly reduces parameters (100M vs. 164-172M) and GPU memory footprint (10.9G vs. 20.6G), enabling the model to scale to larger datasets and model sizes.

2. **Multi-Ratio Random Masking**:

    - Function: Randomly mask audio and visual patches with different ratios (0%–50%) in each training iteration.
    - Mechanism: Audio-visual instances within each mini-batch are randomly assigned different masking ratios. For an efficient GPU implementation, these are selected from a predefined set of discrete ratios, ensuring that instances with the same number of unmasked tokens can be packed together.
    - Design Motivation: A fixed masking ratio requires a trade-off between efficiency and accuracy—a high ratio saves GPU memory but loses information, whereas a low ratio preserves information but is computationally expensive. The multi-ratio scheme enables the model to learn across various levels of information completeness, yielding more robust representations while also optimizing efficiency (160 V100 hours vs. 362 hours with the optimal fixed ratio).

3. **Dual Objectives & Mixed Modality Finetuning**:

    - Function: Combine contrastive matching and token reconstruction during pre-training; randomly select audio, visual, or audio-visual inputs during fine-tuning.
    - Mechanism: Contrastive loss is defined as $\mathcal{L}_c = -\frac{1}{B}\sum_{i=1}^B \log \frac{\exp(g(\mathbf{F}_a^i, \mathbf{F}_v^i)/\tau)}{\sum_{j=1}^B \exp(g(\mathbf{F}_a^i, \mathbf{F}_v^j)/\tau)}$; reconstruction loss is $\mathcal{L}_{rec} = \frac{1}{B}\sum_{i=1}^B (\tilde{A}^i - A^i)^2 + (\tilde{I}^i - I^i)^2$; and the total loss is $\mathcal{L} = \mathcal{L}_{rec} + \mathcal{L}_c$. During fine-tuning, half of the iterations randomly select a single modality (audio-only or visual-only), while the other half use bimodal inputs.
    - Design Motivation: Contrastive learning pulls matched audio-visual pairs closer, while the reconstruction objective learns finer-grained cross-modal correlations. Mixed-modality fine-tuning ensures that the shared backbone functions robustly even in missing-modality scenarios.

### Loss & Training

- **Pre-training**: Pre-trained on AudioSet-2M using the Adam optimizer ($lr=1e-4$), with two independent loss scalers for contrastive and reconstruction losses.
- **Fine-tuning**: Fine-tuned on AudioSet-20K/2M/VGGSound with learning rates of $1e-4$, $5e-6$, and $5e-5$ respectively, where the classification head's learning rate is 10 to 100 times larger than that of the encoder.
- Initialization is performed using a ViT pre-trained on ImageNet-21K, and the multimodal layers are initialized with the last two layers of the pre-trained ViT.

## Key Experimental Results

### Main Results

| Method | Audio Encoder | Visual Encoder | V100 Hours | Params | AS-20K mAP | AS-2M mAP | VGGSound Acc |
|------|-----------|-----------|---------|-------|-----------|----------|-------------|
| MBT | AST-B | ViT-B | - | 172M | 43.9 | 49.6 | 64.1 |
| AV-MAE | AST-B | ViT-B | 2854 | 179M | - | 50.0 | 64.2 |
| CAV-MAE | AST-B | ViT-B | 672 | 164M | 42.0 | 51.2 | 65.5 |
| MAViL-Stage2 | AST-B | ViT-B | 5120 | 172M | 44.9 | 53.3 | 67.1 |
| **AVSiam-Base** | Shared ViT-B | Shared ViT-B | **177** | **100M** | 41.6 | 50.1 | 64.9 |
| **AVSiam-Base+** | Shared ViT-B | Shared ViT-B | **450** | **100M** | 43.0 | 51.4 | 66.7 |
| **AVSiam-Large** | Shared ViT-L | Shared ViT-L | **310** | **332M** | 44.1 | 52.1 | 67.1 |
| **AVSiam-Huge** | Shared ViT-H | Shared ViT-H | **800** | **672M** | **45.0** | **54.1** | **68.0** |

### Ablation Study

| Configuration | AS-20K mAP | V100 Hours | Description |
|------|-----------|---------|------|
| Fixed 25% mask | 40.8 | 362 | Optimal fixed ratio |
| Fixed 50% mask | 39.5 | 142 | Medium |
| Fixed 75% mask | 38.6 | 120 | Excessive information loss |
| Pure contrastive learning | 40.4 | 510 | No masking, no reconstruction |
| **Multi-ratio masking (Ours)** | **41.3** | **160** | Best performance and efficiency |
| AVSep-Large (Independent Encoders) | 52.0 (A+V) | - | Params 640M, GPU 20.6G |
| **AVSiam-Large (Shared Encoder)** | **52.1 (A+V)** | - | Params 332M, GPU 10.9G |

### Key Findings

- AVSiam-Huge achieves SOTA on all datasets (AS-20K: 45.0 mAP, AS-2M: 54.1 mAP, VGGSound: 68.0%), requiring only 15% of the pre-training time of MAViL.
- The shared encoder significantly outperforms independent encoders on audio-visual retrieval (VGGSound A$\rightarrow$V: 20.4 vs. 12.8 R@1), because the shared encoder projects both modalities into a more unified latent space.
- AVSiam far outperforms CAV-MAE in missing-modality scenarios: visual-only on VGGSound is 46.0% vs. 27.3%, and audio-only is 55.7% vs. 51.8%.
- Utilizing ViT as a shared backbone outperforms AST: AST performs slightly better on audio (+1 mAP), but much worse on visual and audio-visual inputs.

## Highlights & Insights

- **Extreme Simplicity**: No complex, modality-specific designs are used. A single standard ViT processes both modalities, offering a simple yet remarkably effective approach.
- **Scalability**: Benefiting from high parameter efficiency, AVSiam can easily scale to ViT-Huge and larger datasets, with scaling gains surpassing those of independent-encoder methods.
- **Shared Space Advantage**: t-SNE visualizations indicate that AVSiam maps audio and visual features into a space where semantics are more separable and modalities are better aligned, a feat difficult to achieve with independent encoders.
- **Throughput**: AVSiam processes 75.4 samples per second vs. 22.5 for CAV-MAE and 3.84 for MAViL, rendering it nearly 20 times faster.

## Limitations & Future Work

- At the Base scale, a performance gap remains between AVSiam and CAV-MAE/MAViL on AudioSet-2M, which needs to be bridged by scaling data or model sizes.
- Averaging the 3-channel to 1-channel weights for the audio projection layer is a simplistic approach; better adaptation mechanisms may exist.
- The framework has only been validated on classification and retrieval tasks; more complex downstream tasks like audio-visual question answering, event localization, and segmentation have not been explored.
- The range of masking ratios (0%–50%) in multi-ratio masking is fixed, which may not be optimal for all scenarios.

## Related Work & Insights

- **vs. CAV-MAE**: CAV-MAE uses independent AST and ViT encoders, resulting in a high computational cost (672 hours). AVSiam reaches comparable performance using a shared ViT in significantly less time (177 hours), and far outperforms it on retrieval tasks.
- **vs. MAViL**: MAViL is the strongest current method but requires 5120 V100 hours for a two-stage training. AVSiam-Huge outperforms it using only 800 hours, proving that weight sharing combined with an efficient training strategy is far more practical.
- **vs. Independent Encoder Baseline (AVSep)**: Under the same settings, the shared encoder achieves comparable or even better performance with 1.92 times fewer parameters and nearly 2 times less GPU memory, strongly validating the feasibility of the shared paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ Although the idea of weight-sharing is not entirely novel, its systematic validation and engineering implementation in the audio-visual field are highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Very comprehensive, covering classification, retrieval, ablation, scalability, missing modalities, throughput, and visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive efficiency comparison charts, and solid experimental analysis.
- Value: ⭐⭐⭐⭐⭐ Significantly lowers the entry barrier for audio-visual learning, enabling more researchers to participate, with SOTA performance proving that effectiveness does not need to be compromised.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SiNGER: A Clearer Voice Distills Vision Transformers Further](../../ICLR2026/audio_speech/singer_a_clearer_voice_distills_vision_transformers_further.md)
- [\[ECCV 2024\] Listen to Look into the Future: Audio-Visual Egocentric Gaze Anticipation](listen_to_look_into_the_future_audio-visual_egocentric_gaze_anticipation.md)
- [\[ECCV 2024\] CoLeaF: A Contrastive-Collaborative Learning Framework for Weakly Supervised Audio-Visual Video Parsing](coleaf_a_contrastive-collaborative_learning_framework_for_weakly_supervised_audi.md)
- [\[ECCV 2024\] Label-Anticipated Event Disentanglement for Audio-Visual Video Parsing](label-anticipated_event_disentanglement_for_audio-visual_video_parsing.md)
- [\[ACL 2025\] Investigating and Enhancing Vision-Audio Capability in Omnimodal Large Language Models](../../ACL2025/audio_speech/investigating_and_enhancing_vision-audio_capability_in_omnimodal_large_language_.md)

</div>

<!-- RELATED:END -->
