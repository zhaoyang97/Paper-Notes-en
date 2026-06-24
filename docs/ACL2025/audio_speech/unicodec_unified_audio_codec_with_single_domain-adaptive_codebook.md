---
title: >-
  [Paper Note] UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook
description: >-
  [ACL 2025][Audio & Speech][Audio Codec] UniCodec proposes a unified audio codec using a single domain-adaptive codebook. Through partitioned domain codebooks and a domain Mixture-of-Experts (MoE) strategy, it achieves outstanding reconstruction and semantic representation performance across three domains: speech, music, and sound.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Audio Codec"
  - "Single Codebook"
  - "Domain-Adaptive"
  - "Mixture of Experts"
  - "Semantic Learning"
date: 2026-05-08
content_hash: e267a3072b3f82a0
---

# UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook

**Conference**: ACL 2025  
**arXiv**: [2502.20067](https://arxiv.org/abs/2502.20067)  
**Code**: [GitHub](https://github.com/Jiang-Yidi/UniCodec)  
**Area**: Audio & Speech  
**Keywords**: Audio Codec, Single Codebook, Domain-Adaptive, Mixture of Experts, Semantic Learning

## TL;DR

UniCodec proposes a unified audio codec using a single domain-adaptive codebook. Through partitioned domain codebooks and a domain Mixture-of-Experts (MoE) strategy, it achieves outstanding reconstruction and semantic representation performance across three domains: speech, music, and sound.

## Background & Motivation

Neural Audio Codecs (NACs) are the cornerstone of audio language models, responsible for mapping continuous waveforms to discrete tokens. Current technologies face the following challenges:

1. **Complexity of Multi-layer RVQ**: Mainstream methods like Encodec and DAC use multi-layer Residual Vector Quantizers (RVQ), generating multiple parallel hierarchical token streams, which increases the decoding complexity and latency of downstream language models.
2. **Difficulty in Unified Modeling with a Single Codebook**: Recent trends shift towards single-layer quantizers (such as WavTokenizer, BigCodec). However, when using a single codebook to handle speech, music, and sound domains simultaneously, performance degrades significantly due to the huge distributional differences among domains. The unified version of WavTokenizer lags significantly behind its domain-specific counterparts in music and audio domains.
3. **Insufficient Semantic Representation**: Discrete tokens often lack high-level semantic information. Existing methods rely on extra pretrained semantic encoders (such as HuBERT) for distillation, which increases training complexity and makes it difficult to support unified multi-domain modeling.
4. **Inherent Contradiction between Reconstruction and Semantics**: Semantic features focus on high-level abstraction, while reconstruction features emphasize fine-grained details. Both need to be optimized simultaneously within a single codebook.

## Method

### Overall Architecture

UniCodec is based on the WavTokenizer architecture, adopting an encoder-quantizer-decoder VQ-VAE structure. The encoder consists of convolutional blocks and Transformer layers, the quantizer uses a single domain-adaptive codebook, and the decoder reconstructs the audio signal. The training is conducted in two stages: the acoustic training stage (reconstruction loss + adversarial loss) and the semantic training stage (adding contrastive learning loss).

### Key Designs

1. **Partitioned Domain-Adaptive Codebook**: The 16,384 codebook entries are partitioned into three dedicated regions: speech domain (indices 0-4095), music domain (4096-8191), and sound domain (8192-16383). The sound domain is allocated more entries because general sounds have a broader distribution. During training, only the codebook entries corresponding to the target domain are updated. During inference, no domain ID is provided, allowing the quantizer to autonomously learn domain features and select the nearest tokens from the entire codebook.

2. **Domain Mixture-of-Experts Encoder (Domain MoE)**: Inspired by DeepSeekMoE, an MoE structure is introduced into the FFN layer of the Transformer encoder. It sets up 1 shared expert ($Ns=1$) and 3 routed experts ($Nr=3$), activating 1 routed expert ($Kr=1$) at each step. Shared experts capture cross-domain general patterns, while routed experts automatically learn domain-specific features through a sigmoid gating mechanism, achieving a balance between efficiency and performance.

3. **Self-Supervised Masked Prediction Semantic Training**: Inspired by Wav2Vec 2.0, a certain proportion of time steps ($p=0.1$, consecutive $M=5$ steps) are randomly masked after the encoder's convolutional output. The model is required to identify the true convolutional latent representation from $K+1$ candidates via contrastive learning. This method enriches semantic information without any extra modules. The acoustic training is finished first to obtain basic reconstruction capability, after which the more challenging masked prediction objective is introduced.

### Loss & Training

- **Acoustic Stage**: Time-domain and frequency-domain reconstruction loss (L1 Mel distance + multi-scale STFT distance) and adversarial loss from multi-resolution discriminators, identical to WavTokenizer.
- **Semantic Stage**: Content loss ($Lm$) is added on top of the acoustic loss, using cosine similarity to calculate the match between the quantized output and unmasked convolutional representations.
- **Fine-tuning Stage**: After training on large-scale data, high-quality data is used for further fine-tuning to improve reconstruction quality (training on massive noisy data significantly hurts reconstruction capability).
- **Training Scale**: Approximately 80,000 hours of data, 32 A800 80G GPUs, AdamW optimizer ($lr=2e-4$). SimVQ is used to replace traditional VQ to improve codebook utilization.

## Key Experimental Results

### Main Results

Objective reconstruction evaluation (Mel distance ↓, lower is better):

| Model | Unified | TPS↓ | Speech Mel↓ | Music Mel↓ | Audio Mel↓ |
|------|------|------|-----------|-----------|-----------|
| DAC (Multi-layer) | ✓ | 600 | 0.3697 | 0.3578 | 0.4581 |
| Encodec (Multi-layer) | ✓ | 600 | 0.5367 | 0.5565 | 0.7601 |
| WavTokenizer (Speech) | ✗ | 75 | 0.5001 | 0.6586 | 0.5990 |
| WavTokenizer (Unified) | ✓ | 75 | 0.5308 | 0.5435 | 0.5193 |
| **UniCodec** | **✓** | **75** | **0.3442** | **0.3959** | **0.3820** |

Detailed indicators in the speech domain (comparison of single-codebook models):

| Model | PESQ↑ | STOI↑ | F1↑ | UTMOS↑ |
|------|-------|-------|-----|--------|
| BigCodec | 2.687 | 0.929 | 0.948 | 4.037 |
| WavTokenizer (Unified) | 1.838 | 0.872 | 0.918 | 3.612 |
| **UniCodec** | **3.027** | **0.949** | **0.949** | **3.987** |

Subjective MUSHRA test:

| Model | Speech↑ | Music↑ | Audio↑ |
|------|-------|-------|-------|
| Ground Truth | 93.52 | 96.18 | 95.28 |
| WavTokenizer (Unified) | 80.40 | 56.10 | 62.21 |
| **UniCodec** | **90.74** | **77.77** | **82.43** |

### Ablation Study

| Configuration | Speech Mel↓ | Music Mel↓ | Audio Mel↓ | Description |
|------|-----------|-----------|-----------|------|
| UniCodec (Full) | 0.3442 | 0.3959 | 0.3820 | Optimal |
| Inference with domain ID | 0.3474 | 0.3912 | 0.3824 | Almost no difference, proving the effectiveness of autonomous codebook learning |
| W/o fine-tuning stage | 0.4476 | 0.4490 | 0.4366 | Fine-tuning with high-quality data is crucial |
| W/o MoE | 0.4883 | 0.4592 | 0.4548 | MoE is crucial for multi-domain modeling |
| W/o partitioned codebook | 0.4873 | 0.5064 | 0.5135 | Partitioned codebook contributes the most, especially in the audio domain |

### Key Findings

1. **UniCodec, as a unified single-codebook model, outperforms domain-specific single-codebook models**: It surpasses WavTokenizer (speech) in the speech domain and WavTokenizer (music/audio) in the music/audio domains, which was previously considered extremely difficult.
2. **It even outperforms multi-layer RVQ models**: UniCodec (75 TPS) outperforms Encodec (600 TPS) and Mimi (100 TPS) across all three domains, achieving better reconstruction while using only 1/8 of the token rate.
3. **Semantic training enhances semantics while maintaining reconstruction quality**: After removing the semantic stage, the classification accuracy on the ARCH benchmark decreases (e.g., RAVDESS 40.28% $\rightarrow$ 36.81%), but reconstruction metrics are barely affected.
4. **Partitioned codebooks do not require domain IDs during inference**: Ablation studies verify that the codebook can autonomously learn domain features. The minor difference in the music domain stems from the mixed nature of vocal and music elements in songs.

## Highlights & Insights

- **Elegant Design Philosophy**: Without relying on extra SSL encoders, diffusion models, or auxiliary modules, UniCodec addresses the twin challenges of multi-domain unification and semantic enhancement within a single-codebook framework using only three strategies: codebook partitioning, MoE, and self-supervised masked prediction.
- **Hypothesis Validation of Partitioned Codebooks**: Domain IDs are used during training but omitted during inference, which validates that the codebook can autonomously learn domain separation—a highly interesting finding.
- **Two-stage Paradigm of Large-scale Data + High-quality Fine-tuning**: It was discovered that while large-scale noisy data helps generalization, it hurts reconstruction. This is compensated for by high-quality fine-tuning, a valuable observation for other audio models.
- **Breakthrough in Compression Rate and Performance**: Outstanding performance over 600 TPS multi-layer models is achieved at 75 TPS (extremely low bitrate).

## Limitations & Future Work

1. **Insufficient Robustness in Noisy Environments**: Performance degradation remains when modeling noisy and overlapping speech.
2. **Performance Drop in Streaming Scenarios**: Streaming usage has been evaluated, but performance degradation was observed; future improvements are needed.
3. **Ceiling of Semantic Density**: Balancing acoustic and semantic density within a single unified codebook remains a challenge, which is a direction worth investigating deeply.
4. **Lack of Integration with LLMs**: The performance of UniCodec-based audio language models on downstream tasks has not yet been explored.
5. **Fixed Number of Domain Partitions**: Currently fixed to speech/music/sound domains, without exploring finer-grained or more flexible domain partitioning.

## Related Work & Insights

- **WavTokenizer** serves as the direct foundation; UniCodec extends it by incorporating partitioned codebooks, MoE, and semantic training.
- The fine-grained experts + shared experts design of **DeepSeekMoE** is successfully transferred to the audio encoder domain.
- The masked prediction paradigm of **Wav2Vec 2.0** is innovatively applied to semantic enhancement of codecs without requiring extra modules.
- The Transformer encoder design from **Mimi Codec (Moshi project)** is also adopted, but UniCodec achieves better performance at a much lower bitrate.
- This work demonstrates that single-codebook solutions for unified multi-domain audio codecs have matured, laying a solid foundation for building general-purpose audio language models.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 5 |
| Engineering Value | 5 |
| Writing Quality | 4 |
| Overall Score | 4.2 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](audio_token_consistency.md)
- [\[ACL 2025\] Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](spark-tts_an_efficient_llm-based_text-to-speech_model_with_single-stream_decoupl.md)
- [\[ACL 2025\] GigaSpeech 2: An Evolving, Large-Scale and Multi-domain ASR Corpus for Low-Resource Languages](gigaspeech2_low_resource_asr.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[NeurIPS 2025\] MGAudio: Model-Guided Dual-Role Alignment for High-Fidelity Open-Domain Video-to-Audio Generation](../../NeurIPS2025/audio_speech/model-guided_dual-role_alignment_for_high-fidelity_open-domain_video-to-audio_ge.md)

</div>

<!-- RELATED:END -->
