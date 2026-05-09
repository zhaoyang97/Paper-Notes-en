---
title: >-
  [Paper Note] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation
description: >-
  [NeurIPS 2025][Audio & Speech][Simultaneous Speech Translation] This paper proposes SimulMEGA, a framework combining prefix training with a Mixture-of-Experts (MoE) refinement module to achieve unsupervised read/write policy learning. A 500M-parameter model achieves <7% BLEU degradation at 1.5-second latency across simultaneous speech translation in 6 languages, and extends to streaming TTS.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Simultaneous Speech Translation
  - MoE
  - Unsupervised Policy Learning
  - Streaming TTS
  - Multilingual Translation
date: 2026-05-08
content_hash: b4656b3ecf7dca32
---

# SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation

**Conference**: NeurIPS 2025
**arXiv**: [2509.01200](https://arxiv.org/abs/2509.01200)
**Code**: [GitHub](https://github.com/nethermanpro/simulmega)
**Area**: Speech Translation, Simultaneous Interpretation, Mixture of Experts
**Keywords**: Simultaneous Speech Translation, MoE, Unsupervised Policy Learning, Streaming TTS, Multilingual Translation

## TL;DR
This paper proposes SimulMEGA, a framework combining prefix training with a Mixture-of-Experts (MoE) refinement module to achieve unsupervised read/write policy learning. A 500M-parameter model achieves <7% BLEU degradation at 1.5-second latency across simultaneous speech translation in 6 languages, and extends to streaming TTS.

## Background & Motivation
- Simultaneous Speech Translation (SimulST) requires jointly optimizing speech recognition and machine translation under strict latency constraints.
- Existing systems struggle to balance translation quality, latency, and semantic coherence.
- In multilingual many-to-many translation, read/write strategies vary substantially across language pairs, making unified policy learning difficult.
- Large models such as SeamlessM4T show non-trivial performance degradation (~8%) and require substantial parameter counts (2B).

## Method

### Overall Architecture
SimulMEGA consists of four components: (1) a streaming speech encoder, (2) a text decoder, (3) a global routing gate, and (4) a MoE refinement module.
- Two-stage training: offline pre-training in Stage 1, simultaneous training in Stage 2.

### Key Designs

#### Streaming Speech Encoder
- Hybrid design: 20 Chunk Autoregressive (Chunk-AR) blocks + 4 Non-Autoregressive (NAR) blocks.
- Chunk-AR blocks use cached KV mechanisms to improve inference efficiency.
- NAR blocks capture global context to ensure translation quality.
- A learnable End-of-Stream (EoSt) token is prepended before the NAR blocks.

#### MoE Refinement Module (training only)
- $N_{refiner}=6$ layers, each containing two experts:
    - **Prefix Expert $E_p$**: standard cross-attention over prefix encodings.
    - **Global Expert $E_g$**: two-layer MLP over temporally average-pooled global embeddings (information bottleneck design).
- **Global Routing Gate**: two-layer MLP + Sigmoid outputting $p \in [0,1]$ to determine expert weights.
    - Small $p$ → prefix information is sufficient → Write.
    - Large $p$ → more input is needed → Read.

#### Preventing Global Information Leakage
- Previous-Output Attention replaces self-attention, so each position only attends to prior decoder outputs.

### Loss & Training
**Stage 1** (offline pre-training):
- Standard S2TT objective $\mathcal{L}^{offline}$.
- Chunk-AR blocks use LoRA ($\alpha=64$) to preserve Whisper encoder capabilities.
- 1 million training steps.

**Stage 2** (simultaneous training):
$$\mathcal{L}^{total} = \mathcal{L}^{offline} + 0.2 \cdot \mathcal{L}^{refiner} + 0.2 \cdot \mathcal{L}^{prefix} + 0.01 \cdot \mathcal{L}^{norm}$$
- $\mathcal{L}^{refiner}$: cross-entropy loss of the MoE refiner → learns read/write policy.
- $\mathcal{L}^{prefix}$: reinforces prefix translation capability (applied only at confident positions where $p < \lambda$).
- $\mathcal{L}^{norm}$: routing score normalization → promotes consistency across tasks and languages.
- Pre-Sigmoid Gaussian noise ($\sigma=1$) encourages discretization of the gating decisions.

#### Inference Policy
$$\text{Action} = \begin{cases} \text{Write} & \text{if } p_{t,i} < \lambda \\ \text{Read} & \text{otherwise} \end{cases}$$
The MoE refinement module is not used at inference time, incurring zero additional overhead.

## Key Experimental Results

### Main Results: Offline BLEU Comparison

| Model | Parameters | CoVoST2 X-EN | CoVoST2 EN-X | Fleurs X-X |
|-------|-----------|-------------|-------------|------------|
| SeamlessM4T Large-v2 | 1.5B | 38.3 | 40.8 | 19.6 |
| S2T Base (Ours) | 561M | 37.0 | 38.9 | 25.1 |
| Seamless-S2T (simultaneous) | 2.0B | 35.3 (-7.8%) | 37.6 (-7.8%) | 18.1 (-7.7%) |
| **SimulMEGA-S2T** | **561M** | **36.9 (-0.3%)** | **38.5 (-1.0%)** | **24.7 (-1.7%)** |

### Streaming TTS Comparison

| Method | LibriSpeech WER | SIM | AL | SeedTTS WER |
|--------|----------------|-----|----|----|
| CosyVoice2-ZS | 2.44 | 0.658 | 22.3 | 1.62 |
| CosyVoice2-S-ZS | 5.31 | 0.651 | 18.5 | 7.98 |
| **SimulMEGA-TTS** | **2.54** | **0.661** | **1.2** | **1.90** |

### Ablation Study

| Design Choice | Effect |
|--------------|--------|
| No pre-Sigmoid noise | Score range too wide; unstable low-latency performance |
| $\sigma=3$ noise | Routing scores become overly deterministic; reduced flexibility |
| No score normalization | Scores cluster in 0.5–0.8; separate threshold tuning required per task/language |
| Remove $\mathcal{L}^{offline}$ | Overall performance drops ~1 BLEU |
| Remove $\mathcal{L}^{prefix}$ | Negligible performance degradation |

### Key Findings
- SimulMEGA consistently outperforms all baselines across all three evaluation settings.
- Only 3–5% degradation at 2-second AL and <3% at 3-second AL.
- Seamless degrades 9–17% under the same conditions.
- SimulMEGA-TTS achieves extreme streaming conditions at the text-unit level (<1.2-second AL) with WER comparable to offline CosyVoice2.
- In end-to-end S2ST, SimulMEGA-S2S incurs less than 200ms additional AL compared to S2TT.

## Highlights & Insights
1. **Zero inference overhead**: The MoE refinement module is used only during training; the inference-time architecture is identical to the offline model.
2. **General framework**: The same framework supports both S2TT and streaming TTS tasks.
3. **Unsupervised policy learning**: The routing gate's $p$ values naturally learn read/write decisions without manual policy design.
4. **Multilingual robustness**: A single threshold configuration generalizes across all language pairs.

## Limitations & Future Work
- Inherent limitations of cascade systems: token mismatch between S2TT and TTS (Whisper vs. Qwen2).
- TTS currently supports only Chinese and English.
- Maximum input duration is 30 seconds; a VAD model is still required for segmentation.
- Future directions: unsegmented continuous generation and end-to-end S2ST systems.

## Related Work & Insights
- Wait-k policies are overly rigid; DiG-SST/ED-ATT/AlignATT exhibit instability.
- Seamless's Monotonic Multihead Attention (MMA) leads to a large performance gap.
- MoE is employed here as a "policy discoverer" rather than a conventional "capacity expander" — a novel usage.

## Rating
- Novelty: ⭐⭐⭐⭐ (novel use of MoE for policy learning)
- Technical Depth: ⭐⭐⭐⭐⭐ (complete system design + comprehensive ablation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (6 languages × multiple baselines × S2TT/TTS/S2ST)
- Writing Quality: ⭐⭐⭐⭐ (clear and thorough)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation](../../AAAI2026/audio_speech/reina_regularized_entropy_information-based_loss_for_efficient_simultaneous_spee.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[NeurIPS 2025\] MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition](mome_mixture_of_matryoshka_experts_for_audio-visual_speech_recognition.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)

</div>

<!-- RELATED:END -->
