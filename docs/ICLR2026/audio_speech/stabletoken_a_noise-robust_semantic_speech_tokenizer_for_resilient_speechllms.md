---
title: >-
  [Paper Note] StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs
description: >-
  [ICLR 2026][Audio & Speech][Semantic speech tokenizer] Addressing the fragility of semantic speech tokenizers where bit-level noise causes dramatic token sequence jumps, StableToken introduces the Voting-LFQ architecture with "multi-branch quantization + differentiable bit-level majority voting" alongside "noise-aware consensus training." This reduces the Unit Edit Distance under noise from 26.17% to 10.17% (a 60%+ relative reduction) and significantly boosts the robustness o…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Semantic speech tokenizer"
  - "noise robustness"
  - "bit-level voting"
  - "consensus training"
  - "SpeechLLM"
date: 2026-05-08
content_hash: d19259b074d76dac
---

# StableToken: A Noise-Robust Semantic Speech Tokenizer for Resilient SpeechLLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=17DNmdQ9aU](https://openreview.net/forum?id=17DNmdQ9aU)  
**Code**: https://github.com/Tencent/StableToken  
**Area**: Speech / Audio  
**Keywords**: Semantic speech tokenizer, noise robustness, bit-level voting, consensus training, SpeechLLM

## TL;DR
Addressing the fragility of semantic speech tokenizers where bit-level noise causes dramatic token sequence jumps, StableToken introduces the Voting-LFQ architecture with "multi-branch quantization + differentiable bit-level majority voting" alongside "noise-aware consensus training." This reduces the Unit Edit Distance under noise from 26.17% to 10.17% (a 60%+ relative reduction) and significantly boosts the robustness of downstream SpeechLLMs in ASR/SER/TTS tasks.

## Background & Motivation

**Background**: Modern SpeechLLMs typically use a discrete speech tokenizer to convert continuous audio into token sequences for the LLM. **Supervised semantic tokenizers** (e.g., S3 Tokenizer, CosyVoice2, GLM-4-Voice-Tokenizer) have become mainstream by embedding a VQ quantizer within an end-to-end ASR model to produce low-bitrate, semantically aligned tokens highly compatible with LLMs.

**Limitations of Prior Work**: The authors find that these "semantic" tokenizers are extremely fragile. Even subtle acoustic perturbations—inaudible to humans at high SNR—cause large-scale changes in output token sequences (as shown in Figure 1). These instabilities disrupt speech-text alignment, forcing the LLM to learn from inconsistent input streams, leading to sharp performance degradation in real-world noisy environments.

**Key Challenge**: Vulnerability stems from two root causes. First, **architectural flaws**: single-path quantization lacks fault tolerance, where perturbations near quantization boundaries are inevitably amplified into entirely different tokens. Second, **remote supervision**: standard ASR losses only supervise the final transcript, remaining indifferent to the intermediate token stability. Consequently, models converge to "functionally correct but representationally fragile" solutions.

**Goal**: To address both issues simultaneously: providing architectural redundancy for fault tolerance and introducing explicit supervision for token invariance under noise.

**Key Insight**: While "offline ensemble voting" might seem to improve robustness, it is impractical due to high inference costs, misaligned quantization boundaries across independent models, and coarse token-level voting. Similarly, naive "token-level consistency losses" suffer from unstable gradients on discrete codes. These failures suggest a need for a **synergistic architecture and training design**.

**Core Idea**: Integrate voting inside the quantizer at the **bit level**. This uses multi-branch parallelism and differentiable bit-level majority voting for "inherent fault tolerance," supported by "consensus training" where a minority of branches process noisy inputs and are forced to align with the "clean consensus" of the majority.

## Method

### Overall Architecture
StableToken follows the paradigm of embedding a semantic tokenizer within an end-to-end ASR model. A pretrained speech encoder (Whisper-large-v3) encodes audio into hidden states, followed by mean pooling to obtain compact representations $h \in \mathbb{R}^D$ per timestep. The quantizer then converts $h$ into discrete tokens for an ASR text decoder. The critical modification is replacing the single-path quantizer with a multi-branch **Voting-LFQ module** and applying **noise-aware consensus training**.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Audio"] --> B["Whisper Encoder<br/>+ Mean Pooling → h"]
    B --> C["Voting-LFQ Module<br/>n-way Parallel Projections + Bit-level Voting"]
    B -->|Training: Input Perturbed<br/>h′ to Minority Branches| D["Noise-aware Consensus Training<br/>Multi-view + Consensus Loss"]
    D -.Constraints.--> C
    C --> E["Stable Speech Tokens"]
    E --> F["Text Decoder / Downstream SpeechLLM"]
```

### Key Designs

**1. Voting-LFQ Module: Replacing Fragility with Redundancy via Bit-level Majority Voting**

This design eliminates the "architectural flaw" of single-path quantization. Instead of a direct $h$-to-token mapping, it uses $n$ parallel linear projection layers to create independent "views." For the $i$-th branch, $p_i = W_i h + b_i$ is calculated, and each $p_i$ is binarized into $B_i \in \{-1, +1\}^d$ using the $\mathrm{sign}$ function, with end-to-end training enabled via the **Straight-Through Estimator (STE)**.

Crucially, voting occurs at the **bit level** rather than the token level. During training, the bits from all branches are averaged per dimension $j$ to obtain a soft score $(s_{\text{final}})_j = \frac{1}{n}\sum_{i=1}^{n}(B_i)_j$. This score represents the consensus/confidence for that specific bit. During inference, the final consensus bit is $(B_{\text{final}})_j = \mathrm{sign}((s_{\text{final}})_j)$. These are mapped to $\{0, 1\}$ to form a binary index $k \in \{0, \dots, 2^d-1\}$ (where $d=13$ for a vocabulary of 8192). Using an **odd number of branches** ensures a strict majority rule. Bit-level voting is superior to token-level voting because even if a majority of branches are perturbed, the correct token can still be recovered if bit errors are sparse across the sequence.

**2. Noise-aware Consensus Training: Aligning Noisy Minority to Clean Majority Consensus**

This design addresses the "supervision gap." For each forward pass, a perturbed version $w' = A(w)$ is generated using waveform-level augmentation $A(\cdot)$. These are processed to get $h$ and $h'$. **A random minority subset of $k$ branches ($k < n/2$) receives the noisy representation $h'$, while the remaining $n-k$ majority branches receive the clean representation $h$**. The **consensus loss** supervises the internal representations by penalizing each branch's deviation from the dynamic global average $\bar{p}_{\text{all}} = \frac{1}{n}\sum_{j=1}^{n}p_j$:

$$\mathcal{L}_{\text{consensus}} = \frac{1}{n}\sum_{i=1}^{n}\lVert p_i - \bar{p}_{\text{all}}\rVert_2^2.$$

Since the majority branches see clean input, they anchor $\bar{p}_{\text{all}}$ to the "clean" representation. The noisy minority branches are thus forced to learn representations consistent with the clean consensus, effectively **learning to ignore perturbations**. The loss is applied to the continuous vectors $p_i$ rather than discrete codes, ensuring smooth and effective gradients.

### Loss & Training
The final objective combines the ASR task loss, consensus loss, and standard LFQ regularization:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{ASR}} + \lambda_1 \mathcal{L}_{\text{consensus}} + \lambda_2 \mathcal{L}_{\text{commitment}} + \lambda_3 \mathcal{L}_{\text{codebook}},$$

where $\mathcal{L}_{\text{ASR}}$ is cross-entropy for transcription, $\mathcal{L}_{\text{commitment}}$ aligns hidden states with quantized representations, and $\mathcal{L}_{\text{codebook}}$ ensures uniform codebook usage. The model is pretrained on 150k hours of diverse speech at a 25Hz frame rate, using $N=5$ branches for the main experiments.

## Key Experimental Results

### Main Results
**Tokenizer-level Noise Robustness (UED%, lower is better, FLEURS benchmark):**

| Type | Model | Vocab | Gaussian | Real Noise | Real (OOD) | Avg. |
|------|------|------|------|---------|-----------|------|
| SSL | R-Spin | 2048 | 21.56 | 15.08 | 14.75 | 16.48 |
| Supervised | S3 Tokenizer | 4096 | 35.40 | 23.88 | 24.58 | 26.17 |
| Supervised | GLM-4-Voice-Token. | 16384 | 42.44 | 27.67 | 28.62 | 31.10 |
| Supervised | CosyVoice2 | 6561 | 54.67 | 31.76 | 32.13 | 38.66 |
| **Ours** | **StableToken** | **8192** | **12.93** | **10.65** | **10.96** | **10.17** |

StableToken reduces average UED to 10.17%, a **61% relative reduction** compared to S3 (26.17%). It even outperforms R-Spin (16.48%), a specialized robust SSL model, despite having a larger vocabulary (8192).

**Reconstruction Fidelity (WER↓ / MOS↑, LibriSpeech + SEED):**

| Model | LS-clean WER | LS-other WER | LS-clean MOS | SEED-zh MOS |
|------|--------------|--------------|--------------|-------------|
| GLM-4-Voice-Token. | 4.04 | 9.33 | 4.07 | 4.10 |
| S3 Tokenizer | 5.78 | 13.38 | 3.40 | 3.31 |
| CosyVoice2 | 4.25 | 9.68 | 3.36 | 3.58 |
| **StableToken** | **3.84** | **7.99** | **4.09** | **4.18** |

The leap in robustness does not sacrifice reconstruction quality, achieving the best WER and MOS scores.

**Downstream SpeechLLM (Qwen2.5-3B backbone, CHiME-4 ASR / SEED-TTS):**

| Tokenizer | CHiME-4 Test-Real WER↓ | SEED-TTS-ZH WER↓ | SEED-TTS-ZH MOS↑ |
|-----------|------------------------|------------------|------------------|
| CosyVoice2 | 59.83 | 9.89 | 3.37 |
| GLM-4-Voice | 51.08 | 5.26 | 3.85 |
| **StableToken** | **35.90** | **3.02** | **4.08** |

Downstream ASR on CHiME-4 shows a ~30% reduction in WER compared to the next best baseline. The performance gap widens as noise levels increase.

### Ablation Study
Sequential ablation (UED% averaged over noises, WER on LibriSpeech-Other):

| Config | Gaussian UED | Real (OOD) UED | LS-Other WER | Description |
|------|---------|---------------|--------------|------|
| StableToken (Full) | 12.93 | 10.96 | 4.68 | Full model |
| w/o Consensus Loss | 24.80 | 17.43 | 4.88 | Explicit consistency removed; robustness drops sharp |
| w/o Noise-aware Tr. | 30.77 | 21.51 | 5.52 | Multi-view removed; semantic fidelity degrades |
| w/o Multi-branch | 34.53 | 24.47 | 5.85 | Reverted to single-path; worst performance |

**Impact of Branch Count $N$**: Increasing $N$ from 3 to 5 significantly improves robustness and semantic preservation. $N=7$ offers marginal gains at higher computational cost, so $N=5$ is chosen.

### Key Findings
- **Consensus Loss is the primary contributor**: Its removal causes OOD UED to jump from 10.96% to 17.43%, confirming that "forced explicit alignment" is the core source of stability.
- **Synergistic Components**: Consensus loss ensures stability; noise-aware multi-view training protects semantic fidelity (WER increases without it); and the multi-branch architecture provides the structural foundation for both training and inference-time error correction.
- **Higher Noise, Higher Advantage**: While all tokenizers perform similarly on clean audio, the gap between StableToken and baselines increases significantly as SNR decreases.
- **Bit-level Correction is Verifiable**: Case studies reveal that even when a branch produces a wrong token due to noise, bit-level voting restores the correct token if the bit errors are sparse.

## Highlights & Insights
- **Internalizing Bit-level Voting**: By moving ensemble voting inside the quantizer at the bit dimension, the model avoids the inference overhead and quantization boundary alignment issues of traditional offline ensembles.
- **Architecture/Training Co-design**: Multi-branching is not just about capacity; it provides the structural "views" necessary for the consensus loss to operate.
- **Continuous Consensus Loss**: Applying the loss to continuous pre-quantization vectors avoids the unstable gradients inherent in discrete token constraints, offering a transferable strategy for other discrete representation tasks.
- **Zero-cost Robustness**: Achieving a 60% UED reduction while maintaining SOTA WER/MOS with negligible inference overhead makes it highly attractive for engineering deployment.

## Limitations & Future Work
- Robustness gains rely on waveform-level augmentations $A(\cdot)$ during training. Performance on structural distortions outside the training distribution (e.g., heavy reverberation, packet loss) requires further exploration.
- Bit-level correction assumes "sparse bit errors." The breakdown point where noise becomes too extreme for the majority rule is not yet systematically mapped.
- The vocab index is directly read from $\{0, 1\}^d$. Whether the exponential mapping or bit-defined structure limits codebook utilization/semantic structure warrants further analysis.

## Related Work & Insights
- **Vs Single-path Tokenizers (S3 / CosyVoice2 / GLM-4)**: These models leverage ASR loss but suffer from boundary instability. StableToken introduces fault tolerance via multi-branch voting.
- **Vs Offline Ensembles**: StableToken is computationally efficient and operates at a finer granularity (bits vs tokens).
- **Vs Discrete Consistency**: Avoiding discrete output constraints leads to more stable training gradients.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models](../../ICML2026/audio_speech/focus_then_listen_an_empirical_study_of_plug-and-play_audio_enhancer_for_noise-r.md)
- [\[ICLR 2026\] CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition](ctc-dro_robust_optimization_for_reducing_language_disparities_in_speech_recognit.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[ICLR 2026\] Hierarchical Semantic-Acoustic Modeling via Semi-Discrete Residual Representations for Expressive End-to-End Speech Synthesis](hierarchical_semantic-acoustic_modeling_via_semi-discrete_residual_representatio.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ICML 2026\] Focus Then Listen: An Empirical Study of Plug-and-Play Audio Enhancer for Noise-Robust Large Audio Language Models](../../ICML2026/audio_speech/focus_then_listen_an_empirical_study_of_plug-and-play_audio_enhancer_for_noise-r.md)
- [\[ICLR 2026\] Improving Black-Box Generative Attacks via Generator Semantic Consistency](improving_black-box_generative_attacks_via_generator_semantic_consistency.md)
- [\[ICLR 2026\] Hierarchical Semantic-Acoustic Modeling via Semi-Discrete Residual Representations for Expressive End-to-End Speech Synthesis](hierarchical_semantic-acoustic_modeling_via_semi-discrete_residual_representatio.md)
- [\[ACL 2025\] On the Robust Approximation of ASR Metrics](../../ACL2025/audio_speech/on_the_robust_approximation_of_asr_metrics.md)

</div>

<!-- RELATED:END -->
