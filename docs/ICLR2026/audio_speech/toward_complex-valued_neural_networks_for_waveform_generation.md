---
title: >-
  [Paper Note] Toward Complex-Valued Neural Networks for Waveform Generation
description: >-
  [ICLR 2026][Audio & Speech][Complex-valued neural networks] This paper proposes ComVo, the first iSTFT vocoder to employ complex-valued neural networks (CVNNs) in both the generator and discriminator. It stabilizes train…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Complex-valued neural networks"
  - "iSTFT vocoder"
  - "phase quantization"
  - "GAN"
  - "waveform generation"
date: 2026-05-08
content_hash: 97e3f0932bb7d2f5
---

# Toward Complex-Valued Neural Networks for Waveform Generation

**Conference**: ICLR 2026
**arXiv**: [2603.11589](https://arxiv.org/abs/2603.11589)
**Code**: [https://hs-oh-prml.github.io/ComVo/](https://hs-oh-prml.github.io/ComVo/)
**Area**: Speech Synthesis / Vocoder
**Keywords**: Complex-valued neural networks, iSTFT vocoder, phase quantization, GAN, waveform generation

## TL;DR
This paper proposes ComVo, the first iSTFT vocoder to employ complex-valued neural networks (CVNNs) in both the generator and discriminator. It stabilizes training via a phase quantization layer and introduces a block-matrix computation scheme that reduces training time by 25%, achieving synthesis quality superior to real-valued baselines such as Vocos on LibriTTS.

## Background & Motivation

**Background**: iSTFT vocoders (e.g., Vocos, iSTFTNet) directly predict complex-valued spectrograms in the frequency domain and synthesize waveforms via iSTFT, avoiding the complexity and latency associated with sample-by-sample generation and learned upsampling.

**Limitations of Prior Work**: All existing iSTFT vocoders rely on real-valued neural networks (RVNNs), treating the real and imaginary parts of the complex spectrogram as two independent channels. This separation breaks the inherent coupling between the real and imaginary parts, which jointly determine magnitude and phase.

**Key Challenge**: RVNNs cannot directly model the algebraic structure of the complex domain (e.g., complex multiplication, rotation), leading to inaccurate phase modeling. Controlled experiments show that CVNNs achieve JSD (Jensen-Shannon Divergence) values 64% and 81% lower than RVNNs for magnitude and phase, respectively, when synthesizing complex-valued distributions.

**Key Insight**: CVNNs represent inputs, activations, and weights as complex numbers, naturally capturing cross-dependencies between real and imaginary parts. However, CVNNs have never been explored in vocoders—primarily due to challenges in designing complex-domain nonlinearities and achieving training efficiency.

**Core Idea**: Construct both the generator and discriminator using CVNNs to form a complete complex-domain adversarial training framework; employ phase quantization as an inductive bias to stabilize training; and adopt a block-matrix computation scheme to improve efficiency.

## Method

### Overall Architecture
Input Mel spectrogram (imaginary part initialized to zero) → complex-valued ConvNeXt generator predicts complex-valued STFT spectrum → iSTFT synthesizes the waveform. The discriminator comprises a complex-valued multi-resolution discriminator (cMRD, operating directly on complex-valued spectra) and a real-valued multi-period discriminator (MPD, operating on waveforms).

### Key Designs

1. **Complex-Valued Generator**:

    - Based on the Vocos architecture, with all Conv1d and LayerNorm layers replaced by complex-valued counterparts.
    - Split GELU activation: GELU is applied separately to the real and imaginary parts of the complex features, preserving the ConvNeXt block structure.
    - End-to-end processing in the complex domain maintains real–imaginary interaction throughout.

2. **Phase Quantization Layer**:

    - For a complex feature $z = re^{i\theta}$, the phase is discretized into $N_q$ uniform levels: $\theta_q = \frac{2\pi}{N_q} \cdot \text{round}(\frac{N_q}{2\pi}\theta)$
    - A straight-through estimator (STE) is used to maintain differentiability.
    - Purpose: constrains the phase range of intermediate representations, serving as regularization against phase drift and guiding the network toward more structured phase patterns.

3. **Complex-Valued Multi-Resolution Discriminator (cMRD)**:

    - Multiple sub-discriminators operate at different STFT resolutions.
    - Takes complex-valued spectrograms directly as input, rather than concatenating real and imaginary parts as separate channels.
    - Adversarial loss is computed separately on the real and imaginary parts, ensuring that gradient feedback respects the complex-domain structure.

4. **Block-Matrix Computation Scheme**:

    - The complex-valued operation $z' = Wz$ (where $W = W_r + iW_i$, $z = x + iy$) is rewritten as $\begin{bmatrix} \text{Re}(z') \\ \text{Im}(z') \end{bmatrix} = \begin{bmatrix} W_r & -W_i \\ W_i & W_r \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}$
    - Four separate real-valued matrix multiplications are fused into a single block-matrix multiplication, reducing redundant computation.
    - Implemented via a custom autograd function, applicable in both forward and backward passes.
    - Reduces training time by approximately 25%.

### Loss & Training
- Adversarial loss: cMRD (complex domain) + MPD (waveform domain)
- Feature matching loss + multi-resolution STFT reconstruction loss
- Trained on LibriTTS train-clean-100/360 + train-other-500 at 24 kHz sampling rate

## Key Experimental Results

### Main Results (LibriTTS test-clean + test-other)

| Model | UTMOS↑ | PESQ↑ | MR-STFT↓ | MOS↑ | CMOS↑ |
|-------|--------|-------|----------|------|-------|
| HiFi-GAN | 3.35 | 2.94 | 1.05 | 4.00 | -0.09 |
| iSTFTNet | 3.36 | 2.81 | 1.10 | 3.98 | -0.04 |
| BigVGAN | 3.52 | 3.61 | 0.90 | 4.00 | -0.01 |
| Vocos (RVNN) | 3.60 | 3.72 | 0.87 | 4.04 | +0.02 |
| **ComVo (CVNN)** | **3.75** | **3.89** | **0.83** | **4.07** | **+0.10** |
| Ground Truth | 3.87 | - | - | 4.08 | +0.14 |

### Ablation Study

| Configuration | UTMOS↑ | PESQ↑ | Notes |
|---------------|--------|-------|-------|
| ComVo (full) | 3.75 | 3.89 | Full CVNN + phase quantization + block matrix |
| w/o phase quantization | 3.63 | 3.75 | Unstable training, phase drift observed |
| w/o cMRD (MPD only) | 3.58 | 3.68 | Lacking complex-domain adversarial feedback |
| RVNN baseline (matched params) | 3.60 | 3.72 | Fair comparison: CVNN advantage is clear |
| Block matrix vs. naïve implementation | Same quality | Same quality | Mathematically equivalent but 25% faster |

### Key Findings
- CVNNs outperform parameter-matched RVNNs on both objective and subjective metrics (UTMOS +0.15, PESQ +0.17).
- Phase quantization is a critical component—its removal causes a 0.12 drop in UTMOS, demonstrating that phase regularization is essential for stable complex-valued training.
- The block-matrix scheme reduces training time by 25% without any loss in synthesis quality.

## Highlights & Insights
- **First fully complex-valued vocoder**—employing CVNNs in both the generator and discriminator, establishing a paradigm for complex-domain adversarial training.
- **Phase quantization** serves as a concise and effective inductive bias—discretizing continuous phases provides the network with "phase anchors," preventing phase inconsistencies during training.
- **Block-matrix computation** cleverly exploits the structured nature of complex-valued operations, transforming an ostensibly inefficient requirement into a computational cost comparable to real-valued implementations.
- Preliminary experiments (GAN for synthesizing complex-valued distributions) validate the theoretical advantages of CVNNs in complex-domain modeling, providing controlled experimental support for subsequent vocoder design decisions.

## Limitations & Future Work
- CVNN layers require approximately 2× the parameters of RVNN layers (separate weights for real and imaginary parts); while the block-matrix optimization improves computation, memory overhead remains higher.
- Rigorous fair comparison is conducted only against Vocos; comparisons with a broader set of vocoders (e.g., BigVGAN v2, Descript Audio Codec) are limited.
- The selection criterion for the phase quantization level $N_q$ is not systematically ablated; the impact of different values on synthesis quality remains unclear.
- Validation within an end-to-end TTS system is absent—it remains to be seen whether vocoder-level improvements translate into gains in overall TTS quality.

## Related Work & Insights
- **vs. Vocos**: Both share the iSTFT framework, but Vocos uses RVNNs while ComVo uses CVNNs—the key difference lies solely in whether the network is natively complex-valued.
- **vs. HiFi-GAN/BigVGAN**: These models generate waveforms directly in the time domain, whereas ComVo generates in the frequency domain followed by iSTFT, representing a fundamentally different architectural paradigm.
- **CVNN literature**: Complex-valued networks have been applied to MRI reconstruction and radar classification; ComVo is the first to introduce them into audio generation.
- **Inspiration**: Any signal processing task where signals naturally exist in complex form (e.g., RF signals, optical imaging) could benefit from a similar CVNN substitution.

## Rating
- Novelty: ⭐⭐⭐⭐ First fully complex-valued iSTFT vocoder; phase quantization design is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers objective and subjective evaluation, controlled experiments, and ablations, though the baseline comparison could be broader.
- Writing Quality: ⭐⭐⭐⭐ The narrative from preliminary experiments to the full system is coherent and well-structured.
- Value: ⭐⭐⭐⭐ Advances the application of complex-valued networks in audio generation and lays the groundwork for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](../../NeurIPS2025/audio_speech/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ACL 2026\] Hard to Be Heard: Phoneme-Level ASR Analysis of Phonologically Complex, Low-Resource Endangered Languages](../../ACL2026/audio_speech/hard_to_be_heard_phoneme-level_asr_analysis_of_phonologically_complex_low-resour.md)
- [\[ACL 2026\] Anchored Cyclic Generation: A Novel Paradigm for Long-Sequence Symbolic Music Generation](../../ACL2026/audio_speech/anchored_cyclic_generation_a_novel_paradigm_for_long-sequence_symbolic_music_gen.md)

</div>

<!-- RELATED:END -->
