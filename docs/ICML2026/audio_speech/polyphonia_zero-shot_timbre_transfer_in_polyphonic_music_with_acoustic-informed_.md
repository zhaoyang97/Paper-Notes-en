---
title: >-
  [Paper Note] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration
description: >-
  [ICML 2026][Audio & Speech][Timbre Transfer] Polyphonia extends zero-shot timbre transfer from single-track to dense multitrack mixing. It utilizes the Ideal Ratio Mask (IRM) obtained via blind source separation as an ex…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Timbre Transfer"
  - "Attention Calibration"
  - "Ideal Ratio Mask"
  - "Multitrack Mixing"
  - "AudioLDM 2"
date: 2026-05-08
content_hash: 1a5c348f1649adaf
---

# Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration

**Conference**: ICML 2026  
**arXiv**: [2605.10203](https://arxiv.org/abs/2605.10203)  
**Code**: None  
**Area**: Diffusion Models / Music Generation / Zero-shot Editing / Audio Signal Processing  
**Keywords**: Timbre Transfer, Attention Calibration, Ideal Ratio Mask, Multitrack Mixing, AudioLDM 2

## TL;DR
Polyphonia extends zero-shot timbre transfer from single-track to dense multitrack mixing. It utilizes the Ideal Ratio Mask (IRM) obtained via blind source separation as an external acoustic prior to perform "source interpolation + acoustic modulation" within the pre-softmax attention logits. This allows the target stem's (e.g., vocals) spectrum to be replaced with a new timbre (e.g., violin) while strictly preserving the background accompaniment, achieving a 15.5% improvement in target alignment compared to the SOTA.

## Background & Motivation
**Background**: Text-to-music diffusion models (AudioLDM 2, Stable Audio) can generate high-fidelity music from text, but a gap remains for professional production—**refined editing control**. Among these, "stem-specific timbre transfer" (replacing the timbre of one track in a multitrack mix while keeping the rest unchanged) is the most useful yet challenging subtask.

**Limitations of Prior Work**: Existing zero-shot editing approaches fail in two ways. (1) **Vanilla cross-attention methods** (MusicGen, DDPM-Friendly, SDEdit): While cross-attention captures semantics, its spectral resolution is insufficient. In dense mixes, target words and background spectra are entangled, leading to diffuse attention maps and **boundary leakage**, where the background is regenerated along with the target. (2) **Feature preservation methods** (Melodia, SteerMusic, MusicMagus): These use self/cross-attention injection or energy gradients for "rigid preservation." However, in dense mixes, the features to be preserved are inherently entangled; rigid preservation conflicts with editing goals, resulting in **target misalignment** where the target timbre fails to manifest.

**Key Challenge**: Unlike images with opaque pixels where each pixel belongs to "target XOR background" and cross-attention naturally separates them, audio is a **spectral superposition**. The same time-frequency bin can simultaneously carry multiple stems, meaning no binary mask is available—the query vector $Q$ expresses "mixed features" rather than discrete objects. Consequently, cross-attention responds to both target and non-target keys, preventing precise localization.

**Goal**: (1) Identify an objective, zero-shot computable "target spectral envelope" prior to compensate for the insufficient spectral resolution of cross-attention; (2) Use this prior within the attention mechanism to achieve simultaneous "target alignment" and "non-target preservation"; (3) Establish standardized evaluation for stem-specific timbre transfer.

**Key Insight**: Since internal attention is unreliable (Fig. 2(b) left shows diffuse CA maps for vocals even with correct conditions), the focus shifts to external acoustic knowledge. The **Ideal Ratio Mask (IRM)** $G_\text{IRM}=\sqrt{|S_\text{tgt}|^2/(|S_\text{tgt}|^2+|S_\text{con}|^2)}$ from speech enhancement serves as a natural probability-level "target energy ratio," obtainable zero-shot via Blind Source Separation (BSS).

**Core Idea**: Inject the IRM as a soft acoustic prior into the pre-softmax attention logits of the diffusion U-Net. This performs "source interpolation" in Self-Attention/LoA-CA to retain the background and "acoustic modulation" in Text-CA to focus on the target.

## Method

### Overall Architecture
Input: Multitrack mix log-mel spectrogram $X_0\in\mathbb{R}^{T\times F}$ + target prompt $Y_\text{tgt}$ (e.g., "violin"). The base model is AudioLDM 2 (VAE + 16-layer T-UNet with Self-Attention and dual Cross-Attention: Text-CA and Language-of-Audio CA). The dual-path pipeline consists of:

1. **Acoustic Prior Extraction**: Use BSS to decompose $X_0$ into estimated target $\tilde S_\text{tgt}$ and non-target $\tilde S_\text{con}$. Construct $G_{X_0}=\sqrt{\mathcal{M}(|\tilde S_\text{tgt}|^2)/(\mathcal{M}(|\tilde S_\text{tgt}|^2)+\mathcal{M}(|\tilde S_\text{con}|^2))}$ (where $\mathcal{M}$ is the Mel filterbank), downsampled to each LDM layer resolution as $G$.
2. **Inversion**: DDPM inversion projects $X_0$ into the latent space, caching source hidden features $\mathcal{H}(X_0)$ (including the source energy matrix $E_\text{src}$ of SA/LoA-CA).
3. **Edit**: During the T-UNet forward pass, use Acoustic-Informed Attention Calibration to perform: (a) **Source Interpolation** (using $G$ to weighted-fuse current features with $E_\text{src}$ in SA and LoA-CA pre-softmax logits, using the source for background and current for target); (b) **Acoustic Modulation** (using $G \otimes$ target token mask as a bias in Text-CA logits to force attention onto the target spectrum).
4. **Decoding**: Returns to the waveform via the VAE decoder after iterative denoising.

### Key Designs

1. **IRM-based Probabilistic Acoustic Prior $G$ replacing binary mask**:

    - **Function**: Transitions the "target envelope" for audio editing from unreliable internal attention to a robust external BSS-derived probabilistic prior.
    - **Mechanism**: A naive approach $G_\text{norm}=\mathcal{N}(|\tilde S_\text{tgt}|)$ only considers loudness and ignores background energy, leading to high-energy background regions being misidentified as the target. Instead, the **Ideal Ratio Mask** $G_\text{IRM}=\sqrt{|\tilde S_\text{tgt}|^2/(|\tilde S_\text{tgt}|^2+|\tilde S_\text{con}|^2)}\in[0,1]$ represents the "proportion of target energy at a time-frequency point." This naturally suppresses guidance in background-dominant areas. Finally, mapping to the AudioLDM 2 input space via the Mel filterbank yields $G_{X_0}$, downsampled to $G_z^l$.
    - **Design Motivation**: In images, pixels are discrete objects (unique mask), whereas audio time-frequency bins are superpositions where **no binary mask exists**. The probabilistic soft mask provided by IRM respects the physical nature of audio while providing the model with a continuous "what to change vs. what to keep" instruction. Since BSS is pre-trained, the process remains zero-shot.

2. **Selective Pre-Softmax Source Interpolation (SA & LoA-CA)**:

    - **Function**: Strictly preserves the structure and texture of non-target components in self-attention and LoA cross-attention.
    - **Mechanism**: Cache the source attention energy (pre-softmax logit) $E_\text{src}\in\mathcal{H}(X_0)$. During editing, mix via $G$-weighting: $E_\text{mix}=(1-G)\odot E_\text{src}+G\odot Q K^\top/\sqrt{d}$, then apply softmax $\text{Attn}_\text{itp}=\text{softmax}(E_\text{mix})V$. Crucially, mixing occurs in **logit space** rather than averaging probabilities post-softmax. The non-linearity of softmax preserves structural sparsity (which tokens are strong/weak) more effectively, whereas post-softmax mixing smears the distribution and introduces entropy.
    - **Design Motivation**: Traditional prompt-to-prompt methods perform replacement on post-softmax probabilities, which works for images but destroys the sparsity of source attention in audio. Pre-softmax interpolation inherits the source logit peaks in background regions (small $G$) and allows for new Q-K decisions in target regions (large $G$). LoA represents global acoustic texture and requires rigid preservation like SA. Shannon entropy analysis (Fig. 5) shows that pre-softmax interpolation follows source entropy closely in SA and is sharper than post-softmax for LoA.

3. **Acoustic Modulation: IRM as Inductive Bias for Text-CA**:

    - **Function**: Forces the attention of the "target token" in Text-CA onto the spectral regions identified by the IRM, eliminating semantic diffusion.
    - **Mechanism**: Construct a target token mask $\mathbf{m}^\text{text}\in\{0,1\}^{L_y}$ where $\mathbf{m}_i^\text{text}=1$ IFF token $i$ is the target subject (e.g., "violin"). The flattened acoustic prior $\mathbf{g}=\text{Flatten}(G)\in\mathbb{R}^{L_z}$ and $\mathbf{m}^\text{text}$ form a spatio-textual bias $\mathbf{B}=\mathbf{g}\otimes\mathbf{m}^\text{text}\in\mathbb{R}^{L_z\times L_y}$. This is injected into the pre-softmax logit: $E_\text{bias}=Q K^\top/\sqrt{d}+\lambda\cdot\mathbf{B}$. This selectively boosts logits at the intersection of "high target energy latent positions × target semantic tokens."
    - **Design Motivation**: Vanilla cross-attention diffuses in dense mixes, spreading target token attention to the background. Adding the BSS-derived spatio-textual bias ensures the target token is only amplified in its "rightful" spectral region. A scalar $\lambda$ controls modulation intensity, where the continuity of $G$ and non-linearity of softmax create a natural transition between editing and preservation zones.

### Loss & Training
Completely **training-free**: AudioLDM 2 parameters are frozen. All modifications occur in the attention paths during inversion/editing. Algorithm 1 summarizes the pipeline. Pre-trained 4-stem separators like Demucs are used for BSS; non-standard targets (e.g., piano) are handled via "Others" and target-to-stem mapping.

## Key Experimental Results

### Main Results
Evaluated on PolyEvalPrompts: 1,170 editing tasks across MusicDelta and MUSDB18-HQ test sets. Objective metrics: CLAP (text alignment, higher is better), CQT1-PCC (rhythm/melody fidelity, higher is better), LPAPS (perceptual similarity, lower is better), FAD/KAD (quality distribution, lower is better). Subjective metrics (1-5 scale): TTA (Target Timbre Alignment), CTI (Content Temporal Integrity), GAC (Global Audio Coherence).

| Dataset | Method | CLAP↑ | CQT1-PCC↑ | LPAPS↓ | FAD↓ | TTA↑ | GAC↑ |
|--------|------|-------|-----------|--------|------|------|------|
| MusicDelta | SDEdit | 0.119 | 0.090 | 6.907 | 1.914 | 1.13 | 1.46 |
| MusicDelta | MusicGen | 0.377 | 0.069 | 6.142 | 1.331 | 3.59 | 3.62 |
| MusicDelta | Melodia | 0.380 | 0.513 | 3.540 | 0.715 | 3.22 | 3.47 |
| MusicDelta | SteerMusic | 0.317 | **0.556** | 3.614 | 0.738 | 3.16 | 3.32 |
| MusicDelta | **Ours** | **0.437** | 0.547 | 4.096 | 0.949 | **3.80** | **3.69** |

CLAP (target alignment) improved by ~15.5% over the strongest baseline. Subjective TTA/GAC scores were also highest. CQT1-PCC was comparable to the best, indicating background preservation.

### Ablation Study

| Configuration | Change | Phenomenon |
|------|---------|------|
| Full Polyphonia | IRM + Pre-Softmax SI + Acoustic Modulation | Best overall balance across all metrics |
| $G_\text{norm}$ replaces IRM | Normalized magnitude instead of ratio | Massive distortion in high-energy background areas |
| W/O Source Interpolation | Only Acoustic Modulation | Significant loss of background structure (CQT1-PCC drops) |
| W/O Acoustic Modulation | Only SI | Semantic leakage; CLAP and TTA decline |
| Post-Softmax SI | Mixing in probability space | SA entropy rises (structure loss); LoA loses sharpness |
| Separate-Edit-Remix | Independent editing then summation | Lower SongEval coherence; target sounds "detached" from accompaniment |

### Key Findings
- **IRM is superior to $G_\text{norm}$**: Relying solely on target loudness misidentifies "quiet target/loud background" regions, causing non-target distortion. IRM's "target energy ratio" concept is central to non-target integrity.
- **Pre-Softmax injection outperforms Post-Softmax**: Entropy analysis shows Pre-Softmax keeps SA close to the source and keeps LoA sharper, confirming that linear mixing followed by non-linear amplification is the correct order.
- **Separate-Edit-Remix is infeasible**: Summing independently generated waveforms lacks contextual coherence. Holistic editing with IRM guidance is required for acoustic unity.
- **Audio vs. Visual distinctions**: The fundamental difference—binary occlusion masks in images vs. spectral superposition in audio—explains why image editing techniques fail for music.

## Highlights & Insights
- **Unified Diagnosis and Prescription**: The paper clarifies the failure mode of "semantic-acoustic misalignment" (Fig. 2) showing diffuse CA vs. sharp IRM, and then provides a dual-calibration solution.
- **Interdisciplinary adaptation of IRM**: Successfully repurposing IRM from speech enhancement as an injection prior for diffusion models allows decades of BSS research to benefit zero-shot editing.
- **Pre-Softmax injection as a transferable trick**: Any diffusion editing scenario requiring hierarchical control (e.g., regional image editing) can benefit from the pre- vs. post-softmax entropy analysis provided here.
- **PolyEvalPrompts Benchmark**: A standardized set of 1,170 tasks and 10 metrics turns "stem-specific timbre transfer" into a reproducible scientific problem.

## Limitations & Future Work
- **Reliance on external BSS**: BSS models are trained for specific stems (vocals/drums/etc.). Instruments like guzheng or synthesizers fall into "Others," reducing localization precision.
- **Semantic parsing for token masks**: Currently uses rule-based identification; complex prompts may lead to missing modifiers in the token mask.
- **Hyperparameter $\lambda$**: Requires manual tuning and lacks an adaptive mechanism for different instrument pairs.
- **Backbone narrowness**: The method was only demonstrated on AudioLDM 2; generalizability to architectures like Stable Audio remains to be proven.
- **Indirect timbre metrics**: CLAP is an indirect measure of timbre; more specialized music timbre embeddings (e.g., OpenL3) could be utilized.

## Related Work & Insights
- **vs. SDEdit / DDIM Inversion**: Global approaches lack localization, regenerating the background. Polyphonia uses IRM gating to restrict changes to the target spectrum.
- **vs. Melodia / SteerMusic**: These rely on rigid attention injection, which is flawed when attention is already contaminated in dense mixes. This work uses an external IRM for a clean boundary.
- **vs. Instruct-MusicGen**: Polyphonia is zero-shot and training-free, lowering the barrier for engineering.
- **vs. PPAE (Xu 2024)**: While PPAE targets sparse audio events, Polyphonia handles dense multitrack music with much higher overlap complexity.
- **Insight**: Any domain involving "dense multi-source superposition" (e.g., multi-speaker TTS, seismic horizon generation) could benefit from IRM-like soft masks combined with attention bias.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of IRM for diffusion audio editing; dual-path design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 1,170 tasks, 10 metrics, and 7 baselines; thorough ablation but lacks multi-backbone experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is exceptionally clear; excellent use of figures and formulas.
- Value: ⭐⭐⭐⭐ Provides an immediately usable solution for the music production community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MusicDET: Zero-Shot AI-Generated Music Detection](musicdet_zero-shot_ai-generated_music_detection.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[ICML 2026\] VocSim: A Training-Free Benchmark for Zero-Shot Content Identity Recognition of Single-Source Audio](vocsim_a_training-free_benchmark_for_zero-shot_content_identity_in_single-source.md)
- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)

</div>

<!-- RELATED:END -->
