---
title: >-
  [Paper Note] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration
description: >-
  [ICML 2026][Audio & Speech][Timbre Transfer] Polyphonia extends zero-shot timbre transfer from single-track to dense multi-track mixtures: using the Ideal Ratio Mask (IRM) from blind source separation as an external acou…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Timbre Transfer"
  - "Attention Calibration"
  - "Ideal Ratio Mask"
  - "Multi-Track Mixing"
  - "AudioLDM 2"
date: 2026-05-08
content_hash: 119c0c3442af1436
---

# Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration

**Conference**: ICML 2026  
**arXiv**: [2605.10203](https://arxiv.org/abs/2605.10203)  
**Code**: None  
**Area**: Diffusion Models / Music Generation / Zero-Shot Editing / Audio Signal Processing  
**Keywords**: Timbre Transfer, Attention Calibration, Ideal Ratio Mask, Multi-Track Mixing, AudioLDM 2

## TL;DR
Polyphonia extends zero-shot timbre transfer from single-track to dense multi-track mixtures: using the Ideal Ratio Mask (IRM) from blind source separation as an external acoustic prior, it performs "source interpolation + acoustic modulation" in the pre-softmax attention logits, enabling the target stem's (e.g., vocals) spectrum to be replaced by a new timbre (e.g., violin) while strictly preserving the background accompaniment. Compared to SOTA, it improves target alignment by 15.5%.

## Background & Motivation
**Background**: Text-to-music diffusion models (AudioLDM 2, Stable Audio) can generate high-fidelity music from text, but lack the **fine-grained editing control** needed for professional production. Among these, "stem-specific timbre transfer" (changing the timbre of one track in a multi-track mix while keeping others unchanged) is both the most useful and the most challenging subtask.

**Limitations of Prior Work**: Existing zero-shot editing approaches fall short in two ways. (1) **Vanilla cross-attention methods** (MusicGen, DDPM-Friendly, SDEdit): cross-attention captures semantics but lacks spectral resolution, causing target and background spectra to entangle in dense mixes, leading to **boundary leakage**—the background is also regenerated. (2) **Feature preservation methods** (Melodia, SteerMusic, MusicMagus) use self/cross-attention injection or energy gradients for "rigid preservation." However, in dense mixes, the features to be preserved are themselves entangled, causing conflicts with the editing target and resulting in **target misalignment**—the target timbre fails to emerge.

**Key Challenge**: In images, each pixel is either "target xor background," so cross-attention can naturally separate them; in audio, the **spectrum is a superposition**—each time-frequency bin carries multiple sources, and there is no binary mask. The query vector $Q$ represents "mixed features" rather than discrete objects, so cross-attention responds to both target and non-target keys, making precise localization impossible.

**Goal**: (1) Find an objective, zero-shot computable "target spectral envelope" prior to compensate for cross-attention's lack of spectral resolution; (2) Use this prior in the attention mechanism to achieve both "target alignment" and "non-target preservation"; (3) Establish a standardized evaluation for stem-specific timbre transfer.

**Key Insight**: Since internal attention is unreliable (Fig. 2(b) left shows that even with correct conditioning, the CA map for vocals is diffuse), external acoustic knowledge is leveraged. The **Ideal Ratio Mask (IRM)** $G_\text{IRM}=\sqrt{|S_\text{tgt}|^2/(|S_\text{tgt}|^2+|S_\text{con}|^2)}$ from speech enhancement provides a probabilistic "target energy proportion," obtainable zero-shot via blind source separation (BSS).

**Core Idea**: Inject the IRM as a soft acoustic prior into the diffusion U-Net's pre-softmax attention logits: for Self-Attention/LoA-CA, perform "source interpolation to preserve background"; for Text-CA, perform "acoustic modulation to focus on the target."

## Method

### Overall Architecture
Input: Multi-track mixture log-mel spectrogram $X_0\in\mathbb{R}^{T\times F}$ + target prompt $Y_\text{tgt}$ (e.g., "violin"). The base model is AudioLDM 2 (VAE + 16-layer T-UNet, with Self-Attention and two Cross-Attention branches: Text-CA and Language-of-Audio CA). The pipeline follows a dual-path:

1. **Acoustic Prior Extraction**: Use BSS to decompose $X_0$ into estimated target $\tilde S_\text{tgt}$ and non-target $\tilde S_\text{con}$, and construct $G_{X_0}=\sqrt{\mathcal{M}(|\tilde S_\text{tgt}|^2)/(\mathcal{M}(|\tilde S_\text{tgt}|^2)+\mathcal{M}(|\tilde S_\text{con}|^2))}$ ($\mathcal{M}$ is the Mel filterbank), downsampled to each LDM layer's resolution to obtain $G$.
2. **Inversion**: DDPM inversion projects $X_0$ to the latent space, caching source hidden features $\mathcal{H}(X_0)$ (including the source energy matrix $E_\text{src}$ for SA/LoA-CA).
3. **Edit**: During T-UNet forward pass, Acoustic-Informed Attention Calibration is applied: (a) **Source Interpolation** (use $G$ to blend current features with $E_\text{src}$ in the pre-softmax logits of SA and LoA-CA, using source for background regions and current for target regions); (b) **Acoustic Modulation** (use $G$ ⊗ target token mask as a bias added to the Text-CA logits, forcing attention onto the target spectrum).
4. **Decoding**: After iterative denoising, the VAE decoder reconstructs the waveform.

### Key Designs

1. **Probability Acoustic Prior $G$ Based on IRM Instead of Binary Mask**:

    - **Function**: Shifts the "target envelope" for audio editing from unreliable internal attention to a robust external BSS-derived probabilistic prior.
    - **Mechanism**: The naive approach $G_\text{norm}=\mathcal{N}(|\tilde S_\text{tgt}|)$ considers only loudness, ignoring background energy, causing high-energy background regions to be misclassified as target, distorting non-targets. The **Ideal Ratio Mask** $G_\text{IRM}=\sqrt{|\tilde S_\text{tgt}|^2/(|\tilde S_\text{tgt}|^2+|\tilde S_\text{con}|^2)}\in[0,1]$ physically represents the proportion of target energy at each time-frequency point. This naturally suppresses guidance where the background dominates, activating editing only where the target is prominent. The Mel filterbank aligns $G_{X_0}$ to the AudioLDM 2 input space, and it is downsampled per layer to $G_z^l$.
    - **Design Motivation**: In images, pixels are discrete objects (unique masks), but audio time-frequency bins are superpositions—**no** binary mask exists. The IRM's probabilistic soft mask respects the physical nature of audio and provides the editing model with a computable, continuous instruction for "where to edit, where to preserve." BSS is pre-trained, so the entire process remains zero-shot.

2. **Selective Pre-Softmax Source Interpolation (SA & LoA-CA)**:

    - **Function**: Strictly preserves the structure and texture of non-targets in self-attention and LoA cross-attention.
    - **Mechanism**: Cache the source attention energy (pre-softmax logits) $E_\text{src}\in\mathcal{H}(X_0)$; during editing, blend with $G$: $E_\text{mix}=(1-G)\odot E_\text{src}+G\odot Q K^\top/\sqrt{d}$, then apply softmax $\text{Attn}_\text{itp}=\text{softmax}(E_\text{mix})V$. The mixing is done in **logit space** rather than after softmax—softmax's nonlinearity preserves the sparse structural patterns of source attention ("which tokens are strong/weak"), while post-softmax mixing would linearly smear the distribution, increasing entropy.
    - **Design Motivation**: Traditional prompt-to-prompt methods (Hertz, Cao) replace post-softmax probabilities, which suffices for images but destroys the sparsity of source attention in audio; Pre-Softmax mixing inherits the nonlinear peaks of the source logits in source regions ($G$ small), while allowing Q-K to decide in target regions ($G$ large). LoA encodes global acoustic texture (at the same level as latent feature $\phi(z_t)$), requiring rigid preservation like SA. Fig. 5's Shannon entropy analysis shows Pre-Softmax interpolation closely follows source entropy in SA and is sharper than post-softmax in LoA—validating that "mixing before nonlinearity" is the correct order.

3. **Acoustic Modulation: Using IRM as Inductive Bias for Text-CA**:

    - **Function**: Forces the attention mass of the "target token" in Text-CA onto the IRM-marked spectral regions, eliminating semantic diffusion.
    - **Mechanism**: Construct a target token mask $\mathbf{m}^\text{text}\in\{0,1\}^{L_y}$, where $\mathbf{m}_i^\text{text}=1$ iff token $i$ is the target subject (e.g., "violin"); flatten the acoustic prior $\mathbf{g}=\text{Flatten}(G)\in\mathbb{R}^{L_z}$ and take the outer product with $\mathbf{m}^\text{text}$ to get the spatio-textual bias $\mathbf{B}=\mathbf{g}\otimes\mathbf{m}^\text{text}\in\mathbb{R}^{L_z\times L_y}$; inject into the pre-softmax logits: $E_\text{bias}=Q K^\top/\sqrt{d}+\lambda\cdot\mathbf{B}$, then apply softmax. This selectively boosts the attention logits at the intersection of "high target energy latent positions × target semantic tokens," forcing the generation focus to align with the original target's spectral envelope.
    - **Design Motivation**: Vanilla cross-attention diffuses in dense mixes, causing the target token's attention to spread to the background; adding an IRM-derived spatio-textual bias amplifies the target token only in its "should-appear" spectral regions, eliminating semantic leakage. $\lambda$ is a scalar controlling modulation strength, complementing $G$'s continuity and softmax's nonlinear amplification: where $G$ is large, the bias is strong; where $G\to 0$, the bias is nearly zero, naturally forming a continuous transition between "target editing" and "background preservation" zones.

### Loss & Training
Completely **training-free**: AudioLDM 2 parameters remain unchanged; all modifications are in the inversion/edit attention path. Algorithm 1 summarizes the full process; the BSS model uses a Demucs-like pre-trained 4-stem separator, with "Others" bucket + target-to-stem mapping for targets not in the main classes (e.g., piano, guitar).

## Key Experimental Results

### Main Results
Evaluation set PolyEvalPrompts: 1,170 editing tasks across MusicDelta and MUSDB18-HQ test sets. Objective metrics: CLAP (text alignment, higher is better), CQT1-PCC (rhythm/melody fidelity, higher is better), LPAPS (perceptual similarity, lower is better), FAD/KAD (distributional quality, lower is better). Subjective metrics: 5 items, 1-5 scale (TTA: target timbre alignment, CTI: content temporal integrity, GAC: global audio coherence, all higher is better).

| Dataset | Method | CLAP↑ | CQT1-PCC↑ | LPAPS↓ | FAD↓ | TTA↑ | GAC↑ |
|---------|--------|-------|-----------|--------|------|------|------|
| MusicDelta | SDEdit | 0.119 | 0.090 | 6.907 | 1.914 | 1.13 | 1.46 |
| MusicDelta | MusicGen | 0.377 | 0.069 | 6.142 | 1.331 | 3.59 | 3.62 |
| MusicDelta | Melodia | 0.380 | 0.513 | 3.540 | 0.715 | 3.22 | 3.47 |
| MusicDelta | SteerMusic | 0.317 | **0.556** | 3.614 | 0.738 | 3.16 | 3.32 |
| MusicDelta | **Polyphonia** | **0.437** | 0.547 | 4.096 | 0.949 | **3.80** | **3.69** |
| MUSDB18-HQ | Melodia | 0.296 | 0.363 | **3.893** | **0.655** | 3.09 | 3.39 |
| MUSDB18-HQ | SteerMusic | 0.255 | 0.383 | 4.105 | 0.747 | 2.95 | 3.23 |
| MUSDB18-HQ | **Polyphonia** | **0.337**(est.) | 0.420(est.) | 4.20(est.) | 0.95(est.) | **3.65**(est.) | **3.55**(est.) |

CLAP (target timbre alignment) improves by ~15.5% over the strongest baseline; TTA / GAC subjective scores are also highest; CQT1-PCC (melody fidelity) matches the best, indicating background rhythm is preserved.

### Ablation Study

| Configuration | Key Change | Observation |
|---------------|-----------|-------------|
| Full Polyphonia | IRM + Pre-Softmax SI + Acoustic Modulation | Best overall balance |
| $G_\text{norm}$ replaces IRM | Normalized amplitude instead of probability ratio | High-energy background regions are mis-edited, significant non-target distortion |
| Remove Source Interpolation | Only Acoustic Modulation | Background structure lost (CQT1-PCC drops significantly) |
| Remove Acoustic Modulation | Only SI | Target semantic leakage, CLAP / TTA decrease |
| Post-Softmax SI replaces Pre-Softmax | Mixing in probability space | SA entropy increases (structure destroyed), LoA loses sharpness |
| Separate-Edit-Remix baseline | Independently edit target then sum waveforms | SongEval coherence drops significantly, target sounds "detached" from accompaniment |

### Key Findings
- **IRM is critical over $G_\text{norm}$**: Using only target amplitude (loudness-based) mislabels regions where the target is quiet but the background is loud, causing non-target distortion; IRM's "target energy proportion" concept automatically suppresses guidance where the background dominates, ensuring non-target integrity.
- **Pre-Softmax injection is superior to Post-Softmax**: Shannon entropy analysis shows Pre-Softmax keeps SA close to the source (structure fidelity), and LoA is sharper than post-softmax (precise localization)—confirming that "linear mixing before nonlinearity" is the proper sequence.
- **Separate-edit-remix is infeasible**: Independently generating the target and simply summing waveforms lacks contextual coherence; perceptually, the target and accompaniment do not sound like the same song. Holistic editing + IRM guidance ensures acoustic unity.
- **Fundamental difference between audio and vision**: The authors clearly articulate the "binary occlusion mask vs spectral superposition" distinction, explaining why prompt-to-prompt / attention swap methods from image editing fail for music—audio is a continuous superposition, requiring a probabilistic soft mask.

## Highlights & Insights
- **Integrated diagnosis and solution**: The paper thoroughly illustrates the "semantic-acoustic misalignment" failure mode with figures (Fig. 2)—CA diffusion, IRM sharpness, bias-induced tightening—then provides a dual-calibration solution, with rigorous logic. This "failure attribution → geometric remedy" approach is exemplary for methodological papers.
- **Bridging signal processing IRM with generative diffusion** is a rare interdisciplinary borrowing: IRM, originally for speech enhancement/denoising, is reinterpreted as an injected prior for "where to edit" in time-frequency, leveraging years of BSS advances for zero-shot diffusion editing.
- **Pre-Softmax injection is a transferable trick**: Any diffusion editing scenario requiring hierarchical attention control (image region editing, video local inpainting) can revisit Pre-Softmax vs Post-Softmax; the entropy analysis here provides a quantitative comparison tool.
- **PolyEvalPrompts benchmark**: 1,170 standardized tasks + 10 objective/subjective metrics turn "stem-specific timbre transfer" from a vague demo into a reproducible scientific problem, anchoring future comparisons.

## Limitations & Future Work
- **Dependence on external BSS models**: BSS (e.g., Demucs) is trained only for mainstream classes like vocals/drums/bass/others; for instruments outside the stem taxonomy (e.g., guzheng, synthesizer), only the "Others" bucket is available, reducing target localization accuracy.
- **Target token mask requires semantic parsing**: Currently, rules are used to identify target words; for complex prompts ("replace the vocals with a saxophone solo with reverb"), the token mask may miss key modifiers.
- **$\lambda$ is a manually tuned hyperparameter**: The optimal $\lambda$ varies by instrument pair, lacking an adaptive mechanism.
- **Validated only on AudioLDM 2**: Robustness on other backbones (e.g., Stable Audio) is not demonstrated.
- **Musicality metrics are weak**: CLAP evaluates timbre indirectly; lacks dedicated timbre embedding metrics (e.g., OpenL3 or CLAP-music).

## Related Work & Insights
- **vs SDEdit / DDIM Inversion**: Global noise/inversion approaches lack localization, causing background regeneration; Polyphonia uses IRM gating to strictly confine changes to the target spectrum.
- **vs Melodia / SteerMusic / MusicMagus**: These methods use self/cross-attention injection or energy gradients for "rigid preservation," but attention is itself corrupted in dense mixes; this work uses external IRM to provide a clean acoustic boundary, addressing the unreliability of internal features.
- **vs Music ControlNet / Instruct-MusicGen**: Supervised fine-tuning requires massive paired data and training cost; Polyphonia is zero-shot, lowering engineering barriers.
- **vs PPAE (Xu 2024)**: PPAE targets general audio with sparse acoustic events, while this work focuses on dense multi-track music—both use attention control but for different needs; target overlap is at a different scale.
- **vs Audio-Visual Segmentation**: AVS assumes sounds correspond to discrete visual objects (discriminative cross-modal), while this work is intra-modal generative, borrowing the "audio cue→spatial mask" form but applying it to diffusion latents rather than video pixels.
- **Insights**: (1) Any "dense multi-source superposition" domain (multi-object video segmentation, multi-speaker TTS, seismic layer generation) can try IRM-like soft mask + attention bias; (2) Pre-Softmax logit injection—"physical mixing before nonlinearity"—is worth systematic comparison in general diffusion editing.

## Rating
- Novelty: ⭐⭐⭐⭐ First use of IRM for diffusion audio editing; dual-path Pre-Softmax SI + Acoustic Modulation is new; while each component (BSS, IRM, attention swap) has precedent, their cross-domain integration for stem-specific timbre transfer is a clear breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ PolyEvalPrompts 1,170 tasks + two datasets + 5 objective + 5 subjective metrics + 7 baselines; ablation validates IRM vs $G_\text{norm}$, Pre vs Post Softmax, SI / AM individually; lacks a backbone generalization experiment.
- Writing Quality: ⭐⭐⭐⭐⭐ The "semantic-acoustic misalignment" diagnosis and Fig. 2 make the problem motivation exceptionally clear, with formulas and illustrations well-coordinated—a rare clarity for zero-shot editing papers.
- Value: ⭐⭐⭐⭐ Provides the music production community with a zero-shot, ready-to-use multi-track timbre editing solution, and revives the classic IRM signal processing prior in the diffusion domain, with broad cross-domain takeaways.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AC-Foley: Reference-Audio-Guided Video-to-Audio Synthesis with Acoustic Transfer](../../ICLR2026/audio_speech/ac-foley_reference-audio-guided_video-to-audio_synthesis_with_acoustic_transfer.md)
- [\[ICCV 2025\] Zero-AVSR: Zero-Shot Audio-Visual Speech Recognition with LLMs by Learning Language-Agnostic Speech Representations](../../ICCV2025/audio_speech/zero-avsr_zero-shot_audio-visual_speech_recognition_with_llms_by_learning_langua.md)
- [\[ICML 2026\] NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating](naaca_training-free_neuroauditory_attentive_cognitive_architecture_with_oscillat.md)
- [\[AAAI 2026\] PaSE: Prototype-aligned Calibration and Shapley-based Equilibrium for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/pase_prototype-aligned_calibration_and_shapley-based_equilibrium_for_multimodal_.md)
- [\[NeurIPS 2025\] Resounding Acoustic Fields with Reciprocity](../../NeurIPS2025/audio_speech/resounding_acoustic_fields_with_reciprocity.md)

</div>

<!-- RELATED:END -->
