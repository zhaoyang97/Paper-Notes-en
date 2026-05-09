---
title: >-
  [Paper Note] Hear What Matters! Text-conditioned Selective Video-to-Audio Generation
description: >-
  [CVPR 2026][Video Understanding][Selective Audio Generation] SelVA introduces the text-conditioned selective video-to-audio (V2A) generation task. Through a learnable supplementary token [SUP] and a self-supervised video mixing strategy, the model generates only the user-specified target sound from multi-source videos guided by text prompts, surpassing existing methods in audio quality, semantic alignment, and temporal synchronization.
tags:
  - CVPR 2026
  - Video Understanding
  - Selective Audio Generation
  - Video-to-Audio
  - Text Conditioning
  - Cross-Modal Attention
  - Self-Supervised Video Mixing
date: 2026-05-08
content_hash: 91b22e9af7b5e109
---

# Hear What Matters! Text-conditioned Selective Video-to-Audio Generation

**Conference**: CVPR 2026  
**arXiv**: [2512.02650](https://arxiv.org/abs/2512.02650)  
**Code**: [https://jnwnlee.github.io/selva-demo/](https://jnwnlee.github.io/selva-demo/)  
**Area**: Video Understanding / Audio Generation  
**Keywords**: Selective Audio Generation, Video-to-Audio, Text Conditioning, Cross-Modal Attention, Self-Supervised Video Mixing

## TL;DR

SelVA introduces the text-conditioned selective video-to-audio (V2A) generation task. Through a learnable supplementary token [SUP] and a self-supervised video mixing strategy, the model generates only the user-specified target sound from multi-source videos guided by text prompts, surpassing existing methods in audio quality, semantic alignment, and temporal synchronization.

## Background & Motivation

1. **State of the Field**: V2A generation has achieved notable progress; models such as MMAudio and ReWaS can generate temporally synchronized audio from video content. Existing methods typically produce a holistic audio track containing all sound sources simultaneously.

2. **Limitations of Prior Work**: In professional audio production (Foley), sound designers need to record each sound source independently and mix them separately. However, existing V2A models can only generate a single mixed audio track; even minor adjustments to one sound require re-synthesizing the entire audio, severely limiting practical usability.

3. **Root Cause**: Existing methods rely on frozen visual encoders that extract features encompassing all objects in the video, including information irrelevant to the target sound. Consequently, the generator cannot selectively produce only the target sound. Text is used merely as auxiliary semantic context rather than as an explicit sound-source selector.

4. **Paper Goals**: (1) How can text serve as an explicit selector to extract visual features relevant solely to the target sound from multi-source videos? (2) How can selective generation capability be trained without single-source annotated data? (3) How can an efficient video encoder fine-tuning strategy be designed to avoid spurious correlations in the attention mechanism?

5. **Starting Point**: Inspired by the human auditory selective attention mechanism—humans can focus on a specific sound source in noisy environments—the model should similarly focus on specific sound sources in video guided by text. The authors also observe the high-norm artifact problem in ViT (attention concentrating on irrelevant tokens) and propose absorbing these spurious attention weights with additional tokens.

6. **Core Idea**: Reposition text prompts as explicit modulators of video features; suppress irrelevant visual activations via learnable [SUP] tokens; and achieve annotation-free selective audio generation through a self-supervised video mixing strategy.

## Method

### Overall Architecture

SelVA consists of two main modules: (1) a text-conditioned video encoder $\mathcal{F}$, and (2) a multimodal-conditioned selective audio generator $\mathcal{G}$. Given video $V$ and text description $T_i$, the model objective is $A_i = \mathcal{G}(\mathcal{F}(V, \mathbf{t}_i), \mathbf{t}_i)$, i.e., generating only the sound corresponding to the text description. Training follows a two-stage strategy: Stage 1 uses teacher-student distillation to train the video encoder to selectively extract features guided by text; Stage 2 freezes the video encoder and trains the audio generator.

### Key Designs

1. **Text-Guided Cross-Attention Block**:

    - **Function**: Lightweight cross-attention layers are inserted into the frozen video encoder (Synchformer) to allow visual features to be modulated by text semantics.
    - **Mechanism**: Cross-attention layers are inserted after the spatial-temporal attention blocks of Synchformer, using the video hidden vector $\mathbf{h_v}$ as Query and text embedding $\bar{\mathbf{t}}$ as Key/Value: $\mathbf{h_{vt}} = \text{Cross-Attn}(Q=\mathbf{h_v}, K=\bar{\mathbf{t}}, V=\bar{\mathbf{t}})$. A learnable spatial attention pooling layer then yields the final video feature $\mathbf{v}$.
    - **Design Motivation**: Unlike prior approaches that freeze visual encoders entirely, parameter-efficient fine-tuning enables the encoder to filter visual information according to textual intent, retaining only semantics relevant to the target sound.

2. **Learnable Supplementary Token [SUP]**:

    - **Function**: Suppresses spurious correlation activations in cross-attention, directing attention toward the target sound source described by the text.
    - **Mechanism**: $N=5$ learnable tokens are prepended to the text embedding: $\mathbf{t}_{\texttt{[SUP]}} = [\texttt{[SUP]} \oplus \mathbf{t}]$. The concatenated sequence serves as Key/Value in cross-attention. These extra tokens absorb high-norm attention weights that would otherwise erroneously fall on non-target motion patches.
    - **Design Motivation**: Naive cross-attention causes the model to attend to non-target object motions in the video (e.g., generating cat sounds because a cat moves nearby). Without [SUP], attention disperses across the entire frame; with [SUP], attention precisely focuses on the target region. Inserting supplementary tokens on the text side is more efficient than adding extra tokens to the visual token sequence, which would increase computation across all encoder layers.

3. **Self-Supervised Video Mixing Strategy**:

    - **Function**: Constructs training samples without single-source annotations, enabling the model to selectively extract target-source visual features from mixed videos.
    - **Mechanism**: Two videos $V_{\text{tar}}$ and $V_{\text{pair}}$ are randomly selected and horizontally concatenated with mixing ratio $\lambda \sim \text{Beta}(\alpha, \alpha)$ to form a mixed video $V = [V_{\text{tar}} \oplus V_{\text{pair}}]$. The audio-text pair of one video is randomly chosen as the training target. The mixing probability is 0.75, with a minimum target video ratio of $\lambda > 0.2$.
    - **Design Motivation**: Real-world videos commonly contain multiple sound sources without separation annotations. Borrowing ideas from audio-visual source separation, this mixing strategy constructs pseudo multi-source samples in a self-supervised manner, teaching the model to identify and select target visual regions via text cues.

### Loss & Training

- **Stage 1 (Video Encoder Training)**: Teacher-student distillation. The teacher model (original Synchformer) receives single-source video $V_{\text{tar}}$ and generates pseudo-label features $\mathbf{v}_{\text{tar}}$; the student model receives the mixed video and target text, minimizing the L2 loss $\|\mathcal{F}_S([V_{\text{tar}} \oplus V_{\text{pair}}], \mathbf{t}_{\text{tar}}) - \mathcal{F}_T(V_{\text{tar}})\|^2$.
- **Stage 2 (Generator Training)**: The video encoder is frozen; only the video feature projection layer $W_{\mathbf{v}}$ and adaLN modules $W_\gamma, W_\beta$ of the MM-DiT generator are fine-tuned using a conditional flow matching (CFM) objective. Inference uses an Euler solver with 25 steps and CFG strength $\gamma = 4.5$.
- Both stages fine-tune only 14% of each model's parameters (encoder: 19M; generator: 22M).

## Key Experimental Results

### Main Results

Evaluated on the self-constructed VGG-MonoAudio benchmark, comprising 67 single-source videos and 1,071 mixed test pairs (560 inter-class + 511 intra-class):

| Method | FAD↓ | KAD↓ | IS↑ | CLAP↑ | IB↑ | DeSync↓ |
|--------|------|------|-----|-------|-----|---------|
| ReWaS | 70.4 | 4.937 | 6.23 | 0.200 | 0.2454 | 1.364 |
| VinTAGe | 50.5 | 1.309 | 11.51 | 0.283 | 0.2850 | 1.292 |
| MMAudio-S-16k | 56.7 | 0.874 | 11.54 | 0.270 | 0.3135 | 0.802 |
| VOS+MMAudio | 60.0 | 0.878 | 12.11 | 0.291 | 0.3010 | 0.991 |
| **SelVA** | **51.7** | **0.676** | **13.07** | **0.292** | **0.3251** | **0.721** |

SelVA achieves state-of-the-art or near-state-of-the-art across all key metrics, with particularly significant advantages in temporal synchronization (DeSync 0.721) and audio quality (KAD 0.676).

### Ablation Study

| Configuration | DeSync↓ (Inter) | DeSync↓ (Intra) | Notes |
|---------------|-----------------|-----------------|-------|
| SelVA (Full) | 0.721 | 0.639 | Complete model |
| w/o Video Enc. FT | 0.868 | 0.734 | Removing encoder fine-tuning severely degrades temporal sync |
| w/o V2A Gen. FT | 0.736 | 0.651 | Removing generator fine-tuning degrades audio quality |
| w/o [SUP] tokens | 0.756 | 0.676 | Removing SUP degrades temporal alignment |
| w/o two-stage | 0.823 | 0.777 | Joint training degrades both semantic and temporal alignment |

### Key Findings

- Video encoder fine-tuning contributes most to temporal synchronization (DeSync rises from 0.721 to 0.868 without it), confirming that teaching the encoder text-guided feature selection is central.
- The [SUP] token primarily improves temporal alignment by suppressing erroneous tracking of non-target motions, with limited impact on audio quality and semantics.
- Joint training (without two stages) causes the model to take shortcuts—substituting visual sound events with text semantics—thereby disrupting temporal synchronization.
- In human evaluation, the VOS baseline achieves CLAP scores comparable to SelVA, yet human-perceived text-audio alignment is substantially lower (3.78 vs. 4.53), exposing the limitations of automatic metrics.

## Highlights & Insights

- **The [SUP] token design is particularly elegant**: rather than inserting tokens into the visual sequence and increasing encoder computation, they are prepended to the text sequence as part of the Key/Value, absorbing spurious attention in cross-attention with negligible computational overhead yet significant effect.
- **The self-supervised video mixing strategy** eliminates the need for any single-source annotated data, enabling direct training on in-the-wild datasets such as VGGSound with strong scalability.
- **The necessity of two-stage training**: decoupling feature extraction and sound generation avoids training instability caused by circular dependencies between the two modules.

## Limitations & Future Work

- VGGSound training data is noisy (containing background sounds and off-screen sounds); cleaner data or improved data filtering could substantially boost performance.
- Text labels are typically simple "noun + verb" structures, limiting the model's ability to handle complex textual descriptions (e.g., distinguishing "male voice singing" from "male voice hiccupping").
- The video encoder occasionally fails to continuously track target motion changes, resulting in residual sound-substitution artifacts.
- The evaluation benchmark VGG-MonoAudio is small (only 67 videos), and generalizability awaits further validation.

## Related Work & Insights

- **vs. MMAudio**: MMAudio does not use text modality for control; SelVA augments it with a text-conditioned video encoder, preserving MMAudio's strong generative capability while adding selective control.
- **vs. VOS+MMAudio**: VOS relies on segmentation models for spatial masks, struggles with diffuse sounds (rain, wind), and incurs high computational cost; SelVA achieves more flexible control using text alone.
- **vs. ReWaS / VinTAGe**: These methods use text as auxiliary semantic context but do not modulate visual features with text, resulting in notably weaker temporal alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First V2A method to perform explicit sound-source selection using text alone; the [SUP] token design is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Quantitative, qualitative, human evaluation, and ablation studies are all included, though the benchmark scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear, method description is thorough, and figures are informative.
- **Value**: ⭐⭐⭐⭐ Addresses a genuine need in practical audio production with good application prospects.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GoalForce: Teaching Video Models to Accomplish Physics-Conditioned Goals](goal_force_teaching_video_models_to_accomplish_physics-conditioned_goals.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[CVPR 2026\] StreamReady: Learning What to Answer and When in Long Streaming Videos](streamready_learning_what_to_answer_and_when_in_long_streaming_videos.md)
- [\[AAAI 2026\] Causality Matters: How Temporal Information Emerges in Video Language Models](../../AAAI2026/video_understanding/causality_matters_how_temporal_information_emerges_in_video_language_models.md)

<!-- RELATED:END -->
