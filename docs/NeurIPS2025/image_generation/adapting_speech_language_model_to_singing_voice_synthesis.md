---
title: >-
  [Paper Note] Adapting Speech Language Model to Singing Voice Synthesis
description: >-
  [NeurIPS 2025 (Workshop)][Image Generation][Speech Language Model] This paper adapts a 1.7B-parameter TTS-pretrained Speech Language Model to the Singing Voice Synthesis (SVS) task via score tokenization, multi-stream LM prediction, conditional flow matching refinement, and a vocoder. Using only 135 hours of synthesized singing data, the system achieves performance comparable to dedicated SVS systems.
tags:
  - NeurIPS 2025 (Workshop)
  - Image Generation
  - Speech Language Model
  - SVS
  - Flow Matching
  - Codec
  - Singing Voice Synthesis
date: 2026-05-08
content_hash: 18c52fcfcef09934
---

# Adapting Speech Language Model to Singing Voice Synthesis

**Conference**: NeurIPS 2025 (Workshop)
**arXiv**: [2512.14657](https://arxiv.org/abs/2512.14657)
**Code**: [https://tsukasane.github.io/SLMSVS/](https://tsukasane.github.io/SLMSVS/)
**Area**: Image Generation
**Keywords**: Speech Language Model, SVS, Flow Matching, Codec, Singing Voice Synthesis

## TL;DR
This paper adapts a 1.7B-parameter TTS-pretrained Speech Language Model to the Singing Voice Synthesis (SVS) task via score tokenization, multi-stream LM prediction, conditional flow matching refinement, and a vocoder. Using only 135 hours of synthesized singing data, the system achieves performance comparable to dedicated SVS systems.

## Background & Motivation
**Background**: Speech Language Models (SLMs) have emerged as a unified paradigm for speech tasks such as TTS, ASR, and speech enhancement, yet their generalization capability to singing voice synthesis remains unexplored.

**Limitations of Prior Work**:
- Public SVS datasets are extremely scarce due to copyright restrictions and high annotation costs, making it infeasible to train large models from scratch.
- SVS inputs are structured musical scores (phonemes + pitch + duration), which are considerably more complex than the plain text inputs used in TTS.
- Codec decoders pretrained on speech cannot faithfully resynthesize singing voice, imposing a hard performance ceiling.

**Key Challenge**: The generalization potential of large-scale SLMs versus the scarcity of SVS data.

**Goal**: To investigate whether a TTS-pretrained SLM can be adapted to SVS at low cost.

**Key Insight**: Tokenize the score-based conditions and incorporate them into the SLM vocabulary, then fine-tune the model and apply flow matching for acoustic refinement.

**Core Idea**: Leverage a TTS-pretrained SLM combined with flow matching refinement to address the low-resource challenge in singing voice synthesis.

## Method

### Overall Architecture
**Input**: Musical score (phonemes + MIDI pitch + duration) and a speaker prompt. The pipeline proceeds as follows: (1) the score is tokenized into 50 FPS discrete tokens; (2) audio is encoded into multi-stream tokens via a codec encoder and an SSL model; (3) the LM predicts the target token sequence; (4) flow matching maps the LM-predicted codec tokens to a mel spectrogram; (5) a HiFi-GAN vocoder synthesizes the final waveform.

### Key Designs

1. **Score Tokenization (`svs_lb`)**:

   - **Function**: Encodes phonemes, MIDI pitch, and duration into frame-level discrete tokens.
   - **Mechanism**: Each frame is represented as a (phoneme\_token, pitch\_token) tuple; duration is implicitly encoded via repetition count: $\text{repeat} = (\text{end} - \text{start}) \times \text{fps}$. A new `svs_lb` modality is introduced to extend the TTS vocabulary.
   - **Design Motivation**: Maintains consistency with the SLM's token prediction paradigm and reuses the TTS pretrained encoder.

2. **Multi-stream LM Token Prediction**:

   - **Function**: Uses the 1.7B SLM to predict concatenated SSL and 8-layer codec tokens.
   - **Mechanism**: Built on ESPNet-SpeechLM; the model takes score conditions and a speaker prompt as input and is trained with cross-entropy loss over frame-level SSL+codec tokens.
   - **Design Motivation**: SSL tokens capture high-level semantics while codec tokens encode acoustic details; concatenating both combines their complementary strengths.

3. **Flow Matching Refinement**:

   - **Function**: Refines the noisy codec tokens predicted by the LM into clean mel spectrograms.
   - **Mechanism**: Conditional Flow Matching (CFM) starts from Gaussian noise and learns a velocity field conditioned on codec tokens and pitch signals to transport samples toward the target mel distribution. A linear interpolation path is used: $\psi_t(x|x_1) = (1-t)x + tx_1$.
   - **Design Motivation**: Tokens directly predicted by the LM are noisy, causing temporal discontinuities and perceptual artifacts; furthermore, the codec decoder pretrained on speech cannot faithfully resynthesize singing. Flow matching circumvents both bottlenecks.

### Loss & Training
- **LM fine-tuning**: Cross-entropy loss, maximizing $P(s|m,p)$.
- **Flow matching**: MSE loss on the conditional velocity field.
- A HiFi-GAN vocoder with parameters aligned to the codec STFT is additionally trained.

## Key Experimental Results

### Main Results
Evaluated on the ACE-Opencpop dataset (135 hours of synthesized singing voice).

| Method | F0\_RMSE↓ | F0\_CORR↑ | MCD↓ | PER↓ | SingMOS↑ |
|---|---|---|---|---|---|
| XiaoiceSing | 71.67 | 0.62 | 11.47 | **0.09** | 3.88 |
| TokSing | 55.83 | 0.67 | **6.77** | 0.19 | **4.08** |
| LM + Flow + Voc (ours) | 62.79 | 0.60 | 7.86 | 0.36 | 4.09 |

The proposed system achieves SingMOS (perceptual quality) on par with the best dedicated SVS system, TokSing.

### Ablation Study

| Configuration | MCD↓ | PER↓ | SingMOS↑ | Notes |
|---|---|---|---|---|
| LM + CD (codec decoder) | 8.26 | 0.56 | 3.65 | Direct codec decoding; poor quality |
| LM + Flow1 + CD | 8.44 | 0.45 | 3.64 | Flow refinement but still uses codec decoder |
| LM + Flow1 + Voc | **7.86** | **0.36** | **4.09** | Flow + dedicated vocoder; best overall |
| CD Resynthesis (upper bound) | 5.84 | 0.19 | 3.95 | Upper bound of the codec decoder |

### Key Findings
- The codec decoder is the dominant bottleneck, as a decoder pretrained on speech is ill-suited for singing voice.
- Flow matching refinement combined with a dedicated vocoder yields substantial quality improvements (SingMOS: 3.65 → 4.09).
- LM + Flow + Voc even surpasses the codec resynthesis SingMOS upper bound (4.09 vs. 3.95), indicating that flow matching can compensate for codec decoder deficiencies.
- PER remains higher than dedicated systems, suggesting room for improvement in lyric intelligibility.

## Highlights & Insights
- **Cross-task generalization of SLMs**: Adapting a TTS SLM to SVS with only 135 hours of data validates the generalization potential of large pretrained models.
- **Flow matching as a decoding bridge**: Elegantly resolves the domain mismatch of the pretrained codec decoder, serving as a general-purpose *domain gap bridging* strategy.
- **Complementary two-stage design**: The LM handles sequence modeling (temporal structure) while flow matching handles acoustic quality (spectral detail), with clearly delineated responsibilities.

## Limitations & Future Work
- PER remains high (0.36 vs. TokSing's 0.19), indicating insufficient lyric pronunciation clarity.
- F0-related metrics are relatively low (0.60 vs. 0.67), leaving room for improvement in pitch tracking accuracy.
- Evaluation is limited to Mandarin singing (Opencpop); multilingual generalization has not been verified.
- The 135 hours of training data are synthesized; performance on real singing recordings remains unknown.

## Related Work & Insights
- **vs. TokSing**: A dedicated SVS system that achieves better objective metrics but comparable perceptual quality. The proposed approach benefits from reusing TTS pretraining.
- **vs. XiaoiceSing**: A conventional end-to-end SVS system with the lowest PER but highest MCD and weaker timbre modeling.
- **vs. ESPNet-SpeechLM**: The proposed method builds on its framework and demonstrates its multi-task extensibility.

## Rating
- **Novelty**: ⭐⭐⭐ The idea of adapting SLMs to SVS is interesting, but the technical contribution is incremental.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablations are thorough, but evaluation is limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐ Concise and clear (workshop paper).
- **Value**: ⭐⭐⭐ Validates cross-task generalization of SLMs and offers insights for low-resource speech generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] ScaleDiff: Higher-Resolution Image Synthesis via Efficient and Model-Agnostic Diffusion](scalediff_higher-resolution_image_synthesis_via_efficient_and_model-agnostic_dif.md)
- [\[NeurIPS 2025\] FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks](focalcodec_low-bitrate_speech_coding_via_focal_modulation_networks.md)
- [\[NeurIPS 2025\] Riemannian Consistency Model](riemannian_consistency_model.md)
- [\[NeurIPS 2025\] MMaDA: Multimodal Large Diffusion Language Models](mmada_multimodal_large_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
