---
title: >-
  [Paper Note] Privacy-preserving Prosody Representation Learning
description: >-
  [ACL2026][Audio & Speech][Prosody representation] This paper proposes a self-supervised prosody encoder that uses the glottal source as input. By employing F0 speaker normalization and an adversarial speaker loss to redu…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "Prosody representation"
  - "speaker disentanglement"
  - "self-supervised learning"
  - "privacy protection"
  - "speech security"
date: 2026-05-08
content_hash: 32b7b4d09e8baeda
---

# Privacy-preserving Prosody Representation Learning

**Conference**: ACL2026  
**arXiv**: [2606.00407](https://arxiv.org/abs/2606.00407)  
**Code**: https://github.com/kpeverson/speaker_disentangled_prosody  
**Area**: Speech Privacy / AI Security  
**Keywords**: Prosody representation, speaker disentanglement, self-supervised learning, privacy protection, speech security  

## TL;DR
This paper proposes a self-supervised prosody encoder that uses the glottal source as input. By employing F0 speaker normalization and an adversarial speaker loss to reduce identity leakage, it outperforms raw prosody and HuBERT baselines in phrase boundary detection, syllable prominence, and pitch reconstruction, while reducing VoxCeleb1 speaker identification accuracy from HuBERT's 0.64 to 0.14.

## Background & Motivation
**Background**: Prosody in speech includes non-lexical information such as pitch, energy, pauses, and duration lengthening. It is essential for expressing information focus, irony, self-correction, interrogative intonation, and excitement levels, making it critical for both speech understanding and generation. Modern speech models often use self-supervised representation learning to obtain general speech representations, but these representations typically conflate lexical content, prosody, and speaker identity.

**Limitations of Prior Work**: Traditional prosodic features rely on statistics of F0, energy, and duration as well as phoneme alignment. However, F0 extraction, forced alignment, and energy features are susceptible to noise, speaker variations, and recording conditions. While self-supervised models like HuBERT are effective for certain prosody tasks, they do not explicitly protect speaker privacy.

**Key Challenge**: Acoustic-prosodic cues inherently carry speaker information, such as average pitch, glottal characteristics, and voice quality. If a model requires prosodic expressiveness without identity information, learning directly from raw speech or raw prosodic features exposes users to privacy risks like speaker identification, voice cloning, and deepfakes.

**Goal**: To learn an explicit prosody representation that preserves linguistically relevant prosodic events and local pitch dynamics while minimizing speaker identity information. The authors aim to demonstrate that speaker disentanglement does not have to come at the expense of downstream prosodic performance.

**Key Insight**: The paper draws on the masked prediction and span boundary objectives of ProsodyBERT/HuBERT but replaces the input with an estimated glottal waveform and incorporates speaker disentanglement strategies into the training objectives and hidden-unit target construction.

**Core Idea**: Redesign the input, targets, and losses of the prosody encoder around "removing lexical content, removing identity, and retaining prosody," rather than performing post-hoc privacy filtering on existing speech representations.

## Method
The model is a frame-based prosody encoder with an architecture similar to HuBERT-base: it processes inputs through a convolutional module followed by a Transformer to output frame-level representations. Training is self-supervised using hidden units clustered from acoustic-prosodic features instead of transcriptions.

### Overall Architecture
First, the system estimates the glottal source from raw speech. The authors use LPC inverse filtering to extract the glottal source; low-energy non-speech frames return the raw waveform to avoid LPC artifacts. This is followed by a 1 kHz low-pass filter to reduce lexical information leakage.

Second, hidden units are constructed offline. Each frame's acoustic-prosodic features include periodicity $P$, speaker-normalized $\log F0$, $\Delta\log F0$, and the first mel-frequency cepstral coefficient $c_1$. These features undergo corpus-level z-normalization before being clustered via k-means to generate frame-level labels.

Third, the prosody encoder learns local prosody cues via masked prediction, suprasegmental patterns through a span-boundary objective, and suppresses speaker identity using an adversarial speaker identification loss.

Finally, the trained encoder is frozen, and only the final encoder output layer is used for downstream tasks. The authors evaluate representation capability on three prosody tasks—pitch reconstruction, phrase boundary detection, and syllable prominence detection—and evaluate privacy leakage via VoxCeleb1 speaker identification.

### Key Designs
1.  **Glottal Source Input and Low-pass Filtering**:
    - **Function**: Reduces lexical content before it enters the model while retaining prosody-related glottal and voice quality information.
    - **Mechanism**: Estimating the glottal waveform via LPC inverse filtering. Frames with energy below $10^{-4}$ bypass inverse filtering to avoid instability from unreliable LPC coefficients. A 1 kHz low-pass filter further reduces lexical cues.
    - **Design Motivation**: If the input retains too much speech content or speaker detail, the adversarial loss alone struggles to remove all privacy information. Shifting privacy protection to the input layer reduces the model's chance of learning identity shortcuts.

2.  **Speaker-normalized Hidden Units**:
    - **Function**: Ensures the masked prediction targets themselves carry less information about the speaker's average pitch.
    - **Mechanism**: Hidden units are derived from clustering $[P, \log F0, \Delta\log F0, c_1]$, where $\log F0$ is centered by subtracting the speaker's weighted average log pitch (using periodicity $P$ as weights). The authors also use $c_1$ instead of energy to decrease sensitivity to recording conditions.
    - **Design Motivation**: If self-supervised targets retain speaker-specific pitch ranges, the model will be pulled back toward identity information despite input processing. Target normalization reduces leakage from the supervision signal.

3.  **Masked/Span Objectives with Adversarial Speaker Loss**:
    - **Function**: Simultaneously learns local and long-range prosodic structures while suppressing speaker identifiability.
    - **Mechanism**: Total loss is $L=L_{mp}+\alpha_{sb}L_{sb}+\alpha_{spk}^{adv}L_{spk}^{adv}$. $L_{mp}$ is the HuBERT-style masked hidden-unit cross-entropy; $L_{sb}$ predicts the center label of a masked span using the unmasked boundary frames; $L_{spk}^{adv}$ uses gradient reversal to train a speaker classifier while forcing the encoder to learn anti-speaker features.
    - **Design Motivation**: Pure masked prediction tends to retain speaker cues; pure adversarial loss may damage prosody tasks. Combining them ensures the representations are constrained by the primary task to remain prosodically informative while being constrained by the adversarial term to be unreadable for identity.

### Loss & Training
The model is trained on the transcribed portion of GigaSpeech. Since the corpus lacks speaker labels, the authors extract utterance-level embeddings using a pretrained speaker encoder and cluster them into 1,000 pseudo-speaker labels for normalization and adversarial objectives. Pitch and periodicity are extracted using `torchcrepe`. Training is performed using `fairseq` on 4 NVIDIA A40 or L40 GPUs for 500K steps with a batch size of approximately 30 per GPU. The checkpoint with the lowest validation loss is frozen for downstream tasks.

## Key Experimental Results

### Main Results
On prosody modeling tasks, the proposed encoder variants generally outperform raw prosody and HuBERT-base. The most significant improvement is in syllable prominence detection, where the paper reports a 15% F1 gain over HuBERT-base.

| Model / Setting | Speaker-normalized $\log F0$ | Adv speaker loss | Phrase boundary F1 | Syllable prominence F1 | Pitch MSE | 0-mean Pitch MSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Most frequent class | No | No | 0.00 | 0.00 | N/A | N/A |
| HuBERT-base | No | ✗ | 0.79 | 0.74 | 0.056 | 0.011 |
| Raw prosody | ✓ | No | 0.49 | 0.66 | N/A | N/A |
| Ours | ✗ | ✗ | 0.82 | 0.86 | 0.027 | 0.012 |
| Ours | ✓ | ✗ | 0.82 | 0.86 | 0.048 | 0.012 |
| Ours | ✗ | ✓ | 0.73 | 0.82 | 0.024 | 0.012 |
| Ours | ✓ | ✓ | 0.82 | 0.85 | 0.025 | 0.008 |

Speaker disentanglement is evaluated using VoxCeleb1 speaker identification accuracy; lower scores indicate less readable identity information in the final representation.

| Model / Setting | Speaker-normalized $\log F0$ | Adv speaker loss | Speaker ID accuracy |
| :--- | :---: | :---: | :---: |
| HuBERT-base | No | ✗ | 0.64 |
| Ours | ✗ | ✗ | 0.41 |
| Ours | ✓ | ✗ | 0.42 |
| Ours | ✗ | ✓ | 0.22 |
| Ours | ✓ | ✓ | 0.14 |

### Ablation Study
The two disentanglement strategies serve different purposes: speaker-normalized targets show little impact on SID accuracy when used alone but achieve the lowest identity leakage when combined with adversarial loss. Adversarial loss alone significantly reduces identifiability but harms phrase boundary F1 when speaker-normalized targets are absent.

| Ablation Config | Prosody Performance | Privacy Performance | Explanation |
| :--- | :--- | :--- | :--- |
| No speaker norm, No adv | phrase 0.82, prominence 0.86 | SID 0.41 | Better than HuBERT, but identity info remains |
| With speaker norm, No adv | phrase 0.82, prominence 0.86 | SID 0.42 | Target normalization alone is insufficient |
| No speaker norm, With adv | phrase 0.73, prominence 0.82 | SID 0.22 | Privacy gain is clear, but downstream prosody suffers |
| With speaker norm, With adv | phrase 0.82, prominence 0.85 | SID 0.14 | Best privacy-utility trade-off |

### Key Findings
- The proposed encoder improves phrase boundary F1 from HuBERT's 0.79 to 0.82 and syllable prominence F1 from 0.74 to 0.85/0.86, showing that removing identity does not require sacrificing prosodic event modeling.
- The final combination (speaker normalization + adversarial loss) reduces SID accuracy to 0.14, substantially lower than HuBERT-base (0.64) and the non-disentangled version (0.41).
- The authors report that the adversarial objective provides a 46% relative SID reduction, while both strategies combined provide a 66% relative reduction.
- For 0-mean pitch reconstruction, the fully disentangled model achieves an MSE of 0.008, outperforming HuBERT-base (0.011), indicating it is particularly adept at modeling local pitch dynamics rather than speaker-specific pitch offsets.
- Using only the final encoder layer is a privacy-oriented choice; SUPERB typically uses all intermediate layers, which might reintroduce identity information.

## Highlights & Insights
- Instead of treating "privacy protection" as post-processing, the paper modifies the input, target, and loss simultaneously. This end-to-end disentanglement is more robust than simply adding an adversarial classifier.
- Using the glottal source as input is insightful: it retains prosody and voice quality cues while minimizing lexical leakage via low-pass and energy thresholding, aligning with the principle of "minimal sufficient representation."
- The most compelling result is the trade-off of the complete model: lowest SID accuracy with no loss in phrase boundary F1 and syllable prominence remaining significantly higher than HuBERT. This suggests privacy and utility are not strictly zero-sum in this context.
- Such a prosody representation can benefit expressive TTS, speech understanding, and dialogue systems while providing a safer intermediate representation for privacy-sensitive scenarios.

## Limitations & Future Work
- Pseudo-speaker labels were used during training instead of real ones. The authors note that if real speaker metadata for GigaSpeech were available, normalization and adversarial loss might be even more effective.
- Evaluation focused on local linguistic prosodic events. Paralinguistic tasks such as emotion, sarcasm, and health diagnostics have not yet been systematically evaluated.
- Understanding tasks used hand transcriptions. Testing with ASR transcriptions would be more realistic but requires more complex scoring.
- No human subjective evaluation has been performed in speech generation or voice conversion tasks. Whether privacy-preserving representations affect naturalness and expressiveness remains to be validated.
- The current model is not causal and cannot be used for streaming generation; while transitioning to a causal framework is straightforward, it requires implementation and testing.
- Privacy evaluation assumes an attacker with limited data; the ethical statement acknowledges that stronger recognition algorithms could be trained if an attacker possesses more speaker data.

## Related Work & Insights
- **vs. HuBERT / wav2vec 2.0**: General SSL representations help prosody tasks but do not explicitly disentangle identity; this work uses the HuBERT architecture with prosody-oriented inputs/targets/losses.
- **vs. ProsodyBERT**: ProsodyBERT uses hidden units and span boundaries for prosody; this work inherits that approach but adds glottal source inputs and speaker disentanglement.
- **vs. PE-Wav2vec**: PE-Wav2vec also uses glottal waveforms; this paper integrates privacy objectives into the training framework.
- **vs. Information Bottleneck / Pitch Shifting / Adversarial Loss**: These are existing disentanglement techniques. The contribution here is combining speaker-normalized targets and adversarial loss within a prosody SSL framework, validated with dual prosody-SID metrics.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Explicitly combines prosody SSL with privacy protection; the design is targeted and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Solid coverage of prosody modeling, speaker ID, and ablations, but lacks generation tasks and broader paralinguistic evaluation.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear methodology, concise tables, and honest discussion of limitations.
- **Value**: ⭐⭐⭐⭐☆ Highly relevant for privacy-sensitive speech systems, especially as an intermediate representation for downstream applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning](../../ICLR2026/audio_speech/emotionthinker_prosody-aware_reinforcement_learning_for_explainable_speech_emoti.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[ACL 2026\] Protecting Bystander Privacy via Selective Hearing in Audio LLMs](protecting_bystander_privacy_via_selective_hearing_in_audio_llms.md)
- [\[ACL 2026\] ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment](immersivetts_environment-aware_text-to-speech_with_multimodal_diffusion_transfor.md)
- [\[ACL 2026\] How Tokenization Limits Phonological Knowledge Representation in Language Models and How to Improve Them](how_tokenization_limits_phonological_knowledge_representation_in_language_models.md)

</div>

<!-- RELATED:END -->
