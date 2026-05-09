---
title: >-
  [Paper Note] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization
description: >-
  [NeurIPS 2025][Audio & Speech][Sound Source Localization] This work systematically reveals that AI SSL models suffer from severe visual bias—degrading to near-random performance under audio-visual conflict—and proposes EchoPin, a neuroscience-inspired model (HRTF filtering + cochleagram + stereo audio) that substantially outperforms prior methods on AudioCOCO and exhibits a human-like horizontal-over-vertical localization accuracy asymmetry.
tags:
  - NeurIPS 2025
  - "Audio & Speech"
  - Sound Source Localization
  - Modality Bias
  - Cross-modal Conflict
  - Neuroscience-Inspired
  - HRTF
  - Cochleagram
date: 2026-05-08
content_hash: 74b70e4987c33d32
---

# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization

**Conference**: NeurIPS 2025
**arXiv**: [2505.11217](https://arxiv.org/abs/2505.11217)
**Code**: Publicly available as declared in the paper (GitHub)
**Area**: Audio & Speech
**Keywords**: Sound Source Localization, Modality Bias, Cross-modal Conflict, HRTF, Cochleagram, AudioCOCO

## TL;DR

Through six controlled audio-visual conditions and human psychophysical experiments, this work systematically reveals that existing AI sound source localization (SSL) models suffer from severe visual bias—degrading to near-random performance under audio-visual conflict—and proposes EchoPin, a neuroscience-inspired model combining HRTF filtering, ERB cochleagram representation, and stereo audio. EchoPin substantially outperforms prior methods on the newly constructed AudioCOCO dataset and, without any human behavioral supervision, exhibits a human-like horizontal-over-vertical localization accuracy asymmetry.

## Background & Motivation

**Background**: Sound source localization (SSL) is a fundamental multimodal task that associates sounds with spatial locations in visual scenes. Recent approaches based on contrastive learning (e.g., DenseAV) and cross-modal attention (Transformer architectures) have achieved promising results under standard congruent conditions; however, virtually all prior work is evaluated under the idealized assumption of audio-visual semantic and spatial consistency.

**Limitations of Prior Work**: (1) Existing datasets are heavily biased—dominated by large, centrally located objects—allowing models to achieve competitive performance via visual shortcuts without genuinely integrating audio information; (2) the vast majority of methods rely on mono audio, discarding binaural spatial cues; (3) there is a systematic lack of investigation into model behavior under non-ideal conditions such as audio-visual conflict or modality absence.

**Key Challenge**: When audio-visual signals conflict (e.g., a dog bark is heard while a car is visible at the expected source location), humans can flexibly prioritize audition to localize accurately. Whether AI models can do the same—or whether they are misled by visual dominance—remains an open question.

**Goal**: (1) Quantify modality bias in AI models across diverse audio-visual conditions; (2) provide human baseline comparisons; (3) construct an unbiased dataset and a biologically inspired model to narrow the human–machine gap.

**Key Insight**: The work is motivated by neuroscience, specifically mimicking the human peripheral auditory processing chain (pinna HRTF → cochlear frequency decomposition → binaural spatial cues) to design both a data synthesis pipeline and the model frontend.

**Core Idea**: Employing HRTF spatial filtering, ERB-based cochleagram frequency decomposition, and stereo audio configuration to simulate the human peripheral auditory system—simultaneously removing dataset bias and endowing the model with the capacity to exploit genuine spatial auditory cues.

## Method

### Overall Architecture

The EchoPin system operates on three levels: (1) **AudioCOCO dataset**—spatially aware stereo audio is synthesized via a Unity 3D simulator under six controlled experimental conditions; (2) **Human psychophysical experiments**—14 participants provide human baselines under identical conditions; (3) **EchoPin model**—a pipeline of HRTF filtering → ERB cochleagram → dual-encoder contrastive learning. The model takes a static image and two-channel stereo audio as input and predicts the sound source location in the image.

### Key Designs

1. **AudioCOCO Dataset and Depth-Aware Stereo Audio Synthesis**
   - **Function**: Provides high-quality, spatially balanced, and bias-free audio-visual training and test data.
   - **Mechanism**: Twelve categories of sound-producing objects are selected from MSCOCO and stratified into three size tiers by object area ratio (Size1: 0–5%, Size2: 5–15%, Size3: 15–30%), with trivially large objects (>30%) excluded. DepthAnything is used for monocular depth estimation; Unity then constructs a 3D scene based on pixel positions and depth, and synthesizes spatially rendered stereo audio from a simulated listener with a 0.17 m interaural distance using physically based sound propagation. The test set comprises six conditions: Congruent, ConflictVCue, AbsVCue, AOnly, VOnly, and MultiInstLoc.
   - **Design Motivation**: Existing datasets (e.g., FlickrSoundNet, VGGSound) suffer from large-object-centered bias that enables models to learn visual shortcuts rather than genuine audio-visual alignment; mono audio further precludes spatial localization cues.

2. **HRTF Filtering + ERB Cochleagram Frontend**
   - **Function**: Transforms raw stereo waveforms into a representation faithful to human peripheral auditory processing.
   - **Mechanism**: Stereo signals are first convolved with direction-dependent HRTFs from the KEMAR dummy head dataset, encoding interaural time differences (ITDs) and interaural level differences (ILDs). The HRTF-filtered 16 kHz stereo signal is then decomposed using 66 ERB (Equivalent Rectangular Bandwidth) filters into a cochleagram (tensor of shape 66 × 160,000 × 2), preserving pitch, timbre, and spatial cues.
   - **Design Motivation**: Conventional mel-spectrograms discard the fine-grained spectral features introduced by HRTFs and the binaural spatial cues essential for localization; ERB filterbanks more faithfully replicate the frequency selectivity and temporal dynamics of the cochlea.

3. **Dual-Encoder Contrastive Learning Architecture**
   - **Function**: Independently encodes auditory and visual features before performing semantic and spatial alignment.
   - **Mechanism**: Building upon the IS3 dual-stream 2D CNN architecture, 1D convolutional kernels first integrate binaural channel information, after which separate visual and audio encoders extract feature maps. A cosine similarity heatmap is computed for sound source localization. Training employs Triplet Loss (to pull matched audio-visual embeddings together) and CIoU Loss (to penalize spatial deviation between predicted and ground-truth bounding boxes).
   - **Design Motivation**: Decoupled encoding allows each modality to develop independent representations; contrastive learning is naturally suited to cross-modal alignment, while CIoU Loss provides direct supervision for spatial localization accuracy.

### Loss & Training

- **Triplet Loss**: Semantic alignment—minimizes the embedding distance of matched audio-visual pairs while maximizing the distance of mismatched pairs.
- **CIoU Loss**: Spatial alignment—penalizes geometric deviation between predicted source bounding boxes and ground-truth annotations.
- All weights except the first-layer 1D convolution are initialized from the pretrained IS3 model, followed by end-to-end fine-tuning.
- Training is conducted solely under the Congruent condition; all six conditions are used for evaluation to assess generalization.

## Key Experimental Results

### Main Results

Audio localization accuracy (A-Acc) under the multi-instance condition:

| Model | Size1 | Size2 | Size3 |
|-------|-------|-------|-------|
| Random | 1.6% | 9.1% | 21.3% |
| IS3 | 4.8% | 7.9% | 22.4% |
| CAVP | 2.9% | 7.5% | 20.4% |
| AVSegformer | 2.5% | 7.3% | 20.2% |
| **EchoPin** | **4.5%** | **24.1%** | **47.1%** |
| Human | 25.7% | 36.4% | 38.6% |

### Ablation Study

A-Acc comparison across mono/stereo and cochleagram/mel configurations (averaged over Congruent, ConflictVCue, AbsVCue, and AOnly conditions):

| Configuration | Size1 | Size2 | Size3 |
|---------------|-------|-------|-------|
| IS3 (mono, mel, standard dataset) | 3.0% | 13.9% | 28.7% |
| EchoPin-M (mono, cochleagram, AudioCOCO) | 3.6% | 15.8% | 31.4% |
| EchoPin-S (stereo, mel, AudioCOCO) | 5.3% | 17.0% | 35.2% |
| **EchoPin (stereo, cochleagram, AudioCOCO)** | **9.7%** | **31.3%** | **47.6%** |

### Key Findings

- **Severe visual bias**: IS3 and CAVP degrade to near-random A-Acc under ConflictVCue, while human participants and EchoPin remain significantly above chance.
- **AI cannot localize by audio alone**: Under AOnly conditions, IS3 essentially fails, while EchoPin maintains limited capability but remains far below human performance.
- **Visual shortcuts exposed**: Under VOnly (no audio), AI models still achieve above-chance V-Acc, revealing a tendency to localize "visually salient sound-producing objects" (people, animals) rather than genuinely leveraging audio.
- **Stereo is critical**: EchoPin outperforms EchoPin-M (mono) by 16.2 percentage points on Size3.
- **Cochleagram outperforms Mel**: EchoPin outperforms EchoPin-S (mel) by 12.4 percentage points on Size3.
- **Emergent human-like asymmetry**: EchoPin exhibits a human-like pattern of horizontal localization accuracy (86.1% of trials within 6°) exceeding vertical accuracy, attributable to the stronger binaural horizontal-plane spatial cues encoded by the HRTF. Rotating the interaural axis by 90° (EchoPin-Ro) reverses the asymmetry, confirming its structural origin.

## Highlights & Insights

- **First systematic quantification of SSL modality bias**: The experimental design—six controlled conditions × three object sizes × 8+ models + human baselines—is exceptionally comprehensive.
- **An elegant neuroscience-to-engineering loop**: Designing the model frontend from the biological peripheral auditory pathway (pinna HRTF → cochlear ERB) not only improves performance but also gives rise to human-like behavioral asymmetries without any human behavioral supervision—a phenomenon rarely observed in current multimodal AI models.
- **Rigorous diagnosis of dataset bias**: The work demonstrates that the "large, centered objects" bias in existing SSL datasets is the root cause of visual shortcut learning; AudioCOCO systematically eliminates this through size stratification and spatial diversification.
- **EchoPin-Ro experiment**: The causal origin of the horizontal–vertical asymmetry is verified by rotating the binaural axis, a cleverly controlled manipulation that substantially strengthens the causal inference.

## Limitations & Future Work

- **Large gap on small targets**: EchoPin achieves only 4.5–9.7% on Size1 compared to 25.7% for humans (a 4–5× gap), indicating severely insufficient exploitation of weak spatial cues.
- **Residual visual interference under conflict**: EchoPin's performance under ConflictVCue degrades more than under AbsVCue, suggesting the model has not yet learned to actively suppress visual information during conflict as humans do.
- **Synthetic data limitations**: Unity-synthesized stereo audio still differs from real recordings; complex acoustic effects such as sound diffraction and environmental reverberation are not modeled.
- **Static scenes only**: The work addresses static images paired with synthesized audio and does not consider dynamic sound sources or temporal cues in video.
- **Limited object categories**: Only 12 sound-producing categories are included; generalization to open-world scenarios remains to be verified.

## Related Work & Insights

- **vs. IS3/CAVP/AVSegformer**: These methods perform reasonably on standard datasets but are fundamentally compromised by visual shortcuts, as revealed under conflict conditions. EchoPin addresses this through joint improvements in data and model design.
- **vs. ImageBind/LanguageBind**: Large-scale pretrained multimodal models also fail to show advantages on SSL tasks, demonstrating that scale cannot substitute for task-specific sensory modeling.
- **Implications for VLM research**: Multimodal large language models may exhibit analogous modality biases; the controlled experimental paradigm introduced here can be directly transferred to other multimodal tasks.
- **Transferability of the auditory frontend**: The HRTF + ERB cochleagram pipeline can be applied to audio-visual source separation, spatial audio generation, and robotic auditory perception.
- **Methodological value of AudioCOCO**: The spatially aware audio synthesis pipeline based on a 3D simulator and depth estimation is generalizable to any dataset construction task requiring audio-visual spatial alignment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The neuroscience-inspired SSL model and the systematic modality bias analysis framework are highly original, though the core model architecture is an incremental extension of IS3.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Six controlled conditions × three sizes × 8+ models + human psychophysical experiments + comprehensive ablations + the EchoPin-Ro validation experiment constitute a textbook-level experimental design.
- **Writing Quality**: ⭐⭐⭐⭐ — The paper is clearly structured with rich figures and rigorous condition descriptions, though the overall length is substantial.
- **Value**: ⭐⭐⭐⭐ — The work uncovers a fundamental deficiency in multimodal models and provides solutions with both scientific insight and engineering value, offering broad implications for multimodal AI research.

---
title: >-
  [Paper Notes] Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization
description: >-
  [NeurIPS 2025][Speech][Sound Source Localization] Systematically reveals that AI SSL models suffer from severe visual bias—degrading to random performance under audio-visual conflict—and proposes EchoPin (HRTF filtering + cochleagram + stereo audio), which substantially outperforms prior methods on AudioCOCO and exhibits a human-like horizontal-over-vertical localization accuracy asymmetry.
tags:
  - NeurIPS 2025
  - Speech
  - Sound Source Localization
  - Modality Bias
  - Cross-modal Conflict
  - Neuroscience-Inspired
  - HRTF
  - Cochleagram
---

# Seeing Sound, Hearing Sight: Uncovering Modality Bias and Conflict of AI Models in Sound Localization

**Conference**: NeurIPS 2025
**arXiv**: [2505.11217](https://arxiv.org/abs/2505.11217)
**Code**: [GitHub](https://github.com/) (publicly available as declared in the paper)
**Area**: audio_speech / multimodal
**Keywords**: Sound Source Localization, Modality Bias, Cross-modal Conflict, Neuroscience-Inspired, HRTF, Cochleagram

## TL;DR
This work systematically reveals that AI SSL models suffer from severe visual bias—degrading to near-random performance under audio-visual conflict—and proposes EchoPin, a neuroscience-inspired model (HRTF filtering + cochleagram + stereo audio) that substantially outperforms prior methods on AudioCOCO and exhibits a human-like horizontal-over-vertical localization accuracy asymmetry.

## Background & Motivation

1. **Background**: SSL is a fundamental multimodal perception task that associates sounds with their spatial origins in visual scenes. Existing multimodal models (contrastive learning, cross-modal attention, etc.) perform well under standard congruent conditions.
2. **Core Problem**: No prior work has systematically examined AI model behavior under audio-visual conflict—specifically, whether models prioritize audition over vision as humans do, or exhibit visual dominance.
3. **Key Finding**: Humans remain robust under conflict and even audio-only conditions by prioritizing auditory information, whereas AI models are heavily vision-biased and collapse to random-level performance under conflict.

## Method

### Overall Architecture
Three main contributions: (1) AudioCOCO dataset + 6 experimental conditions; (2) human psychophysical experiment baselines; (3) EchoPin neuroscience-inspired model.

### Key Design 1: AudioCOCO Dataset
- 12 sound-producing object categories from MSCOCO, stratified into 3 size tiers by target area ratio (Size1: 0–5%, Size2: 5–15%, Size3: 15–30%)
- Unity 3D simulator + DepthAnything depth estimation: synthesizes **spatially rendered stereo audio** from pixel positions and estimated depth
- Training set: 4,953 images → 9,360 audio-image pairs; test set: 5,500 images → 18,864 pairs
- **6 experimental conditions**: Congruent, ConflictVCue, AbsVCue, AOnly, VOnly, MultiInstLoc

### Key Design 2: EchoPin Model
- **HRTF filtering**: Applies direction-dependent head-related transfer functions from the KEMAR dummy head dataset to simulate direction-specific spectral shaping by the pinna, head, and torso
- **Cochleagram**: An ERB filterbank transforms the HRTF-filtered stereo waveform into a cochleagram (66 channels × 160k time steps × 2 ears), providing a more faithful representation of peripheral auditory processing than mel-spectrograms
- **Dual-encoder architecture**: A 2D CNN dual-stream architecture based on IS3, with independent visual and audio encoders whose outputs are fused for localization
- **Training losses**: Triplet Loss (semantic alignment) + CIoU Loss (spatial alignment)

### Key Design 3: Human Psychophysical Experiment
- 14 participants, 2,100 trials, conducted in a laboratory environment with stereo headphones
- Compared directly against AI models under identical six conditions

## Key Experimental Results

### A-Acc under Congruent Condition (Size2)

| Model | Size1 | Size2 | Size3 |
|-------|-------|-------|-------|
| Random | 1.6% | 9.1% | 19.8% |
| IS3 | 4.8% | 7.9% | 22.4% |
| **EchoPin** | **4.5%** | **24.1%** | **47.1%** |
| Human | 25.7% | 36.4% | 38.6% |

### Key Comparative Findings
- **ConflictVCue**: IS3 degrades to near-random; EchoPin remains significantly above chance
- **AOnly**: Humans can still localize; IS3 essentially fails; EchoPin maintains limited capability
- **VOnly**: AI models achieve above-chance V-Acc without audio → visual bias exposed
- **Mono vs. Stereo**: EchoPin (stereo) outperforms mono by 16.2% on Size3 A-Acc (47.6% vs. 31.4%)
- **Cochleagram vs. Mel**: EchoPin (cochleagram) outperforms EchoPin-S (mel) by 12.4% on Size3

### Human-like Asymmetry
EchoPin exhibits a human-like pattern of horizontal localization accuracy exceeding vertical accuracy, arising from the stronger horizontal-plane spatial cues provided by the binaural stereo + HRTF configuration.

## Highlights & Insights
1. **Systematic modality bias analysis**: First work to quantitatively reveal visual bias in SSL models using six controlled conditions
2. **Neuroscience-driven design**: HRTF + cochleagram combination faithfully emulates the human peripheral auditory system and gives rise to emergent human-like behavior
3. **Human–machine comparison**: Psychophysical experiments provide reliable human baselines
4. **AudioCOCO dataset**: Mitigates shortcut learning problems (e.g., large-object-centered bias) present in existing datasets

## Limitations & Future Work
1. The gap between EchoPin and humans on small targets (Size1) remains large (4.5% vs. 25.7%), indicating insufficient exploitation of weak spatial cues
2. EchoPin is still misled under ConflictVCue conditions—less robust than the human auditory-priority strategy
3. The work is limited to static images + synthesized audio and does not address dynamic sound sources in real-world video

## Related Work & Insights
- **vs. IS3**: IS3 uses mono audio and standard training data, resulting in heavy visual bias; EchoPin addresses this through stereo + HRTF + cochleagram + spatially balanced data
- **vs. CAVP/AVSegformer**: Both exhibit severe visual bias and perform comparably to IS3 under multi-instance conditions
- **vs. ImageBind/LanguageBind**: Large-scale pretrained models also fail to show advantages on SSL tasks

## Implications
- Relevant to the VLM community: multimodal large models may harbor analogous modality biases
- The HRTF + cochleagram auditory frontend is transferable to audio-visual separation, spatial audio synthesis, and robotic auditory perception
- The controlled dataset construction methodology of AudioCOCO is generalizable to other multimodal domains

## Rating
- Novelty: ⭐⭐⭐⭐ Neuroscience-inspired SSL model + systematic modality bias analysis
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Human experiments + multi-model comparison + 6 conditions + ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures
- Value: ⭐⭐⭐⭐ Reveals an important deficiency in multimodal models and provides actionable solutions

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Generating Physically Sound Designs from Text and a Set of Physical Constraints](generating_physically_sound_designs_from_text_and_a_set_of_physical_constraints.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ICCV 2025\] How Would It Sound? Material-Controlled Multimodal Acoustic Profile Generation for Objects](../../ICCV2025/audio_speech/how_would_it_sound_material-controlled_multimodal_acoustic_profile_generation_fo.md)

<!-- RELATED:END -->
