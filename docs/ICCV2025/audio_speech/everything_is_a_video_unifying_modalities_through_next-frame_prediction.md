---
title: >-
  [Paper Note] Everything is a Video: Unifying Modalities through Next-Frame Prediction
description: >-
  [ICCV 2025][Audio & Speech][multimodal unification] This paper reformulates multimodal learning tasks involving text, images, audio, and video as a unified next-frame prediction problem—rendering all inputs and outputs as sequences of 64×64 video frames—and demonstrates that a single Transformer model without any modality-specific encoders can handle cross-modal tasks, validating the radical yet feasible "everything is a video" unified representation paradigm.
tags:
  - ICCV 2025
  - "Audio & Speech"
  - multimodal unification
  - next-frame prediction
  - task reformulation
  - video generation
  - unified modality representation
date: 2026-05-08
content_hash: ac2742e616144d63
---

# Everything is a Video: Unifying Modalities through Next-Frame Prediction

**Conference**: ICCV 2025
**arXiv**: [2411.10503](https://arxiv.org/abs/2411.10503)
**Code**: None
**Area**: Audio & Speech
**Keywords**: multimodal unification, next-frame prediction, task reformulation, video generation, unified modality representation

## TL;DR

This paper reformulates multimodal learning tasks involving text, images, audio, and video as a unified next-frame prediction problem—rendering all inputs and outputs as sequences of 64×64 video frames—and demonstrates that a single Transformer model without any modality-specific encoders can handle cross-modal tasks, validating the radical yet feasible "everything is a video" unified representation paradigm.

## Background & Motivation

**Background**: Multimodal learning requires integrating text, images, audio, and video to accomplish tasks such as visual question answering, cross-modal retrieval, and caption generation. Dominant approaches rely on modality-specific encoders (e.g., ViT for images, Transformer for text) followed by late fusion, necessitating dedicated encoders and inter-modal alignment strategies for each modality.

**Limitations of Prior Work**: Modality-specific encoder designs constrain scalability and flexibility—every new modality requires a new encoder and fusion strategy; different modalities occupy distinct representation spaces, making cross-modal knowledge transfer difficult; and architectural complexity grows with the number of modalities.

**Key Challenge**: NLP has already achieved a "unified interface" paradigm in which all NLP tasks can be reformulated as text generation (prompt-based learning), enabling a single LLM to handle translation, summarization, question answering, and more. Multimodal learning has yet to achieve an analogous unification—different modalities still demand different processing pipelines.

**Goal**: To extend the task-reformulation idea from NLP to the multimodal domain by identifying a "supertask" that unifies all modalities, enabling a single model to handle diverse multimodal tasks without any modality-specific components.

**Key Insight**: The authors observe that text can be rendered as image frames (one frame per token) and audio can be converted into spectrograms; consequently, all modalities can theoretically be converted into visual frame sequences without information loss.

**Core Idea**: Reformulate all multimodal tasks as next-frame prediction—inputs and outputs are uniformly represented as sequences of 64×64 RGB video frames, with separator frames delineating input from output, so that the model needs only to learn next-frame prediction to handle cross-modal tasks.

## Method

### Overall Architecture

The framework combines task reformulation with a video prediction model. Each modality is first encoded into a video frame sequence: text tokens are rendered as individual frames (fixed-width font, 64×64), audio is converted to spectrogram frames, images are resized to 64×64, and video retains its original frame sequence. The complete sequence consists of input frames, a separator frame, and output frames; the model autoregressively predicts the next frame. At inference time, the final generated frames are decoded via OCR or used directly to obtain the answer.

### Key Designs

1. **Modality-Unified Reformulation**:

    - **Function**: Encodes inputs and outputs of different modalities uniformly as 64×64 RGB video frame sequences.
    - **Mechanism**: Text → each token rendered as one frame (fixed-width font filling 64×64); image → resized to 64×64 as one or more frames; audio → converted to a spectrogram frame; video → original frames used directly (possibly downsampled). Each task sequence follows the format [input frames…] [separator frame] [output frames…]. For example, SST-2 sentiment classification: [each word rendered as a frame] [|] [positive/negative rendered as a frame]; CIFAR-10: [64×64 image] [|] [class name frame].
    - **Design Motivation**: Text and audio can both be converted to visual representations without information loss (text rendering and spectrograms), making video frame sequences a universal representation that can theoretically encompass all modalities. The separator frame gives the model a clear signal that input has ended and output generation should begin.

2. **Spatial-Temporal Transformer Video Prediction Model**:

    - **Function**: Autoregressively predicts the next frame in a video sequence.
    - **Mechanism**: Input video frames are divided into non-overlapping 8×8 patches, linearly embedded, and augmented with spatial and temporal positional encodings. The model adopts a U-Net-style encoder-decoder architecture: the encoder progressively reduces resolution via patch merge operations, a global spatial-temporal Transformer operates at the lowest resolution, and the decoder restores resolution via patch unmerge with skip connections. Temporal attention uses causal masking to enforce the autoregressive property. Embedding dimension $K=512$.
    - **Design Motivation**: The U-Net-style design enables global attention at low resolution (reducing computational cost) while preserving high-resolution details through skip connections. Causal temporal attention ensures that predictions depend only on past frames.

3. **Cross-Modal Knowledge Transfer Mechanism**:

    - **Function**: Achieves implicit knowledge transfer across modalities through a shared frame-level representation.
    - **Mechanism**: Since all modalities share the same visual input space (64×64 frames) and the model contains no modality-specific components, the frame-level prediction capabilities acquired from one modality naturally transfer to others. For instance, learning the regularities of text-rendered frame-to-frame prediction benefits other tasks involving textual output.
    - **Design Motivation**: This is the core advantage of unified representation—sharing a representation space across modalities eliminates cross-modal alignment problems and allows the model to accumulate generalizable knowledge across tasks within a unified framework.

### Loss & Training

The model is trained with Multi-Scale Structural Similarity (MS-SSIM) loss, using a learning rate of $3 \times 10^{-4}$, AdamW optimizer, dropout 0.1, and batch size 8–32 (adjusted according to sequence length). Each task is trained independently (single-task), with no pretrained weights. All models are trained on a single A100 GPU for at most 7 GPU-days.

## Key Experimental Results

### Main Results

| Task | Dataset | OCR F1 / Acc | Other Metrics | Baselines |
|------|---------|--------------|---------------|-----------|
| Text Classification | SST-2 | 76.8 / 75.5 | — | SOTA 91.3 (pretrained); no-pretrain baseline comparable |
| Image Classification | CIFAR-10 | 89.1 / 89.1 | — | ViT+pretrain 99.5; PCANet 77.1 |
| Video Classification | TinyVIRAT | 30.4* (macro F1) | — | ResNet50 29.1; WideResNet 32.6 |
| Audio Classification | AudioMNIST | 96.9 / 97.1 | — | AlexNet 95.82 |
| Video QA | CLEVRER | 52.4 / 52.5 | — | LSTM 34.7; LSTM+CNN 51.8 |
| Object Tracking | LaSOT | — | IoU 0.63 | Direct tracking; stable performance |
| Video Colorization | TinyVIRAT | — | CDC 0.0169; colorfulness 73.1 | Original data 70.6 |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Text truncation effect | Restricting SST-2 to ≤20 tokens improves F1 to 80.0 |
| Attention visualization | Spatial attention focuses on key objects/text; temporal attention focuses on information-dense frames |
| Audio error analysis | Most common confusion: *four* vs. *five* (phonetically similar) |
| Tracking degradation | Bounding box erosion in later frames due to accumulated autoregressive pixel errors |
| Colorization trade-off | Higher color diversity than ground truth (73.1 vs. 70.6) but poorer temporal consistency |

### Key Findings

- **Audio classification achieves the best performance**: 97.1% accuracy surpasses the AlexNet baseline, demonstrating that spectrograms are highly effective visual representations for audio classification.
- **Usable performance is achievable without pretraining**: All tasks are trained from scratch, and results approach or exceed no-pretrain baselines in most cases, validating the feasibility of the reformulation paradigm.
- **Attention analysis confirms cross-modal understanding**: The model attends to sentiment-bearing keywords (*nightmare*, *painful*) in sentiment classification, and to question keywords (*color*, *metal*) and object trajectories in CLEVRER.
- **Primary bottleneck is OCR decoding of text outputs**: A substantial portion of training time is spent learning to generate readable text; pretraining could address this issue.

## Highlights & Insights

- **Radical unification principle**: The "everything is a video" paradigm is pushed to the extreme—one frame per text token, audio converted to spectrograms. Although this may appear brute-force, it establishes an important point: visual frame sequences can indeed serve as a universal cross-modal representation without information loss.
- **Simplicity through the absence of modality-specific components**: The model contains no components designed for any particular modality—the same Transformer processes text frames and image frames alike. This thoroughgoing unification is not achieved by other multimodal models such as FLAVA or GPT-4V.
- **Theoretical grounding for multimodal foundation models**: If next-frame prediction can serve as a cross-modal "supertask," then—analogous to the role of LLMs in NLP—a sufficiently large video prediction model could in principle become a foundation model spanning all modalities.

## Limitations & Future Work

- Tasks are trained independently; joint multi-task training has not been validated, yet it is essential for realizing a truly unified model.
- The 64×64 resolution limits the model's ability to handle fine-grained tasks such as small-object detection and long-text understanding.
- The text rendering → OCR decoding pipeline introduces additional errors, and OCR itself is imperfect.
- No pretraining is employed—the paper acknowledges that significant training time is spent learning "how to output text," a problem that pretraining could substantially alleviate.
- Future directions include large-scale multi-task pretraining, higher-resolution inputs, and extension to additional modalities (e.g., tactile sensing, point clouds).

## Related Work & Insights

- **vs. UNITER/UniT**: UNITER preprocesses images with RNNs before feeding them to a Transformer; UniT uses independent encoders with a shared decoder. This paper contains no modality-specific components whatsoever, representing a more thoroughgoing unification.
- **vs. data2vec**: data2vec projects multiple modalities into a shared latent space. This paper converts all modalities into the same visual input space directly—unification is performed at the *input* level rather than at the *representation* level.
- **vs. GPT-4V/FLAVA**: These foundation models also process visual inputs via ViT, but text and vision still follow separate processing pipelines. This paper demonstrates that complete unification at the model level is achievable.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The "everything is a video" unification paradigm is bold and highly original, fundamentally rethinking modality representation.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers 7 tasks, but all are simple benchmarks with no joint multi-task training experiments.
- **Writing Quality**: ⭐⭐⭐⭐ The presentation is clear, with detailed descriptions of the reformulation approach for each task.
- **Value**: ⭐⭐⭐⭐ High value as a proof of concept, offering a new direction for unified multimodal foundation models, though substantial performance improvements remain to be achieved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](../../NeurIPS2025/audio_speech/unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)
- [\[NeurIPS 2025\] AudSemThinker: Enhancing Audio-Language Models through Reasoning over Semantics of Sound](../../NeurIPS2025/audio_speech/audsemthinker_enhancing_audio-language_models_through_reasoning_over_semantics_o.md)
- [\[ICCV 2025\] MUG: Pseudo Labeling Augmented Audio-Visual Mamba Network for Audio-Visual Video Parsing](mug_pseudo_labeling_augmented_audio-visual_mamba_network_for_audio-visual_video_.md)
- [\[NeurIPS 2025\] Target Speaker Extraction Through Comparing Noisy Positive and Negative Audio Enrollments](../../NeurIPS2025/audio_speech/target_speaker_extraction_through_comparing_noisy_positive_and_negative_audio_en.md)

</div>

<!-- RELATED:END -->
