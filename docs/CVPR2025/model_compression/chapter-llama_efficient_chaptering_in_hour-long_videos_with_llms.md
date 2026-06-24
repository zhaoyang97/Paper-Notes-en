---
title: >-
  [Paper Note] "Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs"
description: >-
  [CVPR2025][Model Compression][Paper Note] Academic paper note for "Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs".
tags:
  - CVPR2025
  - Model Compression
date: 2025-03-31
content_hash: be53ec85f16c138a
---
# Chapter-Llama: Efficient Chaptering in Hour-Long Videos with LLMs

**Authors**: Lucas Music, Stanislas Music, Antoine Yang, Cordelia Schmid, Ivan Laptev  
**Institutions**: École des Ponts ParisTech / Inria / Google DeepMind  
**Conference**: CVPR 2025  

## Background & Motivation

Video chaptering is the task of automatically segmenting long videos into semantically coherent chapters and generating chapter titles, which is crucial for video browsing, search, and understanding. With the explosive growth of long video content (1 hour+) on platforms like YouTube and Bilibili, the demand for automatic chaptering has become increasingly urgent:

**Computational Bottleneck of Long Video Processing**: Hour-long videos contain hundreds of thousands of frames. Even when sampling at a low frame rate (e.g., 1 fps), thousands of frames still need to be processed. The context windows and computational resources of existing Video LLMs are insufficient to support such large-scale visual inputs.

**Inefficiency of Uniform Sampling**: Traditional methods sample a fixed number of frames (e.g., 100 frames) at equal intervals. However, the information density at chapter boundaries is much higher than within chapters, meaning uniform sampling wastes a significant amount of computation on redundant frames.

**Underutilized Complementarity of Visual and Speech Information**: Long videos typically contain rich speech information (e.g., explanations, dialogues). These speech signals naturally mark semantic transition points in the content, but existing methods mainly rely on visual features.

**Limitations of Prior Work**:
   - The performance of seq2seq models like Vid2Seq drops sharply when processing ultra-long videos.
   - General Video LLMs (e.g., Gemini) possess long-context capabilities but lack optimization specifically for the chaptering task.
   - Sliding window-based methods struggle to capture global semantic relationships across windows.

Chapter-Llama proposes a speech-guided frame selection strategy, achieving chaptering performance that surpasses existing methods using an extremely small number of frames (10.3 frames vs. 100 frames).

## Method

### Overall Architecture

Chapter-Llama consists of three core components: Speech-Guided Frame Selection, Visual-Language Encoding, and LLM-based chapter generation.

### Speech-Guided Frame Selection

**Core Idea**: Guiding visual frame selection using semantic transition points in speech transcripts.

**Mechanism**:

1. **Speech Transcription**: Use ASR (Automatic Speech Recognition) to obtain timestamp-aligned transcripts of the video.
2. **Text Semantic Segmentation**: Compute semantic similarity between adjacent speech segments using sentence embeddings.
3. **Transition Point Detection**: Detect sharp drops in the semantic similarity sequence, which correspond to content transitions.
4. **Frame Selection**: Select representative frames near each semantic transition point.

| Frame Selection Strategy | Avg. Frames | F1 ↑ |
|-----------|---------|------|
| Uniform Sampling 100 frames | 100 | 38.2 |
| Uniform Sampling 50 frames | 50 | 35.6 |
| Random Sampling 10 frames | 10 | 28.7 |
| **Speech-Guided (Ours)** | **10.3** | **45.3** |

### Visual-Language Encoding

The selected keyframes extract visual features through a visual encoder (CLIP ViT-L/14), which are then organized into a multimodal input sequence along with the speech transcript of the corresponding time period:

$$\text{Input} = [\text{SYS}] \oplus \bigoplus_{i=1}^{K} [\text{IMG}_i, \text{TIME}_i, \text{SPEECH}_i]$$

where $K \approx 10.3$ is the number of selected keyframes.

### LLM Chapter Generation

**Model Selection**: Llama-3.1-8B + LoRA Fine-Tuning

| Configuration Item | Setting |
|--------|------|
| Base Model | Llama-3.1-8B |
| Fine-Tuning Method | LoRA (rank=16, alpha=32) |
| Training Time | 40 minutes |
| Training Hardware | 4× H100 GPU |
| Output Format | JSON (timestamps + chapter titles) |

**Prompt Design**:

After receiving the multimodal input, the model generates structured chapter outputs:
```json
{"chapters": [
  {"start": "00:00:00", "title": "Introduction to..."},
  {"start": "00:05:32", "title": "Method overview..."},
  ...
]}
```

## Experimental Results

### Main Results (VidChapters-7M Validation Set)

| Method | F1 ↑ | Model Size | Frames |
|------|------|---------|------|
| Vid2Seq | 26.7 | 0.3B | 100 |
| VideoLLaMA2 | 31.2 | 7B | 32 |
| LLaVA-Video | 35.8 | 7B | 64 |
| Gemini-1.5-Pro (zero-shot) | 42.2 | >1T | All |
| **Chapter-Llama (Ours)** | **45.3** | 8B | **10.3** |
| Gain vs. Vid2Seq | **+69.8%** | - | - |

### Comparison with Gemini Series

| Model | F1 | Setting | Cost |
|------|-----|------|------|
| Gemini-1.5-Flash (zero-shot) | 38.7 | API | ~$0.5/video |
| Gemini-1.5-Pro (zero-shot) | 42.2 | API | ~$2.0/video |
| **Chapter-Llama** | **45.3** | Local | ~$0.01/video |

Chapter-Llama not only outperforms the zero-shot results of Gemini-1.5-Pro (+3.1 F1) but also operates at a running cost that is lower by two orders of magnitude.

### Analysis by Video Duration

| Video Duration | Vid2Seq F1 | Chapter-Llama F1 | Gain |
|----------|-----------|-----------------|------|
| < 10 min | 32.1 | 47.8 | +48.9% |
| 10-30 min | 27.4 | 45.6 | +66.4% |
| 30-60 min | 23.8 | 44.1 | +85.3% |
| > 60 min | 19.2 | 42.7 | +122.4% |

As the video duration increases, the advantage of Chapter-Llama becomes more pronounced, proving the effectiveness of speech-guided frame selection on long videos.

### Ablation Study

| Component | F1 |
|------|-----|
| Visual Only (Uniform 100 Frames) | 38.2 |
| Speech Transcript Only | 41.5 |
| Visual + Speech (Uniform Sampling) | 42.1 |
| **Visual + Speech-Guided Frame Selection** | **45.3** |

## Highlights & Insights

1. **Speech-Guided Frame Selection**: Guiding visual frame sampling using semantic transition points in speech, processing hour-long videos with an average of 10.3 frames, representing a 10x efficiency improvement.
2. **Extremely Efficient Training**: Training for only 40 minutes on 4x H100 GPUs is sufficient to outperform the trillion-parameter Gemini-1.5-Pro.
3. **Multimodal Fusion**: Effectively combining the complementary advantages of visual and speech information.
4. **Strong Generalization**: Demonstrates robust performance across videos of various lengths, especially ultra-long videos (1 hour+).

## Efficiency Analysis

| Metric | Vid2Seq | Gemini-1.5-Pro | Chapter-Llama |
|------|---------|----------------|---------------|
| Input Frames | 100 | ~3600 (1fps) | 10.3 |
| Inference Time / Video | ~15s | ~60s | ~5s |
| Trainable Parameters | 300M | Non-trainable | 4.2M (LoRA) |
| Training Cost | Multiple days/GPUs | N/A | **40min/4×H100** |

## Limitations & Future Work

- Highly dependent on the quality of speech transcription, limiting effectiveness on videos without speech (e.g., music videos, silent documentaries).
- ASR may produce errors in noisy environments or multilingual scenarios, affecting the quality of frame selection.
- The quality of generated chapter titles is limited by the capabilities of the LLM, which may be inaccurate for domain-specific content.
- The integration of other textual information, such as on-screen subtitles or comments in the video, has not been explored.

## Related Work & Insights

- Vid2Seq: A seq2seq-based dense video captioning model.
- Gemini-1.5: Google's long-context multimodal large model.
- LLaVA-Video: A LLaVA-based video understanding model.
- Llama-3.1: Meta's open-source large language model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Charm: The Missing Piece in ViT Fine-Tuning for Image Aesthetic Assessment](charm_the_missing_piece_in_vit_fine-tuning_for_image_aesthetic_assessment.md)
- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](../../ACL2025/model_compression/iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] Efficient Long Context Language Model Retrieval with Compression](../../ACL2025/model_compression/efficient_long_context_language_model_retrieval_with_compression.md)
- [\[CVPR 2025\] Distilling Long-tailed Datasets](distilling_long-tailed_datasets.md)
- [\[ICML 2025\] MKA: Memory-Keyed Attention for Efficient Long-Context Reasoning](../../ICML2025/model_compression/mka_memory-keyed_attention_for_efficient_long-context_reasoning.md)

</div>

<!-- RELATED:END -->
