---
title: >-
  [Paper Note] FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing
description: >-
  [NeurIPS 2025][Model Compression][speech compression] FastLongSpeech is proposed to compress redundant speech representations via an iterative fusion strategy and to transfer short-speech capabilities to long-speech scenarios through dynamic compression training, enabling large speech-language models (LSLMs) to efficiently process long speech without long-speech training data, achieving state-of-the-art performance on long-speech QA with a 70% improvement in inference efficie…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "speech compression"
  - "long-speech processing"
  - "iterative fusion"
  - "CTC"
  - "dynamic compression training"
date: 2026-05-08
content_hash: d3abe5880937ebb4
---

# FastLongSpeech: Enhancing Large Speech-Language Models for Efficient Long-Speech Processing

**Conference**: NeurIPS 2025
**arXiv**: [2507.14815](https://arxiv.org/abs/2507.14815)  
**Code**: Available (github.com/ictnlp/FastLongSpeech)  
**Area**: Model Compression / Speech Model Efficiency
**Keywords**: speech compression, long-speech processing, iterative fusion, CTC, dynamic compression training

## TL;DR

FastLongSpeech is proposed to compress redundant speech representations via an iterative fusion strategy and to transfer short-speech capabilities to long-speech scenarios through dynamic compression training, enabling large speech-language models (LSLMs) to efficiently process long speech without long-speech training data, achieving state-of-the-art performance on long-speech QA with a 70% improvement in inference efficiency.

## Background & Motivation

LSLMs such as Qwen2-Audio perform well on short-speech tasks, but face two major challenges when processing long speech (>30 seconds):

**Scarcity of training data**: Long-speech alignment and instruction fine-tuning data are extremely limited and costly to construct.

**High computational cost**: Speech representation sequences are typically more than 4× longer than their text equivalents, causing the computational cost of LLM inference to surge dramatically for long speech.

Limitations of existing approaches:
- **Cascaded methods** (ASR→LLM): error propagation and loss of paralinguistic information
- **NTK-RoPE**: extends the positional encoding limit but does not compress the sequence, leaving computation unchanged (61.21 TFLOPs)
- **Naive compression** (random sampling, average pooling): significant information loss and poor generation quality
- Only a few LSLMs (e.g., Gemini) handle 30-minute speech by constructing large-scale long-speech datasets, at prohibitive cost

**Core Problem**: How can LSLMs efficiently process long speech using only short-speech data?

## Method

### Overall Architecture

FastLongSpeech builds upon Qwen2-Audio with an additional speech extractor module:

```
Raw speech s → Audio encoder (Whisper) → Speech representation h (25Hz)
→ Extractor (CTC decoder + Iterative Fusion) → Compressed representation h' (length ≤ L)
→ LLM → Text response y
```

For long-speech inference: the input is first segmented into 30-second chunks → each chunk passes through the audio encoder independently → the outputs are concatenated into a full speech representation → iterative fusion compresses it to the target length $L$.

### Key Designs

#### 1. Iterative Fusion

**Core Idea**: Progressively merge redundant frames while retaining frames with high information density. Each iteration halves the sequence length until the target length $L$ is reached.

**Two metrics**:

**Content Density**: The sum of non-blank token probabilities from the CTC decoder output, measuring the textual information content of each frame:
$$d_j = \sum_{a_j \neq \epsilon} p_{ctc}(a_j \mid h_j)$$

**Inter-frame Similarity**: Cosine similarity between adjacent frames:
$$e_{j,j+1} = \frac{h_j h_{j+1}}{|h_j| |h_{j+1}|}$$

**Iterative Process**:
1. Compute the current length $T^{(m)}$ and target length $T^{(m+1)} = \lfloor T^{(m)}/2 \rfloor$ (if $T^{(m)} > 2L$), otherwise $T^{(m+1)} = L$
2. Compute the number of frames to eliminate: $r^{(m)} = T^{(m)} - T^{(m+1)}$
3. Identify the $r^{(m)}$ most similar adjacent frame pairs
4. Group consecutively identified frames into spans; within each span, merge frames into a single frame via content-density-weighted fusion
5. Repeat until sequence length $\leq L$

**Key Advantage**: The progressively shrinking receptive field (halved each round) preserves semantic information better than one-shot compression; content density guides retention of high-information frames.

#### 2. Dynamic Compression Training (DCT)

**Problem**: The LLM has only been exposed to speech representations at their original length; feeding compressed representations directly causes a distributional mismatch.

**Solution**: During training, the target length $L$ is sampled randomly, enabling the LLM to adapt to compressed representations at varying compression ratios:

$$L_{dct} = -\sum_{L \sim \mathcal{U}(\mathbf{L})} \log p(\mathbf{y} \mid \mathbf{x}, \text{IF}(\mathbf{h}, L))$$

where $L$ is sampled uniformly from $\{750, 400, 200, 100, 50, 25, 12\}$. This trains the LLM to handle inputs ranging from uncompressed to 60× compressed.

### Loss & Training

**Two-stage training**:

**Stage 1 — CTC Training**:
- Only the CTC decoder is trained to learn content density estimation for speech frames
- Training data: LibriSpeech 960h + MLS 3000h (ASR data)
- All other modules are frozen

**Stage 2 — Dynamic Compression Training**:
- The LLM component of Qwen2-Audio is fine-tuned with LoRA
- Training data: OpenASQA (5.9kh) + LibriSQA (360h) + Common Voice (1.7kh), all short speech (<30s)
- CTC decoder is frozen
- Speech window $L = 750$ (original Qwen2-Audio setting)

## Key Experimental Results

### Main Results: Long-Speech Spoken QA

| Method | Score (↑) | Note |
|--------|-----------|------|
| Random | 2.54 | Random frame sampling |
| Similar (MostSim) | 3.08 | Merge most similar frames |
| AvgPool | 3.10 | Average pooling |
| NTK-RoPE | 3.44 | Extended positional encoding (no compression) |
| **FastLongSpeech** | **3.55** | Iterative fusion + DCT |

Under the same speech window as NTK-RoPE, FastLongSpeech achieves the best performance on long-speech understanding.

### Inference Efficiency

| Method | Score | TFLOPs (↓) | Time/s (↓) |
|--------|-------|-----------|------------|
| NTK-RoPE | 3.44 | 61.21 | 4.80 |
| Cascaded (Whisper+LLM) | 3.75 | n/a | 17.23+1.38 |
| **FastLongSpeech** | **3.55** | **26.44** | **1.47** |

Computation is reduced by 57%, inference is 3.3× faster than NTK-RoPE and 7× faster than the cascaded baseline.

### Short-Speech Efficiency (LibriTTS OpenASQA)

| Method | Score (↑) | TFLOPs (↓) |
|--------|-----------|-----------|
| Baseline (Qwen2-Audio) | 3.73 | 9.79 |
| Ours ($L=400$) | **3.80** | 8.54 |
| Ours ($L=200$) | **3.87** | **5.64** |
| Ours ($L=100$) | 3.71 | 4.17 |

On short speech, the proposed method matches or exceeds the baseline at half the computational cost.

### Ablation Study

| Method | Score (↑) |
|--------|-----------|
| FastLongSpeech (full) | **3.55** |
| w/o DCT | 3.33 (−0.22) |
| w/o Iterative Fusion (one-shot compression) | 3.41 (−0.14) |
| w/o Content Density (uniform-weight fusion) | 3.28 (−0.27) |

All three components contribute significantly. Content density guidance contributes the most, underscoring the importance of distinguishing informative from redundant frames.

### Key Findings

1. Iterative (multi-round halving) compression outperforms one-shot compression: progressively expanding the receptive field better aggregates semantic content.
2. CTC content density is an effective measure of frame informativeness, guiding retention of high-information frames.
3. Dynamic compression training successfully transfers short-speech capabilities to long-speech scenarios without requiring any long-speech training data.
4. For ASR tasks, WER increases by only 0.23 over Qwen2-Audio at a low compression ratio ($L=400$), but degrades substantially at high compression ($L=100$), indicating that the optimal compression ratio is task-dependent.
5. The method generalizes directly to Qwen2.5-Omni without DCT retraining.

## Highlights & Insights

- **Zero long-speech training data**: Long-speech processing capability is transferred via dynamic compression training using only short-speech data (<30s).
- **Information-aware compression**: The CTC output distribution naturally provides frame-level information density, proving more effective than simple similarity measures or random sampling.
- **Flexible efficiency–quality trade-off**: The target length $L$ can be adjusted freely to balance inference efficiency and generation quality.
- **LongSpeech-Eval benchmark**: A novel evaluation benchmark for long-speech understanding is constructed, filling a gap in the field.

## Limitations & Future Work

1. **Only evaluated on Qwen2-Audio**: Generalizability to other LSLMs remains to be verified.
2. **Long-speech training data still has potential**: As long-speech data accumulates, direct training may outperform compression-based transfer.
3. **CTC decoder introduces additional parameters**: Although relatively lightweight, it increases system complexity.
4. **Significant ASR degradation at high compression ratios**: The trade-off between content fidelity and efficiency requires further optimization.
5. Promising future directions include end-to-end joint training of the CTC decoder and LLM, and adaptive compression ratio selection (dynamically adjusting $L$ based on speech complexity).

## Related Work & Insights

- **SpeechPrune / FastAdaSP**: Token selection/pruning strategies; the fusion strategy in FastLongSpeech preserves more information by merging rather than discarding frames.
- **StreamUni**: Segmentation strategies for real-time speech translation; iterative fusion can be viewed as an offline counterpart for efficient compression.
- **NTK-RoPE**: A classical approach for extending context length, but without reducing computational cost.
- **Insight**: The high redundancy of speech (frame rate far exceeding the text token rate) naturally accommodates compression, and CTC blank probabilities serve as an excellent proxy for frame-level redundancy.

## Rating

- **Novelty**: ★★★★☆ (the combination of iterative fusion, CTC content density, and dynamic compression training is novel)
- **Technical Depth**: ★★★★☆ (clear problem decomposition; each component has well-motivated design with theoretical grounding)
- **Experimental Thoroughness**: ★★★★☆ (multi-task evaluation, ablation study, and efficiency analysis are comprehensive, though limited to a single base model)
- **Value**: ★★★★★ (no long-speech data required, plug-and-play, substantial inference speedup — high practical applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Compress, Gather, and Recompute: REFORMing Long-Context Processing in Transformers](compress_gather_and_recompute_reforming_long-context_processing_in_transformers.md)
- [\[NeurIPS 2025\] Data Efficient Adaptation in Large Language Models via Continuous Low-Rank Fine-Tuning](data_efficient_adaptation_in_large_language_models_via_continuous_low-rank_fine-.md)
- [\[ICML 2025\] LaCache: Ladder-Shaped KV Caching for Efficient Long-Context Modeling of Large Language Models](../../ICML2025/model_compression/lacache_ladder-shaped_kv_caching_for_efficient_long-context_modeling_of_large_la.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)

</div>

<!-- RELATED:END -->
