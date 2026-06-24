---
title: >-
  [Paper Note] Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering
description: >-
  [ACL 2025][Audio & Speech][Music AVQA] Sparsify proposes a three-level sparsification strategy (sparse masking + adaptive sparse merging + key-subset selection) for Music Audio-Visual Question Answering (Music AVQA). It achieves SOTA on both MUSIC-AVQA and v2.0 benchmarks (81.75%/81.30%), reduces training time by 28.32%, and retains 74% of full-data performance using only 25% of the data.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Music AVQA"
  - "sparse learning"
  - "multimodal QA"
  - "token merging"
  - "data efficiency"
date: 2026-05-08
content_hash: ddc464ae24858c2c
---

# Sparsify: Learning Sparsity for Effective and Efficient Music Performance Question Answering

**Conference**: ACL 2025  
**arXiv**: [2506.01319](https://arxiv.org/abs/2506.01319)  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Music AVQA, sparse learning, multimodal QA, token merging, data efficiency

## TL;DR
Sparsify proposes a three-level sparsification strategy (sparse masking + adaptive sparse merging + key-subset selection) for Music Audio-Visual Question Answering (Music AVQA). It achieves SOTA on both MUSIC-AVQA and v2.0 benchmarks (81.75%/81.30%), reduces training time by 28.32%, and retains 74% of full-data performance using only 25% of the data.

## Background & Motivation
**Background**: Music AVQA requires models to understand instrumental performance details (gestures, rhythm, phrases) in continuous and dense audio-visual streams and answer questions regarding sound sources, counting, and temporal sequence.

**Limitations of Prior Work**:
   - Existing methods (AVST, LAVisH, DG-SCT) rely on dense representations, making it difficult to effectively isolate key information from continuous audio-visual signals.
   - Feature extraction and inference lack efficient redundancy reduction mechanisms.
   - All training samples are treated equally without any prioritization strategy, leading to training inefficiency.

**Key Challenge**: The dense and continuous nature of music performance data results in massive redundancy, yet simple temporal pruning may discard fine-grained temporal and semantic information.

**Core Idea**: Simultaneously introduce sparsity at the representation, token, and sample levels to improve performance while enhancing efficiency.

## Method

### Overall Architecture
Based on the AMUSE encoder, Sparsify integrates three sparsification strategies in an end-to-end pipeline: (1) Sparse Masking randomly masks 50% of the visual and audio patches during the first 3 epochs; (2) Adaptive Sparse Merging dynamically filters and merges redundant tokens using IQR throughout training; (3) Key-subset Selection identifies high-value training samples to reduce data scale.

### Key Designs

1. **Sparse Masking**:

    - Function: Randomly mask 50% of visual (image patches) and audio (mel spectrogram patches) modalities during the first 3 epochs of pre-training.
    - Mechanism: A unified masking design maintains cross-modal consistent sparsity.
    - Design Motivation: Forcing the model to learn to extract key information from incomplete inputs in the early stage, similar to the concept of MAE, which reduces early-stage training computations.

2. **Adaptive Sparse Merging**:

    - Function: Dynamically identify and merge redundant tokens based on cross-modal attention.
    - Mechanism: Evaluate token importance using attention scores $\mathbf{a} = \text{softmax}(QK^T/\sqrt{d})V$. Filter and retain the upper quartile of tokens as key tokens via IQR. Merge the remaining tokens into the nearest key token based on similarity $\text{Sim}(\mathbf{tok}_i, \mathbf{tok}_j) = \mathbf{k}_i \cdot \mathbf{k}_j^T$ through clustering.
    - Design Motivation: IQR is more robust than a fixed pruning ratio, adaptively accommodating varying levels of redundancy across different samples.

3. **Key-subset Selection**:

    - Function: Identify the most valuable training samples to reduce data scale.
    - Mechanism: Two-stage classification — samples with a loss higher than the mean are classified as hard samples (D₁), while the rest are easy samples (D₂). Hard samples are aggregated epoch-by-epoch, with their importance weighted using an exponential decay weight $w_g = r^{g-1}$. The InfoBatch method scales gradients to prune redundant easy samples. Finally, the top-n samples are selected to form the key subset D₃.
    - Design Motivation: Prioritizing the training of hard samples accelerates convergence, while the exponential decay ensures higher weights for recently hard samples.

## Key Experimental Results

### Main Results (MUSIC-AVQA)

| Method | Audio QA | Visual QA | AV QA | **Overall** |
|------|----------|-----------|-------|-------------|
| AVST | 73.87 | 74.40 | 65.82 | 71.59 |
| LAVisH | 76.86 | 76.29 | 77.62 | 76.10 |
| DG-SCT | 76.34 | 82.08 | 67.48 | 74.62 |
| **Sparsify** | **80.38** | **84.43** | **79.89** | **81.75** |

The improvement on AV QA is the most significant (+12.41 vs DG-SCT).

### Efficiency Comparison

| Configuration | Training Time | Note |
|------|---------|------|
| Dense baseline | 173h | 100% |
| **Sparsify (full)** | **124h** | **-28.32%** |
| 25% key-subset | - | 74% performance retained |

### MUSIC-AVQA v2.0

| Method | Overall |
|------|--------|
| DG-SCT | 74.53 |
| **Sparsify** | **81.30** (+6.77) |

### Key Findings
- **AV QA achieves the most significant improvement** (+12.41/+9.71), demonstrating that sparsification effectively reduces cross-modal redundant interference.
- **25% data retains 74% performance**, indicating that Key-subset Selection effectively identifies high-value samples.
- **Improvements are especially notable on comparative and temporal questions**: Comparative +13.9, Temporal +12.75, showing that sparsification assists the model in focusing better on key temporal points.

## Highlights & Insights
- **Orthogonality of the three-level sparsification**: Representation-level (masking), token-level (merging), and sample-level (selection) tackle different dimensions of redundancy without interfering with each other.
- **IQR adaptive threshold** is more robust than fixed-ratio pruning and can be transferred to other multimodal tasks requiring token merging.
- **The exponential decay strategy in Key-subset Selection** is a simple yet effective variant of curriculum learning.

## Limitations & Future Work
- Validated only on the Music AVQA dataset; generalization to other dense audio-visual tasks is unknown.
- Lack of hyperparameter sensitivity analysis on the 50% masking rate.
- The Key-subset algorithm involves multiple hyperparameters (k, r, G) and lacks sufficient ablation studies.
- No comparison with the latest LLM-based multimodal methods (e.g., VideoLLM).

## Related Work & Insights
- **vs DG-SCT**: DG-SCT models audio-visual relationships using dense graph convolutions. Sparsify employs sparse strategies to reduce redundancy, significantly outperforming it on AV QA (+12.41).
- **vs LAVisH**: LAVisH freezes pre-trained encoders but yields lower overall performance. Sparsify's sparse strategy leverages encoder capacity more effectively.
- **Insight**: Token merging strategies can be transferred from VLM efficiency techniques (such as those evaluated in EffiVLM-Bench).

## Rating
- Novelty: ⭐⭐⭐⭐ The three-level sparsification framework is a novel combination in the Music AVQA field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated on two benchmarks, compared with multiple baselines, including efficiency analysis.
- Writing Quality: ⭐⭐⭐ Methodology is clearly described, but ablation studies are not sufficiently deep.
- Value: ⭐⭐⭐ The domain is relatively narrow (Music AVQA), but the sparsification concepts are generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Music Audio-Visual Question Answering Requires Specialized Multimodal Designs](../../ACL2026/audio_speech/music_audio-visual_question_answering_requires_specialized_multimodal_designs.md)
- [\[ACL 2026\] Jamendo-MT-QA: A Benchmark for Multi-Track Comparative Music Question Answering](../../ACL2026/audio_speech/jamendo-mt-qa_a_benchmark_for_multi-track_comparative_music_question_answering.md)
- [\[ICLR 2026\] Query-Guided Spatial-Temporal-Frequency Interaction for Music Audio-Visual Question Answering](../../ICLR2026/audio_speech/query-guided_spatial-temporal-frequency_interaction_for_music_audio-visual_quest.md)
- [\[AAAI 2026\] SpikCommander: A High-Performance Spiking Transformer with Multi-View Learning for Efficient Speech Command Recognition](../../AAAI2026/audio_speech/spikcommander_a_high-performance_spiking_transformer_with_multi-view_learning_fo.md)
- [\[NeurIPS 2025\] Ethics Statements in AI Music Papers: The Effective and the Ineffective](../../NeurIPS2025/audio_speech/ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)

</div>

<!-- RELATED:END -->
