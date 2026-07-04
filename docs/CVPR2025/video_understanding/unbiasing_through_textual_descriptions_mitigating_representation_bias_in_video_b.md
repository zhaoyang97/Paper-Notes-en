---
title: >-
  [Paper Note] Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks
description: >-
  [CVPR 2025][Video Understanding][Video benchmark unbiasing] Ours proposes the UTD method, which leverages VLMs and LLMs to generate video frame textual descriptions to systematically analyze object, temporal, and commonsense biases in video benchmarks, and constructs unbiased test splits to make video understanding evaluation more robust and unbiased.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video benchmark unbiasing"
  - "representation bias"
  - "textual descriptions"
  - "VLM/LLM"
  - "dataset analysis"
date: 2026-05-08
content_hash: ce637f40f2c4d084
---

# Unbiasing through Textual Descriptions: Mitigating Representation Bias in Video Benchmarks

**Conference**: CVPR 2025  
**arXiv**: [2503.18637](https://arxiv.org/abs/2503.18637)  
**Code**: [https://utd-project.github.io/](https://utd-project.github.io/) (Project Page)  
**Area**: Video Understanding  
**Keywords**: Video benchmark unbiasing, representation bias, textual descriptions, VLM/LLM, dataset analysis

## TL;DR

Ours proposes the UTD method, which leverages VLMs and LLMs to generate video frame textual descriptions to systematically analyze object, temporal, and commonsense biases in video benchmarks, and constructs unbiased test splits to make video understanding evaluation more robust and unbiased.

## Background & Motivation

1. **Background**: Video understanding relies on extensive benchmark datasets (such as UCF101, Kinetics, etc.) to evaluate model performance, but existing studies have pointed out that these benchmarks may suffer from representation bias—where models can make correct predictions by merely identifying objects or looking at a single frame.
2. **Limitations of Prior Work**: Some unbiasing schemes (e.g., human segmentation with black background replacement, action transfer to unrelated scenes) introduce visual artifacts or domain shifts, making unbiased datasets rarely adopted by the mainstream. Meanwhile, works focusing on temporal bias mainly construct new benchmarks rather than systematically analyzing existing ones.
3. **Key Challenge**: Object representation bias benefits models with strong object recognition capabilities, masking their true abilities in temporal reasoning and action understanding, which compromises the fairness of evaluation.
4. **Goal**: (1) Design a scalable and automated method to measure and analyze various representation biases in video benchmarks; (2) construct unbiased test splits for existing benchmarks; (3) systematically evaluate 30 SOTA models.
5. **Key Insight**: Since directly modifying videos introduces artifacts, textual descriptions can serve as an intermediate representation—using VLMs to generate frame descriptions and then LLMs to extract specific concepts (only objects, only activities, or only verbs), thereby precisely controlling the representation content.
6. **Core Idea**: Use textual descriptions as proxy representations to measure bias, and construct unbiased splits by excluding highly biased samples, without modifying any video data.

## Method

### Overall Architecture

Input video frame sequence → VLM (LLaVA-1.6-Mistral-7B) to generate frame-by-frame textual descriptions → LLM to extract different concept categories (objects/activities/verbs) → Combine into different temporal configurations (single frame / mean / sequence) → Compute bias metrics using text embedding models → Exclude highly biased samples to construct unbiased splits.

### Key Designs

1. **Text Description-Based Representation Bias Analysis Framework**

    - **Function**: Precisely measure representation biases across three independent dimensions: conceptual bias (whether objects alone are sufficient for prediction), temporal bias (whether a single frame is sufficient), and commonsense vs. dataset bias.
    - **Mechanism**: A VLM is used to generate detailed descriptions $d_{n,i} = d(f_{n,i})$ for each frame, and then an LLM extracts object lists $o_{n,i}$, activities $a_{n,i}$, and verbs $\nu_{n,i}$ from the descriptions based on specific prompts. These textual descriptions can be combined into different temporal configurations: middle frame, max-score frame, frame average, or frame sequence ("Frame 1: ... Frame 2: ..."). A pretrained text embedding model (such as E5) is then used to measure zero-shot classification/retrieval performance as the bias metric $M(D, \phi)$.
    - **Design Motivation**: Compared to directly cropping or patching specific content in videos, textual descriptions can isolate specific concepts (objects/actions) more cleanly, avoiding information leakage and artifacts; furthermore, the textual format naturally fits the zero-shot reasoning capabilities of LLMs.

2. **Commonsense Bias vs. Dataset Bias Separation**

    - **Function**: Distinguish between reasonable commonsense reasoning (e.g., "predicting playing piano when seeing a piano") and spurious correlations in the dataset (e.g., "mirrors and flowers always appearing in makeup videos").
    - **Mechanism**: Commonsense bias is measured using a zero-shot text embedding model (without access to the training set), while dataset bias is measured by training a linear classifier on the training set text embeddings. The difference between the two represents pure dataset bias. Experiments show that performance improves significantly after training (e.g., UCF101 objects from 63.3% to 80.3%), indicating a large number of spurious object-label correlations in the dataset.
    - **Design Motivation**: Commonsense bias itself might be reasonable (as objects and actions indeed correlate in daily life), but dataset bias is a harmful spurious correlation, which needs to be treated separately.

3. **Automatic Debiasing Strategy Based on Consistency Voting**

    - **Function**: Automatically identify and exclude test samples with high object bias to construct an unbiased evaluation set.
    - **Mechanism**: Utilize 3 text embedding models with different prompts $\times$ 3 bootstrapped training sets = 9 models in total. For each sample, it is excluded only when all 9 models consistently judge that "the sample can be correctly classified/retrieved solely based on objects." The exclusion ratio is automatically determined by the severity of the bias. A class-balanced version of the unbiased split is also additionally constructed.
    - **Design Motivation**: Multi-model voting ensures the stability of the unbiasing results and avoids mistakenly excluding normal samples due to the random fluctuations of a single model.

### Loss & Training

This work does not involve end-to-end training of new models—rather, it is an analysis framework. The linear classifier is trained with standard cross-entropy, and the text embedding model uses pretrained weights (E5-large-v2). The key parameter is the voting threshold (excluded only when all 9/9 models are consistent).

## Key Experimental Results

### Main Results

The analysis covers 12 video datasets, including 6 classification datasets (UCF101, SSv2, K400/600/700, MiT) and 6 retrieval datasets (MSRVTT, DiDeMo, ActivityNet, LSMDC, YouCook2, S-MiT).

| Dataset | Object Bias (seq) | Activity Bias (seq) | Verb Bias (seq) | Retained % after Unbiasing |
|--------|-------------|-------------|-------------|-----------|
| UCF101 | 63.3% | 67.4% | 50.8% | 27% |
| SSv2 | 5.3% | 6.4% | 5.8% | 93% |
| K400 | 45.9% | 45.2% | 24.8% | 45% |
| K700 | 37.0% | 36.7% | 17.6% | 52% |
| MiT | 21.0% | 21.0% | 16.2% | 76% |

### Ablation Study

| Configuration | UCF101 Original | UCF101 UTD-split | Performance Drop |
|------|-----------|-----------------|---------|
| VideoMAE-B-K400 | 89.2 | 81.3 | -7.9 |
| VideoMAE-B-UH | 88.4 | 79.2 | -9.2 |
| VideoMAE2-B | 89.7 | 82.5 | -7.2 |
| InternVid-B | 92.1 | 82.2 | -9.9 |

### Key Findings

- UCF101 suffers from the most severe bias (with 73% of samples showing object bias), whereas SSv2 is the cleanest (only 7%), consistent with the design intention of SSv2 focusing on manipulative actions.
- Looking solely at objects yields an accuracy of 63.3% on UCF101 (only 5.3% on SSv2), indicating that the "video understanding capability" tested by many benchmarks is actually testing object recognition.
- Dataset bias far exceeds commonsense bias (UCF101 objects: 63.3 $\rightarrow$ 80.3, +17), demonstrating that training sets contain a large number of spurious object-label correlations.
- The performance of all 30 models drops after unbiasing, with larger models like InternVid experiencing more severe degradation (-9.9), suggesting that larger models might be better at taking shortcuts by exploiting object biases.

## Highlights & Insights

- **The idea of using text as an intermediate representation is highly clever**: Using textual descriptions to precisely isolate concepts such as objects, actions, and verbs is much cleaner than pixel-level operations (cropping, inpainting). This idea can be generalized—it is worth attempting to use textual descriptions to analyze biases in other modalities (e.g., image benchmarks/multimodal benchmarks).
- **Robust unbiasing strategy via 9-model consistency voting**: Avoids manually setting thresholds, letting the unbiasing ratio be determined by the data itself.
- **Quantitatively confirmed uniqueness of SSv2**: Explains from a data perspective why model performance variance on SSv2 differs from patterns observed in other datasets.

## Limitations & Future Work

- The quality of VLM descriptions directly affects the accuracy of bias analysis. Currently, only LLaVA-1.6-Mistral-7B is used; stronger VLMs might discover more fine-grained biases.
- Currently, only object unbiasing has been performed, whereas temporal or activity unbiasing has not—future work can further construct unbiased splits targeting different bias dimensions.
- The analysis is conducted offline in a static manner. Exploring how to leverage bias information during the training phase (e.g., re-weighting biased samples) has not yet been investigated.
- Textual descriptions may lose visual details (such as object textures or motion speeds), potentially leading to underestimated bias.

## Related Work & Insights

- **vs. HAT/Diving48**: These methods unbias by modifying videos (segmenting persons) or designing specific datasets. The advantage of UTD is that it does not modify any videos and is applicable to any existing benchmarks.
- **vs. TempAct/TemporalBench**: These works construct new benchmarks specifically to evaluate temporal understanding, whereas UTD analyzes the extent of temporal bias in existing benchmarks, making them complementary.
- This paper provides a systematic analysis tool for video benchmarks, which can serve as a standard pipeline for evaluating the quality of new datasets.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of unbiasing through textual descriptions is novel but not overly complex.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed analysis with 12 datasets, 30 models, and multiple dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear structured formulation, but heavily packed with equations making it slightly heavy to read.
- Value: ⭐⭐⭐⭐ Provides useful unbiased splits and analysis tools, offering a practical contribution to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](../../NeurIPS2025/video_understanding/seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)
- [\[CVPR 2025\] SEAL: SEmantic Attention Learning for Long Video Representation](seal_semantic_attention_learning_for_long_video_representation.md)
- [\[CVPR 2025\] Heterogeneous Skeleton-Based Action Representation Learning](heterogeneous_skeleton-based_action_representation_learning.md)
- [\[CVPR 2025\] Learning Audio-Guided Video Representation with Gated Attention for Video-Text Retrieval](learning_audio-guided_video_representation_with_gated_attention_for_video-text_r.md)
- [\[CVPR 2025\] H-MoRe: Learning Human-centric Motion Representation for Action Analysis](h-more_learning_human-centric_motion_representation_for_action_analysis.md)

</div>

<!-- RELATED:END -->
