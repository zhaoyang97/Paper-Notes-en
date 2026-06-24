---
title: >-
  [Paper Note] Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity
description: >-
  [CVPR 2025][Video Understanding][Video Anomaly Understanding] This paper proposes Holmes-VAU, constructing HIVAU-70k, a video anomaly understanding benchmark with over 70k multi-granularity annotations. It also designs an Anomaly-focused Temporal Sampler (ATS) that enables multimodal VLMs to focus on anomaly-dense regions, significantly outperforming existing methods on long-term video anomaly detection and reasoning tasks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Anomaly Understanding"
  - "Multi-granularity Annotation"
  - "Anomaly-focused Temporal Sampling"
  - "Multimodal Large Language Models"
  - "Hierarchical Instruction Data"
date: 2026-05-08
content_hash: b8448d635ce2e682
---

# Holmes-VAU: Towards Long-term Video Anomaly Understanding at Any Granularity

**Conference**: CVPR 2025  
**arXiv**: [2412.06171](https://arxiv.org/abs/2412.06171)  
**Code**: [https://github.com/pipixin321/HolmesVAU](https://github.com/pipixin321/HolmesVAU)  
**Area**: Image Segmentation  
**Keywords**: Video Anomaly Understanding, Multi-granularity Annotation, Anomaly-focused Temporal Sampling, Multimodal Large Language Models, Hierarchical Instruction Data

## TL;DR

This paper proposes Holmes-VAU, constructing HIVAU-70k, a video anomaly understanding benchmark with over 70k multi-granularity annotations. It also designs an Anomaly-focused Temporal Sampler (ATS) that enables multimodal VLMs to focus on anomaly-dense regions, significantly outperforming existing methods on long-term video anomaly detection and reasoning tasks.

## Background & Motivation

1. **Background**: Video Anomaly Understanding (VAU) is a core task for applications such as video surveillance, violent content analysis, and autonomous driving. Traditional methods mainly focus on frame-level anomaly scoring, treating anomaly detection as a closed-set prediction problem. Recently, multimodal approaches have begun to combine visual and textual information, utilizing VLMs for anomaly-related instruction tuning and text generation.

2. **Limitations of Prior Work**: Existing VAU datasets usually provide annotations at a single temporal granularity—either clip-level or video-level. Consequently, models can only understand instantaneous anomalies (e.g., explosions, fights) or complex events requiring long-term context (e.g., theft, arson), but not both. Furthermore, existing methods generally employ uniform sampling when processing long videos, which is prone to missing key anomalous frames or introducing excessive redundant computation.

3. **Key Challenge**: The lack of hierarchical, multi-granularity anomaly annotation data prevents models from simultaneously understanding anomalies at both the short-term perception and long-term reasoning levels. Additionally, the uniform sampling strategy allocates equal attention to both anomalous and normal frames, which is sub-optimal for long videos.

4. **Goal**: (1) Construct a multi-granularity hierarchical anomaly understanding dataset; (2) design an efficient anomaly sampling strategy for long videos.

5. **Key Insight**: The authors observe that anomalous frames typically contain more information and exhibit greater variations; thus, more frames should be adaptively sampled from anomaly-dense regions. Meanwhile, a semi-automatic annotation engine combines LLMs with human segmenting to efficiently generate multi-level annotations.

6. **Core Idea**: Construct clip/event/video three-level anomaly instruction data using a semi-automatic engine, and combine it with an Anomaly-focused Temporal Sampler to enable VLMs to efficiently handle long-video anomalies.

## Method

### Overall Architecture

The overall pipeline of Holmes-VAU is as follows: given a long input video, visual tokens of each frame are first extracted using a frozen visual encoder (ViT of InternVL2). Then, the Anomaly-focused Temporal Sampler (ATS) adaptively selects $N$ keyframes. The visual tokens of the selected frames are mapped to the language feature space via a projector and concatenated with text prompts before being fed into a pretrained Large Language Model, ultimately generating anomaly descriptions and analysis text. The training consists of two steps: first, train the anomaly scorer using frame-level labels; second, fine-tune the VLM using LoRA on all instruction data in HIVAU-70k.

### Key Designs

1. **HIVAU-70k Semi-Automatic Annotation Engine**:

    - **Function**: Efficiently build a video anomaly understanding benchmark containing 70,000+ multi-granularity annotations.
    - **Mechanism**: Completed in three steps. (1) Hierarchical Video Decoupling—human annotators mark the temporal boundaries of anomalous events, and events are further sliced into clips of random lengths, yielding a total of 5,443 videos, 11,076 events, and 55,806 clips. (2) Hierarchical Free-text Annotation—LLaVA-Next-Video is utilized to generate detailed descriptions for each clip, and then an LLM aggregates clip descriptions into event-level summaries (including judgment, description, and analysis), which are finally summarized into video-level summaries. (3) Hierarchical Instruction Construction—free texts are paired with predefined anomaly-related question templates to form QA-formatted instruction data. The entire annotation process only requires 5 annotators and approximately 20 hours for video segmentation.
    - **Design Motivation**: Pure human annotation is excessively costly and unscalable, whereas pure automatic annotation offers uncontrollable quality. By assigning "segmentation" to humans and "description and reasoning" to LLMs, followed by human auditing, both quality and efficiency are well balanced.

2. **Anomaly-focused Temporal Sampler (ATS)**:

    - **Function**: Adaptively select $N$ keyframes from $T$ frames of a long video, allowing the VLM to focus on anomaly-dense regions.
    - **Mechanism**: ATS consists of two components. (a) Anomaly Scorer $\phi_s$: a lightweight VAD network based on UR-DMU that predicts an anomaly score $s_i$ for each frame's CLS token. (b) Density-Aware Sampler: treats anomaly scores as a probability mass function, calculates the cumulative distribution function $S_{cumsum}(t) = \sum_{i=1}^{t}(s_i + \tau)$, and uniformly samples $N$ points along the cumulative axis, which are mapped back to the temporal axis to retrieve the sampled frame indices. Here, $\tau=0.1$ controls the sampling uniformity—a larger $\tau$ leads to near-uniform sampling, whereas a smaller $\tau$ focuses more on anomalous regions.
    - **Design Motivation**: Uniform sampling misses key anomalous frames, whereas Top-$K$ sampling loses contextual information (focusing only on local anomalous frames). ATS allocates more sampling points in anomaly-dense regions using a probability density approach while retaining the temporal context of normal regions, thereby balancing coverage and focus.

3. **Instruction Tuning and LoRA Adaptation**:

    - **Function**: Inject anomaly understanding knowledge while preserving the VLM's original general capabilities.
    - **Mechanism**: Uses InternVL2-2B as the base model, freezing the parameters of the visual encoder and projection layer. The language model is fine-tuned using LoRA ($r=64, \alpha=128$), trained for 1 epoch with a batch size of 512 using the AdamW optimizer with cosine decay. The choice of $r$ is determined via ablation experiments—too large an $r$ harms general video understanding capabilities.
    - **Design Motivation**: Full parameter fine-tuning damages the pre-existing capabilities of LLMs and is computationally expensive. LoRA is currently the most mature parameter-efficient fine-tuning method, and $r=64$ achieves the best balance between VAU specialized capability and general capability.

### Loss & Training

Training is performed in two stages. In the first stage, the anomaly scorer is trained using frame-level labels from HIVAU-70k with the standard binary cross-entropy loss $\mathcal{L}_{AS} = -\sum_{i=1}^{T}(s_i \log(\hat{y}_i) + (1-s_i)\log(1-\hat{y}_i))$. In the second stage, the anomaly scorer is frozen, and the VLM's LoRA parameters are fine-tuned using the cross-entropy loss on all instruction data. Note that when evaluating the anomaly detection performance on UCF-Crime and XD-Violence, only the respective training sets are used to ensure fairness.

## Key Experimental Results

### Main Results

**Anomaly Detection Performance Comparison (Table 1)**:

| Method | Type | XD-Violence AP(%) | UCF-Crime AUC(%) |
|------|------|-------------------|-------------------|
| UR-DMU | Weakly-supervised | 81.66 | 86.97 |
| VadCLIP | Weakly-supervised | 84.51 | 88.02 |
| LAVAD | Explainable Multimodal | 62.01 | 80.28 |
| **Holmes-VAU** | **Explainable Multimodal** | **87.68** | **88.96** |

**Anomaly Reasoning Performance Comparison (Table 2), Video-level**:

| Method | Params | BLEU↑ | CIDEr↑ | METEOR↑ | ROUGE↑ |
|------|--------|-------|--------|---------|--------|
| InternVL2 | 8B | 0.145 | 0.035 | 0.101 | 0.122 |
| QwenVL2 | 7B | 0.155 | 0.044 | 0.112 | 0.137 |
| **Holmes-VAU** | **2B** | **0.566** | **1.437** | **0.165** | **0.355** |

### Ablation Study

**Ablation of Hierarchical Instruction Data (Table 3)**:

| Training Data | Clip BLEU | Event CIDEr | Video CIDEr |
|----------|-----------|-------------|-------------|
| C only | 0.984 | 0.120 | 0.106 |
| E only | 0.508 | 1.183 | 0.872 |
| C+E | 0.889 | 1.285 | 0.889 |
| **C+E+V** | **0.913** | **1.519** | **1.437** |

**Ablation of Sampling Strategies (Table 4, $N=16$)**:

| Sampling Method | Video BLEU↑ | Video CIDEr↑ |
|----------|-------------|-------------|
| Top-K | 0.476 | 1.302 |
| Uniform | 0.511 | 1.345 |
| **ATS** | **0.566** | **1.437** |

### Key Findings

- The three-tier hierarchical data each makes distinct contributions: clip-level enhances visual perception, event-level improves event judgment, and video-level boosts long-term reasoning analysis. The combined use of all three yields the best results.
- ATS outperforms both Uniform and Top-$K$ sampling across all frame settings (8/16/32) with acceptable inference latency.
- There exists a sweet spot for the LoRA dimension $r$: $r=64$ achieves the optimal balance between VAU capability and general capabilities, whereas an excessively large $r$ significantly impairs general performance on Video-MME.
- Using only a 2B model substantially outperforms 7-8B general VLMs, demonstrating that the importance of domain-specific instruction data far outweighs model size.

## Highlights & Insights

- **Density-aware Sampling Concept**: Treating anomaly scores as a probability density function and using the cumulative distribution function for non-uniform sampling is an elegant adaptive sampling strategy. This concept can be transferred to any sequence processing scenarios requiring "importance sampling", such as video summarization, keyframe extraction, etc.
- **Hierarchical Design of the Semi-Automatic Annotation Engine**: Decomposing the annotation task into "human coarse-grained segmentation + LLM fine-grained text annotation + human auditing" is a general methodology for large-scale dataset construction, which can be transferred to other video understanding tasks.
- **Small Model + Good Data > Large Model**: The 2B Holmes-VAU significantly outperforms 7-8B general VLMs in anomaly understanding, proving the immense value of domain-specific instruction data.

## Limitations & Future Work

- The dataset is based on UCF-Crime and XD-Violence, which focus heavily on surveillance videos, lacking broader anomaly scenarios like autonomous driving or healthcare.
- The anomaly scorer is based on the UR-DMU architecture and requires frame-level annotations for training, making it inapplicable out-of-the-box in completely unsupervised scenarios.
- The hyperparameter $\tau$ in ATS needs to be set manually; adaptive adjustment holds potential for future exploration.
- Only InternVL2-2B is used as the base model; scaling up to larger models (e.g., 7B/13B) may yield stronger reasoning capabilities.
- For ultra-long videos (spanning hours), the sampled frame number $N=16$ might still be insufficient, requiring more efficient processing approaches.

## Related Work & Insights

- **vs LAVAD**: LAVAD is a training-free method that directly leverages LLMs for anomaly scoring and explanation without domain-specific fine-tuning. In contrast, Holmes-VAU significantly enhances anomaly understanding accuracy through instruction tuning on HIVAU-70k, boosting AP from 62.01% to 87.68%.
- **vs UCA**: UCA provides a video-level anomaly causal analysis dataset but lacks multi-granularity annotations. The three-level structure of HIVAU-70k is far more comprehensive.
- **vs Video-ChatGPT/Video-LLaMA**: These general video VLMs lack specialized knowledge in the anomaly domain and exhibit limited performance in long-term reasoning. Holmes-VAU demonstrates the necessity of domain-specific instruction tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ The multi-granularity anomaly understanding framework and the ATS sampling strategy are novel, though individual components (such as the anomaly scorer and LoRA tuning) are combinations of existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive comparisons against numerous methods are performed across both detection and reasoning dimensions, with ablation studies thoroughly covering data granularity, sampling strategies, and fine-tuning parameters.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the motivation is naturally developed, though some technical details require reference to the appendix.
- Value: ⭐⭐⭐⭐ The HIVAU-70k dataset and ATS sampler provide solid practical value to the video anomaly understanding field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MLVU: Benchmarking Multi-task Long Video Understanding](mlvu_benchmarking_multi-task_long_video_understanding.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](../../NeurIPS2025/video_understanding/vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[CVPR 2025\] ETAP: Event-based Tracking of Any Point](etap_event-based_tracking_of_any_point.md)
- [\[CVPR 2025\] DrVideo: Document Retrieval Based Long Video Understanding](drvideo_document_retrieval_based_long_video_understanding.md)
- [\[CVPR 2026\] Question-guided Visual Compression with Memory Feedback for Long-Term Video Understanding](../../CVPR2026/video_understanding/question-guided_visual_compression_with_memory_feedback_for_long-term_video_unde.md)

</div>

<!-- RELATED:END -->
