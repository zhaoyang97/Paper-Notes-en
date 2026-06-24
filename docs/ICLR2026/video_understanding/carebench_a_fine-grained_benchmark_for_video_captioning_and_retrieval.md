---
title: >-
  [Paper Note] CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval
description: >-
  [ICLR2026][Video Understanding][Video Captioning] CaReBench utilizes 1,000 manually annotated videos—each with captions exceeding 200 words and **explicitly split into spatial and temporal versions**—to establish a benchmark capable of simultaneously evaluating fine-grained video captioning and retrieval. It introduces two new metrics, ReBias and CapST, to quantify the spatiotemporal bias of VLMs, and provides a two-stage SFT baseline, CARE, which unifies captioning and retri…
tags:
  - "ICLR2026"
  - "Video Understanding"
  - "Video Captioning"
  - "Video Retrieval"
  - "Fine-grained Evaluation"
  - "Spatiotemporal Bias"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 6a10940b580c7b3a
---

# CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=OZtGhb9x7C](https://openreview.net/forum?id=OZtGhb9x7C)  
**Code**: https://carebench.github.io (Project Page)  
**Area**: Video Understanding  
**Keywords**: Video Captioning, Video Retrieval, Fine-grained Evaluation, Spatiotemporal Bias, Multimodal Large Language Models

## TL;DR
CaReBench utilizes 1,000 manually annotated videos—each with captions exceeding 200 words and **explicitly split into spatial and temporal versions**—to establish a benchmark capable of simultaneously evaluating fine-grained video captioning and retrieval. It introduces two new metrics, ReBias and CapST, to quantify the spatiotemporal bias of VLMs, and provides a two-stage SFT baseline, CARE, which unifies captioning and retrieval into a single MLLM.

## Background & Motivation
**Background**: Video captioning and video retrieval are the two primary tasks in video-language understanding. Historically, they have been dominated by two distinct model types: retrieval relies on dual-encoders like CLIP to align features, while captioning depends on Multimodal Large Language Models (MLLM) to generate descriptions word-by-word. These tasks were traditionally treated as independent, each with its own benchmarks and evaluation conventions.

**Limitations of Prior Work**: Existing benchmarks fail to support "fine-grained" evaluation. Traditional datasets (MSR-VTT, MSVD, DiDeMo) provide only single-sentence captions of roughly a dozen words. Modern MLLMs output descriptions richer than the reference captions, making it impossible to stress-test their fine-grained capabilities. Recent works focusing on long captions either use GPT-4o for automatic annotation (introducing hallucinations and biases) or, like DREAM-1K, focus solely on actions while lacking simultaneous coverage of static objects and dynamic actions or hierarchical structures. Evaluation metrics are equally problematic: traditional n-gram metrics (CIDEr) cannot evaluate long captions, and LLM-based metrics (AutoDQ evaluates only actions, VDCScore ignores precision) are not comprehensive.

**Key Challenge**: Video understanding inherently requires comprehending both **static scenes** and **dynamic actions**. However, existing annotations and metrics conflate the two, masking a long-overlooked issue: do VLMs truly understand actions, or are they taking "shortcuts" based on scene cues? When captioning and retrieval are merged into a single sentence, it is impossible to distinguish whether a model's score reflects spatial or temporal understanding.

**Goal**: (1) To create a benchmark with sufficiently detailed captions and decoupled spatial/temporal evaluation; (2) To design dedicated metrics quantifying spatiotemporal bias; (3) To verify whether retrieval and captioning can be unified within a single model.

**Key Insight**: The authors observe that retrieval and captioning are essentially the same task—mapping pixel space to a high-dimensional space $\phi: \mathbb{R}^{T\times H\times W\times C}\to\mathbb{R}^{D}$. The difference is merely that captioning maps to the vocabulary space $\mathbb{R}^{D_v}$, while retrieval maps to the embedding space $\mathbb{R}^{D_e}$. Since the mappings share a common origin, it is feasible to unify both tasks within one MLLM.

**Core Idea**: Use hierarchical and spatiotemporal-separated manual annotations to support fine-grained evaluation, employ ReBias/CapST to quantify spatiotemporal bias, and unify captioning and retrieval into one MLLM via two-stage SFT.

## Method

### Overall Architecture
This work comprises two main components: the **benchmark (data + metrics)** and the **baseline model CARE**.

The benchmark pipeline involves: selecting 1,000 videos from the FineAction dataset (106 subcategories, 10-20 videos per category, where videos within the same subcategory have highly similar scenes/actions to test discriminative ability) → two-stage manual annotation (writing four-aspect fine-grained descriptions, then splitting them into pure spatial and pure temporal versions) → defining retrieval metric ReBias and captioning metric CapST based on these decoupled annotations. The tasks derive into three types of retrieval (general / spatial / temporal) and two classes of captioning (objects / events).

The baseline model CARE is built on Qwen2-VL using **two-stage progressive SFT**: Stage-I uses high-quality video-caption pairs to align model output with the fine-grained text space (training captioning capability); Stage-II freezes the vision encoder and uses text-only contrastive learning to shift the model output from vocabulary space to embedding space (training retrieval capability). Upon completion, a 7B model can perform both detailed captioning and feature extraction.

### Key Designs

**1. Four-aspect Hierarchical Annotation: Stress-testing MLLMs**
To address short captions, each video is independently described by two annotators and refined by an expert. Captions are limited to 150-300 words and must cover: **General Overview** (one-sentence summary), **Object Description** (position, color, shape, and relationships of static objects), **Action Description** (actions and their sequence, including style), and **Misc Description** (perspective, video type). The final average length is 227.95 words, 24.2x that of MSR-VTT.

**2. Spatial/Temporal Separation: Exposing Shortcuts**
This is the benchmark's most critical design. In Stage-II, detailed descriptions are **manually split into two pure captions**: Spatial Description (removing all action text, retaining background and objects) and Temporal Description (removing all static details, retaining overview and action sequences). If spatial and temporal information is conflated, models can score high by only looking at the scene. This separation exposes the model's actual understanding of dynamic actions via the performance gap between spatial and temporal retrieval.

**3. ReBias: Quantifying Spatiotemporal Bias**
To fill the gap in quantifying VLM bias, the authors define ReBias to measure the deviation between temporal and spatial recall:

$$B = \left|\,1 - \frac{\bar{R}_{\text{temporal}}}{\bar{R}_{\text{spatial}}}\,\right|$$

where $\bar{R}_{\text{temporal}}$ and $\bar{R}_{\text{spatial}}$ are the average recall for temporal and spatial retrieval, respectively. **Lower is better** (0 indicates balance). This scalar effectively identifies if a model relies on spatial shortcuts.

**4. CapST: Evaluating Objects and Events with Precision and Recall**
To address metrics that only evaluate actions or recall, CapST (Captioning, Spatial + Temporal) uses a strong LLM (DeepSeek-V3) as a judge to extract **events** from temporal captions and **objects** from spatial captions. Natural Language Inference (NLI) then determines the entailment between the ground truth $D_{gt}$ and predicted description $D_{pred}$:

$$R = \frac{N(D_{gt}\xrightarrow{\text{entail}}O_{pred})}{N(O_{pred})}, \qquad P = \frac{N(D_{pred}\xrightarrow{\text{entail}}O_{gt})}{N(O_{gt})}$$

$O_{pred}$ and $O_{gt}$ are elements (objects or events) extracted from predictions and ground truth. While recall measures coverage, precision identifies hallucinations. When extracting elements, the LLM **splits attributes** (e.g., "old man wearing glasses and a blue suit" becomes "old man wearing glasses" and "old man wearing blue suit") to ensure partial credit for partially correct predictions.

### Loss & Training
CARE's two-stage SFT:

- **Stage-I (Fine-grained Captioning Adaptation)**: Uses the prompt "Describe the video in detail." and a mixture of Tarsier Recap and LLaVA-Video-178k data for full-parameter fine-tuning. This aligns output with the fine-grained text space (approx. 400 GPU hours).
- **Stage-II (Retrieval Adaptation)**: Starting from Stage-I, the **vision encoder is frozen**. Using the EOL (Explicit One-word Limitation) prompt "`<sent>` Summary of the above sentence in one word:", the hidden state of the next token is used as the sentence embedding $f_i$. Text-only contrastive learning is performed on NLI datasets to move output to the embedding space (24 GPU hours). Contrastive loss:

$$\mathcal{L} = -\log\frac{e^{\cos(f_i, f_i^+)/\tau}}{\sum_{j=1}^{N}\left(e^{\cos(f_i, f_j^+)/\tau} + e^{\cos(f_i, f_j^-)/\tau}\right)}$$

where $f_i, f_i^+, f_i^-$ are embeddings of sentence $s_i$, its positive sample, and hard negative samples. Stage-II imparts strong video retrieval capabilities despite using only text by leveraging the video-text alignment established in Stage-I.

## Key Experimental Results

### Benchmark Statistics Comparison
CaReBench provides more detailed captions than traditional benchmarks and is the only one to offer hierarchical annotation, static object coverage, and dynamic action coverage through manual labeling:

| Benchmark | Samples | Avg. Duration | Avg. Words | Annotator | Hierarchical | Objects | Actions |
|--------|------|------|------|------|------|------|------|
| MSR-VTT | 1,000 | 15.01s | 9.41 | Human | ✗ | ✗ | ✗ |
| DREAM-1K | 1,000 | 8.9s | 59.3 | Human | ✗ | ✗ | ✓ |
| VDC | 1,000 | 28.18s | 500.91 | GPT | ✓ | ✓ | ✗ |
| **CaReBench** | 1,000 | 14.35s | 227.95 | Human | ✓ | ✓ | ✓ |

### Main Results: Retrieval & Captioning
In General Retrieval (zero-shot), the contrastive-trained MLLM outperforms CLIP-based models. CARE achieves state-of-the-art results:

| Model | T2V R@1 | T2V R@5 | V2T R@1 | V2T R@5 |
|------|------|------|------|------|
| CLIP L/14 | 51.2 | 83.4 | 54.7 | 86.9 |
| InternVideo2 1B | 72.5 | 93.7 | 69.5 | 94.6 |
| **CARE 7B** | **77.0** | **95.6** | **79.0** | **96.8** |

In captioning (CapST), CARE 7B exceeds all open-source models, including Qwen2-VL 72B and Qwen2.5-VL 7B (e.g., Events Overall F1: CARE 35.1 vs. Qwen2.5-VL 7B 31.1), indicating that current large scales do not inherently guarantee fine-grained descriptive ability.

### Ablation Study: Role of Two-stage SFT
"Unified Score" is the average of retrieval R@1 and captioning F1:

| Configuration | Avg. R@1 (Retrieval) | Avg. F1 (Captioning) | Unified Score |
|------|------|------|------|
| Baseline (Qwen2-VL) | 25.6 | 26.8 | 26.2 |
| + Captioning SFT only | 17.6 (−8.0) | 33.8 (+7.0) | 25.7 (−0.5) |
| + Retrieval SFT only | 77.0 (+51.4) | 28.2 (+1.4) | 52.6 (+26.4) |
| **+ Both Stages** | **78.0 (+52.4)** | **33.4 (+6.6)** | **55.7 (+29.5)** |

### Key Findings
- **VLMs rely heavily on spatial shortcuts**: When switching from general to spatial retrieval, VLM performance remains stable (Qwen2-VL −0.20), but it collapses when switching to temporal retrieval (Qwen2-VL −24.70). This confirms they rely on scene cues rather than action information.
- **Mutual gains between tasks**: Retrieval adaptation improved captioning F1 (+1.4), and captioning adaptation improved retrieval R@1 (+1.0), supporting the unified framework hypothesis.
- **Generation remains intact**: Embedding space contrastive learning did not damage vocabulary space generation.
- **Generalization**: CARE remains competitive on out-of-domain benchmarks like TVBench (50.1), surpassing Gemini 1.5 Pro (46.5).

## Highlights & Insights
- **Decoupled annotation as a diagnostic tool**: CaReBench physically splits captions to quantify spatiotemporal bias, turning a latent issue into a measurable metric.
- **Unified logic**: Treating both tasks as $\phi:\mathbb{R}^{T\times H\times W\times C}\to\mathbb{R}^D$ mappings (vocabulary vs. embedding space) provides an elegant theoretical foundation for the single-model approach.
- **Stage-II text-to-video transfer**: Training a retrieval model in 24 hours using only text by leveraging existing fine-grained video-text alignment is a highly efficient "free lunch" strategy.

## Limitations & Future Work
- The authors acknowledge that while CARE quantifies spatiotemporal bias, it **does not solve it**; CARE's ReBias (17.53) remains on par with other MLLMs.
- The benchmark is limited to 1,000 short videos (mostly 5-20s) and does not cover long-form video understanding.
- CapST and ReBias rely on DeepSeek-V3 as a judge, making scores susceptible to the judge's inherent biases.
- Future work involves designing training objectives to reduce ReBias and extending the framework to longer videos.

## Related Work & Insights
- **vs. DREAM-1K / VDC**: CaReBench is the first to combine manual annotation, hierarchical structures, object/action dual coverage, and spatial/temporal separation.
- **vs. Long-CLIP**: While Long-CLIP expands context length for long captions, it relies on LLM-labeled benchmarks. CARE discovers that contrastive-trained MLLMs can outperform specialized CLIP variants.
- **vs. E5-V / VISTA**: CARE demonstrates that captioning and retrieval tasks can mutually enhance each other within a unified MLLM framework.

## Rating
- Novelty: ⭐⭐⭐⭐ The decoupling of spatial/temporal data provides genuine insight into model bias.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive multi-task evaluations, though limited by source data diversity.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and well-defined metrics.
- Value: ⭐⭐⭐⭐ The benchmark, metrics, and unified baseline provide a practical toolkit for VLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/video_understanding/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [\[ICLR 2026\] OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text](omnicvr_a_benchmark_for_omni-composed_video_retrieval_with_vision_audio_and_text.md)
- [\[ICLR 2026\] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[AAAI 2026\] FineVAU: A Novel Human-Aligned Benchmark for Fine-Grained Video Anomaly Understanding](../../AAAI2026/video_understanding/finevau_a_novel_human-aligned_benchmark_for_fine-grained_video_anomaly_understan.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [\[ICLR 2026\] OmniCVR: A Benchmark for Omni-Composed Video Retrieval with Vision, Audio, and Text](omnicvr_a_benchmark_for_omni-composed_video_retrieval_with_vision_audio_and_text.md)
- [\[ICLR 2026\] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding](lets_split_up_zero-shot_classifier_edits_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Fine-VAD: Towards Fine-Grained Video Anomaly Detection via Progressive Cross-Granularity Learning](../../CVPR2026/video_understanding/fine-vad_towards_fine-grained_video_anomaly_detection_via_progressive_cross-gran.md)

</div>

<!-- RELATED:END -->
