---
title: >-
  [Paper Note] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models
description: >-
  [CVPR2026][Multimodal VLM][video scene segmentation] This paper proposes Scene-VLM — the first VLM fine-tuning-based framework for video scene segmentation — which leverages structured multimodal shot representations (vi…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "video scene segmentation"
  - "vision-language models"
  - "multimodal reasoning"
  - "sequential prediction"
  - "confidence estimation"
date: 2026-05-08
content_hash: 1764dc77fe2e95a6
---

# Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models

**Conference**: CVPR2026
**arXiv**: [2512.21778](https://arxiv.org/abs/2512.21778)  
**Code**: N/A  
**Area**: Multimodal VLM
**Keywords**: video scene segmentation, vision-language models, multimodal reasoning, sequential prediction, confidence estimation

## TL;DR
This paper proposes Scene-VLM — the first VLM fine-tuning-based framework for video scene segmentation — which leverages structured multimodal shot representations (visual frames + dialogue + metadata), causal sequential prediction, a context-focus window mechanism, and token logits-based confidence extraction, achieving substantial gains of +6 AP and +13.7 F1 on MovieNet, while demonstrating natural language explanation capability.

## Background & Motivation
Video scene segmentation — partitioning long videos into semantically coherent scenes — is a foundational task in video understanding, critical for applications such as automated structured summarization and semantic retrieval. Formally, a scene comprises consecutive shots sharing location, time, characters, or narrative theme.

Existing encoder-based methods (BaSSL, TranS4mer, MEGA) suffer from three key limitations: (1) **Visual bias**: non-visual signals such as dialogue and character identity are ignored or underutilized; (2) **Point-wise independent prediction**: each shot is classified independently, failing to exploit causal dependencies across consecutive decisions; (3) **Lack of interpretability**: only confidence scores are produced, with no explanation of why a boundary is predicted.

**Core Idea**: Leveraging the multimodal reasoning and text generation capabilities of VLMs, scene segmentation is reformulated as a sequential generation task that outputs "Shot i: Yes/No" in order, naturally enabling causal dependency modeling, multimodal fusion, and interpretability.

## Method

### Overall Architecture
Scene-VLM is built upon fine-tuned Qwen2.5-VL-7B. The input consists of multimodal representations of $N$ consecutive shots (visual frames + dialogue + character IDs), and the output is a scene boundary decision (Yes/No) for each shot within a focus window, with confidence scores extracted from token logits.

### Key Designs

1. **Structured Multimodal Shot Representation**: Each shot $s_i$ is represented by $K=3$ sampled frames, synchronized subtitles, and character information. Each frame is overlaid with a visual shot-ID marker to help the model associate visual content with shot references in the text. This design provides narrative context that vision-centric approaches cannot capture.

2. **Causal Sequential Prediction**: Scene segmentation is transformed from independent classification into sequential generation — the model outputs "Shot i: Yes/No" for multiple shots in order, with each boundary decision causally influencing subsequent ones. This allows the model to leverage prior predictions as context. Attention analysis confirms that the model indeed "trusts" previous predictions, allocating less attention to already-processed shots while focusing more on upcoming ones.

3. **Context-Focus Window Mechanism**: A context window of 20 shots is used, but predictions are only made for the central 10 shots (the focus window). This design ensures each evaluated shot has sufficient past and future evidence, eliminating performance degradation at sequence boundaries (experiments show a sharp F1 drop at boundary positions without the focus mechanism).

4. **VLM Confidence Extraction**: Unlike encoders with classification heads that directly output scores, Scene-VLM extracts softmax logits at the decision token positions: $\text{conf}_i = P(\text{Yes}) / (P(\text{Yes}) + P(\text{No}))$, enabling a controllable precision-recall trade-off.

5. **Interpretability Alignment**: Through targeted fine-tuning on a small set of annotated explanation samples, the model can generate coherent natural language explanations (e.g., "The scene transitions from indoors to outdoors, with changes in both characters and narrative topic") — a capability unavailable in encoder-based approaches.

### Loss & Training
- Standard next-token prediction loss
- Backbone: Qwen2.5-VL-7B
- Training data: MovieNet-318 (190 films for training)

## Key Experimental Results

### Main Results (MovieNet-318)

| Method | F1 ↑ | AP ↑ |
|--------|------|------|
| BaSSL | 47.0 | 57.4 |
| TranS4mer | 48.4 | 60.8 |
| MEGA | 55.3 | 58.6 |
| Chapter-LLaMA | 38.6 | 41.5 |
| **Scene-VLM** | **62.1** | **66.8** |

### Zero-Shot Cross-Domain (BBC Planet Earth)

| Method | AP ↑ |
|--------|------|
| TranS4mer | 43.6 |
| **Scene-VLM** | **45.8** |

### Ablation Study

| Configuration | F1 | AP | Notes |
|---------------|----|----|-------|
| Full model | 62.1 | 66.8 | — |
| w/o visual | 32.0 | 34.7 | Visual is the core signal |
| w/o Shot-ID | 60.8 | 64.1 | Temporal anchoring is valuable |
| w/o subtitles | 61.1 | 62.2 | Subtitles provide complementary signal |
| Visual only | 58.6 | 61.4 | Multimodal fusion yields +3.5 F1 |
| Context 20 + Focus 10 | **62.1** | — | Optimal configuration |
| Context 20 + Focus 1 (point-wise) | 60.1 | — | Sequential prediction outperforms point-wise |
| Context 5 + Focus 5 | 55.8 | — | Larger context is better |

### Model Scale

| Parameters | F1 | AP |
|------------|----|----|
| 1.5B | 55.9 | 58.7 |
| 3B | 59.6 | 62.8 |
| **7B** | **62.1** | **66.8** |

### Key Findings
- Visual input is the most important signal source (removing it causes a 30-point F1 drop), but subtitles and character IDs provide indispensable complementary information.
- Attention analysis reveals that, after length normalization, subtitle and character tokens receive attention comparable to visual tokens.
- The model attends more to subsequent shots than preceding ones — because prior information has already been encoded through output tokens.
- The focus mechanism is critical for boundary positions: without it, F1 drops sharply at sequence edges; with it, performance is consistent across all positions.
- Performance improves monotonically from 1.5B to 7B, with the 7B gain still substantial, suggesting that larger models may continue to benefit.

## Highlights & Insights
- **Paradigm shift**: Transitioning from an encoder-based classification framework to a VLM sequential generation framework simultaneously addresses three long-standing challenges: multimodal fusion, sequential dependency modeling, and interpretability.
- **Confidence extraction technique**: The method of computing normalized confidence from Yes/No logits is simple yet effective, providing a general solution for applying VLMs to all binary classification tasks.
- **In-depth attention analysis**: Reveals the information flow pattern of VLMs during scene boundary prediction — trusting historical predictions while focusing attention on future context.
- Zero-shot generalization to BBC Planet Earth demonstrates that the framework is not limited to the movie domain.

## Limitations & Future Work
- Sampling 3 frames per shot may be insufficient to capture scenes with rapid intra-shot motion.
- A context window of 20 shots may be inadequate for very long films, necessitating hierarchical or memory-augmented extensions.
- Inference speed may be slower than lightweight encoder-based methods (latency is not reported in the paper).
- Interpretability alignment requires manually annotated explanation samples, incurring non-negligible annotation cost.

## Related Work & Insights
- **vs. MEGA**: MEGA also incorporates subtitles and scripts but relies on a fixed fusion strategy with point-wise prediction; Scene-VLM uses end-to-end VLM reasoning for greater flexibility.
- **vs. Chapter-LLaMA**: An LLM-based chapter segmentation method, but it relies solely on textual descriptions without direct visual processing, achieving only 38.6 F1 on movies.
- **vs. TranS4mer**: Models long-range dependencies with self-attention and SSM, but remains an encoder without interpretability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of VLMs to video scene segmentation; paradigm-level innovation addressing multiple long-standing limitations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Exceptionally detailed ablations (modality, window size, frame count, model scale) with in-depth attention analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, intuitive figures, and a complete narrative logic from method to analysis.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm for video structural understanding; the confidence extraction and interpretability designs offer broad transferability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HOG-Layout: Hierarchical 3D Scene Generation, Optimization and Editing via Vision-Language Models](hog_layout_hierarchical_3d_scene_generation_optimization_and_editing.md)
- [\[CVPR 2026\] BUSSARD: Normalizing Flows for Bijective Universal Scene-Specific Anomalous Relationship Detection](bussard_normalizing_flows_for_bijective_universal_scene-specific_anomalous_relat.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[NeurIPS 2025\] Nautilus: A Large Multimodal Model for Underwater Scene Understanding](../../NeurIPS2025/multimodal_vlm/nautilus_a_large_multimodal_model_for_underwater_scene_understanding.md)
- [\[CVPR 2026\] Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism](scaling_the_long_video_understanding_of_multimodal_large_language_models_via_vis.md)

</div>

<!-- RELATED:END -->
