---
title: >-
  [Paper Note] EventGPT: Event Stream Understanding with Multimodal Large Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Event Camera] The first MLLM specifically designed for event camera streams. By employing a three-stage progressive training paradigm (vision-language alignment $\to$ event-language alignment $\to$ instruction tuning), it bridges the massive domain gap between asynchronous event data and language, substantially outperforming general MLLMs in event scene description and VQA.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Event Camera"
  - "MLLM"
  - "Spatio-Temporal Aggregation"
  - "Three-Stage Training"
  - "Event-Language Alignment"
date: 2026-05-08
content_hash: c1bfbec39579de26
---

# EventGPT: Event Stream Understanding with Multimodal Large Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.00832](https://arxiv.org/abs/2412.00832)  
**Code**: [https://github.com/EventGPT](https://github.com/EventGPT) (TBD)  
**Area**: Multimodal VLM  
**Keywords**: Event Camera, MLLM, Spatio-Temporal Aggregation, Three-Stage Training, Event-Language Alignment

## TL;DR
The first MLLM specifically designed for event camera streams. By employing a three-stage progressive training paradigm (vision-language alignment $\to$ event-language alignment $\to$ instruction tuning), it bridges the massive domain gap between asynchronous event data and language, substantially outperforming general MLLMs in event scene description and VQA.

## Background & Motivation

**Background**: Event cameras record intensity changes asynchronously with high temporal resolution, offering unique advantages in high-speed motion and extreme lighting conditions. However, the representation of event data differs significantly from traditional RGB images, causing existing MLLMs (such as LLaVA, Qwen2-VL, etc.) to perform poorly when directly processing event data.

**Limitations of Prior Work**: General MLLMs only score 1.5-2.4/5.0 (out of 5.0) in detailed description on event data, as they are pre-trained on RGB image-text pairs and cannot comprehend the spatio-temporal structure of event streams. Moreover, event data lacks paired language annotation datasets, making it impossible to directly train event-language models.

**Key Challenge**: A representation gap exists between event camera data and natural language—events are asynchronous, sparse spike signals, which are fundamentally different from any visual data seen during MLLM pre-training. How to bridge this gap with minimal annotated data is a key challenge.

**Goal**: To design an event-specific MLLM architecture and training strategy that enables the model to comprehend event streams and perform scene description, reasoning, and visual question answering (VQA) using natural language.

**Key Insight**: Leveraging RGB images as an intermediate bridge—first aligning the vision-language space on RGB image-text pairs (reusing existing data), then aligning event features to the same space using synthetic event-text pairs, and finally fine-tuning on real-world event data.

**Core Idea**: Aligning the representation spaces of event camera data and language models through a three-stage progressive training paradigm ("Image $\to$ Event $\to$ Instruction") and a spatio-temporal aggregation module.

## Method

### Overall Architecture
The event stream is split into $T$ temporal windows to form an event tensor, which is encoded by OpenCLIP ViT-L/14 into spatio-temporal features $\mathcal{Z} \in \mathbb{R}^{T \times S \times D}$. The spatio-temporal aggregator applies average pooling and max pooling along the temporal and spatial dimensions separately and concatenates them to obtain a fused representation. This is then mapped to the input space of the LLM (Vicuna-v1.5) via a linear projection layer and an event-language adapter.

### Key Designs

1. **Spatio-Temporal Aggregator**:

    - **Function**: Extract joint spatio-temporal representations from multi-window event features.
    - **Mechanism**: Average pool $\mathcal{Z} \in \mathbb{R}^{T \times S \times D}$ along the temporal dimension to obtain $\mathcal{Z}_T^{avg} \in \mathbb{R}^{S \times D}$ (aggregating temporal information while retaining spatial structure) and along the spatial dimension to obtain $\mathcal{Z}_S^{avg} \in \mathbb{R}^{T \times D}$ (aggregating spatial information while retaining temporal structure), then concatenate them with their corresponding max-pooled representations into $\overline{\mathcal{Z}} \in \mathbb{R}^{(T+S) \times D}$.
    - **Design Motivation**: The uniqueness of event data lies in its high temporal resolution, requiring simultaneous modeling of spatial characteristics (what is changing) and temporal characteristics (how it changes). Independent dual-path pooling preserves more structural information than direct flattening.

2. **Three-Stage Progressive Training**:

    - **Function**: Progressively bridge the modality gap of RGB $\to$ Event $\to$ Language.
    - **Mechanism**: Stage 1 (Vision-Language Alignment): Train the linear projection layer using LLaVA-Pretrain 558K RGB image-text pairs, with the encoder and LLM frozen. Stage 2 (Event-Language Alignment): Train the spatio-temporal aggregator and the event-language adapter using 1 million synthetic event-text pairs from N-ImageNet-Chat, with the rest frozen. Stage 3 (Instruction Tuning): Fine-tune all parameters using 120k real-world annotations from Event-Chat.
    - **Design Motivation**: Direct training on event-text pairs is ineffective due to the immense modality gap. Utilizing RGB as an intermediate bridge, synthetic data for pre-training, and real data for fine-tuning progressively narrows this gap.

3. **Event-Language Adapter**:

    - **Function**: Provide additional cross-modal alignment beyond the linear projection layer.
    - **Mechanism**: A linear layer introduced in Stage 2 to further map event features into the representation space of the LLM. Ablation studies show that the adapter's contribution (+3.24% DC) is greater than that of the spatio-temporal aggregator (+2.35% DC), indicating that cross-modal alignment represents a larger bottleneck than temporal modeling.
    - **Design Motivation**: Due to the massive distribution gap between event and RGB data, the projection layer initially trained for RGB is insufficient for alignment, necessitating an additional adapter layer.

### Loss & Training
Standard next-token prediction cross-entropy loss is employed. In terms of datasets, two new datasets are constructed: N-ImageNet-Chat (1 million synthetic event-text pairs, generated from N-ImageNet event emulation data) and Event-Chat (120k real-world annotations, sourced from driving scene data in DSEC and E2VID).

## Key Experimental Results

### Main Results

| Model | LLM | N-ImageNet DC/CR/VQA | Event-Chat DC/CR/VQA |
|------|-----|---------------------|---------------------|
| LLaVA-7B | Vicuna | 1.54/1.07/1.88 | 2.20/4.04/3.26 |
| Qwen2-VL-7B | Qwen2 | 1.74/1.46/1.91 | 2.38/4.02/2.91 |
| InternVL2-8B | InternLM | 1.51/1.87/2.08 | 2.37/4.00/3.71 |
| **EventGPT-7B** | Vicuna | **2.39/2.57/2.23** | **3.52/4.09/4.29** |
| **EventGPT-13B** | Vicuna | **2.41/2.81/2.40** | **3.40/4.13/4.26** |

### Ablation Study

| Configuration | DC | CR | VQA | Description |
|------|----|----|-----|------|
| Baseline (No aggregator, no adapter) | 3.40 | 3.97 | 4.15 | Baseline |
| +Spatio-Temporal Aggregator | 3.48 | 4.02 | 4.20 | +2.35% |
| +Event-Language Adapter | 3.51 | 4.05 | 4.25 | +3.24% |
| +Both (Full) | **3.52** | **4.09** | **4.29** | +3.53% |

### Key Findings
- **General MLLMs perform poorly on event data**: The best-performing model, InternVL2-8B, achieves only 2.37/5.0 on the detailed description (DC) of Event-Chat, whereas EventGPT reaches 3.52/5.0 (+48%).
- **The event-language adapter is more crucial than the spatio-temporal aggregator**: The adapter contributes +3.24%, while the aggregator contributes +2.35%, indicating that cross-modal alignment represents a larger bottleneck.
- **The optimal number of temporal windows is $N_w=5$**: Too few windows (3) lose temporal details, while too many (>7) lead to sparse distribution per window.
- **Strong downstream transfer capability**: Text descriptions generated by EventGPT can directly drive GroundingDINO for object detection and GroundedSAM for instance segmentation.

## Highlights & Insights
- The **"image as a bridge" training strategy** cleverly addresses the lack of language annotations for event data. This progressive alignment concept can be extended to align other emerging sensor modalities (e.g., radar, haptics) with language.
- **New Dataset Construction**: N-ImageNet-Chat (1 million) and Event-Chat (120k) provide the first large-scale language-annotated datasets for the event camera community.
- **Practical Value**: Event cameras possess irreplaceable advantages in autonomous driving (tunnels, night-time) and high-speed motion scenarios; EventGPT enables natural language interaction within these environments.

## Limitations & Future Work
- The event encoder utilizes OpenCLIP pre-trained on RGB, where the event-RGB domain gap may limit feature quality. Event-specific pre-training could be explored in future work.
- The spatio-temporal aggregator uses simple average/max pooling. More sophisticated sequence modeling (e.g., Mamba, temporal transformers) might yield better performance.
- Event-Chat is primarily derived from driving scenarios; generalization to indoor, industrial, and sports environments remains unverified.
- The reliability and consistency of the evaluation metrics (1-5 GPT scores) require further validation.

## Related Work & Insights
- **vs E2VID + LLaVA Pipeline**: Two-stage methods that convert events to RGB and then apply MLLM lose the high temporal resolution advantages of event data. EventGPT processes data end-to-end to preserve temporal information.
- **vs Event-Specific Vision Models** (e.g., event object detection, optical flow estimation): These task-specific models cannot generalize to novel tasks. EventGPT achieves open-ended capabilities through a language interface.
- **vs General MLLMs**: Experiments demonstrate that directly employing general MLLMs to process event data is highly ineffective; the domain gap is too massive and necessitates dedicated training.

## Rating
- Novelty: ⭐⭐⭐⭐ The first event camera MLLM, filling an important gap, though the architectural components themselves (aggregator, adapter) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐ Comparisons with general MLLMs are clear, but there is a lack of quantitative comparisons with event-specific methods on downstream tasks.
- Writing Quality: ⭐⭐⭐⭐ The motivation behind the three-stage training is well-explained, and the dataset construction is meticulously described.
- Value: ⭐⭐⭐⭐ Pioneering significance for the event camera community, with valuable dataset contributions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Learning to See through Illumination Extremes with Event Streaming in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/learning_to_see_through_illumination_extremes_with_event_streaming_in_multimodal.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)
- [\[CVPR 2025\] On the Out-of-Distribution Generalization of Multimodal Large Language Models](on_the_out-of-distribution_generalization_of_large_multimodal_models.md)
- [\[CVPR 2025\] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment](task_preference_optimization_improving_multimodal_large_language_models_with_vis.md)
- [\[CVPR 2025\] Cross-modal Information Flow in Multimodal Large Language Models](cross-modal_information_flow_in_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
