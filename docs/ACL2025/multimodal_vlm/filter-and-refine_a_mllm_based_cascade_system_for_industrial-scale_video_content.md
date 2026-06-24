---
title: >-
  [Paper Note] Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation
description: >-
  [ACL 2025][Multimodal VLM][MLLM] TikTok proposes a two-stage cascade content moderation system based on MLLM (Router-Ranker). By filtering 97.5% of compliant traffic using a lightweight embedding retrieval router, only high-risk videos are routed to a fine-tuned LLaVA for fine-grained classification. This improves F1 by 66.5% while reducing deployment costs to 1.5% of full direct deployment.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "MLLM"
  - "Content Moderation"
  - "Cascade System"
  - "Video Classification"
  - "Industrial Deployment"
date: 2026-05-08
content_hash: 15123ca66925acd7
---

# Filter-And-Refine: A MLLM Based Cascade System for Industrial-Scale Video Content Moderation

**Conference**: ACL 2025  
**arXiv**: [2507.17204](https://arxiv.org/abs/2507.17204)  
**Code**: None  
**Area**: Multimodal VLM / Content Moderation  
**Keywords**: MLLM, Content Moderation, Cascade System, Video Classification, Industrial Deployment

## TL;DR
TikTok proposes a two-stage cascade content moderation system based on MLLM (Router-Ranker). By filtering 97.5% of compliant traffic using a lightweight embedding retrieval router, only high-risk videos are routed to a fine-tuned LLaVA for fine-grained classification. This improves F1 by 66.5% while reducing deployment costs to 1.5% of full direct deployment.

## Background & Motivation
The explosive growth of short video platforms (such as TikTok, YouTube Shorts) makes content moderation a critical necessity. Traditional video classification models can handle explicit violations but struggle with implicit harmful content (such as subtle misinformation, suggestive imagery) and context-dependent moderation scenarios. Multimodal Large Language Models (MLLM), with their cross-modal reasoning and contextual understanding capabilities, can theoretically address these pain points, but face two core obstacles:

**Prohibitively high computational cost**: Hundreds of millions of new videos are uploaded daily, making it computationally infeasible to run MLLMs directly on all traffic.

**Adaptation challenge of generation models for classification**: MLLMs are generative by nature. There is no mature strategy to efficiently convert them into discriminative classifiers.

The Key Insight of this work is to leverage the "Recall-Rank" paradigm of recommendation systems, decomposing the moderation task into coarse-grained filtering followed by fine-grained ranking, while proposing an extremely simple generative-to-classifier conversion method that requires very little labeled data.

## Method

### Overall Architecture
The system is split into two stages:

1. **Router (Recall Stage)**: A lightweight model rapidly filters out low-risk videos, only letting suspicious content pass.
2. **Ranker (Ranking Stage)**: An MLLM performs fine-grained reasoning and classification on the high-risk subset.

This hierarchical filtering substantially reduces the volume of videos that the MLLM needs to process, balancing moderation accuracy and scalability.

### Key Designs

1. **Embedding Retrieval-Based Router**:

    - Mechanism: Maintains a pool of high-risk representative "seed videos". New videos are matched against seed videos via semantic similarity to determine if they are high-risk.
    - Unsupervised training, requiring no labeled data.
    - Seed selection strategy: Centroid proximity selection (clustering algorithm) + human selection (moderators choosing "golden seeds").
    - Advantages: Seed pool can be dynamically updated to adapt to new violation patterns, offering high flexibility.
    - Actual performance: Filters out 97.5% of compliant traffic without increasing service latency.

2. **Fine-Tuning Method for MLLM as Ranker**:

    - Backbone: LLaVA (Mistral-7B + ViT-Large + 2-layer MLP projector)
    - Key Innovation: Simplifies the generation task to **single-token prediction**—restricting the model via prompt engineering to output only one token (Yes/No), and extracting the logits of that token to calculate probability scores via softmax.
    - Training requires only about 300K samples (2% of the dataset size of traditional classification models), mixing three types of data in a 1:1:1 ratio: VQA data (from LLaVA-Mix665k subset), video captioning data, and moderation classification data.
    - This is a highly pragmatic conversion strategy: it changes only the training format and output decoding without altering the architecture.

3. **Prompt Design**:

    - Four prompt templates were designed to explore different questioning styles: asking directly for the global label, asking for fine-grained and global labels separately, sequentially asking both to emphasize connection, and providing definitions first before asking.
    - Experiments show that asking the two questions separately (P2) achieves the best performance. A single question provides insufficient information, and combining both labels in a single prompt introduces noise.

### Loss & Training
Two SFT strategies were explored:
- **Multi-task Learning**: Training the three types of data mixed together directly, taking around 20 hours on 8×A100.
- **Mixed Sequential Phased Learning**: Performing visual instruction tuning on VQA + Caption data first, followed by a second-stage SFT on moderation classification data, taking approximately 20.5 hours on 8×A100 in total.

Multi-task training exhibits more robust performance across all prompts, while sequential phased training is more flexible and time-efficient.

## Key Experimental Results

### Main Results

| Model | PR-AUC | ROC-AUC | Max-F1 |
|------|--------|---------|--------|
| X-VLM (Traditional Multimodal Classification) | 30.79 | 65.31 | 36.81 |
| LLaVA (Zero-shot) | 23.17 | 58.59 | 31.32 |
| LLaVA w/ Caption (Zero-shot) | 28.85 | 65.88 | 36.71 |
| Multi-task + P2 (Ours/Best) | **68.73** | **87.68** | **61.29** |

The MLLM solution achieves a 66.50% improvement in F1 over traditional classification models, and fine-tuning yields a 45.55% PR-AUC improvement over the zero-shot baseline.

### Ablation Study

| Configuration | PR-AUC | ROC-AUC | Description |
|------|--------|---------|------|
| Original Output | 68.73 | 87.68 | Multi-task + P2 Baseline |
| Union Probability | 68.78 | **87.83** | Union of fine-grained & global labels |
| Weighted Sum | **68.83** | 87.78 | Weighted fusion, optimal PR-AUC |
| Bayesian Fusion | 68.67 | 87.79 | Slightly below baseline |
| Temperature Tuning (0.2-0.8) | ~Unchanged | ~Unchanged | Temperature has little impact on performance |

### Online Experiments

| Metric | Gain / Performance |
|------|------|
| Auto-Moderation Volume Boost | +41.27% |
| System Accuracy Improvement | +19.16% |
| Router Traffic Filtering Rate | 97.5% |
| Human Labeling Data Reduction | Requires only 2% compared to traditional models |

### Key Findings
- MLLM significantly outperforms traditional classification models in content moderation tasks that require deeper understanding.
- Fine-tuning requires only a minimal amount of labeled data (2%), vastly reducing human effort.
- Cascade deployment compresses computational costs to 1.5% of full deployment, achieving industrial feasibility.
- Embedding visualizations clearly demonstrate that the fine-tuned model learns better decision boundaries.

## Highlights & Insights
- **Simple yet highly effective concept**: Restricting MLLM generation to a single Yes/No token and extracting logits for softmax requires near-zero overhead while fully harnessing the MLLM's semantic comprehension.
- **Leveraging recommendation architectures**: The recall-ranking cascade design naturally fits the "massive compliant + scarce violative" distribution characteristics of moderation tasks.
- **Label-free Router**: Using unsupervised embedding retrieval for first-stage filtering bypasses labeling bottlenecks.
- **Real-world deployment validation**: Not purely academic; successfully deployed and verified in TikTok's real production environment.

## Limitations & Future Work
- Still depends on a small amount of human-labeled data, which may introduce noise.
- The Router's recall rate has an upper limit, risking false negatives (misses).
- Demonstrated only on Mistral-7B level MLLMs; remains to be seen if larger/stronger models offer further gains.
- Adaptability to multilingual and cross-cultural moderation is not discussed.
- Maintenance and update strategies for the seed video library are not detailed.

## Related Work & Insights
- **Recommendation System Architecture Migration**: The recall-ranking paradigm is mature in search/recommendation; this work proves it is equally valid in content moderation. Such cross-domain architectural inspiration is highly valuable.
- **Generative-to-Discriminative Conversion**: Works like Sparse Attention Vectors (Mitra et al., 2025) are exploring similar MLLM-to-classifier adaptation paths; the single-token + softmax method in this paper is significantly more lightweight.
- **Implications for Industrial Deployment**: LLM deployment does not necessarily require full-scale inference; cascade designs combined with precise routing can profoundly cut down costs.

## Rating
- Novelty: ⭐⭐⭐ Individual components are not novel, but their combination delivers a practical industrial solution for content moderation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Spans offline evaluation, ablation, and online A/B testing with convincing data.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with a pragmatic, industrial-focused paper style.
- Value: ⭐⭐⭐⭐ Directly references industrial applications, demonstrating a viable deployment pathway for MLLMs in moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Speaking Beyond Language: A Large-Scale Multimodal Dataset for Learning Nonverbal Cues from Video-Grounded Dialogues](speaking_beyond_language.md)
- [\[CVPR 2026\] Towards Open-Vocabulary Industrial Defect Understanding with a Large-Scale Multimodal Dataset](../../CVPR2026/multimodal_vlm/towards_open-vocabulary_industrial_defect_understanding_with_a_large-scale_multi.md)
- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](../../CVPR2025/multimodal_vlm/efficient_motion-aware_video_mllm.md)
- [\[CVPR 2025\] Video-XL: Extra-Long Vision Language Model for Hour-Scale Video Understanding](../../CVPR2025/multimodal_vlm/video-xl_extra-long_vision_language_model_for_hour-scale_video_understanding.md)
- [\[CVPR 2025\] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents](../../CVPR2025/multimodal_vlm/v-stylist_video_stylization_via_collaboration_and_reflection_of_mllm_agents.md)

</div>

<!-- RELATED:END -->
