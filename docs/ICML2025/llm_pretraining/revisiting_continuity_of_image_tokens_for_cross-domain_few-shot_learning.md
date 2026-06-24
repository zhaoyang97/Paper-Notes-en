---
title: >-
  [Paper Note] Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning
description: >-
  [ICML 2025 Spotlight][LLM Pretraining][cross-domain few-shot] It is discovered that disrupting the continuity of ViT image tokens (preventing smooth transitions between adjacent patch pixels) leads to a significant performance drop in the source domain but only a slight decrease in the target domain. This reveals that the large spatial patterns learned via token continuity are harder to transfer across domains. Based on this, a simple and effective method, ReCIT…
tags:
  - "ICML 2025 Spotlight"
  - "LLM Pretraining"
  - "cross-domain few-shot"
  - "ViT"
  - "token continuity"
  - "spatial patterns"
  - "domain gap"
date: 2026-05-08
content_hash: a6a2b19d2288737c
---

# Revisiting Continuity of Image Tokens for Cross-Domain Few-Shot Learning

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2506.03110](https://arxiv.org/abs/2506.03110)  
**Code**: [github.com/shuaiyi308/ReCIT](https://github.com/shuaiyi308/ReCIT)  
**Area**: LLM Pre-training  
**Keywords**: cross-domain few-shot, ViT, token continuity, spatial patterns, domain gap

## TL;DR
It is discovered that disrupting the continuity of ViT image tokens (preventing smooth transitions between adjacent patch pixels) leads to a significant performance drop in the source domain but only a slight decrease in the target domain. This reveals that the large spatial patterns learned via token continuity are harder to transfer across domains. Based on this, a simple and effective method, ReCIT, is proposed to bridge the domain gap.

## Background & Motivation
**Background**: ViT achieves strong general representations through large-scale pre-training, but still faces challenges in downstream domains with large domain gaps (e.g., medical imaging) where training samples are scarce, known as cross-domain few-shot learning (CDFSL).

**Limitations of Prior Work**: Existing CDFSL methods primarily focus on adaptation and fine-tuning strategies, overlooking an interesting phenomenon in ViTs when processing image tokens.

**Key Challenge**: While Self-Attention is permutation-invariant to token order, the spatial continuity of image tokens still affects the behavior of ViT—disrupting continuity has a large impact on the source domain but a minimal impact on the target domain. This implies that the features learned via continuity are actually the main culprits behind the domain gap.

**Goal**: To explain this phenomenon and leverage it to improve CDFSL.

**Key Insight**: Analysis reveals that token continuity helps ViT learn larger spatial patterns (textures/structures across patches) that are highly source-domain specific and hard to transfer, whereas small patterns within patches are more domain-agnostic.

**Core Idea**: By better disrupting token continuity, the model is guided to rely more on small patterns (intra-patch features) and reduce reliance on large patterns.

## Method

### Overall Architecture
ReCIT (Re-visiting Continuity of Image Tokens) performs controlled disruption of image token continuity during training and inference: (1) shuffling the spatial arrangement of patches; (2) processing patch boundaries to eliminate smooth transitions across patches.

### Key Designs
1. **Continuity Disruption Strategy**: Spatial continuity of image tokens is disrupted by randomly or structurally rearranging patch order. Since Self-Attention is inherently order-agnostic, this primarily affects the patch embedding stage—pixels of adjacent patches no longer transition smoothly, forcing the patch embedding to capture information restricted within each patch. **Design Motivation**: Force the model to learn local features within patches rather than global patterns across patches.

2. **Explanatory Analysis**: Explain why disrupting continuity has a large impact on the source domain but a minimal impact on the target domain from both theoretical and experimental perspectives: (a) on the source domain, large spatial patterns (e.g., textures in ImageNet) learned with continuity contribute significantly to performance; (b) on distant domains (e.g., medical images), these large patterns cannot transfer, leaving only small patterns effective. Thus, removing large patterns causes a large loss on near domains and a small loss on far domains.

3. **Adaptive Control**: The degree of continuity disruption can be adaptively adjusted based on the source-target domain distance—the larger the domain gap, the more thorough the disruption.

### Loss & Training
Standard few-shot learning training (meta-learning or fine-tuning), with the core modification occurring solely during the data preprocessing stage.

## Key Experimental Results

### Main Results (Multiple CDFSL benchmarks)

| Method | CropDisease | EuroSAT | ISIC | ChestX | Average |
|------|------------|---------|------|--------|------|
| Baseline ViT | Medium | Medium | Low | Low | Baseline |
| SOTA Methods | High | High | High | Medium | High |
| **ReCIT** | **Highest** | **Highest** | **Highest** | **Highest** | **SOTA** |

### Ablation Study

| Configuration | Near-domain Performance | Far-domain Performance | Description |
|------|---------|---------|------|
| No continuity disruption | Highest | Baseline | Standard ViT |
| Slight disruption | Slightly decreased | Improved | Partially removes large patterns |
| Full disruption (ReCIT) | Decreased | Highest | Maximizes local features |
| Source evaluation | Significantly decreased | N/A | Verifies the source-domain value of large patterns |

### Key Findings
- Disrupting continuity has a major impact on the source domain performance (significant decrease) but only a minimal impact on distant domains (slight decrease).
- The larger the domain gap, the greater the relative gain of ReCIT.
- Small spatial patterns exhibit better cross-domain transferability than large spatial patterns.

## Highlights & Insights
- Discovered an overlooked phenomenon in ViT and provided a profound explanation.
- The method is extremely simple (only changing patch arrangement) with almost zero additional computational overhead.
- Explained the underlying reason why ViT performance degrades under large domain gaps (non-transferable large patterns).
- Provides a new perspective for understanding the feature hierarchy of ViT.

## Limitations & Future Work
- The optimal choice of continuity disruption degree depends on prior knowledge of the domain gap.
- May have a negative impact when the domain gap is small.
- Only tested on CDFSL; effectiveness in standard few-shot or full-data scenarios is unknown.

## Related Work & Insights
- Provides a new perspective for understanding patch-based processing in ViT.
- Insight: The "transferability" of features is related to their spatial scale—this insight might apply to other architectures.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Discovered and explained an interesting and useful phenomenon.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on multiple CDFSL benchmarks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logical chain from phenomenon to explanation to method.
- **Value**: ⭐⭐⭐⭐ Contributes to both CDFSL and the understanding of ViTs.

---

## Supplementary Reflections

### Relation to Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: (1) the growing demand for deep understanding of LLM internal mechanisms; (2) the increasing importance of model efficiency and accessibility; (3) AI safety and reliability becoming core concerns. From a methodological standpoint, this paper represents a research paradigm shift from "black-box utilization" to "white-box understanding".

### Specific Suggestions for Future Work
1. Integrate the core idea of this paper with other modalities (vision, audio).
2. Consider validating the scalability of findings on larger-scale models and data.
3. Explore the possibility of combining with reinforcement learning and online learning.
4. Develop automated evaluation and optimization toolchains.


---

## Supplementary Reflections

### Relation to Domain Trends
The research direction of this paper is closely related to several grand trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological perspective, this work represents an exploration into the deep mechanisms of LLMs, contributing to the transition from empirically-driven to theoretically-driven research paradigms.

### Specific Suggestions for Future Work
1. Combine the core idea with other modalities (visual, speech, multimodal) to verify cross-modal generalizability.
2. Validate conclusions on larger-scale models (70B+) and newer architectures (e.g., Mixture-of-Experts).
3. Explore the possibility of combining with reinforcement learning and online learning to achieve dynamically adaptive models.
4. Develop automated evaluation and optimization tools to lower the barrier to using the method.
5. Consider cross-disciplinary exploration with LLM alignment research to investigate the co-optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning to Obstruct Few-Shot Image Classification over Restricted Classes](../../ECCV2024/llm_pretraining/learning_to_obstruct_few-shot_image_classification_over_restricted_classes.md)
- [\[ECCV 2024\] Cross-Domain Learning for Video Anomaly Detection with Limited Supervision](../../ECCV2024/llm_pretraining/cross-domain_learning_for_video_anomaly_detection_with_limited_supervision.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](../../ACL2025/llm_pretraining/data_whisperer_data_selection.md)
- [\[CVPR 2025\] ConText-CIR: Learning from Concepts in Text for Composed Image Retrieval](../../CVPR2025/llm_pretraining/context-cir_learning_from_concepts_in_text_for_composed_image_retrieval.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](../../NeurIPS2025/llm_pretraining/learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)

</div>

<!-- RELATED:END -->
