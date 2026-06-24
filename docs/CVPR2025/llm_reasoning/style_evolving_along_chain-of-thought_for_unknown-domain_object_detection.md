---
title: >-
  [Paper Note] Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection
description: >-
  [CVPR 2025][Reasoning][Domain Generalization] This paper proposes a Chain-of-Thought Guided Style Evolution (CGSE) method. By generating three-level progressive style descriptions (word $\rightarrow$ phrase $\rightarrow$ sentence), combined with feature disentanglement and class-specific prototype clustering, CGSE achieves significant performance improvements in domain generalization object detection on five adverse weather scenarios and the Real-to-Art benchmark.
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Domain Generalization"
  - "Object Detection"
  - "Chain-of-Thought"
  - "Style Transfer"
  - "CLIP"
date: 2026-05-08
content_hash: 9803b612185e9626
---

# Style Evolving along Chain-of-Thought for Unknown-Domain Object Detection

**Conference**: CVPR 2025  
**arXiv**: [2503.09968](https://arxiv.org/abs/2503.09968)  
**Code**: [https://github.com/ZZ2490/SE-COT](https://github.com/ZZ2490/SE-COT)  
**Authors**: Zihao Zhang, Aming Wu, Yahong Han  
**Affiliations**: Tianjin University, Xidian University  
**Area**: LLM Reasoning  
**Keywords**: Domain Generalization, Object Detection, Chain-of-Thought, Style Transfer, CLIP

## TL;DR
This paper proposes a Chain-of-Thought Guided Style Evolution (CGSE) method. By generating three-level progressive style descriptions (word $\rightarrow$ phrase $\rightarrow$ sentence), combined with feature disentanglement and class-specific prototype clustering, CGSE achieves significant performance improvements in domain generalization object detection on five adverse weather scenarios and the Real-to-Art benchmark.

## Background & Motivation
**Background**: Single-domain generalization object detection (Single-DGOD) is an emerging task that aims to train a detector using only a single source domain such that it performs well on unseen target domains. Due to the unavailability of target domain data, recent methods leverage the multimodal capabilities of vision-language models (such as CLIP) to estimate cross-domain information via textual prompts, thereby enhancing the model's generalization ability.

**Limitations of Prior Work**: Existing methods rely on one-step prompts, which use a single simple textual prompt to describe the target domain style all at once. However, when dealing with complex, combined styles (such as "night + rain"), one-step prompts fail to effectively synthesize multiple pieces of style information. Experiments show that the performance of one-step prompts drops significantly under complex combined-weather scenarios because they lack the capability to model style compositionality.

**Key Challenge**: Real-world domain shifts typically involve a combination of multiple style factors—such as time of day, weather, lighting, and artistic style overlaid together. One-step prompting cannot enumerate all possible style combinations, while simply concatenating multiple style descriptions fails to capture the interactions between different styles. How to systematically generate diverse and hierarchical style descriptions is the key to improving domain generalization performance.

**Key Insight**: Drawing inspiration from the progressive reasoning of Chain-of-Thought (CoT) in LLMs, this work decomposes the style description generation process into multiple progressively refined levels. Similar to how CoT moves step-by-step from simple to complex reasoning, style descriptions can evolve sequentially from basic words to combined phrases and finally to full sentences.

**Core Idea**: This paper proposes CGSE (Chain-of-Thought Guided Style Evolution), which divides style description generation into three levels: (1) word level—extracting basic style words from a captioning model; (2) phrase level—combining words into style phrases using ChatGPT; and (3) sentence level—generating complete sentences containing detailed scene descriptions. The output of each level guides the generation of the next, forming a progressive evolution of style.

## Method

### Overall Architecture
The proposed method consists of three core modules: (1) the CGSE style description generation pipeline, which progressively generates representation-rich domain style descriptions through three stages; (2) a feature disentanglement module that separates deep features into style and content features, ensuring disentanglement quality via contrastive learning; and (3) class-specific prototype clustering, which maintains learnable style prototypes for each target class and performs style transfer via AdaIN. The detector is based on Faster R-CNN (with ResNet-50/101 or Swin Transformer backbones) and requires only a single 3090 GPU for training.

### Key Designs

1. **Three-stage Style Description Generation (CGSE)**:

    - **Function**: Generating diverse domain style description texts from coarse to fine.
    - **Mechanism**:
        - Stage 1 (Word level): Use a captioning model on source domain images to generate descriptions, extract keywords, and group them into 5 categories—weather, time, style, action, and detail.
        - Stage 2 (Phrase level): Use ChatGPT to freely combine words from different categories into style phrases (e.g., "rainy night", "foggy dawn"), expanding the style combination space.
        - Stage 3 (Sentence level): Further extend phrases into complete scene description sentences using ChatGPT, incorporating rich environmental and visual details.
    - **Design Motivation**: This progressive generation ensures the diversity and hierarchy of style descriptions—the word level covers basic style elements, the phrase level captures style combinations, and the sentence level provides the full scenic context.

2. **Feature Disentanglement**:

    - **Function**: Separating deep features extracted by the detector into style and content information.
    - **Mechanism**: Use two independent branches to extract style and content features separately, and apply contrastive loss to ensure that style and content features are orthogonal in the representation space—specifically, the style and content features of the same image are separated (negative pairs), while content features of different domains but the same class are pulled closer (positive pairs).
    - **Design Motivation**: Domain shift is primarily manifested in the style dimension (e.g., color, texture, lighting), while the content dimension (e.g., object shape, semantics) should remain domain-invariant. Disentanglement allows for the independent manipulation of style features for style transfer without affecting the content representation.

3. **Class-Specific Prototype Clustering**:

    - **Function**: Maintaining a set of learnable style prototypes for each target class.
    - **Mechanism**: Maintain $M$ style prototype vectors. During training, the extracted style features are assigned to the nearest prototype and updated via momentum. Critical styles are injected into features via AdaIN (Adaptive Instance Normalization) by replacing the mean and variance of the content features with the statistics of the prototype styles during inference.
    - **Design Motivation**: Class-specific prototypes avoid a one-size-fits-all style transfer, as different classes (such as "car" vs. "pedestrian") may exhibit distinct style-shift patterns across different domains.

### Loss & Training
- **Detection Loss**: Standard Faster R-CNN loss (classification + regression)
- **Contrastive Disentanglement Loss**: InfoNCE contrastive loss, ensuring orthogonal separation of style and content features
- **Prototype Update**: Momentum update strategy to avoid drastic changes in prototypes
- **Training Efficiency**: Requires only a single NVIDIA 3090 GPU

## Key Experimental Results

### Main Results: Adverse Weather Driving Scenarios

| Method | Day Clear | Night | Dusk Rainy | Night Rainy | Day Foggy | Average |
|------|-----------|-------|-----------|------------|----------|------|
| Faster R-CNN (baseline) | 49.6 | 34.7 | 25.7 | 11.8 | 28.4 | 30.0 |
| SW | 50.3 | 38.5 | 32.8 | 17.7 | 35.0 | 34.9 |
| DIV | 52.8 | 42.5 | 38.1 | 24.1 | 37.2 | 38.9 |
| **Ours (R50)** | **55.4** | **42.0** | **39.2** | **24.5** | **40.6** | **40.3** |
| **Ours (Swin)** | **64.4** | **52.7** | **49.5** | **33.7** | **44.9** | **49.0** |

Real-to-Art Cross-Domain Detection (VOC $\rightarrow$ Art Domains):

| Method | VOC | Comic | Watercolor | Clipart | Average |
|------|-----|-------|------------|---------|------|
| DIV | 83.4 | 31.2 | 55.1 | 37.3 | 51.8 |
| **Ours (R101)** | **87.6** | **36.9** | **60.7** | **42.5** | **56.9** |

### Ablation Study

| Configuration | Source | Night | Dusk Rainy | Night Rainy | Foggy | Average |
|------|--------|-------|-----------|------------|-------|------|
| Baseline | 49.6 | 34.7 | 25.7 | 11.8 | 28.4 | 30.0 |
| +One-step prompt | 52.4 | 36.9 | 28.9 | 14.7 | 32.1 | 33.0 |
| +CGSE (3-stage) | 54.2 | 40.7 | 31.2 | 17.9 | 35.7 | 35.9 |
| +Feature Disentanglement | 54.8 | 41.2 | 36.5 | 21.3 | 38.4 | 38.4 |
| +All (Full Method) | 55.4 | 42.0 | 39.2 | 24.5 | 40.6 | 40.3 |

### Analysis of the Number of Chain-of-Thought Levels

| Number of CoT Levels | Night Rainy | Day Foggy | Average |
|-----------|------------|----------|------|
| 1 (Word-level) | 14.7 | 32.1 | 33.0 |
| 2 (Word + Phrase) | 19.8 | 36.9 | 36.8 |
| 3 (Word + Phrase + Sentence) | **24.5** | **40.6** | **40.3** |
| 4 (With Additional Extension) | 23.1 | 39.8 | 39.5 |
| 5 (With More Extensions) | 22.4 | 38.2 | 38.6 |

3-level CoT is optimal; beyond 3 levels, over-extended style descriptions begin to introduce noise, leading to degraded performance.

### Key Findings
- **CGSE is the core contribution**: Upgrading from one-step prompting to the three-stage CGSE improves the mAP by 3.2 on the most challenging Night Rainy scenario.
- **Complex combined scenarios benefit the most**: The Night Rainy scenario (a combination of night and rain) shows the most significant improvement (11.8 $\rightarrow$ 24.5), which validates the capability of multi-stage style evolution in modeling complex combined domain shifts.
- **Three levels is the optimal trade-off**: Too many levels introduce description noise, whereas too few levels fail to capture sufficient style diversity.
- **Swin Transformer backbone yields significant gains**: A stronger backbone coordinates better with style transfer, resulting in an average improvement of around 9 mAP.

## Highlights & Insights
- **Novel Application of CoT**: Applying the Chain-of-Thought concept to generate style descriptions for data augmentation, rather than for the reasoning process itself, is a novel paradigm.
- **Cost-Friendly**: Training requires only a single 3090 GPU without requiring any target domain data or complex domain adaptation pipelines.
- **Seamless Integration with LLMs**: Utilizing ChatGPT to generate style descriptions represents a lightweight approach to LLM application, requiring no fine-tuning or online LLM inference during detector training.

## Limitations & Future Work
- **Dependence on Captioning Models**: The quality of keyword extraction in Stage 1 depends heavily on the captioning model; low-quality descriptions will inevitably affect the subsequent levels.
- **Manual Tuning of Style Prototype Count**: The selection of $M$ requires manual tuning based on the characteristics of the datasets.
- **Evaluation Limited to Detection**: Whether the method can generalize to other computer vision tasks such as segmentation and classification remains to be explored.
- **Uncertainty of ChatGPT**: There is inherent randomness in the style descriptions generated by the LLM, which may lead to varying results across different runs.

## Related Work & Insights
- **vs. DIV**: DIV utilizes one-step prompting to describe the target-domain style, whereas this work generates a much richer style space through multi-stage progressive style descriptions.
- **vs. CLIP-based DGOD**: While most methods directly use CLIP to encode fixed text descriptions, this work applies a CoT-style mechanism to generate diverse, hierarchical descriptions.
- **vs. Traditional Domain Adaptation**: The proposed method does not require target domain data and achieves generalization through style imagination rather than traditional domain alignment.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of CoT to style description generation is highly novel, and the three-stage progressive design is compelling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated extensively across five weather scenarios and cross-domain art detection with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, and the hierarchical design logic is consistent.
- Value: ⭐⭐⭐ The method is effective but targeting a relatively narrow application scope, primarily focused on single-domain generalization detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Interleaved-Modal Chain-of-Thought](interleaved-modal_chain-of-thought.md)
- [\[CVPR 2025\] Argus: Vision-Centric Reasoning with Grounded Chain-of-Thought](argus_vision-centric_reasoning_with_grounded_chain-of-thought.md)
- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](../../NeurIPS2025/llm_reasoning/latent_chain-of-thought_for_visual_reasoning.md)
- [\[ACL 2026\] Towards Effective In-context Cross-domain Knowledge Transfer via Domain-invariant-neurons-based Retrieval](../../ACL2026/llm_reasoning/towards_effective_in-context_cross-domain_knowledge_transfer_via_domain-invarian.md)
- [\[ICLR 2026\] On The Fragility of Benchmark Contamination Detection in Reasoning Models](../../ICLR2026/llm_reasoning/on_the_fragility_of_benchmark_contamination_detection_in_reasoning_models.md)

</div>

<!-- RELATED:END -->
