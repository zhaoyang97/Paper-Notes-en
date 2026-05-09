---
title: >-
  [Paper Note] Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models
description: >-
  [CVPR 2026][Multimodal VLM][Hierarchical Visual Recognition] This paper proposes TARA, a framework that injects taxonomic hierarchy knowledge into large multimodal models (LMMs) by aligning their intermediate representations with taxonomy-aware features from a biological foundation model (BFM), substantially improving hierarchical visual recognition performance on both known and novel categories.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Hierarchical Visual Recognition
  - Biological Taxonomy
  - Representation Alignment
  - Biological Foundation Model
  - Reinforcement Learning Fine-Tuning
date: 2026-05-08
content_hash: 51f50a01038dd0c4
---

# Taxonomy-Aware Representation Alignment for Hierarchical Visual Recognition with Large Multimodal Models

**Conference**: CVPR 2026
**arXiv**: [2603.00431](https://arxiv.org/abs/2603.00431)
**Code**: [https://github.com/PKU-ICST-MIPL/TARA_CVPR2026](https://github.com/PKU-ICST-MIPL/TARA_CVPR2026)
**Area**: Multimodal VLM
**Keywords**: Hierarchical Visual Recognition, Biological Taxonomy, Representation Alignment, Biological Foundation Model, Reinforcement Learning Fine-Tuning

## TL;DR
This paper proposes TARA, a framework that injects taxonomic hierarchy knowledge into large multimodal models (LMMs) by aligning their intermediate representations with taxonomy-aware features from a biological foundation model (BFM), substantially improving hierarchical visual recognition performance on both known and novel categories.

## Background & Motivation
**Background**: Large multimodal models have demonstrated strong performance on fine-grained visual recognition (FGVR), yet hierarchical visual recognition (HVR) requires models to predict consistent label paths from coarse to fine granularity—a capability that remains underdeveloped.

**Limitations of Prior Work**: LMMs frequently violate taxonomic hierarchies, producing inconsistent predictions along paths such as "Kingdom → Phylum → Class → Order → Family → Genus → Species." This problem is further exacerbated for novel categories absent from the training set.

**Key Challenge**: The visual feature encoders of LMMs lack hierarchical biological priors, preventing consistent recognition across different granularity levels.

**Goal**: To inject taxonomic hierarchy knowledge into LMMs so that they produce hierarchically consistent recognition results for both known and novel categories.

**Key Insight**: Biological foundation models (e.g., BioCLIP2), trained via hierarchical contrastive learning, encode rich biological relationships and can serve as a source of taxonomic knowledge.

**Core Idea**: Align the intermediate visual representations and the first answer token representation of the LMM with the visual features and text label features of the BFM, respectively, thereby injecting taxonomic knowledge into the LMM.

## Method

### Overall Architecture
The input consists of an image and a VQA question specifying a taxonomic level (multiple-choice, 4 options). Without modifying the inference pipeline, TARA injects BFM taxonomic knowledge into the LMM via representation alignment during training. Neither the BFM nor the projection modules are required at inference time.

### Key Designs

1. **Taxonomic Visual Representation Alignment**:

    - **Function**: Aligns the visual token representations at layer $\ell$ of the LMM with the visual features of the BFM.
    - **Mechanism**: A learnable projector $P_V$ maps the LMM's visual representations into the BFM feature space, minimizing the cosine similarity loss $\mathcal{L}_V = -\frac{1}{N}\sum_{i=1}^{N}\text{sim}(P_V(\mathbf{e}^{\text{img}}_{\ell,i}), \mathbf{y}_i^{\text{img}})$.
    - **Design Motivation**: The BFM encodes ecological and taxonomic relationships among species through hierarchical contrastive training; after alignment, the LMM's visual representations inherit this hierarchical structure.

2. **Free-grained Label Representation Alignment**:

    - **Function**: Aligns the hidden state of the first generated answer token of the LMM with the target-granularity label encoded by the BFM.
    - **Mechanism**: A projector $P_T$ maps the first answer token into the BFM text space, minimizing $\mathcal{L}_C = \text{sim}(P_T(\mathbf{e}^{\text{answer}}_m[0]), \mathbf{y}^{\text{label}})$.
    - **Design Motivation**: The same image corresponds to different labels at different granularity levels (experts require species names, while general users may only need "bird"). This alignment enables the model to flexibly map to different granularities according to user intent.

3. **Alternating Training Strategy**:

    - **Function**: Alternates between TARA alignment loss training and No-Thinking RFT.
    - **Mechanism**: No-Thinking RFT omits chain-of-thought reasoning and directly produces concise answers using accuracy-based rewards. Alternating optimization with TARA enables more efficient knowledge injection.
    - **Design Motivation**: Explicit reasoning is unnecessary—and may even be detrimental—for classification tasks; alternating training balances taxonomic knowledge injection with the exploratory capacity of reinforcement learning.

### Loss & Training
The total loss is $\mathcal{L}_{\text{alignment}} = (\mathcal{L}_V + \mathcal{L}_C)/2$, alternated with No-Thinking RFT. Both projectors $P_V$ and $P_T$ are three-layer MLPs with SiLU activations. The BFM and projectors are removed at inference time, incurring no additional overhead.

## Key Experimental Results

### Main Results

| Base Model | RL | TARA | HCA (Plant) | Acc_leaf (Plant) | HCA (Animal) | Acc_leaf (Animal) |
|---------|-----|------|-------------|-----------------|--------------|-------------------|
| Qwen3-VL-2B | ✗ | ✗ | 6.46 | 30.16 | 7.18 | 27.86 |
| Qwen3-VL-2B | ✓ | ✗ | 9.23 | 31.96 | 8.57 | 29.32 |
| Qwen3-VL-2B | ✓ | ✓ | **12.78** | **32.66** | **10.26** | **30.77** |
| Qwen2.5-VL-3B | ✗ | ✗ | 10.89 | 39.73 | 16.70 | 40.26 |
| Qwen2.5-VL-3B | ✓ | ✗ | 17.91 | 44.35 | 21.99 | 46.25 |
| Qwen2.5-VL-3B | ✓ | ✓ | **19.53** | **45.66** | **24.02** | **49.16** |

### TerraIncognita Novel Categories

| Species Type | RL | TARA | Order F1 | Family F1 |
|---------|-----|------|----------|-----------|
| Known | ✗ | ✗ | 17.16 | 10.83 |
| Known | ✓ | ✓ | **41.56** | **25.47** |
| Novel | ✗ | ✗ | 17.16 | 10.83 |
| Novel | ✓ | ✓ | **33.45** | **12.67** |

### Key Findings
- TARA yields consistent and significant improvements across all base models, with the most pronounced gains on the HCA metric (e.g., +3.55% on Qwen3-VL-2B).
- On novel categories in TerraIncognita, TARA improves Order-level F1 by more than 10 points, demonstrating effective generalization to unseen classes.
- The RL+TARA combination outperforms either component alone, indicating their complementary nature.
- The BFM is not required at inference time, introducing no additional computational overhead.

## Highlights & Insights
- **Zero inference overhead**: The BFM and projectors are used exclusively during training and are fully removed at inference time. This allows taxonomic knowledge gains to be obtained "for free," making the approach highly practical.
- **Insight from No-Thinking RFT**: Explicit reasoning may actually be harmful for classification tasks; directly producing answers combined with exploratory RL yields better results. This insight is transferable to other non-reasoning-intensive VLM tasks.
- **Free-grained alignment**: By aligning the first token representation rather than enforcing fixed granularity across all levels, the model can flexibly adjust its recognition granularity according to the user's query.

## Limitations & Future Work
- Experiments are conducted exclusively in the biological taxonomy domain; other hierarchical classification scenarios (e.g., product category taxonomies, document classification) remain unexplored.
- The approach relies on BioCLIP2 as a teacher model; applying TARA to non-biological domains requires identifying a corresponding domain-specific foundation model.
- Only a 1-shot setting is evaluated; the effect of varying the number of few-shot examples on performance is not thoroughly investigated.
- The 4-option VQA setting is considerably simpler than open-set hierarchical classification; the impact of distractor design warrants further analysis.

## Related Work & Insights
- **vs. Fine-R1**: Fine-R1 employs a two-stage framework to learn few-shot FGVR reasoning; TARA directly injects taxonomic knowledge via representation alignment, offering a more lightweight alternative.
- **vs. HCPT**: HCPT performs hierarchically consistent prompt tuning on CLIP; TARA achieves a similar objective on LMMs through BFM alignment and additionally generalizes to novel categories.

## Rating
- Novelty: ⭐⭐⭐⭐ — The approach of injecting BFM knowledge into LMMs is novel, and the zero-inference-overhead design is practically valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Validated across multiple models and datasets with thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rigorous mathematical descriptions.
- Value: ⭐⭐⭐⭐ — Opens a new direction for hierarchical recognition with LMMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Predictive Regularization Against Visual Representation Degradation in Multimodal Large Language Models](predictive_regularization_against_visual_representation_degradation_in_multimoda.md)
- [\[CVPR 2026\] The LLM Bottleneck: Why Open-Source Vision LLMs Struggle with Hierarchical Visual Recognition](the_llm_bottleneck_why_open-source_vision_llms_struggle_with_hierarchical_visual.md)
- [\[CVPR 2026\] CoVFT: Context-aware Visual Fine-tuning for Multimodal Large Language Models](covft_context-aware_visual_fine-tuning_for_multimodal_large_language_models.md)
- [\[CVPR 2026\] HAWK: Head Importance-Aware Visual Token Pruning in Multimodal Models](hawk_head_importance-aware_visual_token_pruning_in_multimodal_models.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)

</div>

<!-- RELATED:END -->
