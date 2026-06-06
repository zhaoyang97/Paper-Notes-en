---
title: >-
  [Paper Note] Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events
description: >-
  [CVPR 2026][Interpretability][Multimodal summarization] This paper proposes CoE, a training-free multimodal summarization framework that constructs a Hierarchical Event Graph (HEG) to guide chain-of-events reasoning. CoE…
tags:
  - "CVPR 2026"
  - "Interpretability"
  - "Multimodal summarization"
  - "training-free"
  - "chain-of-events reasoning"
  - "hierarchical event graph"
  - "cross-domain generalization"
date: 2026-05-08
content_hash: 9a0529d831ecf897
---

# Cut to the Chase: Training-free Multimodal Summarization via Chain-of-Events

**Conference**: CVPR 2026
**arXiv**: [2603.06213](https://arxiv.org/abs/2603.06213)  
**Code**: [GitHub](https://github.com/youxiaoxing/CoE)  
**Area**: Interpretability
**Keywords**: Multimodal summarization, training-free, chain-of-events reasoning, hierarchical event graph, cross-domain generalization

## TL;DR

This paper proposes CoE, a training-free multimodal summarization framework that constructs a Hierarchical Event Graph (HEG) to guide chain-of-events reasoning. CoE surpasses state-of-the-art video CoT baselines across 8 datasets, achieving average gains of +3.04 ROUGE, +9.51 CIDEr, and +1.88 BERTScore.

## Background & Motivation

### Limitations of Prior Work

**Background**: **Importance of Multimodal Summarization (MMS)**: MMS requires generating concise textual summaries from multi-source inputs such as video, text, and images, with applications in instructional videos, lectures, and news broadcasts.

**Dependence on Domain-Specific Supervision**: Existing MMS models (e.g., MLASK, MMSum) rely on large-scale paired data and domain-specific fine-tuning, resulting in poor cross-domain generalization. Experiments show significant performance degradation when models trained on VIEWS are transferred to other datasets.

**Implicit Fusion and Weak Cross-modal Alignment**: Most existing methods perform implicit fusion in latent space, lacking explicit reasoning over visual-textual correspondences, which leads to semantic drift.

**Flat Temporal Modeling**: Video CoT models treat videos as flat sequences of frames or clips, without explicitly modeling hierarchical events and causal transitions, making it difficult to capture global event evolution.

**Potential of MLLMs**: Multimodal large language models have brought breakthroughs in video understanding, yet their direct application to long-video summarization still faces the aforementioned challenges.

**Mechanism**: Replace implicit holistic fusion with explicit hierarchical event modeling to achieve interpretable, training-free, and cross-domain robust summarization.

## Method

### Overall Architecture

CoE consists of four modules: (1) Hierarchical Event Graph (HEG) Construction → (2) Cross-modal Spatial Grounding (CSG) → (3) Event Evolution Reasoning (EER) → (4) Domain-adaptive Summary Generation (DSG).

### Key Designs

#### Hierarchical Event Graph (HEG) Construction

A three-layer structure: Global Event Layer (overall theme) → Sub-event Layer (decomposed into $K$ coherent components) → Entity-Relation Layer (modeling key entities and their interactions). The graph is automatically extracted from text via an LLM.

#### Cross-modal Spatial Grounding (CSG)

Uniformly sampled video frames are divided into short clips $\{C_j\}$. Sub-event nodes in the HEG serve as semantic anchors, and each clip is aligned to its most relevant sub-event. Visually grounded entity-relation triplets are then identified within each clip to construct a visual grounding subgraph $\mathcal{G}_k^{(j)}$.

#### Event Evolution Reasoning (EER)

Adjacent clips sharing consistent subgraphs under the same sub-event are merged into longer temporal segments. Changes between subgraphs of adjacent segments (added/sustained/disappeared entity relations) are compared to derive event trajectory descriptions $\mathcal{D}_p$, capturing narrative evolution.

#### Domain-adaptive Summary Generation (DSG)

Event trajectories are synthesized into an initial summary $\hat{s}_{\text{init}}$, which is then refined using a small set of target-domain reference summaries $\mathcal{Y}_{\text{ref}}$ for lightweight style adaptation, adjusting tone and rhetorical structure.

### Loss & Training

No training is required; thus no loss function is needed. The entire pipeline is driven by VLM/LLM prompting.

## Key Experimental Results

### Main Results: Average Performance across 8 Datasets

| Method | ROUGE↑ | CIDEr↑ | BERTScore↑ |
|--------|--------|--------|------------|
| TCoT | baseline | baseline | baseline |
| CoF | +0.5 | +2.1 | +0.3 |
| ViTCoT | +1.2 | +4.5 | +0.9 |
| CoS | +1.8 | +5.2 | +1.1 |
| **CoE (Ours)** | **+3.04** | **+9.51** | **+1.88** |

### Ablation Study

| Module | Contribution |
|--------|-------------|
| HEG Construction | Provides structured semantic scaffold |
| CSG Cross-modal Grounding | Fine-grained visual-textual alignment |
| EER Event Evolution | Temporal coherence modeling |
| DSG Style Adaptation | Cross-domain linguistic style alignment |

### Key Findings

- CoE maintains stable performance across 8 domains in a zero-shot setting, whereas supervised methods degrade significantly under domain shift.
- Each module contributes independently and complementarily.
- The framework is consistently effective across different MLLM backbones (e.g., GPT-4o, Gemini).
- Performance improves steadily with increasing model scale.

## Highlights & Insights

- The **training-free** design confers strong cross-domain generalization, addressing the long-standing supervision dependency in MMS.
- The hierarchical event graph design is elegant, mirroring human cognition from global theme → sub-events → entity relations.
- The EER module explicitly models causal transitions, surpassing flat temporal modeling approaches.
- Lightweight style adaptation requires only a small number of references to align with target-domain language conventions.

## Limitations & Future Work

- Performance depends on the quality of the underlying MLLM (e.g., GPT-4o), incurring high inference costs.
- The video frame sampling strategy may miss critical content.
- Style adaptation requires a small set of target-domain reference summaries, making it not fully zero-resource.
- HEG construction quality is bounded by the LLM's extraction capability.

## Related Work & Insights

- Compared to video CoT methods such as CoF and ViTCoT, CoE adopts a global event perspective rather than local frame-level reasoning.
- Compared to traditional MMS methods (MLASK, MMSum), CoE requires no training.
- The hierarchical event graph concept is generalizable to tasks such as video understanding and long-document summarization.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[CVPR 2026\] U-F²-CBM: CLIP-Free, Label Free, Unsupervised Concept Bottleneck Models](clipfree_label_free_unsupervised_concept_bottlenec.md)
- [\[CVPR 2026\] Towards Faithful Multimodal Concept Bottleneck Models](towards_faithful_multimodal_concept_bottleneck_models.md)
- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](../../NeurIPS2025/interpretability/vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)

</div>

<!-- RELATED:END -->
