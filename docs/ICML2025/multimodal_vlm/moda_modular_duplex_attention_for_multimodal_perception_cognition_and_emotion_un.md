---
title: >-
  [Paper Note] MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding
description: >-
  [ICML 2025 (Spotlight, Top 2.6%)][Multimodal VLM][Multimodal Large Language Models] To address the "Attention Deficit Disorder" issue characterized by cross-modal attention inconsistency and layer-wise decay in Multimodal Large Language Models (MLLMs), this paper proposes a modular duplex attention mechanism named MODA. By decoupling attention into intra-modal self-refinement and inter-modal interaction pathways, and utilizing a Duplex Aligner combined with adaptive masked at…
tags:
  - "ICML 2025 (Spotlight, Top 2.6%)"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Attention Mechanism"
  - "Cross-Modal Alignment"
  - "Emotion Understanding"
  - "Cognitive Understanding"
date: 2026-05-08
content_hash: 8e0b71c15efe97cb
---

# MODA: MOdular Duplex Attention for Multimodal Perception, Cognition, and Emotion Understanding

**Conference**: ICML 2025 (Spotlight, Top 2.6%)  
**arXiv**: [2507.04635](https://arxiv.org/abs/2507.04635)  
**Authors**: Zhicheng Zhang, Wuyou Xia, Chenxi Zhao, Zhou Yan, Xiaoqiang Liu, Yongjie Zhu, Wenyu Qin, Pengfei Wan, Di Zhang, Jufeng Yang  
**Code**: [https://zzcheng.top/MODA](https://zzcheng.top/MODA)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Large Language Models, Attention Mechanism, Cross-Modal Alignment, Emotion Understanding, Cognitive Understanding  

## TL;DR

To address the "Attention Deficit Disorder" issue characterized by cross-modal attention inconsistency and layer-wise decay in Multimodal Large Language Models (MLLMs), this paper proposes a modular duplex attention mechanism named MODA. By decoupling attention into intra-modal self-refinement and inter-modal interaction pathways, and utilizing a Duplex Aligner combined with adaptive masked attention to implement a "Correct-after-Align" strategy, its effectiveness is validated across 21 perception, cognition, and emotion benchmarks.

## Background & Motivation

### Problem Background
Multimodal Large Language Models (MLLMs) have demonstrated powerful capabilities in integrating multimodal data such as vision and language, emerging as a crucial path towards Artificial General Intelligence (AGI). Existing MLLMs primarily achieve multimodal token mixing and fusion via attention mechanisms, performing well on basic perception tasks. However, higher-level cognitive understanding (such as decision-making and judgment in role-playing and code generation) and emotional understanding (such as sarcasm detection and micro-expression recognition) demand fine-grained cross-modal information fusion that far exceeds the capabilities of current models.

### Limitations of Prior Work
- **Language-dominant bias**: Mainstream methods excessively focus on language-centric instruction tuning, neglecting the full utilization of the visual modality in attention mixing.
- **Suppression of the visual modality**: Studies such as MMVP reveal that existing MLLMs fail to sufficiently activate the visual modality, particularly when processing low-level visual attributes.
- **Poor performance on public benchmarks**: Three SOTA models achieve only around $50\%$ accuracy (close to random guessing) on binary sarcasm detection in the HFM dataset.
- **Cambrian-1** mitigates but does not fundamentally resolve the attention imbalance problem, despite enhancing visual features through a spatial vision aggregator.

### Core Finding: Attention Deficit Disorder
The authors conduct an in-depth analysis of the multimodal token mixing process via the attention mechanism in MLLMs, identifying two critical issues:

**Cross-modal attention inconsistency**: Across different layers of MLLMs, attention allocation is severely skewed, with attention weights heavily biased towards linguistic components.

**Layer-wise attention decay**: As depth increases, attention activation for the visual modality decays layer-by-layer, further exacerbating the modal imbalance.

For instance, in a clip from *The Godfather*, SOTA MLLMs fail to capture fine-grained visual details such as a character's subtle expression or gaze, leading to incorrect emotional interpretation. Quantitative analysis indicates a cross-layer attention discrepancy of up to $63\%$, with attention weights between different modalities differing by up to 10-fold.

### Core Motivation
Multimodal attention inherently suffers from an imbalance between intra-modal self-interaction and cross-modal interaction, which simple unified attention cannot reconcile. A mechanism capable of explicitly separating and regulating these two components is needed to achieve effective cross-modal feature alignment while preserving the independent characteristics of each modality.

## Method

### Overall Architecture
MODA is built upon the standard MLLM architecture (vision encoder + connector + LLM). The core innovation lies in replacing the standard attention layers in the LLM with MOdular Duplex Attention. The overall framework adheres to a "Correct-after-Align" strategy, effectively decoupling modality alignment from cross-layer token mixing.

### Key Designs

#### Key Design 1: Duplex (V/T)-Aligner

As a core component of MODA, the Duplex Aligner maps multimodal tokens into a shared dual-modal representation space.

- **Duplex Space Construction**: Two sets of basis vectors for vision (V) and text (T) are defined to construct their respective Gram matrices.
- **Alignment Phase**: Input vision and language tokens are mapped into these two duplex modal spaces. In the visual space, while vision tokens retain their intrinsic features, language tokens are projected onto the visual space to facilitate cross-modal interaction; the same process applies to the textual space.
- **Bidirectional Interaction**: Through duplex mapping, visual and linguistic modalities interact within their own "home" spaces, avoiding the modality bias inherent in a single shared space.

#### Key Design 2: Modular Attention Decomposition

MODA explicitly decomposes conventional unified attention into two modules:

1. **Self-Modal Attention**:
    - Focuses on capturing intrinsic relationships within a single modality.
    - Computes self-attention independently among vision tokens and language tokens.
    - Preserves the independent representation capability of each modality, preventing cross-modal interference.

2. **Cross-Modal Attention**:
    - Implements feature alignment and information exchange between different modalities.
    - Operates within the duplex spaces mapped by the Duplex Aligner.
    - Ensures that visual and textual features effectively complement each other.

#### Key Design 3: Modular Masked Attention

To further enhance model flexibility, MODA introduces an adaptive masked attention mechanism:

- **Customized Masking Patterns**: Tailored attention masks are designed according to different modalities and task types.
- **Adaptive Focusing**: The model dynamically adjusts its attention allocation across different modalities based on task demands.
- **Correction Phase**: Upon completion of alignment, attention weights are calibrated using masked attention to ensure cross-modal attention consistency, addressing the attention decay issue.

### Loss & Training

MODA adopts a multi-stage training strategy, built on top of pre-trained MLLMs:

- **Stage 1: Alignment Pre-training** — The vision encoder and LLM weights are frozen. The Duplex Aligner and connector are trained using large-scale image-text alignment data to establish the initial mapping of the duplex modal spaces.
- **Stage 2: Instruction Tuning** — The MODA attention layers are unfrozen, followed by end-to-end fine-tuning on diverse instruction-following data, covering perception, cognition, and emotion tasks.
- **Advantages of Modular Design** — The attention modules of MODA can serve as plug-and-play components to replace standard attention layers in existing MLLMs without modifying other parts of the original architecture.

## Key Experimental Results

### Perception Task Evaluation

MODA is comprehensively evaluated on multiple visual perception benchmarks, spanning general visual question answering, visual reasoning, and hallucination detection tasks:

| Benchmark Type | Benchmark Name | Evaluated Capability | MODA Performance |
|---|---|---|---|
| General VQA | MMMU | Multidisciplinary Understanding | Outperforms Baseline |
| General VQA | MMBench | Multi-dimensional Perception | Outperforms Baseline |
| Visual Reasoning | MMVP | Visual Patterns | Significant Improvement |
| Hallucination Detection | POPE | Object Hallucination | Effectively Improved |
| Document Understanding | DocVQA | OCR Understanding | Competitive |
| Fine-grained Perception | Multiple Benchmarks | Detail Recognition | Superior to Baseline |

MODA achieves particularly remarkable improvements on benchmarks requiring fine-grained visual perception such as MMVP, directly validating the effectiveness of resolving the attention deficit disorder.

### Cognitive and Emotion Task Evaluation

MODA achieves qualitative improvements over baselines on high-level understanding tasks:

| Task Type | Benchmark Name | Specific Capability | Key Findings |
|---|---|---|---|
| Sarcasm Detection | HFM | Multimodal Sarcasm | Substantial increase from ~50% random-level |
| Sentiment Analysis | Multiple Benchmarks | Fine-grained Sentiment | Significantly outperforms SOTA |
| Persona Cognition | Role-playing Benchmarks | Decisional Judgment | Consistent improvements |
| Emotional Reasoning | Emotional Reasoning Benchmarks | Causal Inference | Enhanced cross-modal reasoning |
| Micro-expression Recognition | Micro-expression Benchmarks | Subtle Features | Visual attention is obviously improved |

A total of 21 benchmarks comprehensively cover the three major dimensions of perception, cognition, and emotion, with MODA demonstrating consistent superiority across all dimensions.

### Ablation Study Key Findings

- **Necessity of Attention Decomposition**: Removing the self-modal/cross-modal decomposition significantly degrades performance, validating the efficacy of the modular design.
- **Contribution of Duplex Aligner**: Duplex space mapping yields substantial improvements compared to a single shared space mapping.
- **Effect of Modular Mask**: Adaptive masking provides varying degrees of gain across different task types, with emotional tasks benefiting the most.
- **Improvement in Attention Inconsistency**: Visualizations show that MODA successfully alleviates cross-layer attention discrepancy, reducing the $63\%$ fluctuation to a controlled range.

## Highlights & Insights

- **Profound Problem Identification**: This work is the first to systematically define and quantify the "Attention Deficit Disorder" in MLLMs. The discovery of a $63\%$ cross-layer variance and a 10-fold attention gap between modalities has significant diagnostic value.
- **Elegant Design Philosophy**: Decoupling attention into self-modal and cross-modal pathways is simple and effective. The "Correct-after-Align" strategy is highly intuitive.
- **Innovative Duplex Space**: The duplex modal space constructed via the Gram matrix provides a "home-field advantage" for each modality, preventing the modality bias introduced by a single shared space in conventional methods.
- **Comprehensive Task Coverage**: Evaluation across 21 benchmarks spanning perception, cognition, and emotion systematically validates the generalization ability of the proposed method.
- **Plug-and-Play Design**: The MODA attention module can seamlessly replace attention layers in existing MLLMs, offering high engineering friendliness.
- **Spotlight Acceptance**: Accepted as an ICML 2025 Spotlight (Top 2.6%), reflecting high academic recognition.

## Limitations & Future Work

- **Computational Complexity**: Computing duplex attention separately for self-modal and cross-modal interactions theoretically increases computational overhead compared to standard attention, leaving the efficiency-performance trade-off to be fully evaluated.
- **Modality Scalability**: The current design targets the vision-language bi-modality. Scaling up to more modalities (e.g., audio, video) requires redesigning the construction of duplex spaces.
- **Basis Vector Selection**: The initialization and learning strategies of basis vectors in the Duplex Aligner and their impact on final performance warrant further research.
- **Reliance on Training Data**: The multi-stage training strategy requires a large, diverse set of data, which may be less practical for resource-constrained scenarios.
- **Lack of Theoretical Analysis**: The analysis of attention deficit disorder is primarily empirical, lacking a formal theoretical explanation for the mechanism of attention decay.
- **Incomplete Cache**: This note is based on a truncated arXiv cached paper; please refer to the original paper for precise experimental metrics and further details.

## Related Work & Insights

### MLLM Attention Mechanism Enhancements
- **Cambrian-1 (Tong et al., 2024a)**: Introduces a spatial vision aggregator to enhance visual features, yet fails to resolve modality imbalance at the attention mechanism level.
- **MMVP (Tong et al., 2024b)**: Reveals that MLLMs fail to fully activate the visual modality; this work further isolates the issue to the attention mechanism.
- **Flamingo (Alayrac et al., 2022)**: Integrates multiple modalities via cross-attention, which still exhibits modality bias.

### Emotion and Cognitive Understanding
- **EmotionCLIP (Yang et al., 2024)**: Discusses the challenges of emotional understanding in multimodal MLLMs, which this work extends into the cognitive dimension.
- **Role Cognition Research (Wang et al., 2024; Chen et al., 2024)**: Cognitive tasks demand fine-grained cross-modal reasoning, which MODA supports by improving attention allocation.

### Insights
The attention decomposition strategy of MODA can be generalized to other multimodal fusion scenarios. Designing attention mechanisms should avoid simply mixing all modal tokens; instead, it should account for modality specificity, designing dedicated mechanisms for different interaction types (self-modal vs. cross-modal). This design philosophy provides direct inspiration for building stronger multimodal agents, such as emotionally aware dialogue systems.

## Rating

- Novelty: ⭐⭐⭐⭐ — The concept of attention deficit disorder is novel, and the duplex attention design is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 21 benchmarks covering three major dimensions, with extensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — In-depth analysis of problems, clear arguments, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ — A Spotlight paper that provides valuable reference for improving MLLM attention mechanisms, though computational overhead remains to be evaluated.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Seeing is Understanding: Unlocking Causal Attention into Modality-Mutual Attention for Multimodal LLMs](../../ICML2026/multimodal_vlm/seeing_is_understanding_unlocking_causal_attention_into_modality-mutual_attentio.md)
- [\[ACL 2025\] Aligning VLM Assistants with Personalized Situated Cognition](../../ACL2025/multimodal_vlm/aligning_vlm_assistants_with_personalized_situated.md)
- [\[CVPR 2025\] SmartCLIP: Modular Vision-language Alignment with Identification Guarantees](../../CVPR2025/multimodal_vlm/smartclip_modular_vision-language_alignment_with_identification_guarantees.md)
- [\[ACL 2025\] AkaCE: A Multimodal Multi-party Dataset for Emotion Recognition in Movie Dialogues](../../ACL2025/multimodal_vlm/akan_cinematic_emotions_ace_a_multimodal_multi-party_dataset_for_emotion_recogni.md)
- [\[ACL 2026\] Dynamic Emotion and Personality Profiling for Multimodal Deception Detection](../../ACL2026/multimodal_vlm/dynamic_emotion_and_personality_profiling_for_multimodal_deception_detection.md)

</div>

<!-- RELATED:END -->
