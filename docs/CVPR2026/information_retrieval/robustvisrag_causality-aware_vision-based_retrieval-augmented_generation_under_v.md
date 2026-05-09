---
title: >-
  [Paper Note] RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations
description: >-
  [CVPR 2026][VisRAG] This paper proposes RobustVisRAG, a causality-guided dual-path framework that decouples semantic–degradation entanglement in VisRAG by capturing degradation signals via a non-causal path and learning clean semantics via a causal path. Under real-world degradation conditions, the framework achieves improvements of 7.35%, 6.35%, and 12.40% in retrieval, generation, and end-to-end performance, respectively, while preserving performance on clean data.
tags:
  - CVPR 2026
  - VisRAG
  - Robustness
  - Causal Inference
  - Visual Degradation
  - Dual-Path Encoding
date: 2026-05-08
content_hash: faa1ff23b9993e68
---

# RobustVisRAG: Causality-Aware Vision-Based Retrieval-Augmented Generation under Visual Degradations

**Conference**: CVPR 2026
**arXiv**: [2602.22013](https://arxiv.org/abs/2602.22013)
**Code**: [https://robustvisrag.github.io/](https://robustvisrag.github.io/)
**Area**: Information Retrieval
**Keywords**: VisRAG, Robustness, Causal Inference, Visual Degradation, Dual-Path Encoding

## TL;DR
This paper proposes RobustVisRAG, a causality-guided dual-path framework that decouples semantic–degradation entanglement in VisRAG by capturing degradation signals via a non-causal path and learning clean semantics via a causal path. Under real-world degradation conditions, the framework achieves improvements of 7.35%, 6.35%, and 12.40% in retrieval, generation, and end-to-end performance, respectively, while preserving performance on clean data.

## Background & Motivation
**Background**: VisRAG encodes document images directly with a VLM for retrieval and generation, avoiding OCR errors, and has become the dominant paradigm for document question answering.

**Limitations of Prior Work**:
   - Both TextRAG and VisRAG suffer significant performance degradation under corrupted inputs (blur, noise, low light, shadow, etc.).
   - Semantic and degradation factors are entangled in the visual encoder of VisRAG: degradation distorts the embedding space, causing retrieval mismatches and generation instability.
   - Dual failure modes exist: corrupted representations may lead to incorrect document retrieval, and even when retrieval is correct, degradation can mislead generation.

**Key Challenge**: In the representation space of existing VLM encoders, the semantic factor $S$ and degradation factor $D$ are entangled. Since the observed image $X$ is a collider node of $S$ and $D$, conditioning on $Z$ opens the spurious path $S \leftrightarrow D$.

**Goal**: Enable VisRAG to remain robust under degraded inputs without additional inference cost and without compromising performance on clean inputs.

**Key Insight**: A structural causal model (SCM) is used to analyze how degradation propagates through VisRAG, and causal intervention (do-operator) is applied to block the non-causal path.

**Core Idea**: Learn a disentangled representation $Z = [Z_{sem}, Z_{deg}]$ such that the semantic component is invariant to degradation, equivalent to the causal intervention $P(A|do(D=d_0))$.

## Method

### Overall Architecture
A dual-path architecture is introduced into the VLM visual encoder. The Non-Causal Path extracts degradation representations via unidirectional attention, while the Causal Path encodes clean semantics via bidirectional attention. Both paths are jointly optimized through two learning objectives: NCDM and CSA.

### Key Designs

1. **Non-Causal Path**:

    - A learnable non-causal token $z_{nc}^{(0)}$ is introduced.
    - Unidirectional attention constraint: the non-causal token can attend to all patch tokens, but patch tokens cannot attend to the non-causal token.
    - Degradation cues are aggregated as: $z_{nc}^{(l+1)} = z_{nc}^{(l)} + \sum_j \alpha_{nc \leftarrow j}^{(l)} v_j^{(l)}$
    - The final degradation representation is $Z_{deg} = z_{nc}^{(L)}$.
    - **Design Motivation**: Unidirectional attention prevents degradation information from flowing back into semantic tokens, achieving structured isolation.

2. **Causal Path**:

    - Bidirectional attention operates among patch tokens, excluding the non-causal token.
    - Semantic representation: $Z_{sem} = \text{Agg}(x_1^{(L)}, ..., x_T^{(L)})$
    - This path follows the causal route $S \to Z_{sem}$ and is designed to be unaffected by degradation.

3. **Non-Causal Distortion Modeling (NCDM)**:

    - A degradation contrastive learning objective: representations of the same degradation type are pulled closer, while those of different types are pushed apart.
    - $\mathcal{L}_{NCDM} = \max(0, \|Z_{deg}^a - Z_{deg}^p\|_2^2 - \|Z_{deg}^a - Z_{deg}^n\|_2^2 + \delta)$
    - Ensures that the non-causal token genuinely learns to encode degradation feature patterns.

4. **Causal Semantic Alignment (CSA)**:

    - Aligns the semantic representations of degraded and clean images to prevent degradation from leaking into the causal path.
    - Keeps $Z_{sem}$ stable under degradation conditions.

### Loss & Training
NCDM, CSA, and the original retrieval/generation losses are jointly optimized. Both paths produce $Z_{sem}$ and $Z_{deg}$ within the same forward pass, incurring no additional inference overhead.

## Key Experimental Results

### Main Results

| Method | Retrieval (Real-Degrade) | Generation (Real-Degrade) | End-to-End (Real-Degrade) |
|------|:---------:|:---------:|:---------:|
| VisRAG baseline | ~70% | ~55% | ~45% |
| VisRAG-FT (full finetune) | ~73% | ~57% | ~48% |
| Two-Stage Restoration | ~72% | ~56% | ~47% |
| **RobustVisRAG** | **~77%** | **~61%** | **~57%** |
| Gain | **+7.35%** | **+6.35%** | **+12.40%** |

### Distortion-VisRAG Dataset

**QA Pairs**: 367K
**Document Types**: 7 domains (papers, charts, tables, slides, handwritten notes, etc.)
**Synthetic Degradations**: 12 types (blur, noise, compression, etc.)
**Real Degradations**: 5 types (low light, shadow, paper damage, etc.)
**Multiple Severity Levels**: ✓

### Key Findings
- RobustVisRAG does not degrade performance on clean data, indicating that causal disentanglement does not impair normal comprehension.
- Perceptual quality improvements from image restoration (Two-Stage) do not necessarily translate into retrieval or generation gains.
- Full-parameter fine-tuning (FFT) improves degradation robustness but causes forgetting of pretrained knowledge and fails to separate semantics from degradation.
- The end-to-end gain (12.40%) substantially exceeds the individual retrieval and generation improvements, suggesting a compounding effect from improvements at both stages.

## Highlights & Insights
- **Elegant Application of Causal Modeling**: An SCM is used to analyze the degradation propagation path in VisRAG, theoretically deriving the necessity of representation disentanglement, which is then realized through a concrete network design (non-causal token + unidirectional attention). The connection between theory and implementation is tight.
- **Distortion-VisRAG Dataset**: The first benchmark specifically designed for VisRAG under degradation conditions, covering both synthetic and real degradations, filling a critical evaluation gap.
- **Zero Additional Inference Overhead**: The causal and non-causal paths are computed within the same forward pass; only $Z_{sem}$ is used at inference time, making the approach highly practical.

## Limitations & Future Work
- The degradation modeling in the non-causal path relies on simple contrastive learning, which may be insufficient for complex mixed degradations.
- Training requires degradation–clean paired data (or degradation type labels), which can be costly to obtain in practice.
- Generalization to unseen degradation types remains to be verified.
- The current work addresses document images only; robustness to degradation in natural scene images is not explored.

## Related Work & Insights
- **vs. TeCoA / FARE (adversarial robustness)**: These methods target small perturbations under $\ell_p$ norm constraints and are not suited for natural degradations (blur, low light, shadow). RobustVisRAG handles a broader range of degradation types by explicitly learning degradation representations.
- **vs. Image Restoration → VisRAG Pipeline**: Restored images exhibit improved perceptual quality but do not guarantee semantic consistency, whereas RobustVisRAG achieves semantic protection directly at the encoder level.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of causal modeling and dual-path encoder design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ A complete benchmark with both synthetic and real degradations is constructed.
- Writing Quality: ⭐⭐⭐⭐ The causal analysis section is formally rigorous.
- Value: ⭐⭐⭐⭐ VisRAG robustness is an important practical problem.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](../../AAAI2026/information_retrieval/knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[CVPR 2026\] NanoVDR: Distilling a 2B Vision-Language Retriever into a 70M Text-Only Encoder for Visual Document Retrieval](nanovdr_distilling_a_2b_vision-language_retriever_into_a_70m_text-only_encoder_f.md)
- [\[CVPR 2026\] CC-VQA: Conflict- and Correlation-Aware Method for Mitigating Knowledge Conflict in Knowledge-Based Visual Question Answering](cc-vqa_conflict-_and_correlation-aware_method_for_mitigating_knowledge_conflict_.md)
- [\[ICLR 2026\] RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding](../../ICLR2026/information_retrieval/ravenea_a_benchmark_for_multimodal_retrieval-augmented_visual_culture_understand.md)
- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)

<!-- RELATED:END -->
