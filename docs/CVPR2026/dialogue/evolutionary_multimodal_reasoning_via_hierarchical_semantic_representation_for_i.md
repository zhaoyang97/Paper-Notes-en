---
title: >-
  [Paper Note] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition
description: >-
  [CVPR 2026][Dialogue Systems][Multimodal Intent Recognition] This paper proposes HIER, which combines hierarchical semantic representation (a three-level hierarchy of tokens → concepts → relations) with a self-evolutionary reasoning mechanism driven by MLLM feedback, consistently outperforming SOTA methods and leading MLLMs by 1–3% on three multimodal intent recognition benchmarks.
tags:
  - CVPR 2026
  - Dialogue Systems
  - Multimodal Intent Recognition
  - Hierarchical Semantic Representation
  - Self-Evolutionary Reasoning
  - Concept Clustering
  - CoT
date: 2026-05-08
content_hash: e0fd2c5e2b7827b1
---

# Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition

**Conference**: CVPR 2026
**arXiv**: [2603.03827](https://arxiv.org/abs/2603.03827)
**Code**: [GitHub](https://github.com/thuiar/HIER)
**Area**: Dialogue Systems
**Keywords**: Multimodal Intent Recognition, Hierarchical Semantic Representation, Self-Evolutionary Reasoning, Concept Clustering, CoT

## TL;DR

This paper proposes HIER, which combines hierarchical semantic representation (a three-level hierarchy of tokens → concepts → relations) with a self-evolutionary reasoning mechanism driven by MLLM feedback, consistently outperforming SOTA methods and leading MLLMs by 1–3% on three multimodal intent recognition benchmarks.

## Background & Motivation

**Importance of Multimodal Intent Recognition**: Inferring human intent from multimodal signals (text + video + audio) is a core task in human-computer interaction, dialogue systems, and intelligent transportation.

**Neglect of Hierarchical Semantics in Prior Work**: Most existing methods focus on fine-grained multimodal cue fusion while overlooking the inherently hierarchical nature of semantic information, which limits coherent and reliable reasoning.

**Limitations of Static Reasoning Pipelines**: Existing methods rely on fixed reasoning workflows and lack self-evolutionary refinement capabilities, making it difficult to dynamically adapt in complex scenarios.

**Underutilization of MLLM Reasoning Potential**: Although MLLMs possess strong reasoning capabilities, they still struggle with complex multimodal semantics in the absence of fine-grained hierarchical reasoning paths.

**Inspiration from Human Cognition**: Humans first establish situational awareness, then identify salient semantic cues, and finally perform integrative judgment through relational reasoning and iterative self-refinement.

**Preliminary Attempt by LGSRR**: Leveraging LLM reasoning to assist intent understanding has shown promise, but the reasoning process remains shallow and dependent on specific semantic concepts.

## Method

### Overall Architecture

HIER consists of three steps: (1) Multimodal Concept Clustering — clustering tokens into mid-level semantic concepts; (2) Multimodal Relation Selection — employing an IB network with JS divergence to select highly informative inter-concept relations; (3) Evolutionary Multimodal Reasoning — performing hierarchical reasoning via structured CoT combined with a self-evolutionary mechanism.

### Key Designs

#### Multimodal Concept Clustering

A Qwen2-VL encoder is used to extract text tokens $T$ and visual tokens $V$, which are concatenated into a unified sequence $Z$. Spherical K-Means++ (with cosine similarity) is applied for soft clustering. A **label-guided strategy** is introduced, using intent label embeddings as semantic anchors: a convex combination is computed with current centroids weighted by cosine similarity:

$$\tilde{c}_m^{(u)} = \alpha \cdot c_m^{(u)} + (1-\alpha) \cdot \sum_{i=1}^L \text{Weight}_{i,m}^{(u)} y_i$$

#### Multimodal Relation Selection

For all concept pairs $(c_i, c_j)$, relations are encoded via an information bottleneck network as $r_{ij} = \text{MLP}(\text{ReLU}([c_i; c_j]))$. JS divergence is used to quantify the semantic novelty provided by each relation — high-divergence relations capture complementary or emergent semantics beyond individual concepts. The top-$k$ high-divergence relations are retained.

#### Evolutionary Multimodal Reasoning

A structured CoT with three stages is employed: CoT-1 (contextual understanding, operating on token-level input) → CoT-2 (concept analysis, operating on mid-level concepts) → CoT-3 (relational reasoning, operating on high-level relations). In the latter two stages, the model is explicitly prompted to assess the utility of each concept and relation.

#### Self-Evolutionary Mechanism

Concept and relation features are projected into vocabulary logits via a shared generation head. Normalized confidence scores for "Yes/No" tokens are extracted from a reflection prompt and used to dynamically modulate features: $\text{Feature}' = \text{Score} \cdot \text{Feature}$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \mathcal{L}_{\text{relation}}$$

$\mathcal{L}_{\text{task}}$ is the autoregressive language modeling loss, and $\mathcal{L}_{\text{relation}}$ is the cross-entropy loss for intent classification over concepts and relations.

## Key Experimental Results

### Main Results: Comparison on Three Benchmarks

| Method | MIntRec ACC | MIntRec F1 | MIntRec2.0 ACC | MELD-DA ACC |
|--------|------------|-----------|---------------|------------|
| MAG-BERT | 72.40 | 68.29 | 60.38 | 61.08 |
| MulT | 72.31 | 68.97 | 60.66 | 59.99 |
| TCL-MAP | 73.17 | 68.92 | 58.24 | 61.63 |
| SDIF-DA | 71.64 | 68.19 | - | - |
| **HIER (Ours)** | **74.5+** | **71.0+** | **62.5+** | **63.0+** |

### Ablation Study

| Component | Contribution |
|-----------|-------------|
| Concept Clustering | Provides mid-level semantic abstraction |
| Label Guidance | Aligns clustering with intent semantics |
| Relation Selection | Captures higher-order interaction patterns |
| JS Divergence Filtering | Removes redundant relations |
| Self-Evolutionary Mechanism | Dynamically refines features |
| Structured CoT | Deepens hierarchical reasoning |

### Key Findings

- HIER consistently outperforms SOTA methods across all three benchmarks and surpasses direct use of MLLMs (e.g., Qwen2-VL).
- The self-evolutionary mechanism effectively filters out uninformative concepts and relations, improving reasoning robustness.
- The method generalizes to different backbone architectures beyond Qwen2-VL.
- Hierarchical representation yields the greatest benefit for complex, multi-class intent recognition.

## Highlights & Insights

- **Convincing three-level hierarchical design**: The progressive abstraction from tokens → concepts → relations naturally mirrors human cognitive processes.
- The label-guided clustering strategy elegantly bridges unsupervised clustering and task objectives.
- The use of JS divergence for relation selection is theoretically well-motivated — high divergence indicates that a relation introduces new information.
- The self-evolutionary mechanism leverages the MLLM's generation head for feature evaluation without requiring additional annotations.
- This is the first work to establish a multi-level progressive reasoning paradigm for multimodal intent recognition.

## Limitations & Future Work

- The number of concepts $k$ and the relation retention ratio require careful tuning.
- Clustering is performed independently per sample, lacking global semantic consistency across samples.
- The binary Yes/No evaluation in the self-evolutionary mechanism is coarse and may miss subtle distinctions.
- The overall computational cost is substantial, involving concept clustering, relation modeling, and MLLM inference.

## Related Work & Insights

- HIER shares the motivation of LGSRR in leveraging LLM reasoning to enhance intent understanding, but employs deeper and more structured reasoning.
- InMu-Net addresses noisy non-verbal cues; HIER implicitly handles a similar issue through concept clustering.
- The self-evolutionary mechanism intersects with self-alignment approaches such as RLAIF-V and SENA, but operates at the feature level rather than the sample level.
- The framework combining hierarchical representation and self-evolution is generalizable to tasks such as sentiment analysis and dialogue understanding.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TalkSketch: Multimodal Generative AI for Real-time Sketch Ideation with Speech](../../AAAI2026/dialogue/talksketch_multimodal_generative_ai_for_real-time_sketch_ideation_with_speech.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](../../ACL2026/dialogue/author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](../../ICLR2026/dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/dialogue/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ICLR 2026\] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation](../../ICLR2026/dialogue/think-while-generating_on-the-fly_reasoning_for_personalized_long-form_generatio.md)

</div>

<!-- RELATED:END -->
