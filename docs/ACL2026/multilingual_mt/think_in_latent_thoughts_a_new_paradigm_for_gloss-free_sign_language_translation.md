---
title: >-
  [Paper Note] Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation
description: >-
  [ACL 2026][Multilingual & Machine Translation][Sign Language Translation] Ours proposes SignThought, a reasoning-driven gloss-free sign language translation framework. It introduces learnable latent thought slots as an e…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Sign Language Translation"
  - "Gloss-free Translation"
  - "Latent Chain-of-Thought"
  - "Cross-modal Reasoning"
  - "Dual-stream Decoder"
date: 2026-05-08
content_hash: d4105fcfb58a94eb
---

# Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation

**Conference**: ACL 2026  
**arXiv**: [2604.15301](https://arxiv.org/abs/2604.15301)  
**Code**: [GitHub](https://github.com/fletcherjiang/SignThought)  
**Area**: LLM Evaluation  
**Keywords**: Sign Language Translation, Gloss-free Translation, Latent Chain-of-Thought, Cross-modal Reasoning, Dual-stream Decoder

## TL;DR

Ours proposes SignThought, a reasoning-driven gloss-free sign language translation framework. It introduces learnable latent thought slots as an explicit intermediate semantic layer between video and text, achieving the decoupling of semantic planning and visual evidence retrieval through a "plan-then-locate" dual-stream decoder. It outperforms existing gloss-free methods across multiple benchmarks.

## Background & Motivation

**Background**: Sign language translation (SLT) has progressively evolved from gloss-based cascade methods to gloss-free end-to-end video-to-text approaches.

**Limitations of Prior Work**: Existing models implicitly assume that sign language video segments can be directly mapped to spoken language vocabulary. However, a significant portion of meaning in sign language is dynamically generated through productive forms (e.g., classifiers, spatial grammar, and movement modulation), for which no fixed lexical correspondence exists.

**Key Challenge**: SLT is essentially a cross-modal reasoning problem rather than a simple alignment task. Meanings are dispersed across continuous video streams, requiring cross-temporal reasoning for correct comprehension.

**Goal**: To introduce an explicit intermediate semantic representation (latent chain-of-thought) to establish a traceable reasoning bridge between video encoding and text decoding.

**Key Insight**: Drawing an analogy to CoT, but implemented in a continuous latent space rather than a discrete text space—using learnable thought slots to distill semantics from video.

**Core Idea**: $K$ ordered latent thought slots iteratively extract semantics via causal self-attention and Sinkhorn-routed cross-attention to form a directed chain of thought. A dual-stream decoder first queries the chain of thought for semantic planning, then returns to the video to retrieve evidence.

## Method

### Overall Architecture

Consists of three parts: (1) Sign encoder $\rightarrow$ frame-level evidence $\mathbf{E}$; (2) Latent CoT module $\rightarrow$ ordered chain of thought $\mathbf{C}$ (segmentation $\rightarrow$ Sinkhorn binding $\rightarrow$ routing retrieval); (3) Dual-stream decoder (thought attention $\rightarrow$ video attention).

### Key Designs

1.  **Latent Chain-of-Thought Module**:
    - **Function**: Distills ordered semantic reasoning states from dense video features.
    - **Mechanism**: $K$ learnable thought slots are iteratively refined through $L$ layers. Each layer first applies causal self-attention (thought $k$ only attends to $1..k$), then assigns video segments to each thought via Sinkhorn normalization, and finally extracts fine-grained evidence through routing cross-attention.
    - **Design Motivation**: Causal constraints provide a directed structure, while Sinkhorn prevents attention degradation.

2.  **"Plan-then-Locate" Dual-stream Decoder**:
    - **Function**: Separates semantic decision-making from evidence retrieval.
    - **Mechanism**: In each layer: Self-attention $\rightarrow$ Thought cross-attention (planning) $\rightarrow$ Video cross-attention (localization). Thought attention weights are converted into frame-level priors via the routing matrix to guide video attention.
    - **Design Motivation**: Retrieving directly from all frames leads to attention diffusion; determining semantic intent before retrieval is more controllable.

3.  **Large-scale Gloss-free Dataset**:
    - **Function**: Provides training/evaluation data with stronger contextual dependencies.
    - **Mechanism**: Constructs and open-sources a new dataset containing more productive forms.
    - **Design Motivation**: Existing datasets exhibit weak context dependency.

### Loss & Training

Standard sequence-level cross-entropy + thought continuity loss. End-to-end training requiring only sentence-level annotations.

## Key Experimental Results

### Main Results

| Method | PHOENIX14T B@4 | ROUGE |
| :--- | :--- | :--- |
| SLTUNET | 28.47 | 52.11 |
| SignThought | **31.2** | **54.8** |

### Ablation Study

| Config | B@4 $\Delta$ | Description |
| :--- | :--- | :--- |
| w/o Chain of Thought | -2.5 | Direct decoding degradation |
| w/o Causal Constraint | -1.3 | Disordered thoughts |
| w/o Sinkhorn | -1.8 | Attention degradation |
| w/o Dual-stream | -2.1 | Planning and localization coupled |

### Key Findings

- Latent CoT provides a 2-3 BLEU improvement over direct decoding.
- The chain of thought serves as traceable anchors to align text with temporal video regions.

## Highlights & Insights

- The perspective that "sign language translation is reasoning rather than alignment" shifts the modeling paradigm of the field.
- Latent CoT, as a cross-modal interface, is transferable to other continuous-to-discrete cross-modal tasks.
- The Sinkhorn technique used to prevent attention degradation is applicable in other slot-attention scenarios.

## Limitations & Future Work

- The number of thought slots $K$ must be preset; the optimal $K$ may vary with different video lengths.
- Validation is limited to restricted datasets.
- The encoder utilizes Inception features; a more powerful encoder might yield further gains.

## Related Work & Insights

- **vs Gloss-based**: These require expensive annotations, whereas Ours is entirely gloss-free.
- **vs Direct Video-to-Text**: These lack an intermediate reasoning layer, resulting in limited performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to introduce latent CoT into sign language translation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple benchmarks and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Strong motivation but symbol-intensive.
- Value: ⭐⭐⭐⭐⭐ Significant contributions to both sign language translation and cross-modal reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Selective Contrastive Learning For Gloss Free Sign Language Translation](selective_contrastive_learning_for_gloss_free_sign_language_translation.md)
- [\[ACL 2026\] Language on Demand, Knowledge at Core: Composing LLMs with Encoder-Decoder Translation Models for Extensible Multilinguality](language_on_demand_knowledge_at_core_composing_llms_with_encoder-decoder_transla.md)
- [\[ACL 2026\] Language Models Entangle Language and Culture](language_models_entangle_language_and_culture.md)
- [\[ACL 2026\] Hierarchical Policy Optimization for Simultaneous Translation of Unbounded Speech](hierarchical_policy_optimization_for_simultaneous_translation_of_unbounded_speec.md)
- [\[ACL 2026\] Efficient Training for Cross-lingual Speech Language Models](efficient_training_for_cross-lingual_speech_language_models.md)

</div>

<!-- RELATED:END -->
