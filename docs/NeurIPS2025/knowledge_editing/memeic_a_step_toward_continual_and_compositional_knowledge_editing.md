---
title: >-
  [Paper Note] MemEIC: A Step Toward Continual and Compositional Knowledge Editing
description: >-
  [NEURIPS2025][Knowledge Editing][LVLM] This paper proposes MemEIC, a three-tier framework for continual and compositional knowledge editing in large vision-language models (LVLMs)…
tags:
  - "NEURIPS2025"
  - "Knowledge Editing"
  - "LVLM"
  - "Continual Learning"
  - "Compositional Reasoning"
  - "LoRA"
  - "External Memory"
date: 2026-05-08
content_hash: 3fdcafa2a764fd96
---

# MemEIC: A Step Toward Continual and Compositional Knowledge Editing

**Conference**: NEURIPS2025
**arXiv**: [2510.25798](https://arxiv.org/abs/2510.25798)  
**Code**: [MemEIC/MemEIC](https://github.com/MemEIC/MemEIC)  
**Area**: Knowledge Editing
**Keywords**: Knowledge Editing, LVLM, Continual Learning, Compositional Reasoning, LoRA, External Memory

## TL;DR

This paper proposes MemEIC, a three-tier framework for continual and compositional knowledge editing in large vision-language models (LVLMs), combining an external dual-modal retrieval memory (Mem-E), an internal modality-decoupled LoRA adapter (Mem-I), and a brain-inspired Knowledge Connector. MemEIC substantially outperforms existing methods on the newly introduced CCKEB benchmark.

## Background & Motivation

Factual knowledge encoded in LVLMs can become outdated or erroneous, necessitating efficient knowledge editing mechanisms. Existing approaches suffer from three critical limitations:

1. **Unimodal isolated editing**: Most methods address only visual or textual editing in isolation, ignoring the inherently multimodal nature of LVLMs. For instance, they can correct "the identity of a person in an image" (visual editing) or update "a person's job title" (textual editing), but cannot handle both simultaneously.
2. **Lack of continual editing evaluation**: Existing benchmarks do not assess knowledge retention across sequential edits, overlooking catastrophic forgetting.
3. **Lack of compositional reasoning evaluation**: Real-world queries often require integrating both visual and textual edits—e.g., "What position does the person in the photo currently hold?" requires first identifying the person (visual edit) and then retrieving their new title (textual edit).

Existing methods fall into two categories, each with fundamental shortcomings:

- **External memory methods** (e.g., SERAC, IKE): These avoid modifying model parameters, making them suitable for long-term retention, but they directly adopt text-LLM retrieval strategies that ignore visual cues, leading to failures in visual edit retrieval. Furthermore, the model over-relies on internal stale knowledge and performs poorly when retrieved external information conflicts with internal knowledge.
- **Internal memory methods** (e.g., LoRA, MEND): These embed edits into model parameters via fine-tuning and effectively internalize new knowledge, but visual and textual edits share the same parameter space, causing cross-modal interference and representation collapse. Sequential editing also induces catastrophic forgetting.

## Core Problem

How can LVLMs stably retain historical edits and avoid forgetting while also supporting cross-modal compositional reasoning, as they continuously receive interleaved visual and textual knowledge edits?

## Method

### Overall Architecture

MemEIC adopts a three-tier collaborative architecture: query decomposition → external memory retrieval (Mem-E) + internal modality-decoupled adapters (Mem-I) → knowledge fusion via Knowledge Connector.

### 1. Query Decomposition

GPT-4o is used to automatically decompose an input query $Q$ into a visual sub-query $Q_v$ and a textual sub-query $Q_t$, enabling independent per-modality processing. For example, "What position does the person in the photo currently hold?" is decomposed into the visual component "Who is the person in the photo?" and the textual component "What position does [person] currently hold?"

### 2. Modality-Aware External Memory (Mem-E)

Two independent external memory stores are maintained:

- **Textual memory** $M_t = \{(q_i, a_i)\}$: stores textual QA editing pairs.
- **Visual memory** $M_v = \{(I_j, q_j, a_j)\}$: stores visual editing triplets containing images.

The key innovation in retrieval is **multimodal fusion retrieval**: visual queries combine text similarity with image similarity computed by a CLIP encoder, with a weighting coefficient $\alpha = 0.5$. Textual retrieval uses [CLS] representations from DistilBERT.

For compositional queries, the system first retrieves the target entity via visual retrieval, then substitutes the entity name into the placeholder of the textual sub-query, enabling cascaded retrieval.

### 3. Internal Modality-Separated Knowledge Integration (Mem-I)

Inspired by **brain lateralization** (the left hemisphere processes language; the right processes vision), two dedicated LoRA adapters are designed:

- **Visual adapter** $\theta_v$ (analogous to the right hemisphere): handles only visual knowledge updates.
- **Textual adapter** $\theta_t$ (analogous to the left hemisphere): handles only textual knowledge updates.
- The original pre-trained FFN weights $\theta$ remain frozen, preserving pre-edit knowledge.

The appropriate adapter is selectively activated based on query type, and only the corresponding modality's adapter is updated during editing, fundamentally preventing cross-modal interference and representation collapse.

### 4. Knowledge Connector (Corpus Callosum Mechanism)

This component is inspired by the **corpus callosum**, the brain structure connecting the two cerebral hemispheres. When a compositional query activates both adapters simultaneously, the Knowledge Connector facilitates information exchange between token representations of different modalities by adding LoRA adaptations to the $Q$ and $K$ projections in the self-attention module:

$$Q^\ell = h^\ell(W_Q + \mathbb{I}_{v,t} \cdot \Delta W_Q^L)$$

where $\mathbb{I}_{v,t}$ is an indicator function that equals 1 only when both adapters are activated. For unimodal queries, the Connector degenerates to an identity operation, leaving independent modality representations unaffected.

### Loss & Training

- **Stage 1**: The LVLM is frozen; the retrieval and alignment capabilities of the external memory module are trained.
- **Stage 2**: Both adapters are activated, and the Knowledge Connector is trained using an **adversarial retriever** that mixes correct and incorrect evidence, encouraging the model to selectively integrate internal and external evidence and preventing over-reliance on external memory.

## Key Experimental Results

### Benchmark & Setup

- The CCKEB benchmark is introduced by extending VLKEB; each sample pairs a visual edit with a corresponding textual edit.
- Backbone models: LLaVA-1.5 (7B) and MiniGPT-4.
- 500 sequential edits are performed, with evaluation at gaps of 0, 10, 20, 50, and 100.

### Main Results (LLaVA-1.5, averaged across gaps)

| Method | Visual Rel | Text Rel | Comp Rel |
|--------|-----------|----------|----------|
| SERAC | Low (insufficient text retrieval) | Stable | Low (no fusion) |
| LoRA | Perfect at gap=0, sharp drop ~30pt as gap↑ | Same | 62.05 |
| WISE | Relatively stable but poor on visual edits | Relatively stable | Low |
| **MemEIC** | **98.93** | **92.48** | **80.56** |

MemEIC outperforms WISE by +16.94 on Visual Reliability and +32.35 on Compositional Reliability; it surpasses the best baseline (LoRA) by +18.51 on Comp Rel.

### Ablation Study

1. **Incorporating visual cues into external memory is critical**: Mem-E (tex+vis) vs. Mem-E (tex) improves Reliability from 48.02 → 96.51 and Image Locality from 4.02 → 57.10.
2. **Dual LoRA outperforms single LoRA**: Under matched total parameter budgets (r=8×2 vs. r=16), Dual-LoRA improves T-Loc by +17.77% and I-Loc by +2.86% (significant at p<0.05).
3. **The Knowledge Connector is critical for compositional reasoning**:
    - Base+RAG (perfect retrieval): Comp Rel only 64.93%—perfect retrieval alone is insufficient.
    - Dual-LoRA+RAG: 78.16% at gap=0, degrades to 63.39% at gap=100—dual adapters without interaction are insufficient.
    - **Dual-LoRA+RAG+Connector**: **99.21%** at gap=0, still **97.01%** at gap=100—approaching oracle-level performance.

## Highlights & Insights

1. **Novel problem formulation**: The paper is the first to formally define the Continual Compositional Knowledge Editing (CCKE) problem, introducing the CompRel metric and CCKEB benchmark, filling a gap in multimodal knowledge editing evaluation.
2. **Compelling brain-inspired design**: The lateralized dual-LoRA maps naturally to left/right brain hemispheres, and the Knowledge Connector to the corpus callosum—the biological analogy is both intuitive and effective.
3. **Clear separation of responsibilities**: The external memory handles precise retrieval, the internal adapters handle modality-isolated editing, and the Connector handles on-demand fusion, yielding a well-structured architecture.
4. **Clever adversarial training strategy**: Stage 2 trains the Connector with mixed correct/incorrect evidence, effectively mitigating over-reliance on external memory.
5. **Thorough ablation study**: Components are incrementally stacked, clearly demonstrating the contribution of each module.

## Limitations & Future Work

1. **Query decomposition depends on GPT-4o**: This increases inference cost and latency, potentially becoming a bottleneck in deployment; lightweight decomposers could be explored.
2. **Experiments limited to paired settings**: Visual edits are immediately followed by corresponding textual edits; real-world editing orders may be more random and complex.
3. **Limited scale of CCKEB**: Being an extension of VLKEB, the diversity of entity types and relation categories may be insufficient.
4. **Only two backbone models evaluated**: LLaVA-1.5 and MiniGPT-4 are both relatively dated; effectiveness on newer, stronger LVLMs remains to be verified.
5. **Adaptability of the frozen Knowledge Connector**: Once frozen during deployment, fusion performance may degrade if the editing distribution shifts significantly from the training distribution.
6. **Scalability of external memory retrieval**: As the number of edits grows, the efficiency and accuracy of cosine-similarity-based retrieval may decline.

## Related Work & Insights

| Dimension | SERAC | WISE | LoRA/FT | MemEIC |
|-----------|-------|------|---------|--------|
| Editing Paradigm | External memory | Internal (side-FFN) | Internal (fine-tuning) | External + Internal hybrid |
| Multimodal Retrieval | Text only | N/A | N/A | Image-text fusion |
| Forgetting Resistance | Strong (no param change) | Moderate (routing) | Weak (shared space) | Strong (modality separation) |
| Cross-modal Interference | None | Present | Severe | None (dual-LoRA isolation) |
| Compositional Reasoning | Poor | Poor | Moderate | Strong (Knowledge Connector) |
| Continual Editing Stability | Stable | Relatively stable | Sharp degradation | Stable |

**Broader Insights:**

1. The **modality separation + on-demand fusion** paradigm is broadly applicable—beyond knowledge editing, it offers a useful reference for cross-modal interference problems in multimodal continual learning and multi-task learning.
2. **Brain-science-inspired network design**: The lateralization and corpus callosum analogies offer a fresh perspective for multimodal architecture design, generalizable to broader multimodal adapter frameworks.
3. **Adversarial training to reduce retrieval dependency**: Training the model with noisy retrieval results to improve robustness against retrieval errors is a strategy applicable to any RAG system.
4. Strong connections to the **continual learning** literature: dual-LoRA edit isolation is essentially a task/modality-specific parameter expansion strategy, in the same spirit as progressive networks.

## Rating

- Novelty: ⭐⭐⭐⭐ — Clear innovations in both the CCKE problem definition and the brain-inspired three-tier architecture.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Detailed ablations, but backbone models are dated and benchmark scale is limited.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured; the neuroscience analogies enhance readability.
- Value: ⭐⭐⭐⭐ — Establishes a new benchmark and strong baseline for multimodal knowledge editing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)
- [\[ACL 2026\] Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse](../../ACL2026/knowledge_editing/spectral_characterization_and_mitigation_of_sequential_knowledge_editing_collaps.md)

</div>

<!-- RELATED:END -->
