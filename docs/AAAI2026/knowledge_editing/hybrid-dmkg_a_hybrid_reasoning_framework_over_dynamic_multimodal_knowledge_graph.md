---
title: >-
  [Paper Note] Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing
description: >-
  [AAAI2026][Knowledge Editing][Multimodal Knowledge Editing] This paper proposes the MMQAKE benchmark and the Hybrid-DMKG framework, which constructs a dual-channel hybrid reasoning mechanism — combining relation link pre…
tags:
  - "AAAI2026"
  - "Knowledge Editing"
  - "Multimodal Knowledge Editing"
  - "Multihop QA"
  - "Dynamic Knowledge Graph"
  - "Cross-modal Retrieval"
  - "RAG"
  - "Hybrid Reasoning"
date: 2026-05-08
content_hash: 347acb6dab74b868
---

# Hybrid-DMKG: A Hybrid Reasoning Framework over Dynamic Multimodal Knowledge Graphs for Multimodal Multihop QA with Knowledge Editing

**Conference**: AAAI2026
**arXiv**: [2512.00881](https://arxiv.org/abs/2512.00881)
**Authors**: Li Yuan, Qingfei Huang, Bingshan Zhu, Yi Cai, Qingbao Huang, Changmeng Zheng, Zikun Deng, Tao Wang (SCUT et al.)
**Code**: [YuanLi95/Hybrid-DMKG](https://github.com/YuanLi95/Hybrid-DMKG)
**Area**: Knowledge Editing
**Keywords**: Multimodal Knowledge Editing, Multihop QA, Dynamic Knowledge Graph, Cross-modal Retrieval, RAG, Hybrid Reasoning

## TL;DR

This paper proposes the MMQAKE benchmark and the Hybrid-DMKG framework, which constructs a dual-channel hybrid reasoning mechanism — combining relation link prediction with RAG-augmented LVLM inference — over a dynamic multimodal knowledge graph, supplemented by a background reflection decision module. The approach significantly outperforms existing methods on 2–5 hop multimodal knowledge editing QA (H-Acc of 29.90% on LLaVA, surpassing IKE by 13.52 percentage points).

## Background & Motivation

### Multimodal Challenges in Knowledge Editing

Knowledge encoded in large language models may become outdated or erroneous. Knowledge Editing aims to correct such knowledge without affecting unrelated content. With the development of LVLMs, Multimodal Knowledge Editing (MKE) extends editing to text-visual settings. However, existing MKE benchmarks (e.g., VLKEB) primarily evaluate final answer correctness, neglecting the quality of intermediate multihop reasoning and robustness to variations in visual inputs.

### Three Key Limitations of Existing Evaluation

(1) **Lack of intermediate reasoning step evaluation**: Models may arrive at correct final answers via incorrect reasoning paths; evaluating only final answers obscures reasoning errors. (2) **Lack of visual paraphrase robustness evaluation**: Different images of the same entity should yield consistent results, yet existing benchmarks do not test this. (3) **Neglect of answer alias diversity**: "Buenos Ayres" and "Buenos Aires" are semantically equivalent but are not recognized as correct answers interchangeably. These limitations lead to an overestimation of the true reasoning capabilities of MKE methods.

### Special Challenges of Multihop Reasoning for Knowledge Editing

When the first fact in a reasoning chain is edited (e.g., changing a person's name from Roy Bittan to Gustavo Santaolalla), the model must correctly propagate the updated information and apply the revised knowledge in subsequent reasoning steps. This requires the model not only to edit a single fact but also to maintain consistency across the entire reasoning chain — a major challenge for both parameter-update and parameter-retention approaches.

## Core Problem

How can a model, after knowledge editing, correctly apply updated knowledge at each reasoning step in multimodal multihop QA, while remaining robust to variations in visual inputs?

## Method

### Overall Architecture

Hybrid-DMKG is a parameter-free framework comprising four core components: (a) Dynamic Multimodal Knowledge Graph (DMKG) construction and maintenance; (b) LLM-based question decomposition; (c) cross-modal entity retrieval; and (d) DMKG-guided hybrid reasoning.

### DMKG Construction and Update

Each record in the multimodal knowledge graph $\mathcal{G}$ is represented as $(\mathcal{G}_i^e, \mathcal{G}_i^r, \mathcal{G}_i^o)$, with some entities associated with images $\mathcal{G}_i^v$. Upon receiving an editing quadruple $(x, v, o, \tilde{o})$, it is integrated into $\mathcal{G}$ to produce a dynamic graph $\tilde{\mathcal{G}}$, preserving both original and edited facts.

### Question Decomposition

An LLM (without fine-tuning) decomposes a multihop question $Q$ into a sequence of sub-questions:
$$\{q_1, q_2, \ldots, q_n\} = \text{LLM}(Q, P_{\text{Dec}})$$
Visual sub-questions use an [IMAGE] placeholder, and relevant entities are tagged with [ENT] to maintain consistency.

### Cross-modal Entity Retrieval

For visual sub-questions, a cross-modal retrieval model $\text{M}_u$ jointly encodes entity names and images:
$$z_m = \text{M}_u([\tilde{\mathcal{G}}_m^e, \tilde{\mathcal{G}}_m^v])$$
$$s = \text{M}_u([q_1, \tilde{v}])$$
The most relevant entity is retrieved via Top-1 cosine similarity as the answer: $a_1 = \arg\text{Top1}_{m} \frac{s^T z_m}{\|s\|_2 \|z_m\|_2}$

### DMKG-Guided Hybrid Reasoning

For reasoning-type sub-questions, two parallel channels are employed:

**Channel 1: Relation Link Prediction** — A fine-tuned relation extractor $\text{M}_e$ extracts relation keywords $k_2^q$ from the query and computes cosine similarity against candidate relations in the DMKG using Sense2Vec embeddings. If the similarity exceeds threshold $\alpha$, the corresponding entity is selected as the answer $a_2^{\text{link}}$.

**Channel 2: RAG-Augmented LVLM Inference** — Top-K relevant triples are retrieved from the DMKG as context and fed into the LVLM to generate an answer:
$$a_2^{\text{model}} = \text{LVLM}(q_2, \tilde{v}, \mathcal{K}_{\text{Ret}}(q_2, C_2), P_{\text{Ans}})$$

**Background Reflection Decision** — When the two channels produce conflicting answers, background knowledge for both candidate answers is extracted from the DMKG, and the LVLM synthesizes this evidence to select the most credible answer:
$$a_2 = \text{LVLM}(q_2, \tilde{v}, [a_2^{\text{link}}, C_2^{\text{link*}}], [a_2^{\text{model}}, C_2^{\text{modal*}}], P_{\text{Cho}})$$

## Key Experimental Results

### MMQAKE Benchmark Statistics

| Metric | Value |
|--------|-------|
| Number of knowledge edits | 1,278 |
| 2-hop questions | 1,278 |
| 3-hop questions | 1,238 |
| 4-hop questions | 1,193 |
| 5-hop questions | 1,110 |
| Total sub-questions | 11,773 |
| Avg. answer aliases | 9.49 |

### Main Results (Original Image)

| Method | BLIP-2 M-Acc | BLIP-2 H-Acc | LLaVA M-Acc | LLaVA H-Acc | MiniGPT-4 M-Acc | MiniGPT-4 H-Acc |
|--------|-------------|-------------|------------|------------|----------------|----------------|
| FT(QFor) | 3.73 | 0.20 | 4.63 | 0.44 | 4.69 | 0.44 |
| MEND | 0.04 | 0.00 | 0.70 | 0.00 | 0.07 | 0.00 |
| SERAC | 5.75 | 0.00 | 6.58 | 0.00 | 0.27 | 0.00 |
| IKE | 16.64 | 6.16 | 38.93 | 16.38 | 15.48 | 6.14 |
| **Hybrid-DMKG** | **47.55** | **28.88** | **53.75** | **29.90** | **35.86** | **24.73** |

### Ablation Study (LLaVA, Original Image)

| Variant | M-Acc | H-Acc |
|---------|-------|-------|
| Hybrid-DMKG (full) | **53.75** | **29.90** |
| w/o Linking | 47.68 | 23.15 |
| w/o Decision | 52.71 | 28.36 |

### H-Acc by Hop Count (LLaVA)

On 4-hop and 5-hop tasks, Hybrid-DMKG achieves H-Acc above approximately 5%, whereas other methods typically fall below 2%, representing nearly a twofold improvement.

## Highlights & Insights

- **First multimodal multihop knowledge editing benchmark MMQAKE**: Supports 2–5 hop reasoning chains, step-wise evaluation, visual paraphrase robustness testing, and answer alias matching, filling a critical gap in MKE evaluation.
- **Dual-channel hybrid reasoning design**: Relation link prediction handles queries with explicit relations in the DMKG, while RAG-augmented inference compensates for incomplete background knowledge; the two channels are complementary and improve overall robustness.
- **Background reflection decision module**: When the two channels disagree, neighborhood context from the DMKG enables the LVLM to perform reflective decision-making, effectively filtering erroneous candidates.
- **Parameter-free framework**: No modification of LVLM parameters is required; knowledge updates are achieved via an external knowledge graph, avoiding catastrophic forgetting.
- **Substantial gains over baselines**: H-Acc on LLaVA exceeds the strongest baseline IKE by 13.52 percentage points (29.90% vs. 16.38%).

## Limitations & Future Work

- **Heavy reliance on external components**: The system requires multiple external modules — LLM-based question decomposition, CLIP retrieval, relation extractor, Wiki Linker, etc. — resulting in high system complexity where failure in any single module may cause cascading errors.
- **Absolute H-Acc values remain low**: Even the best result (29.90%) indicates that multimodal multihop reasoning is far from solved, with H-Acc dropping to approximately 5% at 5 hops.
- **No support for open-ended QA**: MMQAKE covers only factoid QA and does not address open-ended or generative question answering.
- **Limited DMKG scale**: Current experiments involve approximately 58,000 entities and 686,000 triples; efficiency and accuracy on larger-scale KGs remain to be validated.

## Related Work & Insights

- **vs. IKE**: IKE leverages retrieval-augmented in-context learning to maintain a stable baseline but struggles with multihop knowledge propagation. Hybrid-DMKG enables explicit reasoning chains through structured KG traversal.
- **vs. MEND/SERAC**: Parameter-modification methods nearly completely fail on multihop reasoning (H-Acc ≈ 0%), demonstrating that single-hop editing capability does not generalize to multihop settings.
- **vs. MQUAKE**: MMQAKE extends pure-text multihop knowledge editing evaluation to the multimodal setting, adding dimensions of visual paraphrase robustness and step-wise assessment.

The use of knowledge graphs as external knowledge stores demonstrates unique advantages in knowledge editing scenarios, enabling precise localization and modification of specific triples. The dual-channel reasoning with reflection-based decision-making is a design pattern generalizable to other tasks requiring multi-source evidence fusion. The step-wise evaluation protocol of MMQAKE establishes a more rigorous standard for future multihop reasoning research.

## Rating

- Novelty: ⭐⭐⭐⭐ — First multimodal multihop knowledge editing benchmark combined with a DMKG-based hybrid reasoning framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple backbones, ablation studies, hop-wise analysis, and no-alias control experiments.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; Figures 1 and 2 provide intuitive illustrations.
- Value: ⭐⭐⭐⭐ — Addresses an important problem and contributes a usable benchmark, though absolute H-Acc values still leave substantial room for improvement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](../../NeurIPS2025/knowledge_editing/edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[CVPR 2026\] MoKus: Leveraging Cross-Modal Knowledge Transfer for Knowledge-Aware Concept Customization](../../CVPR2026/knowledge_editing/mokus_leveraging_crossmodal_knowledge_transfer_for.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](../../ACL2026/knowledge_editing/aligning_language_models_with_real-time_knowledge_editing.md)

</div>

<!-- RELATED:END -->
