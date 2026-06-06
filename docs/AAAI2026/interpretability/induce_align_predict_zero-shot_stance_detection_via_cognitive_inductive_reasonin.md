---
title: >-
  [Paper Note] Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning
description: >-
  [AAAI 2026][Interpretability][zero-shot stance detection] This paper proposes the CIRF framework, which abstracts transferable reasoning patterns from LLM-generated first-order logic via unsupervised schema induction (US…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "zero-shot stance detection"
  - "cognitive schema"
  - "first-order logic"
  - "graph kernel"
  - "low-resource"
date: 2026-05-08
content_hash: 02c31ff5717c3c59
---

# Induce, Align, Predict: Zero-Shot Stance Detection via Cognitive Inductive Reasoning

**Conference**: AAAI 2026
**arXiv**: [2506.13470](https://arxiv.org/abs/2506.13470)  
**Code**: None  
**Area**: Interpretability
**Keywords**: zero-shot stance detection, cognitive schema, first-order logic, graph kernel, low-resource

## TL;DR

This paper proposes the CIRF framework, which abstracts transferable reasoning patterns from LLM-generated first-order logic via unsupervised schema induction (USI), and performs explainable zero-shot stance reasoning through structural alignment using a schema-enhanced graph kernel model (SEGKM). The method achieves state-of-the-art performance on three benchmarks while requiring only 30% of labeled data.

## Background & Motivation

**Background**: Zero-shot stance detection (ZSSD) requires inferring the stance of text toward targets unseen during training, which is critical for analyzing rapidly emerging polarized social media topics.

**Limitations of Prior Work**:
- LLM zero-shot prompting underperforms on complex reasoning with limited generalization (GPT-3.5 achieves only 69.8 F1 on SEM16)
- LLM-augmented fine-tuning methods (KAI, FOLAR, etc.) still require substantial labeled data and remain at instance-level pattern matching
- Both paradigms lack explainability and cross-target reasoning generalization

**Key Challenge**: Stance detection requires abstract reasoning beyond surface-level lexical matching (e.g., "increasing health risks" and "undermining economic stability" both instantiate the reasoning pattern "negative consequence → opposition"), yet existing methods either perform surface-level matching or rely heavily on annotations.

**Key Insight**: Schema theory in cognitive science — humans induce generalizable reasoning patterns (schemas) from concrete experiences and apply them to new contexts. This cognitive capability is formalized as unsupervised induction of first-order logic patterns with graph kernel alignment.

## Method

### Overall Architecture

CIRF consists of two core modules: (1) USI (Unsupervised Schema Induction): LLM generates FOL reasoning → interpretation abstraction → clustering into schema graphs; (2) SEGKM (Schema-Enhanced Graph Kernel Model): constructs FOL graphs from input → subgraph kernel matching against schema templates → hierarchical graph representation → stance prediction.

### Key Designs

**1. Unsupervised Schema Induction (USI)**

- **Function**: Unsupervisedly induces abstract, cross-target transferable reasoning patterns from raw text
- **Design Motivation**: Instance-level FOL rules cannot generalize across domains; higher-level reasoning abstraction is needed
- **Mechanism** (four-stage pipeline):
    - **FOL Reasoning Generation**: For each sentence-target pair, the LLM is prompted to generate a first-order logic reasoning chain
    - **FOL Interpretation and Abstraction**: The LLM is prompted to analyze the internal logic of FOL, generate logically equivalent but structurally diverse variants, and then summarize them into generalized templates. Example: `∀x, (is_robot(x) → (helps_humans(x) → must_be_safe(x)))` is abstracted to `∀x, ((is_target(x) ∧ meets_condition(x)) → entails_consequence(x))`
    - **Schema Clustering and Hierarchical Abstraction**: FOL templates are clustered by semantic and reasoning pattern similarity; large clusters are processed via a hierarchical strategy (sub-cluster splitting → intermediate chaining → merging into schema templates)
    - **Schema Graph Construction**: Induced schemas serve as nodes in a multi-relational graph, with edges representing logical relations such as causality, contrast, and entailment

**2. Schema-Enhanced Graph Kernel Model (SEGKM)**

- **Function**: Leverages schema knowledge to enhance the representation of input reasoning structures for explainable zero-shot inference
- **Design Motivation**: Standard GNNs rely on local message passing and struggle to capture reusable high-order reasoning motifs; graph kernels enable better generalization through explicit structural matching
- **Mechanism**:
    - **FOL Graph Construction**: For each input pair $(x, q)$, a FOL reasoning chain is generated and parsed into a FOL graph $G_f=(V_f, E_f)$, where nodes are predicates and edges are logical relations
    - **Schema Subgraph Filters**: $k$-hop subgraphs centered at each node are extracted from schema graph $G^{(j)}$ to form a filter pool $\mathcal{H} = \bigcup_j H^{(j)}$
    - **Relation-Aware Node Embedding**: Nodes and edges are initialized with BERT embeddings; edge semantics are fused via relational projection: $x' = \text{ReLU}(x + \text{Proj}(e))$
    - **Deep Graph Kernel Response**: The $p$-step random walk kernel between an input subgraph and a schema filter is computed as: $\phi_{1,i}(v) = K_p(G_v^f, H_i^{(j)}) = \mathbf{s}^\top W A^{\times p} \mathbf{s}$, where $\mathbf{s} = \text{vec}(X_{G_v^f}' \cdot (X_{H_i^{(j)}}')^\top)$
    - **Schema Graph-Level Selection**: Kernel responses across all nodes are aggregated to select the top-$g$ schema graphs: $S^{(j)} = \sum_{v \in V_f} \frac{1}{|H^{(j)}|} \sum_{H_i^{(j)} \in H^{(j)}} \phi_{1,i}(v)$
    - **Hierarchical Graph Representation**: Multi-layer stacking of kernel feature extraction yields the final graph representation: $\Phi(G_f) = \text{Concat}(\sum_{v \in G_f} \phi_l(v) \mid l=0,1,...,L)$

**3. Stance Prediction**

The final graph representation is fed into a fully connected ReLU layer for three-class classification (Favor/Against/None), trained end-to-end with cross-entropy loss.

### Loss & Training

- Loss function: Cross-entropy loss
- Optimizer: AdamW, batch size 32, learning rate $5 \times 10^{-4}$
- Early stopping (patience = 10), up to 20 epochs, validated every 0.2 epochs
- Schema induction is fully unsupervised; SEGKM is trained on source targets
- Hardware: Single 40GB A100 GPU
- Default LLM: GPT-3.5; GPT-4o and DeepSeek-v3 are also evaluated

## Key Experimental Results

### Main Results

Zero-shot stance detection results (Macro-F1) on SEM16, VAST, and COVID-19:

| Method | Type | SEM16 HC | SEM16 FM | SEM16 LA | VAST All | COVID AF | COVID WA |
|--------|------|----------|----------|----------|----------|----------|----------|
| JointCL | BERT fine-tune | 54.4 | 54.0 | 50.0 | 71.2 | 57.6 | 63.1 |
| GPT-3.5 | LLM prompting | 78.9 | 68.3 | 62.3 | 65.1 | 69.2 | 57.8 |
| COLA | LLM prompting | 81.7 | 63.4 | 71.0 | 73.0 | 65.7 | 73.9 |
| KAI | LLM-augmented | 76.4 | 73.7 | 69.4 | 76.3 | - | - |
| LCDA | LLM-augmented | 79.8 | 70.0 | 69.4 | 80.3 | - | - |
| FOLAR | LLM-augmented | 81.9 | 71.2 | 69.9 | 77.2 | 69.5 | 73.1 |
| LogiMDF | LLM-augmented | 75.1 | 67.9 | 68.0 | 76.7 | 70.4 | 75.4 |
| **CIRF** | **Schema** | **80.1** | **74.7** | **73.9** | **80.9** | **74.1** | **81.0** |
| CIRF (GPT-4o) | Schema | 83.2 | 80.4 | 78.2 | 82.8 | 84.9 | 89.4 |

Average Macro-F1: 76.2 on SEM16 (+1.9 over FOLAR), 80.9 on VAST (+0.6 over LCDA), +3.7 over LogiMDF on COVID-19.

### Ablation Study

| Variant | Effect |
|---------|--------|
| w/o Schema | **Largest performance drop**; removing cognitive schemas severely degrades cross-target generalization |
| w/o SEGKM | Large performance drop; graph kernel alignment is critical for leveraging schema knowledge |
| w/o SE (edge semantics) | Moderate drop; relational information benefits structural matching |
| w/o USI (replace with simple clustering) | Moderate drop; LLM-driven semantic induction outperforms simple clustering |

Performance gaps across all ablated components are more pronounced under the VAST (10%) low-resource setting, indicating that each component becomes increasingly critical when labeled data is scarce.

### Key Findings
- **State-of-the-art across all three benchmarks**, with statistical significance ($p < 0.05$)
- **30% data matches full-data baselines**: with 10% of COVID-19 data, CIRF surpasses LogiMDF by 2.8 points; with 20% of SEM16 data, it surpasses FOLAR by 0.6 points
- **LLM scalability**: Upgrading from GPT-3.5 to GPT-4o improves VAST from 80.9 to 82.8 and COVID WA from 81.0 to 89.4
- FOL-based knowledge outperforms natural language knowledge: CIRF and FOLAR (both using FOL) generally outperform KAI (which uses natural language)
- Schema count has minimal impact on performance (variation < 1 point), suggesting that reasoning can be sufficiently abstracted by a small number of schemas
- Performance remains stable as the top-$g$ selection size varies from 2 to 16, indicating low sensitivity to this hyperparameter

## Highlights & Insights
- **Successful transfer from cognitive science to NLP**: Schema theory is formalized as FOL induction + graph kernel alignment, yielding both theoretical depth and practical effectiveness
- **The four-stage USI pipeline is elegantly designed**: generation → interpretation abstraction → clustering → graph construction, progressively moving from instances to abstractions
- **Graph kernels over GNNs** is a well-motivated design choice — GNN local message passing struggles to capture reusable high-order reasoning motifs
- **30% data matching full-data performance** demonstrates that the induced schemas capture genuinely transferable reasoning structures rather than overfitting to the training distribution

## Limitations & Future Work
- Schema induction depends on LLM quality; scalability to noisy or very large corpora remains unverified
- FOL representations may not capture implicit stance expressions such as rhetoric, irony, and metaphor
- Applicability to multilingual and cross-cultural settings is unexplored
- The computational cost of schema induction (requiring multiple LLM calls) is not quantitatively analyzed

## Related Work & Insights
- **vs. FOLAR**: Both use FOL knowledge, but FOLAR operates at the instance-level FOL rule, while CIRF induces cross-target transferable schemas; CIRF surpasses FOLAR by 1.9 points on SEM16
- **vs. LogiMDF**: LogiMDF also employs logical reasoning but operates at the predicate/word level without modeling relational structure; CIRF surpasses it by 3.7 points on COVID-19
- **vs. KAI**: KAI augments with natural language knowledge; CIRF demonstrates that structured knowledge via FOL + schemas is more effective
- **vs. pure LLM prompting**: GPT-3.5 direct prompting achieves only 65.1 on VAST, while CIRF reaches 80.9, showing that schema-guided reasoning far exceeds surface-level prompting

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing cognitive schema theory into ZSSD is a pioneering cross-disciplinary integration; the USI + SEGKM framework design is original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparison across three benchmarks, complete ablation, low-resource analysis, LLM scalability, hyperparameter sensitivity analysis, and case studies
- Writing Quality: ⭐⭐⭐⭐ The derivation from cognitive motivation to formal methodology is clear and notation is consistent
- Value: ⭐⭐⭐⭐ A practical method for low-resource ZSSD; the transferability of schemas offers a new paradigm for zero-shot NLP

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Unified Reasoning Framework for Holistic Zero-Shot Video Anomaly Analysis](../../NeurIPS2025/interpretability/a_unified_reasoning_framework_for_holistic_zeroshot_video_an.md)
- [\[CVPR 2026\] SubspaceAD: Training-Free Few-Shot Anomaly Detection via Subspace Modeling](../../CVPR2026/interpretability/subspacead_training-free_few-shot_anomaly_detection_via_subspace_modeling.md)
- [\[ICCV 2025\] SVIP: Semantically Contextualized Visual Patches for Zero-Shot Learning](../../ICCV2025/interpretability/svip_semantically_contextualized_visual_patches_for_zero-shot_learning.md)
- [\[ICML 2026\] Singular Vectors of Attention Heads Align with Features](../../ICML2026/interpretability/singular_vectors_of_attention_heads_align_with_features.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](../../NeurIPS2025/interpretability/auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)

</div>

<!-- RELATED:END -->
