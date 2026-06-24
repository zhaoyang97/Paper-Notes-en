---
title: >-
  [Paper Note] GraphNarrator: Generating Textual Explanations for Graph Neural Networks
description: >-
  [ACL 2025][Graph Learning][GNN Explainability] GraphNarrator is the first method to generate natural language explanations for Graph Neural Networks (GNNs). By utilizing saliency graph verbalization for pseudo-label generation, information-theoretic metric-driven expert iteration for self-improvement, and knowledge distillation for training an end-to-end explainer, it achieves faithful, concise, and human-friendly explanations of GNN decisions.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "GNN Explainability"
  - "Natural Language Explanation"
  - "Text-Attributed Graphs"
  - "Expert Iteration"
  - "Knowledge Distillation"
date: 2026-05-08
content_hash: 49963841f0e0278f
---

# GraphNarrator: Generating Textual Explanations for Graph Neural Networks

**Conference**: ACL 2025  
**arXiv**: [2410.15268](https://arxiv.org/abs/2410.15268)  
**Code**: None  
**Area**: Graph Learning  
**Keywords**: GNN Explainability, Natural Language Explanation, Text-Attributed Graphs, Expert Iteration, Knowledge Distillation  

## TL;DR

GraphNarrator is the first method to generate natural language explanations for Graph Neural Networks (GNNs). By utilizing saliency graph verbalization for pseudo-label generation, information-theoretic metric-driven expert iteration for self-improvement, and knowledge distillation for training an end-to-end explainer, it achieves faithful, concise, and human-friendly explanations of GNN decisions.

---

## Background & Motivation

**Background:** Graph representation learning is widely applied in recommendation systems, social networks, and other fields, but the internal decision-making process of GNNs remains opaque. Existing GNN explanation methods primarily provide node- or edge-level importance scores. In the context of Text-Attributed Graphs (TAGs), these token-importance-based explanations are redundant and lack synthesis, making them difficult for humans to comprehend.

**Limitations of Prior Work:** (1) Saliency graph or feature importance methods (e.g., GNNExplainer) only provide importance scores without explaining semantic information; (2) In multi-node subgraphs, token-by-token importance annotations are highly fragmented and redundant; (3) External language models lack awareness of the model's internal decision-making processes, leading to unreliable zero-shot generated explanations.

**Key Insight:** Natural language explanations are abstractive, concise, and readable, making them a more human-comprehensible modality of explanation. However, due to the lack of ground-truth explanation data, the key challenge of "training a high-quality explanation generator without labels" must be addressed.

---

## Method

### Overall Architecture

GraphNarrator employs a three-stage pipeline: (1) saliency explanation generation and verbalization → (2) expert-iteration-driven pseudo-label generator training → (3) knowledge distillation for training an end-to-end explainer.

### Key Designs

**1. Saliency Paragraph Verbalization (Saliency Paragraph):** First, saliency methods (e.g., LRP, Input×Grad) are used to obtain node and token importance scores. Then, the ego-graph is decomposed into a tree structure via BFS, which is then organized into a hierarchical document paragraph using pre-order traversal. The text of each node serves as a section, child nodes serve as subsections, and cross-edges are represented by citation hook-ups. Importance scores are appended following each token, in the form of `token(score)`.

**2. Information-Theoretic Explanation Quality Metrics:** Three training objectives are proposed:
- **Input Faithfulness $f_S$:** Uses PMI to measure the mutual information between the explanation and important input regions, which is approximated via masked token prediction.
- **Output Faithfulness $f_F$:** Uses PMI to measure the mutual information between the explanation and the model's prediction.
- **Brevity $f_B$:** The ratio of explanation length to input length, encouraging concise expressions.

**3. Expert Iteration Self-Improvement:** Closed-loop optimization based on Expert Iteration: measuring explanation quality → filtering high-quality explanations → fine-tuning the pseudo-label generator → generating a new round of candidate explanations.

### Loss & Training

Input faithfulness is estimated by sampling masked token predictions across different thresholds $\tau$:

$$f_S \approx \int_0^1 P(\tau) \cdot \log \frac{P_{MLM}(\mathcal{R}_\tau | \mathcal{G}_{M_\tau}, E)}{P_{MLM}(\mathcal{R}_\tau | \mathcal{G}_{M_\tau})} d\tau$$

The final objective balances $f_S$ (maximization), $f_F$ (maximization), and $f_B$ (minimization) through multi-objective optimization.

---

## Experiments

### Main Results

| Dataset | Method | Simul.↑ | PMI-10%↑ | PMI-20%↑ | PMI-30%↑ | Brevity↓ |
|--------|------|---------|----------|----------|----------|----------|
| DBLP | LLaMA3.1 8B | 0.63 | 0.139 | 0.109 | 0.077 | 0.394 |
| DBLP | GPT-4o | 0.82 | 0.142 | 0.101 | 0.085 | 0.385 |
| DBLP | **GraphNarrator** | **0.95** | **0.155** | **0.108** | **0.085** | **0.354** |
| Cora | GPT-4o | 0.95 | 0.414 | 0.284 | 0.225 | 0.357 |
| Cora | **GraphNarrator** | **0.97** | **0.418** | **0.290** | **0.227** | **0.315** |
| Book-History | GPT-4o | 0.89 | 0.456 | 0.313 | 0.240 | 0.768 |
| Book-History | **GraphNarrator** | **0.96** | **0.533** | **0.374** | **0.291** | **0.506** |

### Ablation Study

The effectiveness of expert iteration is verified by comparing different iteration rounds (see the original paper for details), showing a steady improvement in the quality of pseudo-labels in each round. The balanced configuration of the three training objectives ($f_S$, $f_F$, $f_B$) with top-50% filtering achieves the best performance.

| Ablation Dimension | Observation |
|----------|------|
| W/o Expert Iteration | Both explanation faithfulness and brevity deteriorate |
| Only $f_S$ preserved | Explanations are wordy with poor brevity |
| Only $f_B$ preserved | Explanations are too short with poor faithfulness |

### Key Findings

- GraphNarrator consistently outperforms GPT-4o zero-shot explanations across all datasets, particularly showing a significant advantage in Simulatability (DBLP: 0.95 vs 0.82).
- On the Book-History dataset, PMI-10% increases by 16.9% and brevity improves by 34.1%.
- The small-scale model based on LLaMA3.1 8B surpasses GPT-4o after fine-tuning, validating the effectiveness of Expert Iteration.

---

## Highlights & Insights

- Proposes the first method to generate natural language explanations for GNNs, pioneering a new paradigm for graph model explainability.
- Innovatively converts saliency graph explanations into structured documents (Saliency Paragraphs), allowing LLMs to comprehend graph structural information.
- Proposes an information-theoretic unsupervised metric for explanation quality, enabling the evaluation and optimization of explanations without human annotations.
- Achieves efficient end-to-end explanation generation through Expert Iteration and knowledge distillation.

## Limitations & Future Work

- Relies on saliency methods as the initial signal, whose inherent limitations may propagate to the final explanations.
- Evaluated only on node classification tasks, without being extended to other task types like link prediction or graph classification.
- Pseudo-label generation relies on GPT-4o-mini, which limits applicability in resource-constrained scenarios.
- The human evaluation scale is relatively small (50 samples × 3 annotators), and statistical significance remains to be further validated.

## Related Work & Insights

- **GNN Explainability:** GNNExplainer (Ying et al., 2019), PGExplainer (Luo et al., 2020), etc., provide structural explanations.
- **Natural Language Explanation:** e-SNLI (Camburu et al., 2018), Chain-of-Thought (Wei et al., 2022), etc., target textual models.
- **SMV Methods:** Feldhus et al. (2022) verbalize saliency maps for text classification model explanation.
- **Expert Iteration:** The iterative self-improvement framework of Dong et al. (2023).

---

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |
| Overall Rating | 8/10 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](../../NeurIPS2025/graph_learning/sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[ACL 2025\] Can Graph Neural Networks Learn Language with Extremely Weak Text Supervision?](can_graph_neural_networks_learn_language.md)
- [\[ACL 2026\] From Nodes to Narratives: Explaining Graph Neural Networks with LLMs and Graph Context](../../ACL2026/graph_learning/from_nodes_to_narratives_explaining_graph_neural_networks_with_llms_and_graph_co.md)
- [\[NeurIPS 2025\] Over-squashing in Spatiotemporal Graph Neural Networks](../../NeurIPS2025/graph_learning/over-squashing_in_spatiotemporal_graph_neural_networks.md)
- [\[NeurIPS 2025\] Graph Neural Networks for Interferometer Simulations](../../NeurIPS2025/graph_learning/graph_neural_networks_for_interferometer_simulations.md)

</div>

<!-- RELATED:END -->
