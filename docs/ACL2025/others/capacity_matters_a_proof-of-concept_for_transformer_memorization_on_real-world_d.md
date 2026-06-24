---
title: >-
  [Paper Note] Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data
description: >-
  [ACL 2025 (L2M2 Workshop)][Transformer memorization capacity] Using the SNOMED clinical knowledge graph as the data source, this paper systematically investigates the memorization capacity of decoder-only Transformers on structured data. It reveals that the embedding dimension is the primary factor determining learning speed and capacity, while increasing model depth yields marginal returns, and the Softmax activation function exhibits the most stable performance.
tags:
  - "ACL 2025 (L2M2 Workshop)"
  - "Transformer memorization capacity"
  - "knowledge graph"
  - "embedding dimension"
  - "activation function"
  - "edge deployment"
date: 2026-05-08
content_hash: e9b529e150f773ea
---

# Capacity Matters: A Proof-of-Concept for Transformer Memorization on Real-World Data

**Conference**: ACL 2025 (L2M2 Workshop)  
**arXiv**: [2506.14704](https://arxiv.org/abs/2506.14704)  
**Code**: [Available](https://github.com/um-dacs-nlp/capacity/)  
**Area**: Other  
**Keywords**: Transformer memorization capacity, knowledge graph, embedding dimension, activation function, edge deployment

## TL;DR

Using the SNOMED clinical knowledge graph as the data source, this paper systematically investigates the memorization capacity of decoder-only Transformers on structured data. It reveals that the embedding dimension is the primary factor determining learning speed and capacity, while increasing model depth yields marginal returns, and the Softmax activation function exhibits the most stable performance.

## Background & Motivation

- While Transformers have achieved remarkable success in NLP, the mechanism of **how models store and recall information** (especially factual/structured knowledge) remains unclear.
- Practical application scenario: Deploying small, local Transformers on medical wearable devices (e.g., smart glasses, smart watches) requires the model to **precisely memorize** domain-specific factual knowledge under tight parameter constraints.
- Existing theoretical works (Kim et al., 2023; Kajitsuka & Sato, 2024) have established a mathematical upper bound of $O(d + n + \sqrt{nN})$ for Transformer memorization capacity, but lack empirical validation on **real-world structured data**.
- This paper does not focus on generalization, but rather on the measurement of **memorization capacity under controlled conditions** as a proof-of-concept to bridge theoretical analysis and practical evaluation.

## Method

### Overall Architecture

This paper designs a comprehensive experimental framework: (1) generating a structured dataset from the SNOMED knowledge graph; (2) training a small-scale decoder-only Transformer for token-by-token prediction; (3) using the Maximum Attainable Capacity (MAC) metric to measure memorization capacity.

### Data Generation

#### Triplets Dataset
- Extract **(Concept, Property, Related Concept)** triplets from the SNOMED knowledge graph.
- Filter uninformative properties, and randomly select one related concept when multiple exist for a single (Concept, Property) pair to ensure uniqueness.
- Dataset scale: 50K to 100K samples.

#### Sequences Dataset
- Simulate graph traversal, generating sequences in the form of **(node₁, edge₁, node₂, ..., nodeₙ)**.
- Construct subgraphs using BFS (depth of 5 hops) and generate sequences via random walks, containing 4–6 nodes (3–5 edges) per sequence.
- Uniformly align sequence lengths using zero-padding, using a node mask to distinguish node and edge tokens.
- Dataset scale: 20K to 100K sequences.

### Model Architecture

- **Decoder-only Transformer**: Includes an embedding layer (with learned positional encodings), Transformer decoder layers (multi-head attention), and a linear output layer.
- Parameter scale: 2.9M to 44.5M, primarily scaling with embedding dimension and the number of layers.
- Prediction task: Predict the next concept based on preceding tokens.
- Accuracy definition: Correctly predicted related concepts / Total predictions.

### Key Metric: Maximum Attainable Capacity (MAC)

- MAC measures the **maximum number of samples a model can memorize** given a large library.
- Compared to the Maximum Library Size (MLS) method (which requires iteratively training on datasets of different scales), MAC is computationally more efficient.
- Prior research has validated a strong correlation between MAC and MLS.

### Experimental Setups

**Setup 1 - Effect of Data Scale**: 1 layer, embedding=128, 4 attention heads, ReLU, batch=64, 500 epochs, data scale 50K–100K.

**Setup 2 - Architecture & Activation Functions**: 1/2/4 layers; activation functions ReLU/GELU/RReLU/Softmax; maintain a constant total parameter size ($\text{embedding\_size} = \lfloor \text{base\_params} / \text{n\_layers} \rfloor$), batch=128, 1000 epochs.

**Setup 3 - Interaction between Parameter Size & Depth**: 1/2 layers, base parameters in {16M, 32M, 64M, 128M}, Softmax only, batch=128, 500 epochs.

**Setup 4 - Sequence Memorization**: embedding=64, 4 attention heads, 1/2/4 layers, RReLU and Softmax, batch=128, 400 epochs.

### Loss & Training

- Loss function: **Cross-Entropy Loss**
- Optimizer: **Adam** (lr=0.001)
- Each experiment is repeated 10 times (Setup 1-2) or 3 times (Setup 3-4), reporting the mean $\pm$ 2 standard deviations.
- Computational resources: NVIDIA A100 16GB, training a total of 546 models, approximately 3100 GPU hours.

## Key Experimental Results

### Main Results (Data Scale vs. Memorization Capacity)

| Data Scale | Accuracy (%) | Capacity (MAC) |
|---------|----------|----------|
| 50,000 | 93.62±0.3 | 46,811±149 |
| 60,000 | 92.42±0.2 | 55,455±126 |
| 70,000 | 91.1±1.08 | 63,773±756 |
| 80,000 | 89.63±1.66 | 71,706±1326 |
| 90,000 | 87.24±1.66 | 78,517±2173 |
| 100,000 | 86.78±2.42 | 86,776±2484 |

### Sequence Dataset Memorization Capacity (100K Sequences)

| Activation Function | Layers | Capacity (MAC) | Total Predictions |
|---------|------|----------|---------|
| RReLU | 1 | 166,934±243 | 167,965 |
| RReLU | 4 | 165,271±1,068 | 167,965 |
| Softmax | 1 | 166,992±110 | 167,965 |
| Softmax | 4 | 166,825±319 | 167,965 |

### Ablation Study

- **Embedding Dimension vs. Depth**: A 1-layer model with embedding=16 and a 2-layer model with embedding=16/layer converge at almost the same speed $\rightarrow$ embedding dimension determines the learning speed.
- **Comparison of Activation Functions**: Softmax is the most stable across all configurations; ReLU/RReLU show increased variability in deeper models.
- **Capacity Bottleneck**: On 100K data, the capacity of a 2-layer model with embedding=8 drops to 85,935±153 (vs. ~88,200 for other configurations).

### Key Findings

1. **Embedding dimension is the core factor**: Learning curves of models with different depths are almost identical under the same embedding dimension.
2. **Increasing depth is unhelpful or even detrimental**: For simple structured data, extra layers reduce training speed and final capacity.
3. **Softmax leads across the board**: Softmax outperforms ReLU/GELU/RReLU in terms of capacity, stability, and scaling with depth.
4. **Sequences > Triplets**: Sequence data achieves near-perfect memorization with fewer epochs (100% at 20K sequences).
5. **Presence of a ~70K threshold**: Beyond this scale, training dynamics undergo a qualitative transition, requiring significantly more epochs.

## Highlights & Insights

- Initiates the conversion of a scale medical ontology into a tokenized dataset for memorization research, providing a bridge from theory to practice.
- Counter-intuitive finding: **More layers $\neq$ Better memorization**, which has practical implications for edge device deployment—recommending a shallow-and-wide embedding architecture.
- The data structure itself encodes relational patterns that facilitate memorization (sequences outperform triplets).
- Practical recommendation: Small Transformers on medical wearables should adopt a 1-2 layer architecture paired with a large embedding dimension.

## Limitations & Future Work

- Patterns of unlearned samples remain unanalyzed (which could be improved using curriculum learning or loss re-weighting).
- Not validated on longer sequences or larger data scales.
- Lack of direct evaluation of inter-layer redundancy/similarity (which could be studied using probing and pruning).
- The impact of quantization and sparsification on the architectural recommendations is not considered.
- Alignment between sequence generation methods and clinical reasoning patterns needs enhancement.
- Only evaluated on SNOMED, without verifying generalization to other biomedical representation graphs.

## Related Work & Insights

- Kim et al. (2023) provided a theoretical memorization capacity upper bound of $O(d+n+\sqrt{nN})$ for Transformers, which this paper empirically validates.
- Härmä et al. (2024) evaluated memorization capability using random digit sequences, whereas this paper extends it to structured knowledge graphs.
- Future work could combine sparse autoencoders to decouple memorization from generalization, or leverage curriculum learning to improve the accuracy of unlearned samples.

## Rating

| Dimension | Score |
|------|------|
| Novelty | ⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Overall Recommendation | ⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-Shot Head Swapping in Real-World Scenarios](../../CVPR2025/others/zero-shot_head_swapping_in_real-world_scenarios.md)
- [\[CVPR 2026\] Multi-view Crowd Tracking Transformer with View-Ground Interactions Under Large Real-World Scenes](../../CVPR2026/others/multi-view_crowd_tracking_transformer_with_view-ground_interactions_under_large_.md)
- [\[ICML 2025\] Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Settings](../../ICML2025/others/suitability_filter_a_statistical_framework_for_classifier_evaluation_in_real-wor.md)
- [\[ACL 2025\] Partial Colexifications Improve Concept Embeddings](partial_colexifications_improve_concept_embeddings.md)
- [\[CVPR 2026\] VideoWorld 2: Learning Transferable Knowledge from Real-world Videos](../../CVPR2026/others/videoworld_2_learning_transferable_knowledge_from_real-world_videos.md)

</div>

<!-- RELATED:END -->
