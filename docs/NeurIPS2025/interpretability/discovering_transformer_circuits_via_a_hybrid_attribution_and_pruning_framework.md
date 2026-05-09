---
title: >-
  [Paper Note] Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework
description: >-
  [NeurIPS 2025][circuit discovery] This paper proposes HAP, a Hybrid Attribution and Pruning framework that first applies fast Edge Attribution Patching (EAP) to filter high-potential subgraphs, then runs precise Edge Pruning (EP) on the reduced search space. On the IOI task with GPT-2 Small, HAP achieves a 46% speedup over pure EP while maintaining comparable circuit faithfulness, and successfully recovers S-inhibition heads that EAP alone fails to identify.
tags:
  - NeurIPS 2025
  - circuit discovery
  - attribution patching
  - edge pruning
  - hybrid framework
  - IOI
  - GPT-2
  - mechanistic interpretability
date: 2026-05-08
content_hash: 5bf6deda843c3cb1
---

# Discovering Transformer Circuits via a Hybrid Attribution and Pruning Framework

**Conference**: NeurIPS 2025
**arXiv**: [2510.03282](https://arxiv.org/abs/2510.03282)
**Code**: [GitHub](https://anonymous.4open.science/r/HAP-circuit-discovery)
**Area**: Mechanistic Interpretability / Circuit Discovery / Transformer Analysis
**Keywords**: circuit discovery, attribution patching, edge pruning, hybrid framework, IOI, GPT-2, mechanistic interpretability

## TL;DR

This paper proposes HAP, a Hybrid Attribution and Pruning framework that first applies fast Edge Attribution Patching (EAP) to filter high-potential subgraphs, then runs precise Edge Pruning (EP) on the reduced search space. On the IOI task with GPT-2 Small, HAP achieves a 46% speedup over pure EP while maintaining comparable circuit faithfulness, and successfully recovers S-inhibition heads that EAP alone fails to identify.

## Background & Motivation

**Core goal of mechanistic interpretability**: As LLMs are deployed in high-stakes settings, understanding their internal "black-box" mechanisms has become an urgent necessity. Mechanistic interpretability pursues this goal by identifying sparse subnetworks ("circuits") responsible for specific behaviors.

**Standard paradigm for circuit analysis**: Transformers are represented as computation graphs (nodes = attention heads/MLPs, edges = information flow), within which the minimal subgraph executing a specific task is identified. Manual methods (Wang et al. 2022) have been superseded by automated approaches.

**Computational bottleneck of ACDC**: The earliest automated method, ACDC (Conmy et al. 2023), performs greedy edge-wise ablation search and achieves high faithfulness but requires a large number of forward passes, making it unscalable to large models.

**Speed advantage and faithfulness deficit of EAP**: EAP (Syed et al. 2023) estimates the importance of all edges simultaneously via first-order Taylor approximation, requiring only one backward pass and two forward passes. However, the linear approximation leads to substantially reduced faithfulness, and at high sparsity it tends to discard collaboratively important components.

**Faithfulness advantage and computational cost of EP**: EP (Bhaskar et al. 2024) achieves precise pruning by gradient-optimizing binary masks and delivers excellent faithfulness, having been scaled to CodeLlama-13B. However, it demands substantial GPU resources and extended training time.

**Root cause and opportunity**: EAP is fast but unfaithful; EP is faithful but slow — their strengths are precisely complementary. This raises the question of whether EAP's speed can serve as a coarse filter, followed by EP's precision for fine selection. This is the starting point of HAP.

## Method

### Overall Architecture: HAP (Hybrid Attribution and Pruning)

HAP decomposes circuit discovery into a three-stage pipeline: ① computation graph construction → ② EAP coarse filtering → ③ EP precise pruning. The core idea is to use a fast but coarse attribution method to narrow the search space, then apply a precise but expensive optimization method on the reduced space.

### Key Design 1: Computation Graph Construction

- **Function**: Represents the Transformer model as a directed computation graph.
- **Mechanism**: Nodes correspond to attention layers and MLP layers; edges represent information flow from the output of one node to the input of another. For GPT-2 Small (117M parameters, 12 layers × 12 heads), a complete edge set spanning all attention heads and MLPs is constructed.
- **Design Motivation**: A unified graph representation is the foundation for subsequent attribution and pruning operations, following the standard convention of Bhaskar et al. (2024) to ensure comparability.

### Key Design 2: EAP Fast Coarse Filtering

- **Function**: Computes absolute attribution scores for all edges simultaneously via first-order Taylor approximation, then retains the top-$k$ edges by score.
- **Mechanism**: For each edge $e$, its importance is approximated as:
$$L(\mathbf{x} \mid e_{\text{ablated}}) - L(\mathbf{x}) \approx (e_{\text{clean}} - e_{\text{ablated}})^\top \frac{\partial L(\mathbf{x} \mid e_{\text{clean}})}{\partial e_{\text{clean}}}$$
Scores for all edges are obtained with a single backward pass and two forward passes.
- **Design Motivation**: EAP's computational cost is nearly constant (independent of edge count), making it well-suited for rapidly eliminating clearly unimportant edges. Critically, the filtering threshold is set very conservatively to deliberately retain edges with low individual scores that may nonetheless contribute collaboratively.

### Key Design 3: EP Precise Pruning on the Reduced Search Space

- **Function**: Runs gradient-optimized edge pruning on the subgraph produced by EAP filtering.
- **Mechanism**: EP optimizes binary masks $z \in [0,1]^{N_{\text{edge}}}$ over the reduced search space, minimizing the output divergence between the original and pruned graphs while satisfying the target sparsity constraint $1 - |H|/|G| \geq c$.
- **Design Motivation**: EP's computational cost scales with the size of the search space. After EAP removes a large number of irrelevant edges, the parameter space for EP optimization is substantially reduced, accelerating convergence. Meanwhile, EAP's wide threshold ensures a "safety margin" so that collaborative components such as S-inhibition heads are not discarded during coarse filtering.

### Key Design 4: Wide-Threshold Safety Margin Strategy

- **Function**: Deliberately sets a very low filtering threshold during the EAP stage.
- **Mechanism**: Rather than pursuing high sparsity in the EAP stage, a generous candidate edge set is retained, allowing edges with low individual attribution scores but meaningful contributions to overall circuit function to proceed to the EP stage.
- **Design Motivation**: Collaborative components such as S-inhibition heads are characterized by low individual importance scores yet play a critical inhibitory/coordinative role within the circuit. EAP's linear approximation cannot capture such nonlinear collaborative effects, necessitating a wide threshold to prevent erroneous pruning.

## Loss & Training

The EP optimization objective comprises two components:

1. **Faithfulness loss**: Minimizes the KL divergence between the full model and the pruned subgraph on both clean and corrupted inputs, ensuring that the circuit behavior faithfully reflects the original model.
2. **Sparsity constraint**: Enforces the target sparsity $1 - |H|/|G| \geq c$ via Lagrange multipliers or projection methods.

Training alternates gradient updates using clean and corrupted sample pairs. All experiments were conducted on a single NVIDIA H100 GPU. Hyperparameters for the EP stage follow the settings of Bhaskar et al. (2024).

## Key Experimental Results

### Experimental Setup

- **Model**: GPT-2 Small (117M parameters)
- **Task**: Indirect Object Identification (IOI), formatted as "When Dylan and Ryan went to the store, Dylan gave a popsicle to → Ryan"
- **Dataset**: 200 training examples, 200 validation examples, 36,084 test examples, generated using the templates from Wang et al. (2022)
- **Evaluation metrics**: Accuracy (using the manually discovered circuit as ground truth), Logit Difference, KL divergence, and runtime
- **Hardware**: 1× NVIDIA H100

### Main Results — Table 1: Efficiency and Faithfulness Comparison

| Algorithm | Sparsity | Accuracy ↑ | Logit Diff ↑ | KL ↓ | Runtime (s) ↓ |
|-----------|----------|-----------|-------------|------|--------------|
| EAP | 94±0.5% | 0.698 | 3.13 | – | **4** |
| EP | 94±0.5% | **0.772** | **3.48** | 0.190 | 2921 |
| HAP | 94±0.5% | 0.759 | 3.42 | **0.188** | 1579 |

**Key Findings**: HAP is 46% faster than EP (1579s vs. 2921s), with accuracy only 1.3 percentage points lower (0.759 vs. 0.772), nearly identical KL divergence (0.188 vs. 0.190), and comparable Logit Difference (3.42 vs. 3.48). Relative to EAP, HAP substantially outperforms on all quality metrics.

### Ablation Study — Table 2: IOI Case Study — Retention of S-Inhibition Heads

| Method | Head 7.3 | Head 7.9 | Head 8.6 | Head 8.10 | Full Circuit Recovery |
|--------|----------|----------|----------|----------|-----------------------|
| EAP | ✗ | ✗ | ✗ | ✗ | Incomplete |
| HAP | ✓ | ✓ | ✓ | ✓ | **Complete** |

S-inhibition heads in IOI are responsible for suppressing Name Mover heads from incorrectly attending to the subject near the verb; they have low individual attribution scores but are critical for collaborative circuit function. At 94% sparsity, EAP discards all four S-inhibition heads, whereas HAP successfully recovers the complete functional circuit through its wide-threshold safety margin combined with EP precise pruning.

## Highlights & Insights

- **Elegant compositional strategy**: Serializing EAP and EP exploits the complementary speed/precision advantages of each method — a straightforward yet effective approach.
- **Challenges the assumed inevitability of the speed–faithfulness trade-off**: Demonstrates that strategic two-stage search can substantially accelerate circuit discovery without sacrificing faithfulness.
- **Compelling qualitative evidence**: The S-inhibition head case study provides an intuitive demonstration of HAP's ability to preserve collaborative components, complementing purely quantitative comparisons.
- **Clear practical value**: The 46% speedup is theoretically more pronounced when scaled to larger models.

## Limitations & Future Work

- **Narrow experimental scope**: Validation is limited to a single IOI task on GPT-2 Small (117M); model scale and task diversity are severely insufficient.
- **Unoptimized EAP threshold**: The edge filtering threshold is set heuristically, lacking systematic hyperparameter search and sensitivity analysis.
- **No variance reporting**: Performance variance across multiple runs is not reported despite the stochastic nature of training data generation.
- **Insufficient baseline comparisons**: Comparisons with ACDC, EAP-GP, and other methods are absent.
- **Lack of large-model validation**: Scalability to models such as Llama/CodeLlama remains a theoretical expectation rather than an empirically verified result.

## Related Work & Insights

- **ACDC** (Conmy et al. 2023): The earliest automated circuit discovery method, employing greedy edge-wise ablation search. It achieves high faithfulness but at large computational cost, serving as the primary computational bottleneck reference for this work.
- **EAP** (Syed et al. 2023): A fast attribution method based on first-order Taylor approximation. Extremely efficient but with low faithfulness; serves as the first-stage component of HAP.
- **EP** (Bhaskar et al. 2024): A gradient-based binary mask optimization pruning method with high faithfulness, scaled to CodeLlama-13B; serves as the second-stage component of HAP.
- **IOI Circuit** (Wang et al. 2022): The manually discovered indirect object identification circuit in GPT-2 Small, comprising Duplicate Token, Induction, S-inhibition, and Name Mover head types; serves as the ground-truth reference for this work.
- **EAP-GP** (Zhang & Dong 2025): An improved method addressing gradient saturation in EAP, not included in the comparison of this work.

## Rating

- Novelty: ⭐⭐⭐ The combination strategy is straightforward; methodological innovation is limited but effective. The core contribution lies in demonstrating the feasibility of a two-stage approach.
- Experimental Thoroughness: ⭐⭐ Single model and single task; lack of variance reporting and hyperparameter sensitivity analysis limits persuasiveness.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated presentation, and intuitive IOI case analysis.
- Overall Recommendation: ⭐⭐⭐ Provides a practical engineering solution for scalability in mechanistic interpretability, but requires broader experimental validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Components: Singular Vector-Based Interpretability of Transformer Circuits](beyond_components_singular_vector-based_interpretability_of_transformer_circuits.md)
- [\[NeurIPS 2025\] Causal Head Gating: A Framework for Interpreting Roles of Attention Heads in Transformers](causal_head_gating_a_framework_for_interpreting_roles_of_attention_heads_in_tran.md)
- [\[NeurIPS 2025\] Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models](bigram_subnetworks_mapping_to_next_tokens_in_transformer_language_models.md)
- [\[NeurIPS 2025\] Why Is Attention Sparse in Particle Transformer?](why_is_attention_sparse_in_particle_transformer.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)

</div>

<!-- RELATED:END -->
