---
title: >-
  [Paper Note] The Structure of Relation Decoding Linear Operators in Large Language Models
description: >-
  [NeurIPS 2025][Model Compression][Linear relation decoding] This paper reveals that linear relation embeddings (LREs) in Transformer language models do not encode fine-grained relations but instead extract shared coarse-…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Linear relation decoding"
  - "tensor networks"
  - "knowledge compression"
  - "interpretability"
  - "semantic attributes"
date: 2026-05-08
content_hash: 8b0ae16b27ed6a03
---

# The Structure of Relation Decoding Linear Operators in Large Language Models

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.26543](https://arxiv.org/abs/2510.26543)  
**Code**: [GitHub](https://bit.ly/structure-of-relations)  
**Area**: Model Compression
**Keywords**: Linear relation decoding, tensor networks, knowledge compression, interpretability, semantic attributes

## TL;DR

This paper reveals that linear relation embeddings (LREs) in Transformer language models do not encode fine-grained relations but instead extract shared coarse-grained semantic attributes (e.g., "country," "gender"). A rank-3 tensor network is employed to compress large collections of relation decoding matrices by several orders of magnitude.

## Background & Motivation

Hernandez et al. (2023) demonstrated that relational knowledge in Transformer language models (e.g., "Paris → France") can be approximated by a simple affine transformation $f_R(\mathbf{v}_S) = W_R\mathbf{v}_S + \mathbf{b}_R$, where each relation $R$ corresponds to a $d \times d$ linear relation embedding (LRE) matrix $W_R$.

However, prior work analyzed only individual relations, leaving critical questions open:

**Do relation decoders share structure across relations?** A single LRE matrix in GPT-J ($d=4096$) contains over 16 million parameters; 100 relations would yield 1.6 billion parameters — does a more compact shared representation exist?

**If compressible, why?** Does redundancy arise from semantic overlap or deeper structural regularities?

**Can the compressed model generalize to unseen relations?**

These questions are relevant both to interpretability (understanding how models organize knowledge) and to efficiency (representing large numbers of relations with fewer parameters).

## Method

### Overall Architecture

The study proceeds in three progressive stages:
1. Compressing collections of relation decoding matrices via tensor networks → demonstrating compressibility.
2. Using a cross-evaluation protocol to identify the source of compressibility → discovering attribute-level encoding.
3. Testing tensor network generalization to unseen relations → probing the limits of shared representations.

### Key Designs

1. **Tensor Network Compression**: The $n$ relation matrices of size $d \times d$ are stacked into a rank-3 tensor $\mathbb{R}^{d \times d \times n}$, which is then compressed using two tensor network architectures:

    - **SimpleOrder3Network**: Centers on a small rank-3 tensor $T^0 \in \mathbb{R}^{d_{s'} \times d_{r'} \times d_{o'}}$ connected to the original embedding space via three projection matrices $P^1, P^2, P^3$. Its parameter count $N_{\text{Simple}} = (d \cdot d_{s'} + d \cdot d_{r'} + d \cdot d_{o'}) + d_{s'} \cdot d_{r'} \cdot d_{o'}$ is far smaller than the original $n \cdot d^2$.
    - **TriangleTensorNetwork**: Centers on three interconnected rank-3 tensors, offering greater expressive capacity. For a given relation embedding $\mathbf{v}_r$, the corresponding LRE matrix is produced via tensor contraction.

   Both are trained end-to-end with LLM parameters frozen, optimizing only the tensor network. A key finding is that **a linear tensor network without an additional relation encoder outperforms variants with nonlinear encoders**, indicating that linear structure suffices for efficient compression.

2. **Cross-Evaluation Protocol**: For $k$ relations and their corresponding decoders $\{f_j\}$, a $k \times k$ faithfulness matrix is constructed as $F_{j,l} = \text{faithfulness}(R_l, f_j)$, i.e., applying the decoder of relation $j$ to the subjects of relation $l$. If two relations are semantically close (e.g., "characteristic gender" and "occupational gender"), the cross-evaluation score should be high. This matrix serves as an empirical similarity kernel over relations and reveals pronounced **block structure**.

3. **Attribute Extractor Hypothesis**: The central finding of the cross-evaluation analysis is that LRE matrices are not relation decoders per se, but **attribute extractors**. For instance, the decoders for "capital's country," "food's country of origin," and "landmark's country" are interchangeable, because all three extract the coarse-grained attribute "country." Notably, the "occupational gender" decoder can achieve higher faithfulness on "characteristic gender" than the dedicated decoder. This explains the effectiveness of compression: many ostensibly distinct relations share a small set of attribute extraction patterns.

### Loss & Training

- End-to-end training with frozen LLM parameters; only tensor network parameters are optimized.
- Loss function: $\mathcal{L}_{\mathcal{R}}(T_{s,r,o}) = \sum_{R \in \mathcal{R}} \sum_{(S,O) \in R} CE(\mathbb{1}_O, L_{\text{head}}(T^R_{s,o}(\mathbf{v}_S)))$
- Optimized with SGD over relation–subject–object triples until convergence.

## Key Experimental Results

### Main Results — Compression Performance

| Method | Parameters | Avg. Faithfulness | Note |
|--------|------------|-------------------|------|
| Original 47 LRE matrices | ~788M | 0.41 | Baseline (Jacobian approximation) |
| Per-relation low-rank approximation | ~80M | ~0.35 | Independent compression, no sharing |
| SimpleOrder3Network | <1M | >0.41 | **788× compression with higher faithfulness** |
| TriangleTensorNetwork | <1M | >0.41 | Similar; richer internal structure |
| Tensor network + extra encoder | Slightly more | Slightly lower | Nonlinear encoder is detrimental |

### Generalization Experiment — Math Dataset

| Configuration | Train Faithfulness | Test Faithfulness | Note |
|---------------|--------------------|-------------------|------|
| Tensor network (75% train / 25% test) | 0.992 (±0.012) | 0.96 (±0.031) | Near-perfect generalization |
| Majority class baseline | — | 0.30 | Reference |
| Random relation embeddings | Perfect memorization | ~0 | Confirms reliance on semantic information |
| Random subject/object embeddings | ~0 | ~0 | Confirms reliance on entity representations |

### Ablation Study

| Condition | Result | Note |
|-----------|--------|------|
| Remove semantically similar relations | Still compresses substantially | Compression is not solely due to semantic overlap |
| Nonlinear relation encoder | No improvement | Linear structure is sufficient |
| Extended dataset (79 relations) | Block structure in cross-evaluation more pronounced | Further supports attribute hypothesis |
| GPT-J / Llama 3.1 8B / GPT-NeoX-20B | Consistent findings | Cross-model robustness |

### Key Findings

- The 788M parameters of 47 relation matrices can be equivalently replaced by a tensor network with fewer than 1M parameters, achieving approximately **800× compression**.
- The block structure of the cross-evaluation matrix clearly corresponds to semantic attribute classes (country, gender, antonyms, etc.), with cross-class syntactic similarities also observed (e.g., the "first letter" decoder is effective for "superlative adjective").
- On a dense and structured arithmetic relation dataset, the tensor network generalizes near-perfectly to unseen relations (faithfulness 0.96+).

## Highlights & Insights

- The finding that LREs function as **"attribute extractors rather than relation decoders"** fundamentally reframes the understanding of LREs: models do not store thousands of independent relation mappings, but decode relations through combinations of a small number of attribute extraction patterns.
- The success of tensor networks as compression tools suggests the possibility of **applying analogous tensor network compression to collections of LoRA matrices**.
- The cross-evaluation protocol constitutes a semantic similarity measure independent of embedding similarity, with broad potential applicability to other settings.

## Limitations & Future Work

- Validation is limited to relatively small LLMs (6B, 8B, 20B parameters); applicability to larger or instruction-tuned models remains unknown.
- The relation datasets have limited coverage (47–79 relations), far smaller than the space of relations in human knowledge.
- Generalization to general linguistic relations is limited — successful generalization is observed only for relations that are sufficiently semantically proximate.
- The attribute extractor hypothesis remains primarily empirical and lacks validation at the level of the model's internal mechanisms.

## Related Work & Insights

- Complements Chanin et al. (2024)'s Linear Relational Concepts (LRC), which uses the pseudoinverse of LREs for classification; this paper uncovers shared structure across LREs.
- Tensor networks, originally used in physics to represent quantum states, are here introduced to NLP knowledge representation — a noteworthy cross-disciplinary application.
- The findings also have implications for compressing Mixture-of-Experts (MoE) architectures, where multiple experts may share underlying attribute structures.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The cross-evaluation protocol and attribute extractor hypothesis are original and insightful contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, multiple datasets, thorough ablations and generalization experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Concepts are clearly presented with progressive exposition; figures are effective but occasionally dense.
- **Value**: ⭐⭐⭐⭐⭐ Significantly advances understanding of knowledge structure in LLMs, with dual value for interpretability and compression.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Simple Linear Patch Revives Layer-Pruned Large Language Models](a_simple_linear_patch_revives_layerpruned_large_language_mod.md)
- [\[NeurIPS 2025\] Correlation Dimension of Auto-Regressive Large Language Models](correlation_dimension_of_auto-regressive_large_language_models.md)
- [\[NeurIPS 2025\] Restoring Pruned Large Language Models via Lost Component Compensation](restoring_pruned_large_language_models_via_lost_component_compensation.md)
- [\[NeurIPS 2025\] PermLLM: Learnable Channel Permutation for N:M Sparse Large Language Models](permllm_learnable_channel_permutation_for_nm_sparse_large_language_models.md)
- [\[NeurIPS 2025\] LayerIF: Estimating Layer Quality for Large Language Models using Influence Functions](layerif_estimating_layer_quality_for_large_language_models_using_influence_funct.md)

</div>

<!-- RELATED:END -->
