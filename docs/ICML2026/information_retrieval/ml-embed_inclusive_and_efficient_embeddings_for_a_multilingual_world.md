---
title: >-
  [Paper Note] ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World
description: >-
  [ICML 2026][Information Retrieval & RAG][Matryoshka Representation Learning] ML-Embed extends the Matryoshka concept from one dimension (representation dimension) to **three dimensions**—implementing full-stack nested tr…
tags:
  - "ICML 2026"
  - "Information Retrieval & RAG"
  - "Matryoshka Representation Learning"
  - "Multidimensional Nesting"
  - "MTEB"
  - "Low-resource Languages"
  - "Decoder-based Embeddings"
date: 2026-05-08
content_hash: 2c293ff592afcfa2
---

# ML-Embed: Inclusive and Efficient Embeddings for a Multilingual World

**Conference**: ICML 2026  
**arXiv**: [2605.15081](https://arxiv.org/abs/2605.15081)  
**Code**: https://github.com/codefuse-ai/CodeFuse-Embeddings  
**Area**: Text Embeddings / Multilingual Models / Efficient Training  
**Keywords**: Matryoshka Representation Learning, Multidimensional Nesting, MTEB, Low-resource Languages, Decoder-based Embeddings

## TL;DR
ML-Embed extends the Matryoshka concept from one dimension (representation dimension) to **three dimensions**—implementing full-stack nested training across embedding parameters (MEL), model depth (MLL), and representation dimensions (MRL). By constructing a multilingual training set of 50 million samples covering 282 natural languages and 40 programming languages, the authors released a family of open-source models (140M–8B). These models achieve 1st place in 9 out of 17 MTEB benchmarks, with performance gains of +22.89 in Polish and +6.88 in Vietnamese.

## Background & Motivation

**Background**: Text embeddings serve as the foundation for RAG and semantic search. Current SOTA models mostly adapt decoder LLMs (e.g., E5-Mistral, NV-Embed, Qwen3-Embedding, Gemini-Embedding) into embedding models. While effective, this approach is costly due to explosive training costs, massive inference memory requirements, neglect of low-resource languages, and an increasing trend toward closed-source APIs.

**Limitations of Prior Work**: Three major barriers exist. (1) **Computational Barrier**: Decoder-based embedding models typically have 7B+ parameters, posing high entry barriers for training and deployment. Existing Matryoshka Representation Learning (MRL) only optimizes the **storage** dimension (truncatable output) without reducing training or inference costs. (2) **Language Barrier**: On MTEB, Polish has only 1 model with complete results, Japanese has 11, and Vietnamese has 17, compared to 154 for English and 146 for multilingual tasks. Lower-resource languages receive less attention, creating a vicious cycle. (3) **Transparency Barrier**: Leading models (Qwen3-Embedding, Gemini-Embedding, EmbeddingGemma) are either closed APIs or open-weight without disclosing training data or recipes, making reproducibility difficult.

**Key Challenge**: There is an implicit trade-off between efficiency, performance, and language coverage. Existing MRL-based methods only nest the **output dimension**. However, the true costs of decoder-based models lie in three areas: **Parameters (embedding layers are particularly large in small/multilingual models) + Depth (Transformer layers) + Output Dimension**. Single-dimension nesting is insufficient. While LoRA-based methods save training parameters, they still require loading the full model during inference, failing to solve deployment bottlenecks.

**Goal**: (1) Design a unified framework capable of **simultaneous nested training across three dimensions**, allowing a single training run to produce usable models of various sizes, depths, and dimensions. (2) Build multilingual models using this framework to provide large-scale coverage for low-resource languages. (3) Open-source the data, weights, and code to break transparency barriers.

**Key Insight**: The authors noted a neglected detail: in small decoders like Qwen3-0.6B, the **embedding layer accounts for 1/4 of total parameters** due to the massive multilingual vocabulary. This area, untouched by existing MRL, is low-hanging fruit. Furthermore, Matryoshka Layer Learning (MLL) can address inference depth. Integrating these three aspects into a nested loss creates 3D-ML.

**Core Idea**: **3D-Matryoshka Learning**: In each forward pass, the model simultaneously samples an embedding rank $r'$, network depth $l$, and representation dimension $d'$. The loss function is designed to converge for any combination of these three, yielding a cube-shaped decomposable model space from a single training session.

## Method

### Overall Architecture

ML-Embed employs a two-stage training process within the 3D-ML framework:
- **Data**: 121 public data sources aggregated into 50 million samples, covering 282 natural languages and over 40 programming languages. Data is unified into three contrastive formats (retrieval, clustering, and two-way classification).
- **Training**: Stage 1 builds a semantic foundation on 27 million retrieval samples; Stage 2 performs fine-tuning on 8.3 million mixed samples using task-specific instructions.
- **Objective Function**: Instead of standard InfoNCE, it uses a 3D-ML loss summed over a nested grid of layer × MRL dimensions.
- **Model Family**: A single training run produces six tiers: 140M, 330M, 0.6B, 1.7B, 4B, and 8B.

During each forward pass, the input undergoes: (1) Embedding via an MEL layer with sub-rank $r'$; (2) Processing through the first $l$ transformer layers; (3) Extracting hidden states of each layer through a final LN; (4) Truncating to the first $d'$ dimensions for the contrastive loss. These three samplings occur **simultaneously** during training, forcing the model to provide useful representations under all possible combinations.

### Key Designs

1.  **Matryoshka Embedding Learning (MEL) — Parameter Dimension Nesting**:
    *   **Function**: Uses low-rank decomposition and nested training to compress the massive embedding layer into elastic, deployable matrices, reducing both **trainable** and **total** parameters.
    *   **Mechanism**: Truncated SVD decomposes the original embedding $E \in \mathbb{R}^{v \times d_{model}}$ into $E_A \leftarrow U_r S_r \in \mathbb{R}^{v \times r}$ and $E_B \leftarrow V_r^{\top} \in \mathbb{R}^{r \times d_{model}}$. Only $E_A$ and $E_B$ are updated. The Matryoshka trick involves randomly sampling a sub-rank $r' < r$ from $\{64, 128, 256, 512, 1024\}$ and using $E_{effective} = E_A[:, :r'] E_B[:r', :]$. This forces the model to encode critical information in the leading $r'$ columns. It supports **Compatibility Mode** (reconstructing the standard matrix for zero-code deployment) and **Efficiency Mode** (maintaining low-rank form for reduced memory).
    *   **Design Motivation**: LoRA only saves trainable parameters while requiring the full embedding for inference. MEL saves on both ends. SVD initialization ensures $E_A E_B$ approximates the original matrix, preserving pre-trained knowledge for a smooth transition from full to low-rank fine-tuning.

2.  **Matryoshka Layer Learning (MLL) — Depth Dimension Nesting**:
    *   **Function**: Enables the same weights to be used when truncated at different depths, allowing users to change model size by simply adjusting `num_hidden_layers`.
    *   **Mechanism**: A set of logarithmically spaced layers $\mathcal{L}_{layers} = \{1, 2, 4, 8, 16, 32, L\}$ is selected. For each selected layer $l$, the hidden state $h_l$ passes through $\text{LN}_{final}$ (reusing the final layer norm to maintain consistent scales) before entering the contrastive loss. This is a variant of early-exit training where all exits share the final LN.
    *   **Design Motivation**: Simply removing the last few layers typically leads to severe performance drops because deep semantics are not anchored in shallower layers. Logarithmic intervals force each milestone layer to become a qualified exit point, allowing flexibility in accuracy/latency trade-offs.

3.  **Unified 3D Nested Contrastive Loss + Matryoshka Representation Learning (MRL)**:
    *   **Function**: Combines parameter, depth, and dimension variables into one objective to ensure convergence under any combination.
    *   **Mechanism**: For each selected MLL layer $l$ and each MRL dimension $d' \in \mathcal{D}_{mrl} = \{8, 16, 32, \ldots, d_{model}\}$, truncation and contrastive loss are performed. The representation is $v_{l, d'}(\cdot) = \text{proj}_{d'}(\text{LN}_{final}(h_l(\cdot)))$, and the loss is:
        $$\mathcal{L}_{3D\text{-}ML} = \sum_{l \in \mathcal{L}_{layers}} \sum_{d' \in \mathcal{D}_{mrl}} c_{l, d'} \mathcal{L}_{cl}(q_i, d_i^+, \{d_{i,j}^-\}; v_{l, d'})$$
        where $\mathcal{L}_{cl}$ is standard InfoNCE. The MEL sub-rank $r'$ is sampled per step, while layers and dimensions are enumerated. $c_{l, d'}$ is a tunable weight.
    *   **Design Motivation**: Training separate models costs $N$ times more. Nested training shares the forward pass, updating all exits/dimensions with one gradient. Shared LN is crucial for scale consistency across depths.

### Loss & Training
The total loss is the 3D-ML loss described above. Data is organized into three contrastive formats: retrieval (query, pos, hard negs mined via Qwen3-Embedding-8B), clustering, and two-way classification. Training consists of Stage 1 (27M retrieval samples for semantic foundation) and Stage 2 (8.3M mixed samples with task instructions for fine-tuning).

## Key Experimental Results

### Main Results
Comparison of average scores across 17 MTEB benchmarks against top leaderboard entries:

| Benchmark (# Tasks) | Top-1 | Top-5 | **ML-Embed-8B** | ML-Embed-4B | ML-Embed-1.7B | ML-Embed-0.6B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Multilingual (131) | 72.32 | 69.45 | 66.79 | 65.80 | 63.70 | 61.30 |
| English (41) | 75.97 | 74.61 | 73.26 | 72.89 | 71.19 | 70.01 |
| European (73) | 63.60 | 62.32 | **68.00** | 67.53 | 65.47 | 63.40 |
| Indic (20) | 70.15 | 67.39 | **76.76** | 75.15 | 72.58 | 66.11 |
| German (19) | 59.96 | 55.72 | **66.43** | 65.49 | 63.99 | 61.58 |
| French (25) | 70.37 | 67.25 | **71.91** | 70.97 | 68.94 | 66.64 |
| Polish (17) | 50.95 | n.a. | **73.84** | 73.14 | 71.12 | 68.13 |
| Persian (52) | 71.58 | 65.26 | **71.12 (≈top-1)** | 69.94 | 68.35 | — |
| Vietnamese (50) | 54.74 | 52.37 | **61.62** | 61.20 | 60.27 | — |
| Average | 68.46 | 65.95 | **70.24** | 69.29 | 67.58 | — |

The 8B model ranks 1st in 9 out of 17 benchmarks, with massive gains in low-resource languages: +22.89 in Polish and +6.88 in Vietnamese. Even the 0.6B model achieves 68.13 in Polish, far exceeding the previous Top-1 of 50.95.

### Ablation Study

| Configuration | Avg MTEB | Description |
| :--- | :--- | :--- |
| Full 3D-ML | Best | MEL + MLL + MRL all enabled |
| w/o MEL | Similar, but high cost | No embedding low-rank; significantly higher VRAM |
| w/o MLL | Dropped, loss of flexibility| Inference limited to full depth |
| w/o MRL | Fixed embedding dim | No storage/retrieval flexibility |
| Separate Training| $N \times$ cost | Similar performance but $N \times$ resources |

### Key Findings
*   **Low-resource languages gain the most**: Gains of +22.89 (Polish) and +6.88 (Vietnamese) suggest previous top models lacked sufficient training data for these languages. ML-Embed's "real data distribution" strategy outperforms benchmark-optimized strategies.
*   **Embedding layers are bottlenecks for small models**: MEL compresses Qwen3-0.6B's embedding layer (25% of params) to rank 128, reducing its size to 1/10 without significant MTEB performance loss.
*   **MLL early exits are effective**: The 0.6B model's performance on multilingual benchmarks (61.30) is close to the 1.7B's (63.70), proving shallow exits with shared LN are functional.
*   **3D-ML requires joint sampling**: Nesting must be sampled jointly; independent training of MEL or MLL breaks nested properties (e.g., prefix dimensions are not optimized if only layers are sampled).
*   **Open-source strategy as a contribution**: By providing data, weights, and code, the authors aim to push the community forward, particularly for neglected languages like Polish.

## Highlights & Insights
*   **3D Matryoshka as a concise production abstraction**: While MRL, MLL, and LoRA previously addressed separate concerns, 3D-ML integrates them into a single loss. This "multi-task training, single binary deployment" mode can be extended to all foundation models.
*   **Underrated engineering optimization**: Low-rank embedding layers with Matryoshka nesting cleverly tackle the 25% parameter overhead in multilingual models.
*   **"Real data distribution" vs. "Benchmark optimization"**: Prioritizing actual population and corpus distribution over MTEB-specific tasks leads to superior robustness in long-tail languages.
*   **Shared Final LN**: This detail ensures "representation scale" consistency across depths, preventing contrastive loss instability.

## Limitations & Future Work
*   **Multilingual Ranking**: The 8B model (66.79) still trails the Top-1 (72.32) on the comprehensive multilingual benchmark. The authors note 35/131 tasks are purely English, which favors "pseudo-multilingual" models.
*   **Inference Latency Latency**: Actual throughput/TTFT data for different MLL depths is missing.
*   **MEL Re-factorization Decay**: Performance curves for lower $r''$ values during inference re-factorization are not fully provided.
*   **Training Curriculum**: Whether a specific order (e.g., low-resource followed by high-resource) could further improve performance was not explored.

## Related Work & Insights
*   **vs MRL (Kusupati 2022)**: ML-Embed extends representation nesting to parameters and depth for full lifecycle optimization.
*   **vs Matryoshka Layer Learning (Li 2024)**: ML-Embed integrates MLL and MRL into a joint grid optimization.
*   **vs LoRA/QLoRA**: These only save training parameters; MEL reduces both training and inference footprints.
*   **vs Closed-source SOTAs**: ML-Embed leads in transparency, providing the full recipe unlike Qwen3 or Gemini embeddings.
*   **vs KaLM-Embedding**: ML-Embed offers significantly better coverage of long-tail languages like Arabic and Vietnamese.

## Rating
*   Novelty: ⭐⭐⭐⭐ MEL and 3D joint nesting are novel; MRL/MLL integration is a systematic advancement.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 17 benchmarks and 6 model tiers.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation; however, core ablation details are primarily in the appendix.
*   Value: ⭐⭐⭐⭐⭐ The open-source contribution and low-resource language improvements provide a strong baseline for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Very Efficient Listwise Multimodal Reranking for Long Documents](very_efficient_listwise_multimodal_reranking_for_long_documents.md)
- [\[ICML 2026\] LazyAttention: Efficient Retrieval-Augmented Generation with Deferred Positional Encoding](lazyattention_efficient_retrieval-augmented_generation_with_deferred_positional_.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](../../ICCV2025/information_retrieval/langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ACL 2026\] ReasonEmbed: Enhanced Text Embeddings for Reasoning-Intensive Document Retrieval](../../ACL2026/information_retrieval/reasonembed_enhanced_text_embeddings_for_reasoning-intensive_document_retrieval.md)
- [\[ACL 2026\] CORAL: Adaptive Retrieval Loop for Culturally-Aligned Multilingual RAG](../../ACL2026/information_retrieval/coral_adaptive_retrieval_loop_for_culturally-aligned_multilingual_rag.md)

</div>

<!-- RELATED:END -->
