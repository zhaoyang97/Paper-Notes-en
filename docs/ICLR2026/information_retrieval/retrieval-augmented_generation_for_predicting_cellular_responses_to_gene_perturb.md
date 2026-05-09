---
title: >-
  [Paper Note] Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation
description: >-
  [ICLR2026][RAG] This paper proposes **PT-RAG** (Perturbation-aware Two-stage Retrieval-Augmented Generation), the first application of differentiable retrieval-augmented generation to single-cell gene perturbation response prediction. The framework combines semantic retrieval of candidate perturbations via GenePT embeddings with Gumbel-Softmax-based conditional discrete sampling for cell-type-aware, end-to-end retrieval optimization. PT-RAG surpasses the STATE baseline on the Replogle-Nadig dataset (Pearson 0.633 vs. 0.624), while demonstrating that naïve RAG severely degrades performance (Pearson 0.396 only), establishing that **differentiable, cell-type-aware retrieval** is indispensable in this domain.
tags:
  - ICLR2026
  - RAG
  - differentiable retrieval
  - gene perturbation prediction
  - single-cell transcriptomics
  - Gumbel-Softmax
  - cell-type-aware
date: 2026-05-08
content_hash: 0903d2dac0d63d3a
---

# Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation

**Conference**: ICLR2026  
**arXiv**: [2603.07233](https://arxiv.org/abs/2603.07233)  
**Authors**: Andrea Giuseppe Di Francesco, Andrea Rubbi, Pietro Liò (Sapienza University of Rome / University of Cambridge / Wellcome Sanger Institute)  
**Code**: [github.com/difra100/PT-RAG_ICLR](https://github.com/difra100/PT-RAG_ICLR)  
**Area**: Information Retrieval  
**Keywords**: RAG, differentiable retrieval, gene perturbation prediction, single-cell transcriptomics, Gumbel-Softmax, cell-type-aware

---

## TL;DR

This paper proposes **PT-RAG** (Perturbation-aware Two-stage Retrieval-Augmented Generation), the first application of differentiable retrieval-augmented generation to single-cell gene perturbation response prediction. The framework combines semantic retrieval of candidate perturbations via GenePT embeddings with Gumbel-Softmax-based conditional discrete sampling for cell-type-aware, end-to-end retrieval optimization. PT-RAG surpasses the STATE baseline on the Replogle-Nadig dataset (Pearson 0.633 vs. 0.624), while demonstrating that naïve RAG severely degrades performance (Pearson 0.396 only), establishing that **differentiable, cell-type-aware retrieval** is indispensable in this domain.

---

## Background & Motivation

**Importance of gene perturbation prediction**: Understanding cellular responses to gene perturbations is a central challenge in systems biology, with critical implications for drug discovery, disease modeling, and gene therapy. The combinatorial explosion of high-throughput Perturb-seq data renders exhaustive experimental screening infeasible, necessitating computational prediction methods.

**Limitations of prior work**: Methods such as scGen, CPA, GEARS, and STATE generate predictions solely from control cell states and perturbation identity, without leveraging biological knowledge from related perturbations, which limits generalization to unseen cell types.

**The gap beyond NLP RAG**: While RAG has proven highly successful in NLP, extending it to cell biology poses fundamental challenges—no pretrained retriever exists, no canonical perturbation similarity metric is established, and the "generator" must produce high-dimensional cellular distributions rather than text.

**Risk of naïve retrieval**: Because the same perturbation can elicit drastically different effects across cell types, cell-type-agnostic fixed retrieval provides identical context regardless of cellular state, potentially introducing noise rather than useful information.

**Necessity of differentiable retrieval**: When the retrieval objective itself must be learned (with no prior definition of "relevance"), end-to-end differentiable retrieval becomes essential.

**Core Idea**: Functionally similar perturbations should induce similar cellular responses; differentiable retrieval enables the model to **learn** to dynamically select the most informative reference perturbations for each cellular context, rather than retrieving blindly.

---

## Method

### Overall Architecture: PT-RAG Two-Stage Retrieval-Augmented Generation

PT-RAG adopts a two-stage pipeline: (1) semantic retrieval of candidates using GenePT embeddings to narrow the search space; (2) a differentiable, cell-type-aware selection mechanism based on Gumbel-Softmax.

### Perturbation Representation: GenePT Embeddings

Rather than using one-hot vectors as in prior work, PT-RAG employs GenePT embeddings $h_g^{gpt} \in \mathbb{R}^d$, obtained by encoding NCBI gene descriptions with GPT-3.5, such that functionally similar genes are positioned closer in embedding space. A perturbation database $\mathcal{D} = \{h_p^{gpt}; \forall p \in \mathcal{P}\}$ is constructed (with $|\mathcal{P}| \approx 2009$ in experiments).

### Stage 1: Semantic Retrieval

Top-$K$ candidate perturbations are retrieved via cosine similarity over GenePT embeddings (non-differentiable):

$$\mathcal{R}_{p^{pert}} = \text{TOP}_K(h_{pert}^{gpt}, \mathcal{P}) = \{p_{(1)}, p_{(2)}, \ldots, p_{(K)}\}$$

This reduces the search space from ~2009 perturbations to $K$ semantically relevant candidates.

### Stage 2: Differentiable Retrieval (Key Innovation)

**Encoding**: For each candidate perturbation $k$, an embedding is obtained as $h_k^{cxt} = \text{PertEncoder}(h_{p_{(k)}}^{gpt}) \in \mathbb{R}^{d_h}$.

**Triplet scoring**: A triplet $c_k = [h^{ctrl}; h_{pert}; h_k^{cxt}]$ is constructed to capture the relationship among the cell state, target perturbation, and candidate context:

$$s_k = \text{MLP}_{\text{score}}(\text{LayerNorm}(c_k)) \in \mathbb{R}^2$$

Two logits are output, corresponding to "exclude" and "include."

**Gumbel-Softmax discrete selection**: Straight-Through Gumbel-Softmax is applied to obtain hard binary decisions while preserving differentiability:

$$w_k = \text{GumbelSoftmax}(s_k, \tau)[\texttt{include}] \in \{0, 1\}$$

The forward pass uses $\arg\max$ for hard selection; gradients are computed through soft probabilities in the backward pass.

### Context Aggregation and Generation

Each triplet is projected as $h_k' = \text{MLP}_{\text{proj}}(c_k)$, then aggregated with selection weights:

$$z = \sum_{k=1}^{K} w_k \cdot h_k'$$

The aggregated context is passed to a Transformer Generator to produce the predicted post-perturbation expression profile $\hat{x}^{pert} = \text{TransformerGenerator}(z)$.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{dist}} + \lambda_{\text{sparse}} \mathcal{L}_{\text{sparse}}$$

- **Distribution loss**: Energy Distance $\mathcal{L}_{\text{dist}} = \text{Energy}(\hat{x}^{pert}, x^{pert})$
- **Sparsity regularization**: $\mathcal{L}_{\text{sparse}} = \frac{1}{K}\sum_{k=1}^{K} w_k$, with $\lambda_{\text{sparse}} = 0.1$, preventing mode collapse from selecting all candidates

### Key Differences from Vanilla RAG

| Property | Vanilla RAG | PT-RAG |
|----------|-------------|--------|
| Retrieval | Fixed top-K, non-differentiable | Two-stage; Stage 2 differentiable |
| Cell-type awareness | ❌ Perturbation embedding only | ✅ Conditioned on $[h^{ctrl}; h_{pert}; h_k^{cxt}]$ |
| Context integration | Cross-Attention | Weighted sum (hard selection) |
| Gradient flow | Truncated at retrieval step | End-to-end optimization |

---

## Key Experimental Results

### Dataset & Setup

- **Dataset**: Replogle-Nadig single-gene perturbation dataset; 2,009 unique perturbations; 2,000 highly variable genes
- **Cell types**: K562 (chronic myelogenous leukemia), Jurkat (T-cell lymphoma), RPE1 (retinal pigment epithelium), HepG2 (hepatocellular carcinoma)
- **Evaluation protocol**: Leave-one-cell-type-out cross-validation—trained on 3 cell types, tested on the 4th; 30% few-shot samples from the target cell type, 70% for validation/testing
- **Statistical testing**: Mann-Whitney U test + Benjamini-Hochberg FDR correction

### Main Results

**Table 1: Cross-cell-type generalization performance (mean over 1,635 test perturbations)**

| Metric | STATE | STATE+GenePT | Vanilla RAG | **PT-RAG** |
|--------|-------|-------------|-------------|-----------|
| Pearson DEG ↑ | 0.624† | 0.631 | 0.396† | **0.633** |
| Spearman DEG ↑ | 0.403† | 0.411 | 0.307† | **0.412** |
| MSE ↓ | 0.211 | **0.210** | 0.316† | **0.210** |
| RMSE ↓ | 0.458 | 0.458 | 0.562† | **0.457** |
| MAE ↓ | 0.298† | 0.296 | 0.429† | **0.295** |
| MSE_PCA50 ↓ | 8.43 | 8.42 | 12.64† | **8.39** |
| $W_1$ ↓ | 35.70† | 35.53†† | 48.48† | **35.41** |
| $W_2$ ↓ | 646.1† | 638.7†† | 1189.5† | **633.7** |
| Energy ↓ | 9.41†† | 9.40 | 14.18† | **9.33** |

> † indicates $p<0.01$ after FDR correction; †† $p<0.05$

### Ablation Study: Retrieval Strategy Comparison

**Table 2: Pearson correlation of Vanilla RAG vs. PT-RAG across different values of K**

| K | Vanilla RAG | PT-RAG |
|---|------------|--------|
| 2 | ~0.29 | ~0.62 |
| 5 | ~0.31 | ~0.63 |
| 10 | ~0.33 | ~0.63 |
| 32 | 0.351 | **0.633** |

Regardless of $K$, Vanilla RAG consistently falls far below the baseline; differentiable retrieval is the key factor driving performance recovery.

### Cell-Type-Specific Retrieval Analysis

- Jaccard similarity is computed over 33 genes shared across all 4 cell types
- Pairwise Jaccard similarity across all cell-type pairs is only **0.185–0.196** (mean 0.191), indicating that only ~19% of retrieved perturbations overlap between cell types
- **WARS gene case study**: PT-RAG consistently retrieves aminoacyl-tRNA synthetases (functionally coherent), but the specific selections vary by cell type—Jurkat selects EARS2/DARS/VARS; HepG2 selects SARS2/GART/TARS; K562 selects FARSB/KARS/FARS2; RPE1 selects KARS/GART/TARS/QARS

### Key Findings

1. **Naïve RAG severely degrades performance**: Vanilla RAG achieves a Pearson of only 0.396 (far below STATE without retrieval at 0.624) and $W_2$ of 1189.5 (vs. 646.1 for STATE), demonstrating that indiscriminate retrieval introduces substantial noise.
2. **GenePT embeddings provide modest gains**: STATE+GenePT marginally outperforms STATE on most metrics, with improvements primarily attributable to improved semantic representation.
3. **PT-RAG gains concentrate on distributional similarity**: Improvements in $W_1$ (35.41 vs. 35.70) and $W_2$ (633.7 vs. 646.1) are the most pronounced and statistically significant, indicating that differentiable retrieval primarily helps capture cellular population heterogeneity and distributional structure.
4. **Cell-type awareness is essential**: The ~19% retrieval overlap confirms that the model genuinely learns to select different reference information for different cellular contexts.

---

## Highlights & Insights

- **Pioneer contribution**: This is the first work to extend the RAG paradigm to perturbation response modeling in cell biology, demonstrating that differentiable retrieval is a necessity rather than an option in this domain.
- **Value of negative results**: The failure of Vanilla RAG is itself a significant finding, revealing fundamental differences in RAG requirements across domains.
- **Elegant end-to-end differentiable design**: The combination of triplet scoring and Gumbel-Softmax is concise yet effective, enabling joint optimization of retrieval and generation objectives.
- **Biological interpretability**: The Jaccard analysis and WARS case study demonstrate that the learned retrieval patterns are biologically plausible.

---

## Limitations & Future Work

1. **Computational overhead**: PT-RAG incurs approximately 1.7× the FLOPs per batch compared to the baseline, attributable to the scoring and Gumbel-Softmax mechanisms.
2. **Single-gene perturbations only**: The framework has not been validated for combinatorial perturbations (multi-gene knockouts), chemical compounds, or CRISPR activation/interference settings.
3. **Modest absolute gains**: Compared to STATE+GenePT, the absolute improvements of PT-RAG are small and concentrated primarily in Wasserstein distance metrics.
4. **Limited biological validation**: Analysis of cell-type-specific retrieval patterns is largely qualitative and lacks rigorous experimental biological validation.
5. **Restricted retrieval mechanism**: GraphRAG (exploiting gene regulatory network structure) and multimodal retrieval (integrating sequence, structure, and functional annotations) remain unexplored.

---

## Related Work & Insights

- **Perturbation prediction methods**: scGen (Lotfollahi 2019), CPA (Lotfollahi 2023), GEARS (Roohani 2024), CellOT (Bunne 2023), STATE (Adduri 2025), and CellFlow (Klein 2025) all generate predictions from cell state and perturbation identity without leveraging related perturbation knowledge.
- **Differentiable RAG**: Stochastic RAG (Zamani & Bendersky 2024) and D-RAG (Gao 2025) demonstrate the value of end-to-end optimization in the text domain, but rely on mature similarity metrics and pretrained generators.
- **RAG in biology**: GeneRAG (Lin 2024), scRAG (Yu 2025) use LLMs to retrieve textual annotations; E1 (Jain 2025) augments protein encoders—none address cellular response generation.

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First application of differentiable RAG to cellular perturbation prediction; strong cross-domain innovation
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive multi-metric evaluation, rigorous statistical testing, detailed ablation and Jaccard analysis
- **Writing Quality**: ⭐⭐⭐⭐ — Clear comparison across three method variants; professional figures and tables
- **Value**: ⭐⭐⭐⭐ — Introduces a new tool for computational biology; negative results carry meaningful guidance
- **Overall Recommendation**: ⭐⭐⭐⭐ — Cross-domain innovation combined with in-depth analysis of negative results; modest absolute gains but valuable insights

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Feedback Adaptation for Retrieval-Augmented Generation](../../ACL2026/information_retrieval/feedback_adaptation_for_retrieval-augmented_generation.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[NeurIPS 2025\] Chain-of-Retrieval Augmented Generation (CoRAG)](../../NeurIPS2025/information_retrieval/chain-of-retrieval_augmented_generation.md)
- [\[ICLR 2026\] Attributing Response to Context: A Jensen-Shannon Divergence Driven Mechanistic Study of Context Attribution in Retrieval-Augmented Generation](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)
- [\[NeurIPS 2025\] Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations](../../NeurIPS2025/information_retrieval/retrieval-augmented_generation_for_reliable_interpretation_of_radio_regulations.md)

</div>

<!-- RELATED:END -->
