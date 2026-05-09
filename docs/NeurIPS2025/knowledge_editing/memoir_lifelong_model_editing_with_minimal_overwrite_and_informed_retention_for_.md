---
title: >-
  [Paper Note] MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs
description: >-
  [NeurIPS 2025][Knowledge Editing][Model Editing] MEMOIR introduces a framework that incorporates zero-initialized residual memory matrices into FFN layers, employs TopHash-based sparse masks to confine each edit to a distinct subset of memory parameters, and at inference time conditionally activates stored knowledge by measuring mask overlap. The approach achieves an optimal balance among reliability, generalization, and locality across 15,000 sequential edits.
tags:
  - NeurIPS 2025
  - Knowledge Editing
  - Model Editing
  - Lifelong Learning
  - Sparse Masking
  - Residual Memory
  - Catastrophic Forgetting
date: 2026-05-08
content_hash: bdbcdd9012fc2ec9
---

# MEMOIR: Lifelong Model Editing with Minimal Overwrite and Informed Retention for LLMs

**Conference**: NeurIPS 2025  
**arXiv**: [2506.07899](https://arxiv.org/abs/2506.07899)  
**Code**: [https://github.com/qym7/MEMOIR](https://github.com/qym7/MEMOIR)  
**Area**: LLM Efficiency / Knowledge Editing  
**Keywords**: Model Editing, Lifelong Learning, Sparse Masking, Residual Memory, Catastrophic Forgetting

## TL;DR
MEMOIR introduces a framework that incorporates zero-initialized residual memory matrices into FFN layers, employs TopHash-based sparse masks to confine each edit to a distinct subset of memory parameters, and at inference time conditionally activates stored knowledge by measuring mask overlap. The approach achieves an optimal balance among reliability, generalization, and locality across 15,000 sequential edits.

## Background & Motivation
**State of the Field**: Deployed LLMs require continuous knowledge updates (correcting factual errors, incorporating new information), yet full fine-tuning is costly and prone to forgetting. Model editing methods fall into two categories: non-parametric methods (e.g., GRACE, which stores fixed input–output activation patterns) and parametric methods (e.g., ROME/MEMIT/AlphaEdit, which inject knowledge by modifying model parameters).

**Limitations of Prior Work**: Non-parametric methods are precise and locality-preserving but generalize poorly—they cannot handle semantically similar yet differently phrased queries. Parametric methods generalize better but suffer from **new edits overwriting parameter updates of prior edits** during sequential editing, leading to catastrophic forgetting. MEMIT's reliability drops to near zero after 100 edits; ROME becomes nearly completely ineffective after 1,000 edits.

**Root Cause**: There exists a fundamental trade-off among reliability, generalization, and locality. GRACE achieves near-perfect reliability and locality at the cost of severely degraded generalization (only 0.37 after 1,000 edits); AlphaEdit generalizes reasonably well but its locality degrades sharply under large-scale editing (0.56 after 1,000 edits).

**Paper Goals**: Design a lifelong model editing framework that simultaneously maintains high scores across all three metrics and scales to tens of thousands of sequential edits.

**Starting Point**: Drawing inspiration from continual learning, where **sparsity mitigates forgetting**—if each edit modifies only a small subset of the parameter space and different edits use different subsets, interference is substantially reduced. The key innovation is the TopHash mechanism, which automatically assigns semantically consistent sparse masks.

**Core Idea**: A sparse mask is generated via top-$k$ selection of input activations combined with a fixed random permutation; during editing, gradient updates are restricted to the parameter columns corresponding to the mask; at inference, mask matching determines whether the residual memory is activated.

## Method

### Overall Architecture
MEMOIR introduces a **residual memory matrix** $\mathbf{W}_m$ (zero-initialized, same shape as the original projection matrix $\mathbf{W}_0$) into the FFN layer of a single intermediate Transformer block. During editing, a sparse mask $\mathcal{M}$ restricts input activations to $k \ll D$ active dimensions, and only the corresponding columns of $\mathbf{W}_m$ are updated. At inference, the overlap between the input mask and all masks stored in the edit database determines whether the residual memory is activated. The final output is: $\text{FFN}_{\text{edited}}(\mathbf{a}) = \mathbf{W}_0 \mathbf{a} + \mathbf{W}_m (\mathcal{M} \odot \mathbf{a})$.

### Key Designs

1. **TopHash Sparse Masking Mechanism**:

    - **Function**: Generates a sample-dependent binary mask for each input, determining which parameter columns participate in editing/inference.
    - **Mechanism**: The input activation $\mathbf{a}$ is centered (by subtracting the mean over unrelated samples); the top-$k$ most activated dimensions form $\mathcal{T}(\mathbf{a})$; a fixed random permutation $\pi$ is then applied to yield the final mask $\mathcal{M}(\mathbf{a}) = \pi(\mathcal{T}(\mathbf{a}))$.
    - **Design Motivation**: Top-$k$ selection ensures **semantic consistency** (semantically similar inputs produce similar activation patterns, hence similar masks); the random permutation introduces **diversity** (preventing all edits from concentrating on the same salient features, thereby reducing mutual interference). Centering removes "globally salient" features that are large across all inputs, making masks more discriminative.
    - This is essentially a form of Locality-Sensitive Hashing (LSH), mapping semantically similar inputs to identical or near-identical binary codes.

2. **Editing Phase: Sparse Distributed Knowledge Storage**:

    - **Function**: Each edit updates only the $k$ columns of $\mathbf{W}_m$ corresponding to the mask.
    - **Mechanism**: For the $t$-th edit $(\mathbf{x}_e^t, \mathbf{y}_e^t)$, the mask $\mathcal{M}(\mathbf{a}(\mathbf{x}_e^t))$ is computed, and gradient updates to $\mathbf{W}_m$ are performed under a masked forward pass. Only columns where the mask equals 1 receive gradients.
    - **Design Motivation**: Different edits use different parameter subsets, distributing stored knowledge sparsely across the parameter space and minimizing overwriting. This contrasts with methods such as MEMIT that apply unconstrained updates over the entire parameter space. No large set of unrelated samples is required for regularization (MEMIT requires 100K samples; AlphaEdit also relies on unrelated samples), since locality is ensured by conditional activation at inference time.

3. **Inference Phase: Conditional Knowledge Activation**:

    - **Function**: Determines whether a query input corresponds to an edited sample, a paraphrased variant, or an unrelated sample, and activates the residual memory accordingly.
    - **Mechanism**: An edit mask database is maintained. For a new input $\mathbf{x}$, its mask is computed and the closest match $\mathbf{x}_{\text{match}}$ in the database (by Hamming distance) is retrieved; the overlap ratio $R_{\text{match}} = \frac{1}{N}\|\mathcal{M}(\mathbf{a}(\mathbf{x})) \wedge \mathcal{M}(\mathbf{a}(\mathbf{x}_{\text{match}}))\|_1$ is then calculated.
    - If $R_{\text{match}} \geq \tau$ (threshold), the residual memory is activated using the matched edit's mask; otherwise $\mathbf{W}_m$ is bypassed entirely and only $\mathbf{W}_0$ is used.
    - **Key Detail**: For paraphrased variants, the **matched edit's mask** (rather than the query's own mask) is used, enabling paraphrased inputs to precisely retrieve the knowledge stored in the original edit's corresponding columns.

### Loss & Training
Standard cross-entropy loss is applied to $\mathbf{W}_m$ for a few gradient descent steps per edit. All original model parameters are frozen. Hyperparameters include sparsity $k/D$ and overlap threshold $\tau$.

## Key Experimental Results

### Main Results (ZsRE QA dataset, LLaMA-3-8B-Instruct)

| Method | T=1 Avg | T=100 Avg | T=1000 Avg | Notes |
|--------|---------|-----------|------------|-------|
| Fine-tuning | 0.54 | 0.09 | 0.09 | Complete collapse |
| ROME | 0.97 | 0.06 | 0.03 | Fails after ~100 edits |
| MEMIT | 0.98 | 0.02 | 0.00 | Drops to zero after ~100 edits |
| GRACE | 0.82 | 0.80 | 0.79 | Consistently poor generalization |
| WISE | 0.92 | 0.74 | 0.77 | Insufficient reliability |
| AlphaEdit | 0.96 | 0.88 | 0.72 | Locality degrades after 1,000 edits |
| **MEMOIR** | **1.00** | **0.95** | **0.93** | Best across all metrics |

### Ablation Study

| Configuration | Rel. | Gen. | Loc. | Avg. | Notes |
|---------------|------|------|------|------|-------|
| Full MEMOIR | 0.94 | 0.85 | 1.00 | 0.93 | Complete model |
| w/o permutation $\pi$ | 0.90 | 0.81 | 0.99 | 0.90 | Removing random permutation degrades reliability and generalization |
| w/o top-$k$ (random mask) | 0.96 | 0.41 | 1.00 | 0.79 | Generalization collapses—semantic consistency lost |
| w/o conditional routing | 0.94 | 0.79 | 0.47 | 0.73 | Locality degrades substantially |

### Key Findings
- MEMOIR is the only method to maintain all three metrics above 0.85 under 1,000 sequential edits.
- Performance remains competitive at 15,000 edits, far exceeding all baselines.
- Both components of TopHash (top-$k$ selection and permutation) are indispensable: top-$k$ ensures generalization; permutation ensures diversity.
- Conditional routing is critical for locality (drops from 1.00 to 0.47 when removed).
- MEMOIR achieves top performance across four backbones: LLaMA-3, Mistral, LLaMA-2, and GPT-J.
- The method also reaches state-of-the-art performance on hallucination correction and OOD generalization benchmarks.

## Highlights & Insights
- **The TopHash mechanism** is particularly elegant: top-$k$ selection serves as a natural LSH, since LLM activation patterns inherently encode semantics—this property is exploited for knowledge routing without requiring an auxiliary embedding model or retriever. The random permutation is especially insightful, remapping the "most salient but most congested" features onto "less popular" parameter columns.
- **Zero initialization combined with conditional routing** ensures zero impact on unrelated inputs ($\mathbf{W}_m$ is completely bypassed), achieving perfect locality without any unrelated-sample regularization.
- The editing phase requires only a few gradient descent steps and avoids the costly 100K-sample forward passes needed by MEMIT to estimate covariance matrices, yielding substantially higher computational efficiency.

## Limitations & Future Work
- As the number of edits grows, Hamming distance search over the mask database may slow down (though binary vector search is relatively fast); approximate nearest neighbor search may be necessary at very large scales (millions of edits).
- The sparsity ratio $k/D$ is a hyperparameter requiring tuning—too large increases interference; too small limits expressive capacity.
- Only a single FFN layer's $\mathbf{W}_{\text{proj}}$ is edited; complex knowledge updates requiring modifications across multiple layers may not be adequately handled.
- The choice of threshold $\tau$ is performance-sensitive, with optimal values varying across datasets.
- Performance on true multi-hop reasoning scenarios (requiring multiple edited knowledge pieces to work in concert) has not been evaluated.

## Related Work & Insights
- **vs. GRACE**: GRACE also relies on non-parametric storage but uses exact matching, leading to poor generalization. MEMOIR employs sparse masks for "fuzzy matching," achieving both reliability and generalization.
- **vs. AlphaEdit**: AlphaEdit edits within the null space to preserve locality, but the null space becomes exhausted under long edit sequences. MEMOIR's sparse parameter allocation fundamentally reduces interference.
- **vs. WISE**: WISE introduces external memory but uses a merging strategy to handle conflicts, resulting in insufficient reliability over long sequences. MEMOIR's conditional routing is more precise.
- **vs. Continual Learning Methods**: MEMOIR draws inspiration from sparse subnetwork approaches such as PackNet/SupSup, but TopHash enables sample-dependent mask assignment without any training overhead.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of TopHash and conditional routing is novel, though the individual components (sparsity, LSH, residual connections) are not new in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated across 4 models, multiple benchmarks, 15,000 sequential edits, and detailed ablations—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated, method descriptions are precise, and figures are intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Lifelong model editing is a practical necessity for deployed systems; MEMOIR represents a qualitative advance in scalability.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)
- [\[NeurIPS 2025\] Rethinking Residual Distribution in Locate-then-Edit Model Editing](rethinking_residual_distribution_in_locate-then-edit_model_editing.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](../../ICLR2026/knowledge_editing/fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](../../ICLR2026/knowledge_editing/energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] Rote Learning Considered Useful: Generalizing over Memorized Data in LLMs](../../ICLR2026/knowledge_editing/rote_learning_considered_useful_generalizing_over_memorized_data_in_llms.md)

<!-- RELATED:END -->
