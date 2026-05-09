---
title: >-
  [Paper Note] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization
description: >-
  [ICLR 2026][Knowledge Editing][Massive model editing] This paper identifies the root cause of large-scale model editing failures as structural inconsistency (embedding misalignment) between key embeddings and residual embeddings, and proposes EAMET, which progressively saves optimized residual embeddings and aligns their neighborhood structure to the key embedding space via a dual KL divergence + MSE loss. EAMET outperforms MEMIT by an average of 14% (CounterFact) and 8% (ZsRE) when simultaneously editing 10k facts across 6 LLMs and 3 datasets, while remaining robust in two challenging scenarios: long-prefix inputs and multi-fact editing under the same subject.
tags:
  - ICLR 2026
  - Knowledge Editing
  - Massive model editing
  - embedding alignment
  - MEMIT
  - structural inconsistency
date: 2026-05-08
content_hash: 54b56715220d03fb
---

# EAMET: Robust Massive Model Editing via Embedding Alignment Optimization

**Conference**: ICLR 2026
**arXiv**: [2505.11876](https://arxiv.org/abs/2505.11876)
**Code**: [https://github.com/ybdai7/EAMET-massive-editing](https://github.com/ybdai7/EAMET-massive-editing)
**Area**: LLM NLP / Model Editing
**Keywords**: Massive model editing, embedding alignment, MEMIT, knowledge editing, structural inconsistency

## TL;DR

This paper identifies the root cause of large-scale model editing failures as structural inconsistency (embedding misalignment) between key embeddings and residual embeddings, and proposes EAMET, which progressively saves optimized residual embeddings and aligns their neighborhood structure to the key embedding space via a dual KL divergence + MSE loss. EAMET outperforms MEMIT by an average of 14% (CounterFact) and 8% (ZsRE) when simultaneously editing 10k facts across 6 LLMs and 3 datasets, while remaining robust in two challenging scenarios: long-prefix inputs and multi-fact editing under the same subject.

## Background & Motivation

**Background**: As LLM knowledge becomes outdated post-deployment, model editing techniques aim to revise specific facts without full retraining. Locate-then-edit methods such as MEMIT and PMET modify FFN weights directly to support batch editing, claiming the ability to edit tens of thousands of facts simultaneously.

**Limitations of Prior Work**: Existing methods are overestimated by overly lenient evaluation metrics—these metrics only check whether the target token probability exceeds that of the original token, rather than whether the model actually generates the target entity. Under stricter "practical metrics" (requiring the model's output to precisely contain the target entity), performance degrades sharply for large-scale editing (>1000 edits). Two additional robustness issues arise in practical settings: (a) prepending a 50-token descriptive prefix to edited knowledge drops MEMIT accuracy on LLaMA2-7B from 98.5% to 77.4%; (b) simultaneously editing multiple facts sharing the same subject causes mutual interference and editing failures.

**Key Challenge**: When a large number of facts are jointly edited, the "neighborhood structure" of each fact's residual embedding $r_i$ (the difference between the target memory and original weights) diverges from that of its key embedding $k_i$ (the FFN layer input representation)—i.e., the pairwise similarity ordering among $\{r_i\}$ becomes inconsistent with that among $\{k_i\}$. This misalignment causes information loss in per-fact reconstruction when solving the normal equations jointly.

**Goal**: To maintain per-fact embedding space structural consistency during large-scale batch editing (10k+), thereby preserving high editing success rates and robustness under strict evaluation metrics.

**Key Insight**: The authors proceed from both theoretical and empirical directions. Theoretically, they derive an upper bound on the reconstruction error per fact: $\|e_i\| \leq C_i\sqrt{\frac{1}{2}\mathcal{A}(i)} + |\beta_{ii}|\|r_i\| + \|\varepsilon_i\|$, where $\mathcal{A}(i)$ is the misalignment score. Empirically, on LLaMA2-7B, increasing the number of edits from 200 to 1000 causes the total misalignment score to rise from 79 to 554, while accuracy drops from 98.5% to 86.8%—a strong correlation that validates the theory.

**Core Idea**: During optimization of the target memory for each fact, progressively save the already-optimized residual embeddings and apply a KL divergence + MSE dual loss to constrain the neighborhood structure of each residual to be consistent with the key embedding space.

## Method

### Overall Architecture

EAMET follows the locate-then-edit paradigm of MEMIT. The input is a batch of fact triples $(s_i, rel_i, o_i)$ to be edited, and the output is a parameter update $\Delta$ to the FFN layer $W_{out}^l$. Unlike MEMIT, which jointly optimizes all residuals at once, EAMET iteratively optimizes each fact's residual $r_i$ one at a time, incorporating embedding alignment constraints during optimization. The pipeline consists of three steps: (a) pre-extract key embeddings for all facts and compute pairwise cosine similarity distributions; (b) optimize residual embeddings sequentially—after each residual is optimized it is saved, and subsequent optimizations use previously saved residuals to compute the alignment loss; (c) substitute the aligned residuals into the normal equations to solve for $\Delta$.

### Key Designs

1. **Theoretical Formalization of Embedding Misalignment**:

    - Function: Define and quantify the root cause of performance degradation in large-scale editing.
    - Mechanism: For each fact $i$, collect the cosine similarity distribution $P_r^{(i)}$ between its residual $r_i$ and all other residuals, and the distribution $P_k^{(i)}$ between its key $k_i$ and all other keys. The misalignment is quantified as $\mathcal{A}(i) = KL(P_r^{(i)} \| P_k^{(i)})$. Theorem 1 proves that the reconstruction error upper bound for each fact is proportional to $\sqrt{\mathcal{A}(i)}$. Intuitively, if the nearest neighbors of $r_i$ are $r_3, r_7$, but those of $k_i$ are $k_5, k_9$, then jointly solving $\Delta k_i = r_i$ forces $\Delta$ to combine $k_i$ in incorrect directions, causing reconstruction failure.
    - Design Motivation: Prior work observed that "more edits leads to failure" but provided no quantitative explanation. This formalization identifies a concrete, measurable, and optimizable objective $\mathcal{A}(i)$—reducing it directly reduces the reconstruction error upper bound.

2. **Progressive Residual Saving with KL+MSE Dual-Loss Alignment**:

    - Function: Constrain the spatial structure of each fact's residual during optimization to be consistent with the key embedding space.
    - Mechanism: When optimizing fact $i$ sequentially, the residuals $\{r_j \mid j < i\}$ from previous iterations are already saved. The cosine similarity distribution $P_r^{(i)}$ between $r_i$ and these saved residuals is compared against the corresponding key-side distribution $\bar{P}_k^{(i)}$. The alignment loss has two components: $L_{KL}(i) = KL(P_r^{(i)} \| \bar{P}_k^{(i)})$ for global distributional alignment, and $L_{MSE}(i) = \frac{1}{M} \sum_{j=1}^M \|P_r^{(i,j)} - P_k^{(i,j)}\|^2$ for precise matching of the top-$M$ nearest neighbors in key space. The two terms are complementary—KL governs the overall distributional shape while MSE enforces accurate alignment of the most critical neighbors.
    - Design Motivation: KL alone focuses on distributional divergence and provides insufficient precision for a small number of critical neighbors; MSE alone addresses local structure but ignores the global distribution. Ablation studies confirm that combining both outperforms using either in isolation.

3. **Target Memory Optimization with Prefix Augmentation**:

    - Function: Optimize each fact's residual vector $r_i$ so that the model correctly generates the target entity under diverse prefixes.
    - Mechanism: The total loss is $r_i = \arg\min_{r_i} \left( \frac{1}{N_{FP}} \sum_j -\log P_{G(h_i^L += r_i)}[o_i | f_j \oplus tp(s_i, rel_i)] + \lambda_{KL} L_{KL}(i) + \lambda_{MSE} L_{MSE}(i) \right)$. The first term is the standard NLL loss ensuring the model predicts the target object $o_i$, where $f_j$ are randomly sampled prefixes that encourage learning a more generalizable memory representation; the latter two terms are the alignment regularizers.
    - Design Motivation: The original MEMIT also uses prefix sampling when optimizing $r_i$, but applies no alignment constraints, allowing the optimized residuals to drift freely in the embedding space. The alignment regularizers anchor residuals to positions that are structurally consistent with key space, reducing reconstruction error when substituted into the normal equations.

### Loss & Training

Total loss = NLL editing loss (with prefix augmentation) + $\lambda_{KL} \cdot L_{KL}$ + $\lambda_{MSE} \cdot L_{MSE}$. Optimization proceeds iteratively: optimize fact $i$ → save $r_i$ → when optimizing fact $i+1$, use the first $i$ saved residuals to compute the alignment loss. Parameter updates are still solved in one step via MEMIT's normal equations $\Delta(C_p + K_t K_t^T) = R K_t^T$, with $R = [r_1 | r_2 | \ldots | r_N]$ replaced by the alignment-optimized residual matrix.

## Key Experimental Results

### Main Results (10k fact editing, 6 LLMs, CounterFact dataset)

| Model | Method | Eff.(%)↑ | Gen.(%)↑ | Spe.(%)↑ | Flu.↑ |
|------|------|----------|----------|----------|-------|
| LLaMA2-7B | MEMIT | 24.95 | 22.68 | 63.84 | 506.69 |
| LLaMA2-7B | PMET | 74.22 | 46.45 | 72.47 | 507.10 |
| LLaMA2-7B | **EAMET** | **89.09** | **61.21** | 72.19 | 519.89 |
| LLaMA2-13B | MEMIT | 47.98 | 34.75 | 71.61 | 517.63 |
| LLaMA2-13B | **EAMET** | **92.85** | **60.08** | 77.51 | 530.78 |
| Deepseek-7B | MEMIT | 62.11 | 42.01 | 78.04 | 512.16 |
| Deepseek-7B | **EAMET** | **89.74** | **59.98** | 77.73 | 513.93 |
| Falcon-7B | MEMIT | 89.21 | 60.85 | 77.56 | 519.92 |
| Falcon-7B | **EAMET** | **92.37** | **63.91** | **78.94** | 528.98 |
| LLaMA3-8B | MEMIT | 93.76 | 61.98 | 77.69 | 526.47 |
| LLaMA3-8B | **EAMET** | **93.87** | **63.74** | **79.07** | 533.30 |
| Qwen2.5-7B | MEMIT | 90.06 | 63.86 | 70.53 | 529.27 |
| Qwen2.5-7B | **EAMET** | **90.49** | **64.37** | **72.18** | 536.67 |

### Misalignment Score Comparison (10k edits)

| Model | EAMET (CF/ZS) | MEMIT (CF/ZS) | PMET (CF/ZS) |
|------|---------------|---------------|--------------|
| LLaMA2-7B | 377 / 165 | 11506 / 22245 | 11475 / 11477 |
| Qwen-7B | 374 / 180 | 18498 / 23699 | 18471 / 18463 |
| Deepseek-7B | 520 / 161 | 12135 / 23241 | 12155 / 12046 |
| Falcon-7B | 385 / 181 | 8564 / 17589 | 8602 / 8590 |

### Prefix Robustness (200 edits, LLaMA2-7B)

| Prefix Length | MEMIT Accuracy | EAMET Accuracy | Low $\mathcal{A}$ Group | High $\mathcal{A}$ Group |
|----------|-------------|-------------|-------------------|-------------------|
| 0 token | 98.50% | ~99% | - | - |
| 5 token | 84.15% | ~95% | 94.00% | 46.00% |
| 50 token | 77.40% | ~90% | 90.00% | 45.00% |
| 200 token | 66.50% | ~92% | - | - |

### Key Findings

- **Misalignment is the core signal of editing failure**: EAMET reduces the total misalignment score for 10k edits from 11,506 (MEMIT) to 377 on LLaMA2-7B (CounterFact), a 96.7% reduction, directly validating the effectiveness of alignment optimization.
- **LLaMA2-7B benefits most**: EAMET raises Eff. from 24.95% (MEMIT) to 89.09% on this model, a gain of 64 percentage points, attributable to the most severe original misalignment among all tested models.
- **Insensitivity to editing order**: Randomly shuffling the editing sequence causes EAMET's Eff. to fluctuate by only ~1% on CounterFact and at most 2% on ZsRE.
- **Robustness to multi-fact same-subject editing**: On ZsRE, as the number of facts associated with each subject increases, MEMIT and PMET performance degrades continuously while EAMET remains stable.
- **Scalable to 15k edits**: On Qwen2.5-7B with 15k edits, EAMET achieves 83.66% vs. MEMIT's 77.46%, with the advantage growing at larger scales.

## Highlights & Insights

- **Formal diagnosis of embedding misalignment**: This is the first work to quantitatively explain why large-scale editing fails. The failure is not due to insufficient optimization or limited parameter capacity, but rather the destruction of neighborhood structure between residuals and keys during joint optimization. This insight is particularly elegant in that it transforms a vague "scalability issue" into a concrete, measurable, and optimizable target $\mathcal{A}(i)$.
- **Elegant design of progressive alignment**: The strategy of sequentially optimizing and saving residuals avoids memory explosion from processing 10k residuals simultaneously, while naturally constructing a continuously growing "alignment reference set." This progressive strategy is itself a general large-scale optimization paradigm transferable to other settings requiring spatial structural consistency.
- **Introduction of stricter evaluation metrics**: Replacing probability comparison with whether the actual generated output contains the target entity exposes the overestimation of MEMIT and related methods. The proposal of these "practical metrics" in itself advances evaluation standards across the model editing community.

## Limitations & Future Work

- **Computational overhead of iterative per-fact optimization**: Each fact requires an independent forward and backward pass to optimize $r_i$, resulting in linear time complexity growth for 10k edits. A batch-wise alignment optimization scheme (e.g., using a lightweight alignment network for one-step optimization) could substantially reduce runtime.
- **Restricted to Transformer FFN layer editing**: The framework is tied to the locate-then-edit paradigm and cannot be applied to attention layer editing or adapter-based methods. While the misalignment concept theoretically extends to other parameter spaces, new formalization would be required.
- **Absence of continual editing evaluation**: The paper only evaluates one-shot batch editing and does not test the scenario of continued editing on an already-edited model. Successive editing rounds may lead to accumulated alignment drift.
- **Insufficient hyperparameter sensitivity analysis**: The values of $\lambda_{KL}$ and $\lambda_{MSE}$ may require tuning for different models, and the paper does not provide a systematic sensitivity analysis.

## Related Work & Insights

- **vs. MEMIT**: MEMIT jointly solves the normal equations without alignment constraints, causing residual space structure to be destroyed as edit count grows. EAMET's alignment loss is essentially a regularization of MEMIT—it does not alter the mathematical form of the final parameter update, only improving the quality of the input residuals. This reveals that MEMIT's bottleneck lies in input quality rather than the solver.
- **vs. AlphaEdit**: AlphaEdit addresses knowledge forgetting during sequential editing via null-space constraints to protect previously edited knowledge. EAMET addresses spatial structural consistency during batch editing. The two approaches are orthogonal and could theoretically be combined.
- **vs. PMET**: PMET introduces attention layer parameter modifications beyond the FFN to increase editing capacity, yet still suffers from misalignment (with misalignment scores comparable to MEMIT). EAMET improves residual quality at the source without adding editing layers, and achieves superior results.

## Rating

- Novelty: ⭐⭐⭐⭐ Formalizing embedding misalignment offers a genuinely new perspective, though the method itself (KL+MSE regularization) is relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6 LLMs, 3 datasets, edit scales from 200 to 15k, prefix robustness, same-subject robustness, and editing order sensitivity—comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ The theory–empirics–solution narrative chain is clear, though heavy notation makes the method section moderately difficult to read.
- Value: ⭐⭐⭐⭐ Provides important insights for the model editing community; the misalignment diagnostic tool has independent value in its own right.

- Novelty: ⭐⭐⭐⭐ Original discovery and formalization of embedding misalignment.

- Experimental Thoroughness: ⭐⭐⭐⭐ 6 LLMs × 3 datasets.

- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear.

- Value: ⭐⭐⭐⭐ Addresses a practical bottleneck in large-scale model editing.

## Summary

This paper makes meaningful contributions in its research direction, with the proposed method demonstrating competitive performance across multiple experimental settings.

The technical approach of the core contribution is clearly articulated, the experimental design is sound, and the work provides valuable reference points for subsequent research.

Future work could further explore the applicability and scalability of the method in broader settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](../../ACL2026/knowledge_editing/evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ICLR 2026\] Fine-tuning Done Right in Model Editing](fine-tuning_done_right_in_model_editing.md)
- [\[ICLR 2026\] Energy-Regularized Sequential Model Editing on Hyperspheres](energy-regularized_sequential_model_editing_on_hyperspheres.md)
- [\[ICLR 2026\] Bilinear Representation Mitigates Reversal Curse and Enables Consistent Model Editing](bilinear_representation_mitigates_reversal_curse_and_enables_consistent_model_ed.md)
- [\[ICLR 2026\] GOT-Edit: Geometry-Aware Generic Object Tracking via Online Model Editing](got-edit_geometry-aware_generic_object_tracking_via_online_model_editing.md)

</div>

<!-- RELATED:END -->
