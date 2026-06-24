---
title: >-
  [Paper Note] Large Language Models are Demonstration Pre-Selectors for Themselves
description: >-
  [ICML 2025][LLM Pretraining][in-context learning] FEEDER (FEw yet Essential Demonstration prE-selectoR) is proposed, which is a demonstration pre-selection framework based on "sufficiency" and "necessity" metrics. It leverages the LLM's own capabilities to identify a representative subset from the training data. Across both ICL and fine-tuning scenarios, FEEDER reduces data size by over 20% while maintaining or even improving performance.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "in-context learning"
  - "demonstration selection"
  - "data pre-selection"
  - "sufficiency & necessity"
  - "bilevel optimization"
date: 2026-05-08
content_hash: ce1c608d5a96085f
---

# Large Language Models are Demonstration Pre-Selectors for Themselves

**Conference**: ICML 2025  
**arXiv**: [2506.06033](https://arxiv.org/abs/2506.06033)  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: in-context learning, demonstration selection, data pre-selection, sufficiency & necessity, bilevel optimization

## TL;DR

FEEDER (FEw yet Essential Demonstration prE-selectoR) is proposed, which is a demonstration pre-selection framework based on "sufficiency" and "necessity" metrics. It leverages the LLM's own capabilities to identify a representative subset from the training data. Across both ICL and fine-tuning scenarios, FEEDER reduces data size by over 20% while maintaining or even improving performance.

## Background & Motivation

In-context learning (ICL) enables LLMs to perform downstream tasks without fine-tuning by selecting a few demonstrations as prompt context. The core challenge is **how to select the most representative demonstrations from a large-scale training dataset**.

Existing methods suffer from two major limitations:

**High computational overhead**: Existing demonstration selectors (e.g., methods based on similarity, diversity, or clustering) need to repeatedly retrieve from the entire training set for each test query. As the number of shots increases and selection criteria become complex, the computational cost escalates sharply.

**Ignoring the LLM's own characteristics**: Different LLMs possess distinct capabilities and knowledge domains. Demonstration selection should consider the characteristics of a specific LLM rather than relying on generic similarity or diversity metrics.

The authors observe that instead of selecting from the full training set for every query, it is better to perform a **pre-selection** first using the LLM itself to filter a small yet refined representative subset (the FEEDER subset). Subsequent queries then run demonstration selection solely on this subset, thus enhancing both efficiency and effectiveness.

## Method

### Overall Architecture

FEEDER divides demonstration selection into two stages:

1. **Pre-selection Stage**: A representative subset $\mathcal{D}_{\text{FEEDER}}$ is pre-selected from the full training set $\mathcal{D}_{\text{TRAIN}}$, aiming to make the subset as small as possible while fully representing the entire dataset.
2. **Selection Stage**: Existing demonstration selectors (e.g., Random, Similarity, Diversity) are applied over $\mathcal{D}_{\text{FEEDER}}$ to select n-shot demonstrations for a specific test input.

In addition, FEEDER can be applied to **fine-tuning scenarios**: the LLM is fine-tuned using the pre-selected subset, and a bi-level optimization is employed to alternately optimize the subset selection and model parameters.

### Key Designs

#### 1. **Sufficiency Metric**: Assessing the representative ability of demonstrations

Core Idea: If plugging a sample $(x_n, y_n)$ into the LLM context enables the LLM to generate the correct output for another sample $x_m$, then $(x_n, y_n)$ is deemed **sufficient** for $(x_m, y_m)$.

Formal Definition: $Y_{x_m}=1 \mid \text{plug}((x_n, y_n)); C, S=(Y_{x_m}=0)$

This indicates that under the condition where the LLM initially answers $x_m$ incorrectly, inserting $(x_n, y_n)$ corrects the model's answer. Sufficiency addresses a key question: **Is this demonstration sufficient to represent other samples?**

#### 2. **Necessity Metric**: Assessing the indispensability of demonstrations

Core Idea: If unplugging a sample $(x_n, y_n)$ that is already in the context causes the LLM's output for $x_m$ to change from correct to incorrect, then $(x_n, y_n)$ is deemed **necessary** for $(x_m, y_m)$.

Formal Definition: $Y_{x_m}=0 \mid \text{unplug}((x_n, y_n)); C, S=(Y_{x_m}=1)$

Necessity addresses another question: **Does removing this demonstration cause information loss?** If a demonstration is not necessary, the information it provides is redundant and can be safely removed.

#### 3. **Tree-based Approximation**: Efficient discovery of the FEEDER subset

Directly enumerating all possible subsets has a complexity of $O(2^N)$, which is intractable. The authors design a bottom-up tree-based algorithm:

- **Initialization**: Each training sample is set as a leaf node at the bottom level of the tree.
- **Each Round**: Pairwise nodes at the current level are matched to check their mutual sufficiency using the LLM.
    - If $W_i$ and $W_j$ are mutually sufficient: Retain the one with fewer elements.
    - If sufficiency is unidirectional (e.g., $W_i$ is sufficient for $W_j$): Retain the sufficient one $W_i$.
    - If neither is sufficient for the other: Merge them into $W_i \cup W_j$.
- **Termination**: The algorithm terminates when only one node remains, and the samples contained in this node form the FEEDER subset.

The complexity is $O(RK \log_2 |\mathcal{D}_{\text{TRAIN}}|)$, where $R$ refers to the number of algorithm runs and $K$ is the tree depth. Experiments show that $K=1, R=1$ is sufficient, simplifying the complexity to $O(\log_2 |\mathcal{D}_{\text{TRAIN}}|)$.

The authors also prove **Proposition 1**: Under the assumption of sufficiency transitivity, the subset produced by the tree-based algorithm fully represents the entire training set.

### Loss & Training

The optimization objective under the **ICL setting** is to minimize the size of the FEEDER subset while ensuring its performance on the training set is not inferior to that of the full dataset:

$$\min_{\mathcal{D}_{\text{FEEDER}} \subseteq \mathcal{D}_{\text{TRAIN}}} |\mathcal{D}_{\text{FEEDER}}|, \quad \text{s.t.} \; \mathcal{L}(\mathcal{D}_{\text{FEEDER}}, \mathcal{D}_{\text{TRAIN}}) \leq \mathcal{L}(\mathcal{D}_{\text{TRAIN}}, \mathcal{D}_{\text{TRAIN}})$$

Bi-level optimization under the **fine-tuning setting**:
- **Outer level**: Freeze the LLM and update $\mathcal{D}_{\text{FEEDER}}$ using the tree-based algorithm.
- **Inner level**: Fix $\mathcal{D}_{\text{FEEDER}}$ and fine-tune the LLM parameters.

The two levels alternate iteratively to achieve joint improvement in subset selection and model optimization.

## Key Experimental Results

### Main Results

Experiments cover 6 text classification datasets (SST-2, SST-5, COLA, TREC, SUBJ, FPB), a reasoning dataset (GSM8K), semantic parsing (SMCALFlow), and science QA (GPQA). Evaluated LLMs include GPT-2 (335M/774M), GPT-neo (1.3B), GPT-3 (6B), Gemma-2 (2B), Llama-2 (7B), Llama-3 (8B), and Qwen-2.5 (32B).

**Key results for the ICL setting** (using Gemma-2 (2B) on text classification as an example, with $n=5$ shots):

| Dataset | Selector | $\mathcal{D}_{\text{TRAIN}}$ | $\mathcal{D}_{\text{FEEDER}}$ | Gain |
|--------|--------|------|------|------|
| SUBJ | Similarity | 91.5 | **94.5** | +3.0 |
| SST-2 | Similarity | 80.5 | **83.6** | +3.1 |
| COLA | Similarity | 67.2 | **77.6** | +10.4 |
| GSM8K | Similarity | 20.31 | **22.58** | +2.27 |
| SMCALFlow | Diversity | 27.89 | **32.54** | +4.65 |

**Qwen-2.5 (32B) on GSM8K/GPQA ($n=10$ shots)**:

| Dataset | Selector | $\mathcal{D}_{\text{TRAIN}}$ | $\mathcal{D}_{\text{FEEDER}}$ | Gain |
|--------|--------|------|------|------|
| GSM8K | Similarity | 90.41 | **91.23** | +0.82 |
| GSM8K | Uncertainty | 90.20 | **91.96** | +1.76 |
| GPQA | Similarity | 46.85 | **47.80** | +0.95 |
| GPQA | Diversity | 46.71 | **47.93** | +1.22 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| $K=1, R=1$ (Default) | Optimal cost-effectiveness | Excellent results achieved with just one one-shot inference run and a single round of execution. |
| Increasing $R$ (multiple rounds of execution) | Performance first increases and then decreases | Subsets that are too small limit performance, showing a trade-off between data quantity and quality. |
| Increasing $K$ (deepening the tree) | More robust upward trend | Two-shot inference provides a more stringent filtering mechanism. |
| Bi-level optimization (fine-tuning) | Substantive improvement (e.g., SUBJ: 89.2 to 95.6) | Fine-tuning leverages high-quality subsets more effectively. |
| Repeated training set experiments | FEEDER minimizes the impact of noise | Verifies the robustness of FEEDER under data redundancy scenarios. |

### Key Findings

1. **FEEDER can reduce training data by 20% to 50%** while maintaining or exceeding the performance of the full dataset in ICL.
2. **FEEDER + Similarity is comparable to or outperforms complex selectors such as Diversity/Clustering**, indicating that the pre-selection stage is more critical than the selection strategy itself.
3. **The performance improvement is more pronounced in fine-tuning scenarios**: Under the bi-level optimization framework, FEEDER elevates the 10-shot accuracy of GPT-2 (0.8B) on SUBJ from 94.0 to 95.5.
4. **Significant advantage with larger shot sizes**: When the number of shots increases from 5 to 10, using the full dataset often suffers from performance degradation (due to noise/redundancy), whereas FEEDER effectively mitigates this issue.
5. **Different LLMs require different FEEDER subsets**: Case studies confirm that different LLMs need distinct sufficient and necessary demonstrations for the same fact, validating the necessity of LLM-aware pre-selection.
6. **Near-linear time complexity**: Sublinear or nearly linear correlation with data size. In practical deployment, using $K=1, R=1$ is sufficient, rendering the overhead highly manageable.

## Highlights & Insights

- **Novel "Pre-selection" Concept**: Splitting demonstration selection into pre-selection and selection stages is both intuitive and practical. The pre-selection stage is query-agnostic, allowing all queries to reuse the pre-selected subset after a single computation.
- **Causal Inference Perspective**: Borrowing concepts of sufficiency and necessity from causal inference (linked to intervention operations in do-calculus) provides a solid theoretical foundation for assessing demonstration quality.
- **LLM-aware Design**: The pre-selection process accounts for the capability boundaries of specific LLMs. Different models yield distinct FEEDER subsets, embodying a model-centric approach to data selection.
- **Applicable to Dual Scenarios**: FEEDER serves both ICL and fine-tuning, seamlessly bridged by the bi-level optimization framework.

## Limitations & Future Work

1. **Sufficiency Transitivity Assumption**: The tree-based algorithm relies on the assumption of sufficiency transitivity, which may not hold universally in practice, potentially leading to suboptimal subsets.
2. **Computational Cost of the Pre-selection Stage**: Though the complexity is $O(\log N)$, each sufficiency check requires LLM inference, which still incurs considerable overhead for very large datasets.
3. **Limited Dataset and Task Coverage**: Evaluation has been primarily conducted on text classification and simple reasoning tasks; the effectiveness on generation tasks (e.g., summarization and translation) remains unexplored.
4. **Potential Amplification of Model Bias**: FEEDER relies on the LLM's own judgment to filter data. If the model exhibits bias, the pre-selection process might further reinforce it (acknowledged in the paper's Impact Statement).
5. **Unexplored Hybrids with Advanced Selectors**: Combining FEEDER with retrieval-augmented ICL methods has not been thoroughly investigated.

## Related Work & Insights

- **Demonstration Selection**: FEEDER is orthogonal to existing similarity, diversity, clustering, and uncertainty selectors, serving as an effective preparatory stage for them.
- **Core-set Selection**: The subset selection in FEEDER echoes the literature on core-set selection (Feldman, 2020; Guo et al., 2022), but introduces LLM-specific evaluation criteria.
- **Data-centric AI**: It embodies the paradigm that data quality outweighs data quantity, mirroring other data filtration works such as Alpagasus (Chen et al., 2023).
- **Insights**: This LLM-as-judge paradigm (utilizing an LLM to evaluate the value of data to the LLM itself) can be extended to other data selection scenarios, such as training data cleaning and curriculum learning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The pre-selection stage and sufficiency/necessity framework are creative, though the core idea remains a variant of subset selection.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 8 LLMs (335M to 32B), 9 datasets, 6 selectors, and both ICL and fine-tuning scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured and formally rigorous, though some notations might pose a reading barrier.
- **Value**: ⭐⭐⭐⭐ — Highly practical as a plug-and-play pre-selection module, though performance gains are more prominent on smaller models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](../../ACL2025/llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](../../ACL2025/llm_pretraining/leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Large Vocabulary Size Improves Large Language Models](../../ACL2025/llm_pretraining/large_vocabulary_size_improves_large_language_models.md)
- [\[NeurIPS 2025\] Scalable Fingerprinting of Large Language Models](../../NeurIPS2025/llm_pretraining/scalable_fingerprinting_of_large_language_models.md)
- [\[NeurIPS 2025\] The Curse of Depth in Large Language Models](../../NeurIPS2025/llm_pretraining/the_curse_of_depth_in_large_language_models.md)

</div>

<!-- RELATED:END -->
