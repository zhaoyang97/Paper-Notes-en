---
title: >-
  [Paper Note] Length-Induced Embedding Collapse in PLM-based Models
description: >-
  [ACL 2025 (Findings)][Information Retrieval & RAG][text embedding] Identifies and rigorously proves the "length collapse" phenomenon in PLM-based text embedding models—where long-text embeddings tend to cluster together because self-attention acts as a low-pass filter whose filtering effect intensifies as text length increases, over-suppressing high-frequency information. Proposes the TempScale method to mitigate the distributional discrepancy between short and long text embe…
tags:
  - "ACL 2025 (Findings)"
  - "Information Retrieval & RAG"
  - "text embedding"
  - "length collapse"
  - "self-attention"
  - "low-pass filter"
  - "temperature scaling"
date: 2026-05-08
content_hash: c9913edc2e383d3f
---

# Length-Induced Embedding Collapse in PLM-based Models

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2410.24200](https://arxiv.org/abs/2410.24200)  
**Code**: [GitHub](https://github.com/Yuqi-Zhou/Length_Collapse)  
**Area**: Other  
**Keywords**: text embedding, length collapse, self-attention, low-pass filter, temperature scaling

## TL;DR
Identifies and rigorously proves the "length collapse" phenomenon in PLM-based text embedding models—where long-text embeddings tend to cluster together because self-attention acts as a low-pass filter whose filtering effect intensifies as text length increases, over-suppressing high-frequency information. Proposes the TempScale method to mitigate the distributional discrepancy between short and long text embeddings by scaling down the attention temperature, improving MTEB by 0.94% and LongEmbed by 1.10%.

## Background & Motivation

**Background**: PLM-based embedding models (e.g., BGE, E5, etc.) encode text into fixed-dimensional vectors and are widely used in tasks such as retrieval, classification, and STS.

**Limitations of Prior Work**: Embedding models experience significant performance degradation on long text. For BGE in IMDB classification, accuracy drops from 75.6% to 59.0% (a 16.6% decrease) when the token count increases from [0, 100) to [400, 500). However, the underlying cause remains unclear.

**Key Challenge**: **Why** does long-text embedding perform poorly? It is not simply due to excessive information, but because long-text embeddings become overly similar to each other, losing discriminative power.

**Goal**: (a) Define and empirically verify the Length Collapse phenomenon; (b) Provide a mechanistic explanation from a frequency-domain perspective; (c) Propose a mitigation method.

**Key Insight**: Analyze self-attention in the frequency domain as a low-pass filter (following the analysis of ViTs by Wang et al., 2022) and prove the relationship between the filtering rate and the sequence length $n$.

**Core Idea**: The maximum singular value of the high-frequency component of the attention matrix $\sigma_a$ decreases as $n$ increases $\to$ token representations of long texts tend to homogenize (retaining only the DC component) $\to$ embeddings cluster after pooling = Length Collapse.

## Method

### Overall Architecture
Theoretical analysis: Proves that $\sigma_a$ monotonically decreases with sequence length $n \to$ explains the mechanism of Length Collapse. Empirical validation: Proposes TempScale to adjust the attention temperature to mitigate this phenomenon.

### Key Designs

1. **Frequency-Domain Analysis of Length Collapse**

    - **Lemma 1**: The attention matrix $\mathbf{A} = \text{softmax}(\mathbf{P})$ is a low-pass filter—repeatedly applying $\mathbf{A}$ causes the signal to retain only the DC component.
    - **Theorem 2**: The high-frequency decay rate is controlled by $\sigma_a \|\mathbf{W}_V\|_2$. Smaller $\sigma_a \to$ high-frequency components are suppressed faster.
    - **Theorem 3**: Assuming the query/key distributions are Gaussian, prove that $\sigma_a \leq \sqrt{\frac{n}{2\sqrt{1+e^{-2\sigma_s^2}}(n-1)^{3/2}+1}}$, which **monotonically decreases as $n$ increases**.
    - **Corollary 4**: Larger $n \to$ smaller $\sigma_a \to$ token characteristics homogenize $\to$ embedding cosine similarity increases = Length Collapse.

2. **Impact Analysis of Length Collapse on Downstream Tasks**

    - **Classification/Clustering**: Long-text embeddings cluster at the center $\to$ the KNN classifier biases towards long texts $\to$ accuracy decreases.
    - **Retrieval**: The embedding space of long documents is compressed $\to$ short noisy documents may rank higher than truly relevant long documents.
    - **STS**: High similarity between long text pairs $\to$ unrelated long text pairs also receive high scores, resulting in poor discriminative power.

3. **The TempScale Method**

    - **Function**: Scaled down by temperature $\tau < 1$ after dividing the attention score by $\sqrt{d}$, i.e., $\mathbf{A} = \text{softmax}(\frac{\mathbf{XW}_Q(\mathbf{XW}_K)^\top}{\tau\sqrt{d}})$.
    - **Mechanism**: $\tau < 1$ is equivalent to increasing $\sigma_s$ and lowering the "temperature" of the attention $\to$ makes $\sigma_a$ less sensitive to changes in $n \to$ narrows the filtering rate gap between long and short texts.
    - **Design Motivation**: Analysis of extreme cases—as $\tau \to 0$, the attention becomes one-hot (no filtering but losing aggregation capability); as $\tau \to 1$, it maintains the original behavior. The optimal $\tau$ lies in between.
    - **No retraining required**: Directly modify the attention calculation during inference.

## Key Experimental Results

### Main Results

| Benchmark | Base Model | Original | +TempScale | Gain |
|-----------|-----------|------|-----------|------|
| MTEB (56 tasks) | BGE-base | 63.55 | **64.49** | +0.94% |
| MTEB | E5-large | 66.23 | **67.01** | +0.78% |
| LongEmbed | BGE-base | 42.15 | **43.25** | +1.10% |
| LongEmbed | E5-4K | 56.82 | **57.89** | +1.07% |

### Classification Accuracy by Text Length (BGE on IMDB)

| Token Range | Original | +TempScale |
|-----------|------|-----------|
| [0, 100) | 75.6 | 75.8 |
| [100, 200) | 72.3 | 73.5 |
| [200, 300) | 65.1 | 67.2 |
| [300, 400) | 61.5 | 64.0 |
| [400, 500) | 59.0 | 62.3 |

### Key Findings
- **Length Collapse is a universal phenomenon**: It exists in mainstream embedding models such as BGE, E5, GTE, and ANCE.
- **Pairwise cosine similarity of long texts monotonically increases with length**: Increasing from ~0.3 for [0, 100) to ~0.6 for [400, 500), validating Corollary 4.
- **TempScale improves long texts the most**: The improvement is most significant in the [400, 500) range, while short texts are virtually unaffected.
- **Experimental validation of the relationship between $\sigma_a$ and $n$**: $\sigma_a$ extracted from the last layer of the BGE attention matrix indeed decreases as the text length increases.
- **The optimal $\tau$ lies between $[0.5, 0.8]$**: Too low of a $\tau$ causes the attention to degenerate into one-hot.

## Highlights & Insights
- **Elegance of frequency-domain analysis**: Explaining attention as a low-pass filter from a Fourier perspective, and then deriving the impact of length on the filtering rate—with a clear causal chain: $n \uparrow \to \sigma_a \downarrow \to$ high frequencies are suppressed $\to$ embeddings collapse.
- **Training-free inference-time solution**: TempScale does not require retraining the model, only modifying the attention temperature during inference—making it directly applicable to deployed embedding models.
- **Perfect alignment between theoretical prediction and experiments**: Theorem 3 predicts $\sigma_a$ decreases with $n$, which is experimentally verified in Figure 7; Corollary 4 predicts cosine similarity increases with $n$, which is verified in Figure 1(c).
- **Direct value for RAG and long-text retrieval**: Representation degradation of long-text embeddings is a real-world bottleneck in long-document retrieval. TempScale provides a simple yet effective mitigation.

## Limitations & Future Work
- **Gaussian assumption**: Theorem 3 assumes Gaussian distributions for query/key, which might not be strictly satisfied in actual PLMs.
- **Only a single hyperparameter $\tau$**: A globally uniform temperature; layer-wise or head-wise adaptive temperatures could be considered.
- **Causal attention (decoder-only) not considered**: The analysis targets bidirectional attention (encoders). Length Collapse in causal attention requires extra research.
- **Limited performance gain**: MTEB +0.94% / LongEmbed +1.10%. While the consistency is good, the absolute improvements are modest.
- **Not combined with other long-text methods**: Such as position interpolation, RoPE scaling, etc.

## Related Work & Insights
- **vs. Wang et al. (2022) ViT over-smoothing**: They found that self-attention in ViTs causes over-smoothing as depth increases. This paper finds that a similar effect occurs **as length increases**—both over-smoothing phenomena share the same mechanism.
- **vs. rotary position embedding**: RoPE improves attention's generalization to length via rotation, but does not directly address the low-pass filtering problem—and can be combined with TempScale.
- **vs. chunking strategies**: Splitting a long document into short chunks for individual encoding is an engineering solution, whereas TempScale is a theory-driven model-level solution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define Length Collapse and provide a rigorous frequency-domain proof, with a brand-new derivation for Theorem 3.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on MTEB + LongEmbed across multiple models with visualizations—highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Seamlessly progressive theoretical derivations, intuitive figures (Figure 1 perfectly showcases the problem), and rigorous reasoning.
- Value: ⭐⭐⭐⭐⭐ Provides a fundamental understanding of the long-text issue in embedding models; TempScale is both practical and training-free.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Optimized Text Embedding Models and Benchmarks for Amharic Passage Retrieval](optimized_text_embedding_models_and_benchmarks_for_amharic_passage_retrieval.md)
- [\[ACL 2025\] Semantic Outlier Removal with Embedding Models and LLMs](semantic_outlier_removal_with_embedding_models_and_llms.md)
- [\[ACL 2025\] Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models](sticking_to_the_mean_detecting_sticky_tokens_in_text_embedding_models.md)
- [\[ACL 2025\] When Claims Evolve: Evaluating and Enhancing the Robustness of Embedding Models Against Misinformation Edits](when_claims_evolve_evaluating_and_enhancing_the_robustness_of_embedding_models_a.md)
- [\[ACL 2025\] Collapse of Dense Retrievers: Short, Early, and Literal Biases Outranking Factual Evidence](collapse_dense_retrievers.md)

</div>

<!-- RELATED:END -->
