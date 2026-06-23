---
title: >-
  [Paper Note] EntropyLong: Effective Long-Context Training via Predictive Uncertainty
description: >-
  [ICLR 2026][LLM Efficiency][Paper Note] EntropyLong utilizes the model's own predictive entropy to locate "information gaps," retrieves distant context, and **empirically tests whether it reduces entropy at those positions**. By retaining only dependencies that bring genuine information gain to concatenate 128K training samples, it constructs "verified" long
tags:
  - ICLR 2026
  - LLM Efficiency
date: 2026-05-08
content_hash: b4fc66395e5e0401
---
# EntropyLong: Effective Long-Context Training via Predictive Uncertainty

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=SFXX5Pjl5K](https://openreview.net/forum?id=SFXX5Pjl5K)  
**Code**: To be confirmed (The paper states that the dataset will be open-sourced)  
**Area**: LLM Efficiency  
**Keywords**: Long-context training, predictive entropy, information gain, data construction, model-in-the-loop verification

## TL;DR
EntropyLong utilizes the model's own predictive entropy to locate "information gaps," retrieves distant context, and **empirically tests whether it reduces entropy at those positions**. By retaining only dependencies that bring genuine information gain to concatenate 128K training samples, it constructs "verified" long-range dependencies, significantly outperforming heuristic data construction methods on RULER and LongBench-v2.

## Background & Motivation
**Background**: Architectural innovations (e.g., Longformer, Big Bird, RoPE extrapolation) have pushed theoretical context windows to millions of tokens, but the true bottleneck is **training data**—the lack of samples that enable models to effectively utilize long contexts. Prevailing practices involve direct concatenation of short documents or heuristic synthesis (e.g., Quest via topic-relevant retrieval, NExtLong via inserting distractor documents).

**Limitations of Prior Work**: All heuristic methods **preset** what constitutes a "good long-context sample" (such as semantic similarity or the presence of distractors) but never **directly verify** from the model’s perspective whether such dependencies are actually useful. Recently, RE³SYN utilized a small proxy model's perplexity to rerank candidates, but this represents a document-level, proxy-based criterion that is misaligned with the target model's parametric knowledge and remains too coarse in granularity.

**Key Challenge**: Samples obtained through concatenation may only contain "spurious correlations"—while the model's attention span increases, it does not necessarily learn the ability to integrate information across long distances.

**Goal**: To utilize the model's own predictive uncertainty as a signal to construct long-range dependency data that carries **provable information gain**.

**Key Insight**: High entropy positions = information gaps. If introducing distant relevant context can **empirically reduce** the predictive entropy at that position, it indicates the formation of a genuine long-range dependency that requires remote information integration. Establishing a closed loop of "Information Gap → Entropy Reduction Verification → Valid Dependency" is inspired by active learning and uncertainty-guided data filtering.

## Method

### Overall Architecture
EntropyLong is a four-stage, model-in-the-loop data construction pipeline: first, the base model calculates token-wise predictive entropy for a document and selects high-entropy anchors using adaptive thresholds; next, semantically related contexts are retrieved for each anchor; then, candidate contexts are prepended to **empirically test workshops if entropy reduction exceeds a threshold**, retaining only those that pass; finally, the verified contexts are randomly shuffled and concatenated with the original document into a 128K training sequence.

```mermaid
flowchart LR
    A[Root Document D] --> B[Step 1 Adaptive Threshold<br/>Select High-Entropy Positions τH=μ+ασ]
    B --> C[Step 2 Information-theoretic Retrieval<br/>Retrieve top-K using anchor neighborhood query]
    C --> D[Step 3 Entropy Reduction Verification<br/>Prepend candidates, retain only if ΔI>ε]
    D --> E[Step 4 Strategic Concatenation<br/>Shuffle verified contexts + D]
    E --> F[128K Training Sample<br/>Contains verified long-range dependencies]
```

### Key Designs

**1. Adaptive Threshold for High-Entropy Position Selection: Let the document define "where it is uncertain"**  
Given a document $D=\{x_1,\dots,x_n\}$, the base model $M_\theta$ calculates the Shannon entropy for each position: $H_\theta(x_t|x_{<t})=-\sum_{v\in V}P_\theta(v|x_{<t})\log P_\theta(v|x_{<t})$. Rather than using a global fixed threshold, the threshold is determined by the entropy distribution of **each individual document**: $\tau_H=\mu_H+\alpha\sigma_H$ (practically $\alpha=2.0$). Positions exceeding $\tau_H$ are defined as "high uncertainty positions." This adapts to varying document difficulties, preventing simple documents from being discarded entirely or complex documents from being over-selected.

**2. Information-theoretic Context Retrieval: Using anchor neighborhoods as queries**  
For each high-entropy position $t_i$, $w$ tokens ($w=16$) before and after it are taken as the query $q_i=x_{t_i-w:t_i+w}$. A pre-trained sentence transformer encodes this query to retrieve top-K candidates from a large corpus of over 1B documents based on cosine similarity. Note that this step only **recalls candidates**; high similarity does not equate to utility, leaving the actual selection to the next step.

**3. Entropy Reduction Verification (Mechanism): Only empirical information gain counts**  
The candidate context $C_j$ is prepended to the original document, and the entropy at the high-entropy position is recalculated as $H'_\theta$. The **contextual information gain** is defined as the relative entropy reduction: $\Delta I_{t_i}(C_j,D)=\frac{H_\theta(x_{t_i}|x^D_{<t_i})-H'_\theta(x_{t_i}|x^{[C_j;D]}_{<t_i+|C_j|})}{H_\theta(x_{t_i}|x^D_{<t_i})}$. The context is retained only if $\Delta I_{t_i}>\epsilon$ (set to $\epsilon=0.4$). This step replaces "semantic similarity" with "empirical utility," filtering out spurious dependencies and redundancy. This is the core of the method and the largest contributor in the ablation study. The final dataset achieves an average information gain of $\bar{\Delta I}=0.68$.

**4. Strategic Concatenation: Random shuffling is superior to sequential ordering**  
For a set of verified contexts $\{C_1,\dots,C_m\}$ collected for a document, the sample is constructed as $S=[C_{\pi(1)};C_{\pi(2)};\dots;C_{\pi(m)};D]$. Comparing two permutations—Sequence (original token order) and Shuffle (random permutation $\pi$)—experiments show that random shuffling is slightly better. It prevents the model from learning shortcuts based on "fixed position dependencies," forcing it to genuinely locate information across long distances.

## Key Experimental Results
The base model is Meta-Llama-3-8B, with the RoPE base modified to 200,000,000 to extend to 128K. Using FineWeb-Edu + Cosmopedia, 100K documents were sampled as sources and the full corpus (>1B documents) served as the retrieval pool, generating 4B tokens of 128K sequences. Training was conducted for 1000 steps with a global batch size of 4M tokens. Baselines Quest and NExtLong used the same configuration and 4B tokens.

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Gain |
|--------|------|------|------|---|
| RULER (avg 8K–128K) | Avg Score | 87.37 | 85.22 (NExtLong) | +2.15 |
| RULER @128K | Score | 81.26 | 77.99 (NExtLong) | +3.27 |
| RULER @128K vs Quest | Score | 81.26 | 60.11 (Quest) | +21.15 |
| LongBench-v2 (Post-SFT) | Overall | 27.60 | 24.10 (NExtLong) | +3.50 |
| LongBench-v2 Long Tasks | Score | 31.50 | 23.10 (NExtLong) | +8.40 |

### Ablation Study

| Configuration | Metric (RULER avg) | Description |
|------|------|------|
| Full Method | 87.37 | Includes entropy reduction verification |
| NoVerify (Semantic Similarity Only) | 85.82 | Removed verification, −1.55 overall, −1.79 at 128K |
| α=1.5 (913 tokens) | 82.49 | Too many positions introduce noise |
| α=2.0 (292 tokens) | 87.37 | Optimal balance |
| α=2.5 (83 tokens) | 85.52 | Too few positions lack sufficient data |
| ε=0.2 (62 dependencies) | 85.45 | Loose verification includes weak dependencies |
| ε=0.4 (46 dependencies) | 87.37 | Optimal balance |
| ε=0.6 / 0.8 (29/13 dependencies) | 86.14 / 86.47 | Too strict, insufficient dependencies |

### Key Findings
- **Entropy reduction verification is indispensable**: Performance drops across the board without it, and the **drop increases as context lengthens** (−1.79 at 128K), indicating that spurious dependencies filtered by verification are most harmful in long-range tasks.
- **Clear optimal thresholds exist**: $\alpha=2.0$ and $\epsilon=0.4$ validate the "quality vs. quantity" trade-off hypothesis—it must neither be too loose to admit noise nor too strict to lack training signals.
- **Advantage increases with context length**: EntropyLong's lead over NExtLong becomes more pronounced at longer contexts, confirming that information-theoretic construction indeed enhances remote integration capabilities.
- **Non-monotonicity of window $w$**: $w=8$ performs better at short contexts but degrades at long contexts, while $w=16$ is the most balanced overall.
- **Random shuffling outperforms sequential concatenation**: Shuffling breaks the shortcut of "contexts appearing in a fixed order," forcing the model to genuinely learn remote localization rather than memorizing positional patterns.
- **Data volume is not always better**: While $\alpha=1.5$ selects 913 high-entropy tokens (far more than $\alpha=2.0$'s 292), the inclusion of noisy positions reduces the average score from 87.37 to 82.49, highlighting that "a small amount of verified high-quality dependencies" > "a large amount of unverified dependencies."

## Highlights & Insights
- **Returning the definition of "good data" to the model**: Shifting from human-defined heuristics (similarity/distraction) to model-measured information gain represents a self-supervised, verifiable data construction paradigm.
- **Entropy reduction as an operational criterion for long-range dependency**: For the first time, the "Information Gap → Verification → Valid Dependency" process is formalized into a closed loop, where each dependency has quantifiable evidence with $\bar{\Delta I}=0.68$.
- **Token-level model alignment**: Compared to the document-level proxy perplexity used in RE³SYN, this method operates at the token level and uses the target model itself, avoiding knowledge misalignment and enabling scaling without permutation searching.

## Limitations & Future Work
- The verification step requires a forward pass for each candidate context, making the construction cost scale linearly with the number of top-K retrieved and high-entropy positions, resulting in high pipeline overhead.
- Experiments were only validated on the Llama-3-8B scale; whether the optimal entropy thresholds remain the same for larger models or different architectures is unknown.
- High entropy does not always imply a "need for remote context"—it could also result from inherent ambiguity or noisy tokens. While verification filters some of this, the neighborhood query is still limited by local information.
- $\alpha$, $\epsilon$, and $w$ all require hyperparameter tuning, and their optimal values might change with the target context length (as shown by window non-monotonicity), lacking an automatic selection mechanism.

## Related Work & Insights
- **vs Quest (Coherence-driven)**: Quest assumes that thematic relevance forms useful dependencies. This paper points out that semantic similarity $\neq$ utility, replacing similarity with empirical entropy reduction and leading by 21 points at 128K.
- **vs NExtLong (Distinction-driven)**: NExtLong relies on inserting distractor documents to force the model to differentiate. This paper avoids presetting the "source of difficulty" and directly verifies information gain, showing more significant leads in long-range tasks.
- **vs RE³SYN (Verification-driven)**: Both involve "verification," but RE³SYN uses a small proxy model for document-level perplexity reranking, whereas this paper uses the target model for token-level entropy reduction, ensuring model alignment without needing permutation search.
- **Insight**: Predictive entropy/uncertainty has long been used in active learning to select samples for labeling; this paper applies it **constructively** for unsupervised pre-training data construction, providing a blueprint for automated pipelines where "models autonomously refine their own training data."

## Rating
- Novelty: ⭐⭐⭐⭐ Transforms predictive entropy from a "diagnostic signal" into a "constructive data criterion"; the model-in-the-loop verification of long-range dependencies is clear and original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual benchmarks (RULER + LongBench-v2), systematic ablation of α/ε/w and concatenation strategies, and solid justification for the verification step; however, it is limited to a single base model scale.
- Writing Quality: ⭐⭐⭐⭐ Clear correspondence between theoretical motivation (precondition → principle → hypothesis) and the four stages of the method; well-supported by formulas and flowcharts.
- Value: ⭐⭐⭐⭐ A practical paradigm for long-context data construction with a commitment to open-source the 128K dataset, providing direct value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] NExtLong: Toward Effective Long-Context Training without Long Documents](../../ICML2025/llm_efficiency/nextlong_toward_effective_long-context_training_without_long_documents.md)
- [\[ICLR 2026\] AutoSP: Unlocking Long-Context LLM Training Via Compiler-Based Sequence Parallelism](autosp_unlocking_long-context_llm_training_via_compiler-based_sequence_paralleli.md)
- [\[ICLR 2026\] Let's (not) just put things in Context: Test-time Training for Long-context LLMs](lets_not_just_put_things_in_context_test-time_training_for_long-context_llms.md)
- [\[ICLR 2026\] Wide-In, Narrow-Out: Revokable Decoding for Efficient and Effective DLLMs](wide-in_narrow-out_revokable_decoding_for_efficient_and_effective_dllms.md)
- [\[ICLR 2026\] Long-Context Attention Benchmark: From Kernel Efficiency to Distributed Context Parallelism](long-context_attention_benchmark_from_kernel_efficiency_to_distributed_context_p.md)

</div>

<!-- RELATED:END -->
