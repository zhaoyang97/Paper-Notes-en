---
title: >-
  [Paper Note] Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals
description: >-
  [NeurIPS 2025][Interpretability][Dynamic context cutoff] This paper proposes dynamic context cutoff, which trains lightweight classifiers to detect "information sufficiency signals" encoded in specific Transformer attent…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "Dynamic context cutoff"
  - "attention head probing"
  - "information sufficiency"
  - "inference efficiency"
  - "KV cache"
date: 2026-05-08
content_hash: 9577fae8684cdd73
---

# Knowing When to Stop: Efficient Context Processing via Latent Sufficiency Signals

**Conference**: NeurIPS 2025
**arXiv**: [2502.01025](https://arxiv.org/abs/2502.01025)  
**Code**: [GitHub](https://github.com/ruoyuxie/when-to-stop)  
**Area**: Interpretability
**Keywords**: Dynamic context cutoff, attention head probing, information sufficiency, inference efficiency, KV cache

## TL;DR
This paper proposes dynamic context cutoff, which trains lightweight classifiers to detect "information sufficiency signals" encoded in specific Transformer attention heads, enabling the model to determine when sufficient context has been gathered and terminate processing early. On 6 QA datasets, the method achieves an average accuracy improvement of 3.4% while reducing token consumption by 1.33×.

## Background & Motivation

**Background**: LLMs process input context indiscriminately at inference time, assigning equal computational priority to every token regardless of its actual relevance to the task.

**Limitations of Prior Work**: Existing context compression methods (e.g., the LLMLingua series, RAG) rely on preset fixed compression ratios or fixed numbers of retrieved documents. This one-size-fits-all approach cannot adapt to varying information densities across inputs, leading to either information loss or computational waste.

**Key Challenge**: Humans dynamically decide when to stop reading based on the sufficiency of acquired information, whereas LLMs lack this adaptive capability. Furthermore, the "lost-in-the-middle" phenomenon demonstrates that redundant context can actually degrade accuracy.

**Goal**: Enable LLMs to autonomously assess context sufficiency and dynamically determine the truncation position, achieving both efficiency and performance gains without presetting a compression ratio.

**Key Insight**: Analysis of internal model activations reveals that certain attention heads naturally encode information sufficiency signals, which can be detected with lightweight linear probes.

**Core Idea**: Attention head activations within the model itself embed "sufficiency" judgment signals; reading these signals via probes enables dynamic early stopping.

## Method

### Overall Architecture
Given a complete context $\mathbf{C}$ and query $q$, the context is partitioned left-to-right into $m$ non-overlapping chunks $\{\mathfrak{s}_j\}_{j=1}^m$, forming cumulative context sequences $\mathbf{C}_i = \mathfrak{s}_1 \| \mathfrak{s}_2 \| \dots \| \mathfrak{s}_i$. After each new chunk is processed, a classifier determines whether the current cumulative context is "sufficient"; if so, processing halts and remaining chunks are discarded. KV caches are reused across chunks to avoid redundant computation.

### Key Designs

1. **Probing for Sufficiency Heads**:

    - Function: Identify which attention heads encode information sufficiency signals.
    - Mechanism: For each attention head $(l, h)$ with activation $x_l^h \in \mathbb{R}^D$, a linear classifier $p_\theta(x_l^h) = \sigma(\langle \theta, x_l^h \rangle)$ is trained to predict whether the current cumulative context contains sufficient information (binary classification). Predictive ability is measured by validation F1 score, and the top-$k$ heads are selected.
    - Design Motivation: Experiments reveal that a small number of middle-layer attention heads exhibit substantially higher F1 scores than others (e.g., clear high-F1 hotspots in LLaMA3.2-1B), indicating that internal model representations naturally encode sufficiency semantics.
    - Novelty: Rather than relying on external signals or compression heuristics, the method extracts already-present information from the model's own activations.

2. **Ensemble Sufficiency Classifier**:

    - Function: Build a robust sufficiency classifier based on the top-$k$ attention heads.
    - Mechanism: Multiple lightweight classifiers $\{\mathcal{S}_1, \dots, \mathcal{S}_e\}$ are trained on the top heads using StratifiedKFold (n=5) cross-validation, and combined via AUC-weighted ensembling: $\mathcal{S}_{\text{ensemble}}(\mathbf{C}_i) = \frac{1}{e}\sum_{j=1}^e \mathcal{S}_j(\mathbf{C}_i)$.
    - Decision Rule: Context is deemed sufficient and processing halts when $\mathcal{S}_c(\mathbf{C}_i) \geq \tau$.

3. **Iterative Inference with Cache Reuse**:

    - Function: Efficiently expand context incrementally while evaluating sufficiency.
    - Mechanism: Only the activations of newly added chunks are computed at each step, reusing cached activations $\mathbf{A}_{\text{cache}}^{i-1}$ from the previous step: $\mathbf{A}(\mathbf{C}_i) = f_{\text{model}}(\mathbf{C}_i \setminus \mathbf{C}_{i-1}, \mathbf{A}_{\text{cache}}^{i-1})$.
    - Design Motivation: Non-overlapping chunks combined with KV cache reuse ensure computational efficiency; overlapping chunks would require recomputing activations, eliminating the efficiency advantage.

4. **Self-Prompting for Large Models**:

    - Function: For models with 14B+ parameters, a meta-prompt enables the model to self-assess context sufficiency.
    - Key Findings: Self-prompting yields an F1 of only 52.6 for 1B models, but reaches 83.1 for 70B models, suggesting that sufficiency self-assessment is an emergent capability.

### Loss & Training
- Probe training data: Each cumulative context is labeled as "sufficient" or "insufficient" based on the position of the last token in the gold information span.
- Datasets are carefully balanced so that gold answer positions are uniformly distributed (mean ≈ 0.50, standard deviation 0.25–0.28), ensuring a 50%/50% positive-to-negative sample ratio.
- Hyperparameters: $k=5$ (number of attention heads), 8 classifiers with top-4 selected for ensembling, 10% incremental chunk strategy.

## Key Experimental Results

### Main Results

| Method | LLaMA-1B Avg | Mistral-8B Avg | Qwen-14B Avg | LLaMA-70B Avg | Overall Avg |
|--------|-------------|---------------|-------------|--------------|-------------|
| Full Context | 14.2 | 37.2 | 44.0 | 56.1 | 37.9 |
| BM25 | 13.7 | 35.6 | 36.5 | 41.7 | 31.9 |
| LLMLingua2 | 14.4 | 35.8 | 45.0 | 55.4 | 37.7 |
| Self-Prompt | 8.9 | 30.0 | 45.1 | 59.1 | 35.8 |
| **Ours** | **13.9** | **37.3** | **46.3** | **59.5** | **39.2** |

At a 1.33× token reduction ratio, the proposed method achieves an average accuracy of 39.2%, surpassing both full context (37.9%) and the strongest static compression baseline LLMLingua2 (37.7%).

### Sufficiency Classification Performance

| Model | Fine-Tune | Self-Prompt | Probing (Ours) |
|-------|-----------|-------------|----------------|
| LLaMA3.2-1B | 79.5 | 52.6 | **88.3** |
| Mistral-8B | 69.7 | — | **89.8** |
| Qwen2.5-14B | — | 78.3 | **87.2** |
| LLaMA3.3-70B | — | 83.1 | **91.1** |

The probing method substantially outperforms competing approaches across all model scales without requiring any additional fine-tuning.

### Key Findings
- Probing achieves F1=91.1, far exceeding fine-tuned classifiers (79.5) and self-prompting (83.1), demonstrating that attention head activations are more reliable than the model's surface-level outputs.
- Large models (14B+) exhibit emergent self-assessment capability, whereas small models (1B–8B) must rely on probing.
- Context truncation improves accuracy in certain scenarios, likely alleviating the "lost-in-the-middle" problem.
- The threshold $\tau$ is the most critical hyperparameter, governing the efficiency–performance trade-off.

## Highlights & Insights
- **Exploiting existing internal signals**: The method requires no external models or complex compression logic — it reads attention head activations with linear probes, making the approach remarkably simple. The philosophy that "the model already knows but does not say so" offers broad inspiration.
- **Dynamic vs. static paradigm shift**: Traditional compression methods preset compression ratios, whereas this work allows each input to adaptively determine how much context to process, better reflecting real-world information distributions.
- **Emergent capability discovery**: The ability to self-assess context sufficiency emerges with model scale, providing a new perspective on understanding large model capabilities.
- **Strong practical utility**: Probes are trained once and generalize across tasks; KV cache reuse preserves efficiency; the method is readily deployable in engineering settings.

## Limitations & Future Work
- The current approach assumes information accumulates linearly from left to right, making it unsuitable for scenarios where relevant information is scattered throughout the context (e.g., certain multi-hop reasoning tasks).
- Sufficiency labels are defined based on gold span positions, without accounting for the possibility that different models may require different amounts of context.
- The 10% chunk granularity is fixed; a more adaptive chunking strategy could further improve efficiency.
- The selected attention heads are model-specific, requiring probe retraining when switching models.
- Validation is limited to QA tasks; applicability to generative tasks (e.g., summarization, translation) remains to be explored.

## Related Work & Insights
- **vs. LLMLingua series**: LLMLingua uses a smaller model to filter tokens by entropy under a preset fixed compression ratio. The proposed method is fully dynamic, driven by internal model signals, and achieves higher accuracy at equivalent token reduction rates.
- **vs. RAG**: RAG retrieves a preset top-$k$ number of documents, which is orthogonal to compression methods. The paper finds that RAG performance degrades severely for larger models, whereas the proposed method improves consistently with model scale.
- **vs. KV Cache Compression**: KV cache optimization approximates attention at the computational level and is orthogonal to the context-level truncation proposed here; the two approaches can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ The insight of extracting sufficiency signals from attention head activations is highly inspiring, though linear probing itself is a well-established technique.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets, three model families, and full coverage from 1B to 70B scale, with comprehensive ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is well-organized, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Introduces a new paradigm for inference efficiency with strong engineering utility, though applicability is somewhat scenario-dependent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Forest Before Trees: Latent Superposition for Efficient Visual Reasoning](../../ACL2026/interpretability/forest_before_trees_latent_superposition_for_efficient_visual_reasoning.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Base Models Know How to Reason, Thinking Models Learn When](base_models_know_how_to_reason_thinking_models_learn_when.md)
- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)
- [\[NeurIPS 2025\] ARC-JSD: Attributing Response to Context via Jensen-Shannon Divergence Driven Mechanistic Study](attributing_response_to_context_a_jensen-shannon_divergence_driven_mechanistic_s.md)

</div>

<!-- RELATED:END -->
