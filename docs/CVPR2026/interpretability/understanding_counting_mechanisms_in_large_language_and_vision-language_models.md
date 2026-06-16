---
title: >-
  [Paper Note] Understanding Counting Mechanisms in Large Language and Vision-Language Models
description: >-
  [CVPR 2026][Interpretability][Paper Note] Authors utilize a set of controlled "repetitive object counting" experiments and a self-developed causal probing tool, CountScope, to dissect LLMs and LVLMs layer-by-layer and token-by-token. They find that counting is not a one-time summation but a hierarchical process emerging across layers, driven by "internal count
tags:
  - CVPR 2026
  - Interpretability
date: 2026-05-08
content_hash: 37597f339c81970a
---
# Understanding Counting Mechanisms in Large Language and Vision-Language Models

**Conference**: CVPR 2026  
**arXiv**: [2511.17699](https://arxiv.org/abs/2511.17699)  
**Code**: https://github.com/sharif-ml-lab/counting-mechanisms (Available)  
**Area**: Mechanistic Interpretability / Multi-modal VLM  
**Keywords**: Mechanistic Interpretability, Counting, Activation Patching, Causal Mediation Analysis, Internal Counters

## TL;DR
Authors utilize a set of controlled "repetitive object counting" experiments and a self-developed causal probing tool, CountScope, to dissect LLMs and LVLMs layer-by-layer and token-by-token. They find that counting is not a one-time summation but a hierarchical process emerging across layers, driven by "internal counters" that update incrementally and rely heavily on structural shortcuts like delimiters.

## Background & Motivation
**Background**: Counting is a fundamental capability of LLMs/LVLMs, underpinning tasks such as object enumeration, quantitative description, and arithmetic reasoning. However, extensive prior work shows that models often fail even at simple tasks like "counting a string of repeated items"—suffering from omissions, repetitions, and sensitivity to prompt phrasing.

**Limitations of Prior Work**: Previous research has almost exclusively focused on the **behavioral level**—measuring whether a model "counts correctly" using accuracy or different prompts. Scant attention has been paid to the internal workings of the model to see **exactly where and through what mechanisms** these numerical concepts are represented and updated. In short, the community knows models fail at counting but does not know what the internal counting circuit looks like.

**Key Challenge**: Answering questions about "internal mechanisms" requires causal methods rather than correlational observations. Directly using existing correlational probes (e.g., Logit Lens / Tuned Lens) to extract numerical concepts results in extreme noise and lacks stable patterns, especially in LVLMs due to the gap between text and vision modalities. What is missing is a **causal decoding tool specifically designed for numerical information**.

**Goal**: (1) Locate where counting-related information resides within model tokens, regions, and layers; (2) Characterize how these numerical representations emerge across layers and modalities; (3) Reveal the internal mechanism of count updates (total summation vs. incremental counter).

**Key Insight**: Instead of using uncontrollable latent variables in natural corpora, the authors construct a **fully controlled synthetic dataset** (text lists and synthetic geometric images where object types and quantities are precisely adjusted). They then use activation patching for causal intervention—directly modifying intermediate activations and observing changes in output to determine which components causally decide the final count.

**Core Idea**: Design CountScope—transferring activations of a target token into a minimal "placeholder + counting question" target context. This allows the model itself to decode "what number is hidden in this activation," precisely extracting potential counts encoded within any token, layer, or modality.

## Method

### Overall Architecture
Ours is not a new model architecture but a **mechanistic interpretability experiment and analysis framework**. The goal is to see "what happens internally when LLMs/LVLMs count." The framework consists of three levels: **Controlled Testbed → Causal Probing Tool → Targeted Analyses**.

Controlled Testbed: Models are presented with a sequence of repetitive objects (text like "apple, apple, apple"; visual images containing 1–9 geometric shapes) followed by the question "How many?". Text-side variables include list-first vs. question-first, monotypic vs. polytypic items, and specific vs. general questions. The visual side corresponds with single/multi-type layouts. All variables are precisely controllable.

Analyses follow three categories: **Behavioral Analysis** (accuracy, in appendix), **Observational Analysis** (passive viewing of internal representations via PCA and cosine similarity), and **Causal Mediation Analysis** (active intervention on activations, the core of this paper). The engine for causal analysis is activation patching—swapping or zeroing activations between a source (clean) context $C$ and a target (corrupted) context $C'$ to produce a manipulated $C^*$, then comparing the change in probability of the correct number.

The core tool built upon this is **CountScope** (Section 4.1). Subsequent analysis subsections (Input-level localization, Internal counters, Maximum latent count, Layer-wise emergence, Linear additivity, Delimiter shortcuts) use CountScope/patching to verify specific hypotheses about "how the model counts."

### Key Designs

**1. CountScope: A Causal Probe to "Read" Numerical Information from Activations**

This is the methodological cornerstone, addressing the noise issues in Logit/Tuned Lens. CountScope's approach, inspired by PatchScope, constructs a **minimal target context**: containing one or more placeholder tokens (text or visual) followed by a counting question. The internal activations of a source token/layer/region are patched into these placeholders. The output token's softmax probability then reveals "what number is encoded in that activation." Because the target context is clean and focused only on counting, it **causally** isolates the counting information from other semantic noise.

The evaluation metric provided is **Causal Influence (CI)**, inspired by Indirect Effect. Let $\tilde{r}$ be the expected answer under a hypothesis and $r'$ be the competitive answer:

$$\mathrm{CI}=\tfrac{1}{2}\Big[\big(P(\tilde{r}\,|\,C^{*})-P(\tilde{r}\,|\,C')\big)+\big(P(r'\,|\,C')-P(r'\,|\,C^{*})\big)\Big]$$

High CI supports the hypothesis, while 0 indicates the intervention has only random impact.

**2. Layer-wise Emergence: Early for Small Numbers, Late for Large Numbers**

First revealed by observational analysis (PCA + cosine similarity). Authors recorded hidden representations of tokens for counts 1–9. PCA on item embeddings and cosine similarity matrices across list positions show: **Representations of small numbers are separated in shallow layers, while large numbers only distinguish themselves in deep layers**. Numerical magnitude is encoded progressively along the model hierarchy. Similarity matrices show a clear diagonal structure emerging early for small numbers and extending to larger numbers in deeper layers, following a log-like pattern where larger numbers are closer in PCA space—echoing human number sense.

**3. Input-level Localization: Counts in "Context" Not "Question," Text in Final Item, Vision in Background**

Using offline zero patching and interchange patching, results show: context zeroing causes correct number probability to plummet (LLM $0.73\pm0.02$, LVLM $0.65\pm0.31$), while zeroing the question has almost no impact ($0.03$ / $0.05$). Swapping context activations makes the prediction follow the source (CI LLM $0.61$, LVLM $0.57$). In text tasks, **dropping the activation of the final item causes the biggest drop ($0.95\pm0.04$)**, indicating the final token encodes the total count. In vision tasks, counter-intuitively, a significant portion of the counting signal resides in **background patches** rather than the objects themselves ($0.61$ background vs. $0.42$ foreground at 10×10 patch size). This stems from the global spatial attention of the vision encoder.

**4. Internal Counter Mechanism: Memoryless, Type-independent, Context Maxima, Linearly Additive**

Core mechanistic findings answering "summation vs. incremental counter":

- **Per-Item Latent Count**: Probing items via CountScope reveals each item carries its sequence position; embeddings encode latent numerical order.
- **Continued Counting (Markov-like)**: Patching source final $k$ items into target first $k$ items results in predictions of $\tilde{r}=N_{\text{source}}+N_{\text{target}}-k$. The counter **does not recalculate the sum but continues counting from the source's end**, relying only on the most recent latent count.
- **Type-Specific Resetting**: In polytypic cases with clusters (apple, apple, orange...), the model maintains **one counter per category**. If a category is interrupted and reappears, the counter resets (counting 4 instead of 6), dependent on distance.
- **Maximum Latent Count**: When only swapping final items, if source count > target, prediction $=N_{\text{source}}$. If source < target, prediction $\approx N_{\text{target}}-1$. The model **evaluates the entire context and takes the maximum latent count**.
- **Linear Additivity**: "Position difference vectors" from mean embeddings can shift the decoded number of an item when added to its activation. Latent counts are **approximately linear and decoupled from object type**.

**5. Delimiter Shortcut: Commas Predict Count Better than Objects**

CountScope probing of delimiter embeddings (e.g., commas) shows **delimiters alone encode sufficient position information**, often with higher predictability than element tokens. Forcing all delimiters to the activation of the "first delimiter" causes correct probability to drop significantly (polytypic $0.97\pm0.05$). Models treat **delimiters as structural shortcuts for tracking counts**, explaining failures when delimiters are misused.

### Loss & Training
No models were trained. All experiments were conducted on frozen open-source models: Llama3, Qwen2.5, InternVL3.5, and Qwen2.5-VL. Primary results report on Qwen2.5-7B and Qwen2.5-VL-7B. Evaluation processed on synthetic datasets with precise control over object identity and quantity.

## Key Experimental Results

### Main Results
**Input-level Localization (context vs question)**—Information resides in the context:

| Intervention | Metric | LLM | LVLM |
|------|------|-----|------|
| Zero context | Prob. Drop | $0.73\pm0.02$ | $0.65\pm0.31$ |
| Zero question | Prob. Drop | $0.03\pm0.02$ | $0.05\pm0.03$ |
| Swap context | CI | $0.61\pm0.02$ | $0.57\pm0.12$ |
| Swap question | CI | $0.02\pm0.01$ | $0.03\pm0.04$ |
| Zero final item (text) | Prob. Drop | $0.95\pm0.04$ | — |

**Vision Counting Signal Foreground vs Background** (Mean correct prob. via CountScope):

| Region | 3×3 patch | 6×6 patch | 10×10 patch |
|------|-----------|-----------|-------------|
| Foreground | $0.48\pm0.02$ | $0.46\pm0.01$ | $0.42\pm0.01$ |
| Background | $0.44\pm0.03$ | $0.58\pm0.02$ | $0.61\pm0.01$ |

**Practical Insight: Scaling via Spatial Structures** (Accuracy for 10–20 objects):

| Setting | Accuracy |
|------|--------|
| Unstructured + No CoT (Baseline) | 10.0% |
| Text CoT (No image partition) | 15.3% |
| Image partitioned into 4 (Structured) | **50.4%** |

Dividing the image via lines allows the model to decompose counting into sub-problems, significantly outperforming text CoT.

### Ablation Study
The paper validates experimental hypotheses using CI scores (higher = stronger support):

| Mechanism Hypothesis | Configuration | Key CI | Note |
|----------|------|---------|------|
| Continued Counting | LLM question-first, k=2 | $0.90\pm0.07$ | Most obvious in question-first config |
| Continued Counting | LLM question-last, k=2 | $0.16\pm0.05$ | Barely exists in question-last |
| Maximum Latent Count | LLM question-first, $N_s{>}N_t$, k=2 | $0.80\pm0.13$ | Prediction follows source count |
| Linear Additivity | LLM, Position diff k=3 | $0.85\pm0.11$ | Intervening layers 21-26 shifts count |
| Delimiter Shortcut | Polytypic Delimiter Patch | Prob Drop $0.97$ | Model follows fake delimiter pattern |

### Key Findings
- **Counting is constrained by Transformer depth**: Numbers 1→9 emerge sequentially between layers 5 and 16. Larger numbers require deeper layers, explaining failures in long sequences.
- **Mechanism sensitivity to prompt layout**: Markovian counting and maximum latent count are most significant in **question-first (LVLM: system-prompt)** setups.
- **LVLM sensitivity drop after 5 objects**: Visual models lose internal signal sensitivity after count > 5, whereas text models remain stable up to 10.
- **Localization differences**: Text signals are localized at the final item; visual signals are global and often strongest in the background.

## Highlights & Insights
- **CountScope as a decoding probe**: By patching activations into a minimal context, it causally isolates discrete scalar concepts (order, count).
- **"Memoryless Counter" as an "Aha!" moment**: Realizing the model does not sum but follows a Markovian state machine ($N_{\text{source}}+N_{\text{target}}-k$) clarifies how Transformers process sequences.
- **Delimiters as hidden protagonists**: The discovery that commas are more predictive than items themselves identifies "structural markers" as a primary lever for counting performance.
- **Scalable engineering insight**: Since counting is depth-constrained, spreading the "computation" across space (image partitioning) improved 10-20 object accuracy from 10% to 50% without retraining.

## Limitations & Future Work
- **Lack of full circuit discovery**: The work characterizes mechanisms but does not map the full attention-head/MLP circuit.
- **System-2 mechanisms unexplored**: Internal mechanisms of CoT-based counting remain unknown.
- **Synthetic data bias**: Primary analysis relies on synthetic lists/geometric images; validity in natural scenes with occlusion and high density requires verification.
- **Prompt sensitivity**: Findings vary across prompt structures, raising questions about robustness in arbitrary layouts.

## Related Work & Insights
- **vs Logit/Tuned Lens**: CountScope is a causal tool that outperforms correlational probes in the noisy numerical domain.
- **vs PatchScope**: Ours adapts the "activation transfer" paradigm specifically for counting with a custom target context and CI scoring.
- **vs Behavioral Benchmarks**: Moves beyond "does it count" to "where and how does it count."
- **vs System-2 CoT**: While CoT expands computation via text tokens, Ours suggests "spatial tokens" (image partitioning) as an orthogonal way to distribute counting load.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic causal characterization of counting in LLM+LVLM.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-modal validation; slightly biased toward synthetic data.
- Writing Quality: ⭐⭐⭐⭐ Clear hypotheses, though mechanisms are complex and require careful reading of charts.
- Value: ⭐⭐⭐⭐⭐ Explains depth-constraints and provides practical structural tricks (image partitioning).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](../../ICML2026/interpretability/dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](../../ACL2026/interpretability/fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](../../ACL2026/interpretability/preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](../../ACL2026/interpretability/mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)

</div>

<!-- RELATED:END -->
