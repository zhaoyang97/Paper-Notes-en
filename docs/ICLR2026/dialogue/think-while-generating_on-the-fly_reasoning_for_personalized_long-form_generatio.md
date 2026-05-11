---
title: >-
  [Paper Note] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation
description: >-
  [ICLR 2026][Dialogue Systems][personalized generation] FlyThinker proposes an efficient "think-while-generating" framework that employs a dedicated reasoning model (Reasoner) to generate latent reasoning signals in paral…
tags:
  - "ICLR 2026"
  - "Dialogue Systems"
  - "personalized generation"
  - "long-form generation"
  - "latent reasoning"
  - "think-while-generating"
  - "parallel reasoning"
date: 2026-05-08
content_hash: 6386cd70a70212fb
---

# Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation

**Conference**: ICLR 2026
**arXiv**: [2512.06690](https://arxiv.org/abs/2512.06690)
**Code**: None
**Area**: Dialogue Systems
**Keywords**: personalized generation, long-form generation, latent reasoning, think-while-generating, parallel reasoning

## TL;DR

FlyThinker proposes an efficient "think-while-generating" framework that employs a dedicated reasoning model (Reasoner) to generate latent reasoning signals in parallel at the token level, dynamically incorporating them into a generation model (Generator) to guide personalized long-form generation, while preserving both training and inference efficiency.

## Background & Motivation

Preference alignment enables LLMs to better reflect human expectations; however, existing approaches primarily optimize for group-level preferences, neglecting individual user needs. Personalized long-form generation faces three key challenges:

**Implicit preferences are difficult to infer**: User interests are typically embedded in historical behaviors, and simple prompt customization or fine-tuning is insufficient to effectively reason over such preferences.

**Limitations of "think-then-generate"**: Existing reasoning methods complete all reasoning in a single pass before generation, producing a static analysis. For long-form text, this one-shot reasoning must encompass information for the entire response, making it difficult to learn and unable to adapt to the dynamic evolution of content.

**Efficiency bottleneck**: Although the alternating reasoning-generation paradigm of "think-while-generating" is intuitively appealing, frequent reasoning steps significantly increase both training and inference time.

The core insight of FlyThinker is to decouple reasoning and generation into two independent models and eliminate direct sequential dependencies between reasoning tokens, thereby enabling true parallelization of reasoning and generation.

## Method

### Overall Architecture

FlyThinker consists of two parallel models:
- **Reasoner (R)**: Produces one latent reasoning token at each generation step, conditioned on the query and the previously generated response.
- **Generator (G)**: An augmented LLM that integrates latent reasoning signals into token prediction.

Core formulation:

$$\text{Generation:} \quad (h,x) \xrightarrow{(h,x;\hat{y}_{<1}+r_{<1})} \hat{y}_1 \xrightarrow{(h,x;\hat{y}_{<2}+r_{<2})} \hat{y}_2 \dots$$

$$\text{Reasoning:} \quad (h,x) \xrightarrow{(h,x,\hat{y}_{<1})} r_1 \xrightarrow{(h,x,\hat{y}_{<2})} r_2 \dots$$

### Key Designs

1. **Latent reasoning token generation (Reasoner)**: At each step $t$, the Reasoner extracts latent reasoning from its last-layer hidden state: $r_t = R_\theta^{(-1)}(h,x; \hat{y}_{<t-1})[-1]$. Crucially, $r_t$ **does not depend on previous reasoning tokens** $r_{<t}$, but only on the already-generated response $\hat{y}_{<t-1}$, thereby breaking sequential dependencies among reasoning tokens.

2. **Reasoning signal fusion (Generator)**: The Generator injects reasoning into the token embedding space via additive fusion: $f(\hat{y}_{<t}, r_{<t}) = [e(y_1) + \lambda r_1, \dots, e(y_{t-1}) + \lambda r_{t-1}]$, where $\lambda$ controls the strength of the reasoning signal.

3. **Parallel training**: Since $r_t$ does not depend on $r_{<t}$, the complete target sequence $y$ can be fed into the Reasoner in a single forward pass during training, obtaining reasoning tokens $r^\star = [r_1, \dots, r_T]$ for all positions simultaneously. The Generator can then also compute predictions for all positions in parallel, yielding training efficiency comparable to standard LLM training.

4. **Parallel inference**: During inference, while the Generator predicts the current token, the Reasoner prepares the reasoning token for the next step in parallel. This staggered design eliminates waiting time, resulting in inference latency close to that of a standard non-reasoning LLM.

### Loss & Training

The Reasoner and Generator are jointly optimized using the standard next-token prediction loss:

$$\mathcal{L} = -\sum_{(h,x,y) \in \mathbb{D}} \sum_{t=1}^{|y|} \log P(\hat{Y}_t = y_t \mid h,x, y_{<t})$$

No external reasoning annotations or auxiliary objectives are required; the Reasoner naturally learns to generate useful reasoning signals through end-to-end training.

## Key Experimental Results

### Main Results

**LongLaMP Benchmark (Qwen2.5-3B-Instruct backbone):**

| Method | Product Review (BLEU) | Abstract Gen. (BLEU) | Topic Writing (BLEU) |
|------|---------------------|---------------------|---------------------|
| Non-pers | 1.54 | 4.58 | 1.12 |
| RAG | 3.30 | 3.40 | 1.43 |
| SFT | 3.91 | 5.82 | 3.89 |
| CoT | 3.37 | 5.85 | 3.00 |
| Coconut | 3.32 | 5.24 | 3.07 |
| **FlyThinker** | **4.36** | **6.34** | **4.06** |

FlyThinker outperforms all baselines on every task, achieving approximately 10% BLEU improvement over SFT.

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| Reasoner 3B→1.5B | Performance largely preserved | Moderate size reduction does not harm quality; more training-efficient |
| Reasoner 3B→0.5B | Noticeable drop in ROUGE-L/BLEU | Excessively small Reasoner lacks sufficient capacity |
| $\lambda$=0 (no reasoning) | Degrades to SFT | Reasoning signals are indispensable |
| $\lambda \in [0.2, 2.0]$ | All outperform SFT | Method is robust to the choice of $\lambda$ |
| $\lambda$=5 (too large) | Performance degrades | Overly strong reasoning signals interfere with generation |

### Key Findings

- **Position-sensitive evaluation**: All baseline methods exhibit a significant drop in personalization quality in later segments (tokens 100–300), a phenomenon termed "context drift." FlyThinker maintains high quality in later segments, effectively mitigating preference forgetting in long-form generation.
- **Training efficiency**: FlyThinker's training time is only marginally higher than SFT and far lower than CoT and Coconut.
- **Inference efficiency**: Inference latency approaches that of SFT and is substantially faster than the sequential reasoning of CoT and Coconut.
- **The Reasoner can be scaled down to 1.5B without quality loss**, offering a favorable cost–performance trade-off.

## Highlights & Insights

1. **First efficient realization of the "think-while-generating" paradigm**: Prior instantiations of this concept were impractical due to efficiency concerns; FlyThinker elegantly resolves this via an independent Reasoner and the elimination of sequential reasoning dependencies.
2. **Alignment with human long-form writing behavior**: Humans naturally reason as they write rather than planning everything upfront; FlyThinker's token-level dynamic reasoning is a natural reflection of this process.
3. **Highly elegant engineering**: A single forward pass through the Reasoner during training produces all reasoning tokens; staggered parallelism at inference time introduces virtually no additional overhead.
4. **Effective countermeasure against "context drift"**: Position-sensitive experiments clearly demonstrate that dynamic reasoning substantially improves generation quality in the later portions of long-form outputs.

## Limitations & Future Work

1. **Increased memory footprint**: Although time-efficient, the approach requires maintaining two models simultaneously (Reasoner + Generator), effectively doubling memory consumption.
2. **Limited evaluation metrics**: Only automatic metrics such as ROUGE, BLEU, and METEOR are employed; human evaluation and GPT-based assessment are absent.
3. **Narrow task scope**: Validation is limited to personalized long-form generation; applicability to other tasks requiring dynamic reasoning remains unexplored.
4. **Uninterpretable reasoning content**: Latent reasoning tokens are hidden-state vectors, making it impossible to inspect whether the underlying reasoning logic is sound.

## Related Work & Insights

- **Coconut** (ICLR 2025) conducts reasoning in latent space but follows a think-then-generate paradigm; FlyThinker extends this to think-while-generating.
- **REST-PG** and **R2P** employ explicit reasoning chains for personalization but suffer from low efficiency.
- The **LongLaMP** benchmark establishes a systematic evaluation framework for personalized long-form generation.
- This work offers important insights for the **reasoning-augmented generation** field: reasoning and generation need not be serialized and can be parallelized.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ An efficient realization of think-while-generating; the design that breaks sequential reasoning dependencies is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three tasks, efficiency analysis, position-sensitive evaluation, and ablation studies are included, though human evaluation is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clearly articulated; the three-paradigm comparison is intuitive; formulations are concise.
- **Value**: ⭐⭐⭐⭐ Provides meaningful insights for both personalized generation and reasoning-augmented methods, though the scope of application awaits broader exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](../../ACL2026/dialogue/apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ICLR 2026\] AQuA: Toward Strategic Response Generation for Ambiguous Visual Questions](aqua_toward_strategic_response_generation_for_ambiguous_visual_questions.md)
- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](../../ACL2026/dialogue/agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[CVPR 2026\] Evolutionary Multimodal Reasoning via Hierarchical Semantic Representation for Intent Recognition](../../CVPR2026/dialogue/evolutionary_multimodal_reasoning_via_hierarchical_semantic_representation_for_i.md)

</div>

<!-- RELATED:END -->
