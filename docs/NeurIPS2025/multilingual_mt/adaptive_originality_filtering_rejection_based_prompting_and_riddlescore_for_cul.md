---
title: >-
  [Paper Note] Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation
description: >-
  [NeurIPS 2025][Multilingual & Machine Translation][Adaptive Originality Filtering] This paper proposes Adaptive Originality Filtering (AOF)—a semantic rejection-sampling prompting strategy that filters repetitive or templated outputs via MiniLM embedding cosine similarity, compelling LLMs to generate more novel, diverse, and culturally grounded multilingual riddles. It also introduces the RiddleScore composite evaluation metric (Novelty + Diversity + Fluency + Alignment)…
tags:
  - "NeurIPS 2025"
  - "Multilingual & Machine Translation"
  - "Adaptive Originality Filtering"
  - "RiddleScore"
  - "multilingual riddle generation"
  - "rejection sampling"
  - "cultural adaptation"
date: 2026-05-08
content_hash: 165e8d4f6e214b13
---

# Adaptive Originality Filtering: Rejection-Based Prompting and RiddleScore for Culturally Grounded Multilingual Riddle Generation

**Conference**: NeurIPS 2025
**arXiv**: [2508.18709](https://arxiv.org/abs/2508.18709)  
**Code**: None  
**Area**: NLP Generation / Multilingual Creative Generation
**Keywords**: Adaptive Originality Filtering, RiddleScore, multilingual riddle generation, rejection sampling, cultural adaptation

## TL;DR
This paper proposes Adaptive Originality Filtering (AOF)—a semantic rejection-sampling prompting strategy that filters repetitive or templated outputs via MiniLM embedding cosine similarity, compelling LLMs to generate more novel, diverse, and culturally grounded multilingual riddles. It also introduces the RiddleScore composite evaluation metric (Novelty + Diversity + Fluency + Alignment), achieving a human correlation of $\rho=0.83$.

## Background & Motivation

**Background**: LLMs underperform on creative multilingual tasks such as riddle generation—standard prompting strategies (zero-shot, few-shot, CoT) tend to produce templated and repetitive outputs, with particularly poor cultural adaptation in non-English languages.

**Limitations of Prior Work**: Existing evaluation metrics (BLEU, BERTScore, perplexity) fail to capture the core dimensions of creative quality—novelty, cultural relevance, and richness of rhetorical devices. Riddles require metaphor, misdirection, and cultural resonance, none of which fall within the scope of standard metrics.

**Key Challenge**: LLM pretraining data is skewed toward high-frequency patterns; creative generation demands deviation from these patterns (greater novelty), yet excessive deviation may compromise fluency and answer alignment.

**Goal**: How can LLMs be guided to generate more novel, diverse, and culturally adapted multilingual riddles? How should the quality of such creative generation be evaluated?

**Key Insight**: Enforce novelty within the generation loop through semantic filtering (rejection sampling), while designing a composite metric to balance four quality dimensions.

**Core Idea**: Detect semantic proximity among LLM outputs, reject candidates that are too similar to existing riddles, and thereby compel the model to produce more original content.

## Method

### Overall Architecture
Generation loop: LLM generates candidate riddle → MiniLM encoding → cosine similarity computed against reference set → candidates exceeding threshold $\theta=0.75$ are rejected and regenerated (up to $k$ attempts) → accepted riddles undergo final verification → evaluated with RiddleScore.

### Key Designs

1. **Semantic Rejection Sampling**:

    - Function: For each candidate riddle, cosine similarity with the reference set (BiRdQA corpus) is computed via MiniLM; candidates with similarity $> 0.75$ are rejected.
    - Mechanism: A candidate is accepted only if $\cos(\text{MiniLM}(r_{\text{candidate}}), \text{MiniLM}(r_{\text{reference}})) \leq 0.75$. The threshold 0.75 is determined via sensitivity analysis as optimal—lower values permit template leakage, while higher values increase the failure rate by 14%.
    - Design Motivation: LLMs tend to reproduce riddles seen during pretraining; semantic filtering enforces originality without modifying model parameters.

2. **RiddleScore Composite Metric**:

    - Function: Four-dimensional weighted score $= \alpha \cdot \text{Novelty} + \beta \cdot \text{Diversity} + \gamma \cdot \text{Fluency} + \delta \cdot \text{Alignment}$
    - Mechanism: Novelty = cosine distance from BiRdQA (MiniLM); Diversity = Distinct-2 bigram ratio; Fluency = inverse perplexity from GPT-2; Alignment = BERTScore(riddle, answer). Weights $\alpha=0.30, \beta=0.20, \gamma=0.30, \delta=0.20$ are determined via grid search to maximize Spearman $\rho$ against human ratings.
    - Design Motivation: A single metric cannot capture the multifaceted nature of riddle quality; the weighted combination achieves $\rho=0.83$ on a 120-sample development set, outperforming uniform weighting at 0.71.

3. **Prompt-level Constraints**:

    - Function: Cultural element constraints are injected into the system prompt (e.g., requiring language-specific rhetorical devices, metaphors, and cultural imagery).
    - Mechanism: Prompt templates incorporate language-specific creative constraints (e.g., Japanese prompts require waka-style metaphors), guiding LLMs toward culturally grounded outputs.
    - Design Motivation: Semantic filtering alone is insufficient to ensure cultural adaptation; explicit guidance at the prompt level is necessary.

### Loss & Training
- AOF does not modify model parameters; it is a purely inference-time method.
- Fine-tuning variant: GPT-4o is supervised fine-tuned on BiRdQA with cross-entropy loss.
- Temperature = 0.7, balancing diversity and quality.
- Five languages: English, Chinese, Arabic, Japanese, and French.

## Key Experimental Results

### Main Results (GPT-4o + AOF vs. Other Prompting Strategies)

| Language | Metric | AOF | CoT | Few-Shot | Zero-Shot |
|----------|--------|-----|-----|----------|-----------|
| Japanese | Self-BLEU ↓ | **0.177** | 0.483 | 0.421 | 0.512 |
| Japanese | Distinct-2 ↑ | **0.915** | 0.732 | 0.781 | 0.698 |
| Arabic | RiddleScore ↑ | **+57.1%** | baseline | - | - |
| Chinese (fine-tuned) | RiddleScore ↑ | 0.728 | 0.453 | - | - |

### Ablation Study

| Configuration | RiddleScore | Notes |
|---------------|------------|-------|
| AOF (θ=0.75) | Best | Optimal threshold |
| θ=0.60 | Decreased | Permits template leakage |
| θ=0.85 | Slight decrease | Failure rate increases by 14% |
| Uniform-weight RiddleScore | ρ=0.71 | Suboptimal weighting |
| Optimized-weight RiddleScore | ρ=0.83 | Grid-search optimum |

### Key Findings
- **AOF is most effective for Japanese**: Self-BLEU decreases by 63.4%, indicating that semantic filtering is particularly effective for morphologically rich languages.
- **Fine-tuning + AOF yields larger gains than pretraining + AOF**: Chinese RiddleScore improves by +48.3%, suggesting that AOF better unlocks riddle knowledge already encoded in fine-tuned models.
- **Large cross-lingual variation**: The RiddleScore gain for Arabic (+57.1%) far exceeds its Distinct-2 gain, indicating uneven improvement across quality dimensions in different languages.
- **Human evaluation confirms automatic metric results**: Fluency, creativity, and cultural adaptation all show improvement across evaluation dimensions.

## Highlights & Insights
- **Semantic rejection sampling is a general-purpose creative enhancement strategy**: It requires no fine-tuning, no architectural changes, and adds only a filtering loop at inference time. It is transferable to any creative generation task (poetry, jokes, stories, etc.).
- **RiddleScore's design methodology is generalizable**: Task-specific quality dimensions are decomposed into independently computable sub-metrics, whose weights are then optimized via correlation with human ratings. This "composite metric design methodology" can be applied to other creative NLP tasks.
- **Cultural adaptation requires explicit constraints**: Multilingual models alone are insufficient; cultural knowledge must be injected at the prompt level.

## Limitations & Future Work
- **Validated only on riddle generation**: Although generalizability to broader creative tasks is claimed, it is not empirically verified.
- **Dependence on an external embedding model (MiniLM)**: The quality of semantic filtering is bounded by MiniLM's multilingual capabilities.
- **Generalizability of threshold 0.75**: Different tasks and domains may require different thresholds.
- **Computational overhead**: Rejection sampling may require multiple LLM calls (retry on failure), increasing API costs.
- **Limited evaluation dataset**: Reliance solely on BiRdQA (15K riddles) may be insufficient to cover all cultural backgrounds.

## Related Work & Insights
- **vs. Self-Refine / Reflexion**: These methods iteratively refine outputs using the model's own feedback but do not explicitly filter for semantic similarity; AOF more directly controls novelty.
- **vs. COLD Decoding**: COLD imposes constraints at the decoding level, whereas AOF applies constraints at the post-generation filtering level; the two approaches are complementary.
- **vs. Standard Prompting Strategies (CoT/Few-Shot)**: These strategies optimize the reasoning process but do not control output diversity or novelty; AOF is orthogonal to these approaches.

## Rating
- Novelty: ⭐⭐⭐⭐ Semantic rejection sampling is conceptually straightforward, and RiddleScore follows the common practice of weighted aggregation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 LLMs, 5 languages, 5 prompting strategies, and human evaluation with good overall breadth.
- Writing Quality: ⭐⭐⭐⭐ Well-structured but somewhat verbose in places.
- Value: ⭐⭐⭐⭐ The method is simple and practical, though the application scope is narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Toward Culturally Grounded Natural Language Processing](../../ACL2026/multilingual_mt/toward_culturally_grounded_natural_language_processing.md)
- [\[ACL 2025\] LexGen: Domain-aware Multilingual Lexicon Generation](../../ACL2025/multilingual_mt/lexgen_domain-aware_multilingual_lexicon_generation.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](../../ACL2025/multilingual_mt/exploring_in-context_example_generation_for_machine_translation.md)
- [\[NeurIPS 2025\] XIFBench: Evaluating Large Language Models on Multilingual Instruction Following](xifbench_evaluating_large_language_models_on_multilingual_instruction_following.md)
- [\[NeurIPS 2025\] MERIT: Multilingual Semantic Retrieval with Interleaved Multi-Condition Query](merit_multilingual_semantic_retrieval_with_interleaved_multi-condition_query.md)

</div>

<!-- RELATED:END -->
