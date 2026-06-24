---
title: >-
  [Paper Note] The Impact of Token Granularity on the Predictive Power of Language Model Surprisal
description: >-
  [ACL 2025][LLM (Other)][token granularity] This paper systematically investigates the impact of subword token granularity (vocabulary sizes from 256 to 128K) on the ability of LM surprisal to predict human reading times. It finds that a moderate granularity of ~8K vocabulary performs best for predicting natural reading times (even outperforming GPT-2), while coarser-grained tokens are more sensitive to garden-path syntactic effects, revealing that the optimal tokenization gra…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "token granularity"
  - "subword tokenization"
  - "surprisal"
  - "cognitive modeling"
  - "reading times"
date: 2026-05-08
content_hash: a832c2c225812547
---

# The Impact of Token Granularity on the Predictive Power of Language Model Surprisal

**Conference**: ACL 2025  
**arXiv**: [2412.11940](https://arxiv.org/abs/2412.11940)  
**Code**: [GitHub](https://github.com/byungdoh/ssm-surprisal)  
**Area**: Cognitive Modeling / Computational Psycholinguistics  
**Keywords**: token granularity, subword tokenization, surprisal, cognitive modeling, reading times

## TL;DR

This paper systematically investigates the impact of subword token granularity (vocabulary sizes from 256 to 128K) on the ability of LM surprisal to predict human reading times. It finds that a moderate granularity of ~8K vocabulary performs best for predicting natural reading times (even outperforming GPT-2), while coarser-grained tokens are more sensitive to garden-path syntactic effects, revealing that the optimal tokenization granularity for cognitive modeling does not align with general NLP standards.

## Background & Motivation

**Language model surprisal is a core tool in cognitive modeling.** Under the theoretical frameworks of Hale (2001) and Levy (2008), word-level surprisal ($-\log P(w_i | w_{<i})$) is widely used as a predictor of human word-by-word processing difficulty. In recent years, neural language models such as Transformers have been used to calculate surprisal and fit it to human reading times, providing computational models for understanding predictive language processing.

**While factors affecting surprisal quality have been studied, the fundamental variable of token granularity has been overlooked.** The effects of model architecture and training data on surprisal quality have been investigated (Oh and Schuler, 2023; Shain et al., 2024), but the granularity of subword tokenization—specifically, vocabulary size—on cognitive modeling capability has never been systematically explored.

**Token granularity impacts surprisal quality through two pathways.** First, **the initial bias pathway**: fine-grained tokenization (small vocabulary) splits low-frequency long words into multiple tokens, implicitly encoding word length and frequency information—for instance, when "journey" is split into 7 tokens, its probability under a uniform distribution is 6 orders of magnitude lower than "to". Coarse-grained tokenization (large vocabulary) keeps most words intact, leading to more uniform initial probabilities. Because word length and frequency are key variables affecting human reading processing (Barton et al., 2014; Just and Carpenter, 1980), certain tokenization schemes are naturally better suited for predicting reading times. Second, **the representation quality pathway**: coarse-grained tokens learn representations closer to word-level co-occurrence statistics (similar to Word2Vec), whereas fine-grained tokens scatter a single word across multiple vectors, increasing the difficulty of learning associations between words.

## Method

### Overall Architecture

(1) Train 11 tokenizers with different vocabulary sizes (256 to 128K) using the Unigram LM (ULM) tokenizer → (2) Train language models of three experimental scales based on the Mamba-2 architecture → (3) Evaluate the predictive power of surprisal on 5 reading-time corpora (across 10 metrics) → (4) Evaluate syntactic sensitivity on garden-path syntactic constructions.

### Key Designs

1. **Systematic control of 11 token granularity levels**:
    - Function: Train ULM tokenizers with vocabulary sizes $|V| \in \{256, 512, 1K, 2K, 4K, 8K, 16K, 32K, 48K, 64K, 128K\}$.
    - Mechanism: The ULM tokenizer uses characters as basic units (rather than BPE's bytes) and iteratively prunes the vocabulary by maximizing the joint probability of subword sequences. It is trained on 1 million Wiki-40B articles. $|V|=256$ is close to character-level ("journey" → 7 tokens), and $|V|=128K$ is close to word-level ("journey" → 1 token).
    - Design Motivation: ULM (instead of BPE) is chosen because characters are more interpretable than bytes; a extremely wide range from 256 to 128K is covered to comprehensively map the granularity-quality relation.

2. **Mamba-2 architecture to resolve incomparable sequence lengths**:
    - Function: Train LMs using State Space Models (SSMs) instead of Transformers.
    - Mechanism: Different granularities lead to massive differences in token sequence lengths for the same text (sequence length under $|V|=256$ can be several times that under $|V|=128K$). The self-attention mechanism of Transformers has $O(n^2)$ space complexity, making it unfriendly to long sequences. Mamba-2, based on the linear complexity of SSM, is naturally suited for handling variable sequence lengths. Three scales—Small/Medium/Large—are trained: 6/12/24 layers, 256/512/768 embedding dimensions, with 2.6M/19.8M/88M parameters (excluding embedding layers).
    - Design Motivation: If a Transformer were used with a fixed maximum length, LMs of different granularities would condition on different amounts of context, compromising the fairness of the experiment.

3. **Whitespace probability correction**:
    - Function: Correct issues in normalizing word probabilities derived from subword token probabilities.
    - Mechanism: The ULM tokenizer prepends a whitespace prefix to tokens. Naively computing word probabilities could result in the sum of all word probabilities exceeding 1 (as word end positions are unmarked). The correction method of Oh and Schuler (2024) is applied to redistribute the whitespace probability to the preceding token.
    - Design Motivation: Ensuring that word-level surprisal is theoretically correct in terms of probability.

### Loss & Training

Standard causal LM objective (next-token prediction). Models are trained for one epoch on the full Wiki-40B English training set (5,152,219 training samples, 10,063 batches × 512 samples) using the AdamW optimizer (maximum learning rate $10^{-3}$, cosine annealed to $10^{-5}$), gradient clipping norm=1, and half-precision training on a 48GB RTX 8000 GPU.

## Key Experimental Results

### Main Results—Natural Reading Time Prediction (higher ΔLogLik is better)

| Phase | Best Vocab | ΔLogLik | Worst Vocab | ΔLogLik | GPT-2 Small |
|------|---------|---------|---------|---------|-------------|
| Pre-training (Tokenizer-only) | $|V|=4K$ | 2553 | $|V|=128K$ | 1899 | — |
| Post-training (Small Avg.) | $|V|=8K$ | Highest | $|V|=128K$ | Lowest | Reference Line |
| Post-training (Large Avg.) | Gap narrows | — | — | — | — |
| Post-training (Avg. across 3 scales) | **$|V|=8K$** | **Optimal** | $|V|=128K$ | Lowest | Outperformed by 8K |

### Garden-Path Experiment (larger GPE = higher sensitivity)

| Syntactic Construction | Coarse-grained (large vocab) Trend | Fine-grained (small vocab) Trend | Description |
|---------|-------------------|-------------------|------|
| MV/RR (Main Verb / Reduced Relative) | Larger GPE (~6ms, Small) | Smaller GPE (~2ms, Small) | Coarse-grained shows clear advantage |
| NP/S (Noun Phrase / Sentential Complement) | Similar but weaker trend | — | Differences across constructions |
| NP/Z (Noun Phrase / Transitive-Intransitive) | Similar but weaker trend | — | Differences across constructions |
| Human Actual Effect | — | — | All LMs underestimate by 1-2 orders of magnitude |

### Key Findings

- **Surprisal from untrained tokenizers (tokenizer-only) can already predict reading times**: At $|V|=4K$, ΔLogLik≈2553, indicating that word length and frequency information encoded purely by the tokenizer provides significant predictive power.
- **After training, a ~8K vocabulary is overall optimal**: Averaged over 5 corpora × 10 reading metrics, it even outperforms GPT-2 Small ($|V|≈50K$).
- **Interaction exists between model size and granularity**: Larger models (88M parameters) can partially overcome initial biases, narrowing the gap between different granularities.
- **Optimal vocabulary size varies by task**: Natural reading prefers moderate granularity (8K), whereas garden-path sentences prefer coarse granularity (large vocabulary)—since word-level co-occurrence statistics are more beneficial for learning syntactic relationships.
- **Perplexity and cognitive modeling quality do not perfectly align**: Large vocabularies yield lower perplexity but not necessarily higher ΔLogLik.

## Highlights & Insights

- **Revealing an overlooked critical variable**: Tokenization granularity not only affects NLP performance but also profoundly influences the quality of the model as a cognitive model—which has never been systematically studied before.
- **Comprehensiveness of experimental design**: 11 granularities × 3 model scales × 5 corpora × 10 reading metrics + garden-path experiments provide extremely comprehensive coverage.
- **Pre-trained tokenizers alone can predict reading times**: This finding indicates that human reading processing is highly sensitive to word length and frequency, which tokenizers implicitly encode.
- **Mamba-2 architectural choice**: It elegantly solves the experimental design challenge where sequence lengths of different granularities are incomparable.
- **Different tasks require different optimal granularities**: The dissociated results of natural reading vs. garden-path sentences offer direct guidance for cognitive modeling practices.

## Limitations & Future Work

- The findings are validated only on English data and native English speakers; cross-linguistic generalizability is unknown (e.g., the impact of token granularity in Chinese could be entirely different).
- The model scale is limited (maximum 88M parameters); larger models might fully overcome initial biases, rendering granularity less critical.
- The study focuses solely on cognitive modeling scenarios and does not address the impact of token granularity on NLP application performance.
- The garden-path experiment covers only 3 syntactic constructions (MV/RR, NP/S, NP/Z), showing limited construction coverage.
- All models still heavily underestimate actual human garden-path effects (by 1-2 orders of magnitude); token granularity cannot make up for this fundamental gap.

## Related Work & Insights

- **vs. Nair and Resnik (2023)**: Compared morphological tokenization vs. BPE on surprisal; this study is more systematic, exploring continuous variations across 11 granularities.
- **vs. Giulianelli et al. (2024)**: Derived character-level probabilities from GPT-2 token probabilities; this study directly alters the tokenization granularity from the source.
- **vs. Oh and Schuler (2023b) / Shain et al. (2024)**: Investigated the effect of model size on surprisal quality; this study reveals granularity as another overlooked key variable.
- **Insight**: Future work in cognitive modeling should report tokenization granularity as a standard variable rather than defaulting to GPT-2/Llama's ~50K vocabulary.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically investigates an overlooked and important variable with exquisite experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage spanning 11 granularities × 3 scales × 5 corpora × 10 metrics + garden path.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, complementary experiments, and explicit conclusions.
- Value: ⭐⭐⭐⭐ Offers direct guidance for cognitive modeling practices and holds fundamental research significance for understanding token-cognition relationships.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Clozing the Gap: Exploring Why Language Model Surprisal Outperforms Cloze Surprisal](../../ACL2026/llm_nlp/clozing_the_gap_exploring_why_language_model_surprisal_outperforms_cloze_surpris.md)
- [\[ACL 2025\] Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] Circuit Stability Characterizes Language Model Generalization](circuit_stability_characterizes_language_model_generalization.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)

</div>

<!-- RELATED:END -->
