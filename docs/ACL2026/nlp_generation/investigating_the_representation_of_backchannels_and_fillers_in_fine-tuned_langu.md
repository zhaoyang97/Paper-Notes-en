---
title: >-
  [Paper Note] Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models
description: >-
  [ACL 2026][Text Generation][backchannel] The paper trains BERT, GPT-2, TurnGPT, LLaMA-3 8B, and Qwen-3 8B using MASK, NTP, and TTP fine-tuning tasks on English and Japanese spoken dialogue corpora. It employs t-SNE visua…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "backchannel"
  - "filler"
  - "fine-tuning"
  - "silhouette clustering"
  - "dialogue language models"
date: 2026-05-08
content_hash: f0dd2d2b5ef1a363
---

# Investigating the Representation of Backchannels and Fillers in Fine-tuned Language Models

**Conference**: ACL 2026  
**arXiv**: [2509.20237](https://arxiv.org/abs/2509.20237)  
**Code**: https://github.com/colalao/discourse_markers (Available)  
**Area**: Dialogue Generation / Representation Analysis / Discourse Markers  
**Keywords**: backchannel, filler, fine-tuning, silhouette clustering, dialogue language models

## TL;DR
The paper trains BERT, GPT-2, TurnGPT, LLaMA-3 8B, and Qwen-3 8B using MASK, NTP, and TTP fine-tuning tasks on English and Japanese spoken dialogue corpora. It employs t-SNE visualization and silhouette clustering to quantify the representation quality of backchannels (e.g., *uh-huh*) and fillers (e.g., *um*). The findings indicate that fine-tuning enables these "semantically bleached" function words to be significantly differentiated within the embedding space and allows models to produce diverse backchannels/fillers naturally during NLG, marking a quantifiable step toward "human-like conversational LMs."

## Background & Motivation
**Background**: In dialogue, backchannels (*uh-huh*, *yeah*) and fillers (*uh*, *um*) rank among the most frequent spoken words. However, NLP has traditionally treated them as stop words and removed them during preprocessing—for instance, dependency parsing on Switchboard often excludes them to improve parsing accuracy.

**Limitations of Prior Work**: (a) Mainstream text pre-training corpora contain almost no spoken markers; pre-trained LMs assign high token IDs to them, resulting in near-random embeddings. (b) LMs lacking backchannels/fillers fail to provide feedback or mark cognitive load in dialogue agent tasks, making ASR and chatbots seem less human. (c) While research exists on word-level representation changes after fine-tuning BERT/GPT-2, **systematic studies targeting low-frequency function words are rare**.

**Key Challenge**: Theoretically, backchannels/fillers are semantically bleached (lacking referential meaning), yet pragmatically they carry crucial functions like grounding, turn-taking, and disfluency. Are LMs capable of distinguishing the different pragmatic functions of a single backchannel across various contexts? Or are they genuinely limited to random vectors?

**Goal**: (RQ1) Can fine-tuning improve LM representations of backchannels/fillers? (RQ2) What role does the context window size play? (RQ3) Which models benefit the most? (RQ4) Do different fine-tuning tasks yield distinct results?

**Key Insight**: Clustering quality (silhouette score) naturally measures "whether a single backchannel is represented by multiple sub-pragmatic functions and whether different backchannels are separable." This metric serves as a "microscope" across 4 models × 3 fine-tuning tasks × 3 context settings × 2 languages.

**Core Idea**: Utilizing a suite of silhouette scores, t-SNE, and distance matrices to transform the subjective impression of "whether function words are learned" into statistically verifiable metrics. Representation improvements are cross-validated through NLG generation evaluations to see if they translate to output behavior.

## Method

### Overall Architecture
The workflow consists of "Data Selection → Three Fine-tuning Tasks → Three Context Representation Extractions → Clustering / Distance Matrix / t-SNE / NLG Evaluation."

- **Data**: English datasets include Switchboard + MapTask (~150K utterances, 127,672 backchannels/fillers); Japanese datasets use the BTSJ 1000 Person Natural Conversation Corpus (170,898 units). The top 15 most frequent items are selected for each.
- **Models**: BERT, GPT-2 (EN 768-dim / JP 1024-dim), TurnGPT (based on GPT-2), LLaMA-3 8B, and Qwen-3 8B (latter two 4096-dim + LoRA fine-tuning, rank=16, applied to q_proj/v_proj).
- **Fine-tuning Tasks**: MASK (for BERT), NTP (GPT series + LLMs), and TTP (GPT-2 within the TurnGPT framework). MASK/NTP use an 80/20 data split; TTP uses standard train/val/test splits.
- **Representation Extraction**: A `<ds>` marker is added to sentences containing backchannels/fillers. The final layer hidden state is extracted. Multi-token backchannels are compressed into a single vector via weighted averaging. Finally, PCA reduces dimensions to 100 for $k$-means.

Three context settings: no-context (current sentence only), one-context (one preceding and one following sentence), and full-context (only LLaMA-3 / Qwen-3, concatenating all previous history).

### Key Designs

1. **Comparative Experiments with Three Fine-tuning Tasks**:
    - **Function**: Driving LMs to learn the contextual semantics of backchannels/fillers through different objectives to compare "general language modeling vs. turn-relevance."
    - **Mechanism**: MASK replaces backchannels/fillers with [MASK]/random/original tokens (0.8/0.1/0.1 probability), forcing bidirectional reconstruction. NTP merges sentences from two speakers with `<s1>`/`<s2>` IDs for standard next-token prediction. TTP directs TurnGPT to predict $y^* = \arg\max_y P(y|X)$ (turn-shift probability), leveraging the strong correlation between backchannels and turn-holding.
    - **Design Motivation**: The authors initially hypothesized TTP would perform best due to its relevance to backchannel behavior. However, results showed NTP > TTP, proving that "task-agnostic language modeling" is sufficient to push pragmatic differences into the representation space.

2. **Factorial Design for Three Context Levels + Model Scales**:
    - **Function**: Decoupling "context window" and "model capacity" to determine if more context is beneficial or dilutive, and if larger is always better.
    - **Mechanism**: "No-context" feeds only the sentence containing the backchannel; "one-context" adds neighbors; "full-context" concatenates all history (up to thousands of tokens, supported by LLaMA-3/Qwen-3). Benchmarks include BERT (110M), GPT-2 (124M), and 8B LLMs.
    - **Design Motivation**: The authors discovered a counter-intuitive pattern: "more context leads to lower silhouette scores." Ample context "flattens" function word representations toward the average of nearby content words; thus, backchannels/fillers are most distinguishable under "no-context."

3. **Triple Validation: Silhouette Clustering + Distance Matrix + NLG Assessment**:
    - **Function**: Verifying fine-tuning effectiveness through internal representation quality (silhouette), pairwise separability (distance matrix), and external generation behavior (frequency/diversity/PPL/BERTScore/BLEU).
    - **Mechanism**: Silhouette $s(i)=\frac{b(i)-a(i)}{\max(a(i),b(i))}$ rewards "intra-cluster compactness and inter-cluster separation." Distance matrices visualize Euclidean distances between top-15 items. NLG assessments compare backchannel frequency and diversity during dialogue continuation.
    - **Design Motivation**: Single metrics can be "hacked." High silhouette scores don't guarantee natural generation, and high frequency doesn't guarantee structured representation space. The three layers together substantiate the conclusion that "fine-tuning truly teaches these function words to LMs."

### Loss & Training
MASK uses token-level cross-entropy (at masked positions); NTP uses standard next-token cross-entropy; TTP optimizes binary turn-taking probability (GPT-2 + TurnGPT framework: batch 4, lr 5e-4, dropout 0.3, 15 epochs, selecting the lowest val loss checkpoint). LLaMA-3 / Qwen-3 use LoRA (rank 16, dropout 0.1, q/v_proj only), with training times between 7–15 hours (8×L40 48G).

## Key Experimental Results

### Main Results
Average silhouette scores (bootstrap n=1000, 95% CI):

| Model | Task | Lang | no-ctx Base → FT | one-ctx Base → FT | full-ctx Base → FT |
|---|---|---|---|---|---|
| BERT | MASK | EN | 0.144 → 0.241 | 0.213 → 0.391 | — |
| BERT | MASK | JP | 0.213 → 0.391 | 0.197 → **0.429** | — |
| GPT-2 | NTP | EN | 0.274 → 0.328 | 0.149 → 0.311 | — |
| GPT-2 | NTP | JP | 0.157 → 0.288 | 0.101 → 0.273 | — |
| GPT-2 | TTP | EN | — → 0.289 | — → 0.211 | — |
| GPT-2 | TTP | JP | — → 0.284 | — → 0.261 | — |
| LLaMA-3 8B | NTP | EN | 0.450 → **0.588** | 0.183 → 0.291 | 0.210 → 0.301 |
| LLaMA-3 8B | NTP | JP | 0.257 → 0.450 | 0.179 → 0.335 | 0.318 → 0.408 |
| Qwen-3 8B | NTP | EN | 0.253 → 0.379 | 0.157 → 0.292 | 0.189 → 0.322 |
| Qwen-3 8B | NTP | JP | 0.172 → 0.452 | 0.154 → 0.263 | 0.173 → 0.181 |

English LLaMA-3 with no-context fine-tuning achieved a silhouette score of 0.588 (out of 1.0), the highest overall. Japanese BERT MASK with one-context reached 0.429, nearly equaling the 8B LLMs.

### Ablation Study (NLG Assessment, one-context)

Key metrics for English backchannel/filler generation:

| Model | Diversity ↑ | Frequency ↑ | Perplexity ↓ | BERTScore F1 ↑ | BLEU ↑ |
|---|---|---|---|---|---|
| LLaMA-3 no-FT | 73 | 4.29% | 197.7 | 78.69% | 0.0600 |
| **LLaMA-3 FT** | **83** | **18.61%** | **5.30** | **79.99%** | **0.0697** |
| Qwen-3 no-FT | 87 | 5.33% | 202.1 | 76.02% | 0.0698 |
| Qwen-3 FT | 95 | 9.19% | 91.3 | 79.73% | 0.0800 |
| GPT-2 no-FT | 68 | 6.68% | 158.6 | 79.67% | 0.0544 |
| GPT-2 FT | 90 | 17.43% | 6.98 | 79.99% | 0.0731 |

In Japanese, frequency increased from 0.31% → 7.57% (LLaMA-3), and PPL dropped from 256 → 28.5.

### Key Findings
- **Fine-tuning improves silhouette scores in nearly all (Model × Task × Lang × Context) combinations**: FT curves consistently outperform baseline curves, most notably for English LLaMA-3 no-context (+30% improvement).
- **Context inversely dilutes function word representations**: Silhouette scores generally decrease as the context window grows. The authors interpret this as context causing backchannel representations to be overwhelmed by surrounding content word information—a counter-intuitive yet logical finding for dialogue representation research.
- **NTP slightly outperforms TTP**: Despite the intuitive link between TTP objectives and backchannel function (turn-holding), NTP produced slightly higher silhouette scores. This suggests that general language modeling objectives are sufficient to capture pragmatic differences.
- **Small models are not necessarily inferior**: Japanese BERT MASK with one-context reached 0.429, nearly matching LLaMA-3’s 0.450, indicating that targeted fine-tuning with bidirectional attention can be more cost-effective than scaling model size for this task.
- **Representation quality translates to generation behavior**: After fine-tuning, LLaMA-3's English backchannel frequency quadrupled (4.29% → 18.61%), and PPL plummeted. Qualitative analysis confirms the model inserts *yeah* (acknowledgment) or *um* (disfluency) in appropriate pragmatic positions.
- **Side effects are controllable**: Control experiments on MapTask dialogue act classification showed only minor drops (0.4–1.4 points) for BERT/GPT-2/Qwen, while LLaMA-3 actually improved by 5.8 points, proving specialized fine-tuning does not significantly harm general NLU.

## Highlights & Insights
- **Turning "discarded stop words" into primary research objects**: This is one of the few works to treat backchannels/fillers as first-class citizens across multiple languages, serving as a paradigm reminder for "natural" LM speech: what we discard during cleaning is often the social signal.
- **The silhouette + distance matrix + NLG triad**: This multi-layered evaluation is highly suitable for any research on low-frequency token representations (e.g., dialects, emojis, low-resource pronouns) and can be easily migrated.
- **Context Dilution as a Double-Edged Sword**: The discovery that context can dull backchannel representations is a valuable negative insight. It suggests dialogue systems should differentiate between "semantic tasks" (requiring long context) and "pragmatic tasks" (where less context might preserve marker distinctness).
- **NTP > TTP Overturns Intuition**: A task being more "symptomatic" of a phenomenon doesn't guarantee better results. This supports the view that general language modeling inherently captures turn-taking signals, reducing the need for specialized pre-training tasks.

## Limitations & Future Work
- Authors admit: (a) Restricted to English and Japanese; (b) Limited to 8B models due to compute; (c) Analysis focuses on the final layer; (d) Specialized techniques like surgical fine-tuning were not explored; (e) Lacks comparison with acoustic models (prosodic features are vital); (f) NLG evaluation relied on small-scale qualitative samples.
- Own findings: (g) LLaMA-3's FT frequency (18.61%) might exceed ground truth, indicating a potential "over-compensation" bias leading to verbosity. (h) Qwen-3 showed almost no gain in Japanese full-context (0.173 → 0.181), warranting investigation into multilingual model mechanisms for low-resource function words. (i) Dramatic PPL drops might reflect corpus over-fitting rather than generalized capability.
- Improvement ideas: Explicitly label pragmatic functions for multi-class silhouette evaluation; introduce cross-modal speech training; use RLHF-style naturalness rewards rather than purely supervised fine-tuning.

## Related Work & Insights
- **vs. Qian & Skantze 2024**: They use contrastive feedback embeddings (HuBert/Whisper/BERT) but focus only on small feedback classes. This paper covers broader function word categories and uses a more lightweight FT approach.
- **vs. Mosbach 2020, Merchant 2020**: They study BERT fine-tuning effects on general representations; this work pivots to systematically ignored backchannels/fillers.
- **vs. Skantze 2017 / Ekstedt & Skantze 2020 (TurnGPT)**: While previous work analyzed turn-taking prediction, this paper reverses the focus to investigate the representations of the backchannels/fillers themselves.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically investigating backchannels/fillers and providing a triple-layered evaluation is a "hidden gem," though individual components are established.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 5 models, 3 tasks, 3 contexts, 2 languages, bootstrap CIs, side-effect checks, and NLG behavior.
- Writing Quality: ⭐⭐⭐⭐ Clear RQs, rich visualizations, and detailed appendices; introductory concepts occasionally run long.
- Value: ⭐⭐⭐⭐ Directly relevant to dialogue agents, TTS, and spoken LMs, reminding the community to find value in "garbage" data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Characterizing the Effect of Noise in Language Generation in the Limit](../../ICML2026/nlp_generation/characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)
- [\[ACL 2026\] Children's English Reading Story Generation via Supervised Fine-Tuning of Compact LLMs with Controllable Difficulty and Safety](childrens_english_reading_story_generation_via_supervised_fine-tuning_of_compact.md)
- [\[ACL 2026\] In-depth Research Impact Summarization through Fine-Grained Temporal Citation Analysis](in-depth_research_impact_summarization_through_fine-grained_temporal_citation_an.md)
- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)

</div>

<!-- RELATED:END -->
