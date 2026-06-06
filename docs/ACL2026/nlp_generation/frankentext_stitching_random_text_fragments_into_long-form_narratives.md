---
title: >-
  [Paper Note] Frankentext: Stitching Random Text Fragments into Long-Form Narratives
description: >-
  [ACL 2026][Text Generation][AIGC Detection] Introduces the Frankentext paradigm, which constrains LLMs to stitch random human text fragments into coherent long-form narratives under extreme constraints (90% of tokens cop…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "AIGC Detection"
  - "Mixed-authorship Attribution"
  - "Controllable Text Generation"
  - "AI Text Detector"
  - "Human-AI Collaborative Writing"
date: 2026-05-08
content_hash: 9220763874b7a695
---

# Frankentext: Stitching Random Text Fragments into Long-Form Narratives

**Conference**: ACL 2026  
**arXiv**: [2505.18128](https://arxiv.org/abs/2505.18128)  
**Code**: [GitHub](https://github.com/chtmp223/Frankentext)  
**Area**: AIGC Detection  
**Keywords**: AIGC Detection, Mixed-authorship Attribution, Controllable Text Generation, AI Text Detector, Human-AI Collaborative Writing

## TL;DR

Introduces the Frankentext paradigm, which constrains LLMs to stitch random human text fragments into coherent long-form narratives under extreme constraints (90% of tokens copied verbatim from human writing). This reveals a significant failure of existing AI text detectors in mixed-authorship scenarios (72% of Frankentext is misclassified as human writing).

## Background & Motivation

**Background**: As the quality of LLM-generated text improves, AI text detection has become a critical requirement for academic integrity and content provenance. Existing detectors are primarily based on binary classification assumptions (AI vs. Human).

**Limitations of Prior Work**: A "gray zone" of human-AI collaborative writing exists in reality—where text is neither purely AI nor purely human-authored but a mixture. Existing binary detectors (e.g., Binoculars, FastDetectGPT) cannot effectively identify such mixed texts.

**Key Challenge**: Current detection methods rely on surface-level features (e.g., perplexity, statistical signatures). However, when massive amounts of human text are embedded into AI-generated content, these statistical signatures are diluted, leading to detection failure.

**Goal**: To systematically study an extreme controllable generation paradigm—Frankentext—where LLMs generate coherent narratives while most tokens must be copied verbatim from human writing. This aims to expose detector vulnerabilities and drive the development of fine-grained detection methods.

**Key Insight**: Inspired by Frankenstein—assembling a complete "creature" from "scraps" of different origins. The LLM acts as a composer rather than a writer, selecting, arranging, and stitching coherent stories from thousands of random human text fragments.

**Core Idea**: Through a prompt-based pipeline, LLMs select and stitch randomly sampled human text paragraphs. This generates coherent and relevant narratives while maintaining a specified copy rate (e.g., 90%), posing a fundamental challenge to existing AI detectors.

## Method

### Overall Architecture

The Frankentext pipeline consists of two primary stages: first, 1,500 human text fragments (approx. 103K BPE tokens) are randomly sampled from a large book corpus (Books3, containing 197K books and 156M paragraphs). These are input into an LLM along with a writing prompt to generate a draft under constraints. Second, an iterative editing phase corrects contradictions and incoherence.

### Key Designs

1. **Draft Generation**:
    - **Function**: To have the LLM select and combine random human text fragments into a preliminary narrative.
    - **Mechanism**: Provides the LLM with a writing prompt and 1,500 randomly sampled paragraph-level human text fragments. The LLM is required to generate a ~500-word story where 90% of the content must be copied verbatim from the provided fragments, allowing only minor connectors and transitional phrases.
    - **Design Motivation**: Leverages the LLM's implicit search capability over a combinatorial space to find composable fragments from unrelated human texts, a task nearly impossible for humans.

2. **Iterative Polishing**:
    - **Function**: To fix contradictions, continuity errors, and grammatical issues in the draft.
    - **Mechanism**: Uses the same LLM to identify and apply minimal edits (similar to self-correction). It improves coherence while maintaining verbatim copy constraints and the writing prompt, with up to 3 iterations.
    - **Design Motivation**: Drafts may contain character contradictions, factual conflicts, or irrelevant content. The editing phase increases coherence from $68\%$ to $81\%$.

3. **Copy Rate Control**:
    - **Function**: To ensure the final output meets the preset human text copy proportion.
    - **Mechanism**: Optionally uses ROUGE-L or the AI text detector itself as a feedback signal. If the copy rate is insufficient or detected as AI-generated, the LLM is prompted for revisions.
    - **Design Motivation**: Balances copy rate and coherence while maximizing the probability of evading detection.

## Key Experimental Results

### Main Results

| Model | Copy Rate | Relevance% | Coherence% | Pangram Detect AI% | Binoculars Detect% | FastDetectGPT Detect% |
|:---|:---|:---|:---|:---|:---|:---|
| Gemini-2.5-Pro (vanilla) | - | 100 | 100 | 100 | 52 | 99 |
| Gemini-2.5-Pro (FT) | 75% | 100 | 81 | 4 | 0 | 1 |
| o4-mini (FT) | 68% | 56 | 29 | 24 | 0 | 1 |
| Claude-3.7-Sonnet (FT) | 64% | 95 | 73 | 17 | 2 | 16 |
| DeepSeek-R1 (FT) | 42% | 91 | 72 | 74 | 0 | 0 |
| Qwen-3-32B thinking (FT) | 36% | 91 | 54 | 85 | 52 | 92 |

### Ablation Study

| Configuration | Key Metrics | Remarks |
|:---|:---|:---|
| Remove Editing Phase | Coherence $68\% \rightarrow 81\%$ | Editing significantly improves coherence |
| Use Relevant Paragraphs (FAISS) | Copy rate $38\%$, detection drops | Relevance surprisingly decreases copy rate |
| Use Sentences instead of Paragraphs | Coherence $89\%$, detection $+14\%$ | Shorter units improve coherence but are easier to detect |
| Maximum Context (100K paragraphs) | Copy rate $76\%$, coherence $89\%$ | Best performance but 10x cost |
| Single Book (Monte Cristo) | Coherence $89\%$, relevance $97\%$ | A single source also works effectively |
| Request 25% Human Text | Pangram AI Detect $54\%$ | More AI text $\rightarrow$ Easier to detect |
| Request 90% Human Text | Pangram AI Detect $4\%$ | More human text $\rightarrow$ Harder to detect |

### Key Findings

- Gemini-2.5-Pro is the best model for Frankentext generation, balancing coherence ($81\%$), relevance ($100\%$), and a low detection rate ($4\%$).
- Binoculars and FastDetectGPT fail almost completely against Frankentext (detection rate $< 2\%$). Pangram performs slightly better but still misses $59\%$.
- Human evaluators can identify Frankentext as AI-involved $56\%$ of the time on average, outperforming automated detectors.
- Copy rate is inversely proportional to detection rate: higher human text proportions are harder to detect, though coherence decreases.
- The frequency of AI keywords (e.g., "Elara") in Frankentext drops sharply from 686 occurrences in the vanilla version to 10.

## Highlights & Insights

- **Discovery of the Gray Zone**: Frankentext breaks the "AI vs. Human" binary assumption, revealing a mixed-authorship space that detectors struggle to handle.
- **Cost-Benefit**: Each Frankentext costs only $\$1.32$ (Gemini), significantly lower than the $\$2.50$ per story in the human-AI collaborative dataset CoAuthor, with no complex setup required.
- **Token-level Annotation**: Each Frankentext comes with intrinsic token-level labels for copied vs. generated content, which can be directly used to train mixed-authorship detection models.
- **Unique Human-like Perception**: Evaluators praised Frankentext for its unique "human feel"—imaginative premises, vivid descriptions, and dry humor—precisely because most of its content originates from human writing.

## Limitations & Future Work

- Reliance on large-scale, high-quality, in-domain human text corpora makes it difficult to apply to low-resource languages or specialized fields (e.g., technical manuals).
- The copy rate metric might underestimate the actual proportion of human text.
- This work only exposes the attack surface without proposing specific defense mechanisms.
- Quality in non-fiction domains (e.g., news) still has room for improvement, as generated text leans toward a narrative style.
- Books3 contains copyrighted works, raising issues regarding creative ownership and copyright.

## Related Work & Insights

- **vs. Binoculars/FastDetectGPT**: These perplexity-based detectors fail almost entirely on Frankentext, suggesting surface-level statistical features are insufficient for mixed-authorship text.
- **vs. Pangram**: As a trained classifier, Pangram can partially detect mixed text ($37\%$ labeled as mixed) but still misses $59\%$ of Gemini Frankentext.
- **vs. CoAuthor**: Frankentext provides a cheaper, scalable way to generate mixed-authorship data across various granularities (word-level and sentence-level).
- **vs. Paraphrasing Attacks**: Unlike paraphrasing original text to evade detection, Frankentext uses raw human text directly, representing a completely new attack vector.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes a completely new text generation paradigm, positioning the LLM as a "composer" rather than an "author."
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 model series, 3 detectors, human evaluation, and multiple ablation experiments.
- Writing Quality: ⭐⭐⭐⭐ The paper structure is clear, and the Frankenstein analogy is vivid, though some parts could be further refined.
- Value: ⭐⭐⭐⭐⭐ Serves as a significant warning for the AI text detection field, driving the transition from binary to fine-grained detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model](../../ICLR2026/nlp_generation/fs-dfm_fast_and_accurate_long_text_generation_with_few-step_diffusion_language_m.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)
- [\[ACL 2026\] Can You Make It Sound Like You? Post-Editing LLM-Generated Text for Personal Style](can_you_make_it_sound_like_you_post-editing_llm-generated_text_for_personal_styl.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)

</div>

<!-- RELATED:END -->
