---
title: >-
  [Paper Note] Frankentext: Stitching Random Text Fragments into Long-Form Narratives
description: >-
  [ACL 2026][AIGC Detection][mixed authorship attribution] This paper proposes Frankentext, a paradigm where LLMs stitch random human text fragments into coherent long-form narratives under extreme constraints (90% content…
tags:
  - "ACL 2026"
  - "AIGC Detection"
  - "mixed authorship attribution"
  - "controllable text generation"
  - "AI text detector"
  - "human-AI collaborative writing"
date: 2025-05-08
content_hash: 85c8c2e270f2c192
---

# Frankentext: Stitching Random Text Fragments into Long-Form Narratives

**Conference**: ACL 2026
**arXiv**: [2505.18128](https://arxiv.org/abs/2505.18128)
**Code**: [GitHub](https://github.com/chtmp223/Frankentext)
**Area**: AIGC Detection
**Keywords**: AIGC detection, mixed authorship attribution, controllable text generation, AI text detector, human-AI collaborative writing

## TL;DR

This paper proposes Frankentext, a paradigm where LLMs stitch random human text fragments into coherent long-form narratives under extreme constraints (90% content verbatim-copied from human writing), revealing severe failures of existing AI text detectors in mixed-authorship scenarios (72% of Frankentext is misclassified as human-written).

## Background & Motivation

**Background**: As LLM-generated text quality continues to improve, AI text detection has become critical for academic integrity and content provenance. Existing detectors primarily assume a binary (AI vs. human) classification paradigm.

**Limitations of Prior Work**: In reality, there exists a substantial "gray area" of human-AI collaborative writing—text is neither purely AI nor purely human-authored, but a mixture. Existing binary detectors (e.g., Binoculars, FastDetectGPT) cannot effectively identify such mixed text.

**Key Challenge**: Current detection methods rely on surface features (e.g., perplexity, statistical signatures), but when AI-generated content heavily embeds genuine human text, these statistical features are diluted, causing detection failure.

**Goal**: Systematically study an extreme controllable generation paradigm—Frankentext, where LLMs must generate coherent narratives with most tokens verbatim-copied from human writing, to expose detector vulnerabilities and advance fine-grained detection methods.

**Key Insight**: Inspired by Frankenstein—assembling a complete "creature" from fragments of different sources. The LLM acts as a composer rather than a writer, selecting, arranging, and stitching coherent stories from thousands of random human text fragments.

**Core Idea**: Through a prompt-based pipeline, LLMs select and stitch randomly sampled human text paragraphs while maintaining a specified copy rate (e.g., 90%) and generating coherent, relevant narratives, thereby posing a fundamental challenge to existing AI detectors.

## Method

### Overall Architecture

The Frankentext pipeline has two main stages: first, randomly sample 1500 human text fragments (~103K BPE tokens) from a large book corpus (Books3, containing 197K books and 156M paragraphs), then input them alongside a writing prompt to the LLM for constrained draft generation; second, iteratively edit to fix contradictions and incoherences.

### Key Designs

1. **Draft Generation**:

    - Function: Have LLMs select and combine random human text fragments into a preliminary narrative
    - Mechanism: Provide the LLM with a writing prompt and 1500 randomly sampled paragraph-level human text fragments, requiring it to generate a ~500-word story with 90% content verbatim-copied from given fragments, allowing only minimal connective words and transitional phrases
    - Design Motivation: Leverage LLMs' ability to implicitly search combinatorial spaces, finding composable fragments from unrelated human text—a task nearly infeasible for humans

2. **Iterative Polishing**:

    - Function: Fix contradictions, continuity errors, and grammatical issues in the draft
    - Mechanism: Use the same LLM to identify and apply minimal edits (similar to self-correction), maintaining the verbatim copy constraint and writing prompt while improving coherence, with up to 3 iterations
    - Design Motivation: Drafts may contain character contradictions, factual conflicts, and irrelevant content; the editing stage can improve coherence from 68% to 81%

3. **Copy Rate Control and Detection Feedback**:

    - Function: Ensure final output meets the preset human text copy ratio
    - Mechanism: Optionally use ROUGE-L or AI text detectors themselves as feedback signals; if the copy rate is insufficient or the text is detected as AI-generated, prompt the LLM for revision
    - Design Motivation: Balance between copy rate and coherence while maximizing detection evasion probability

## Key Experimental Results

### Main Results

| Model | Copy Rate | Relevance% | Coherence% | Pangram AI% | Binoculars% | FastDetectGPT% |
|------|--------|---------|---------|---------------|-----------------|-------------------|
| Gemini-2.5-Pro (vanilla) | - | 100 | 100 | 100 | 52 | 99 |
| Gemini-2.5-Pro (FT) | 75% | 100 | 81 | 4 | 0 | 1 |
| o4-mini (FT) | 68% | 56 | 29 | 24 | 0 | 1 |
| Claude-3.7-Sonnet (FT) | 64% | 95 | 73 | 17 | 2 | 16 |
| DeepSeek-R1 (FT) | 42% | 91 | 72 | 74 | 0 | 0 |
| Qwen-3-32B thinking (FT) | 36% | 91 | 54 | 85 | 52 | 92 |

### Ablation Study

| Config | Key Metric | Note |
|------|---------|------|
| Remove editing stage | Coherence 68%→81% | Editing significantly improves coherence |
| Use related paragraphs (FAISS retrieval) | Copy rate only 38%, detection rate drops | Relevance paradoxically reduces copy rate |
| Use sentences instead of paragraphs | Coherence 89%, detection rate +14% | Shorter units improve coherence but are easier to detect |
| Maximum context (100K paragraphs) | Copy rate 76%, coherence 89% | Best performance but 10× cost |
| Single book (Monte Cristo) | Coherence 89%, relevance 97% | Single source also works effectively |
| Require 25% human text | Pangram AI detection 54% | More AI text → easier to detect |
| Require 90% human text | Pangram AI detection 4% | More human text → harder to detect |

### Key Findings

- Gemini-2.5-Pro is the best Frankentext generator, balancing coherence (81%), relevance (100%), and low detection rate (4%)
- Binoculars and FastDetectGPT almost completely fail against Frankentext (detection rate <2%); Pangram performs better but still misses 59%
- Human evaluators can identify 56% of Frankentext as AI-involved on average, outperforming automatic detectors
- Copy rate and detection rate are inversely related: higher human text ratio makes detection harder but decreases coherence
- AI keyword frequency (e.g., "Elara") drops sharply from 686 in vanilla to 10 in Frankentext

## Highlights & Insights

- **Gray area discovery**: Frankentext breaks the "AI vs. human" binary assumption, revealing a mixed-authorship space that detectors struggle with
- **Cost-effectiveness**: Each Frankentext costs only $1.32 (Gemini), much less than the CoAuthor human-AI dataset at $2.50/piece, with no complex setup required
- **Token-level annotation**: Each Frankentext comes with copy-vs-generated token-level labels, directly usable for training mixed-authorship detection models
- **Unique human feel**: Evaluators praised Frankentext for its distinctive "human quality"—imaginative premises, vivid descriptions, and dry humor—precisely because most content genuinely comes from human writing

## Limitations & Future Work

- Depends on large-scale high-quality same-domain human text corpora; low-resource languages and specialized domains (e.g., technical manuals) are difficult to apply directly
- The copy rate metric may underestimate actual human text proportion
- This paper only exposes the attack surface without proposing specific defense mechanisms
- Non-fiction domain (e.g., news) Frankentext quality still has room for improvement, as generated text tends toward narrative style
- Books3 contains copyrighted works, raising creative attribution and copyright concerns

## Related Work & Insights

- **vs Binoculars/FastDetectGPT**: These perplexity-based detectors almost completely fail against Frankentext, demonstrating that surface statistical features are insufficient for mixed-authorship text
- **vs Pangram**: As a trained classifier, Pangram can partially detect mixed text (37% labeled as mixed), but still misses 59% of Gemini Frankentext
- **vs CoAuthor**: Frankentext provides a cheaper, scalable way to generate mixed-authorship data, covering word-level and sentence-level granularities
- **vs Paraphrasing attacks**: Unlike rewriting original text to evade detection, Frankentext directly uses raw human text—an entirely new attack vector

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Proposes an entirely new text generation paradigm, positioning LLMs as "composers" rather than "authors"—a highly novel perspective
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 model families, 3 detectors, human evaluation, multiple ablation experiments with extremely broad coverage
- Writing Quality: ⭐⭐⭐⭐ Clear paper structure with vivid Frankenstein analogy, though some sections could be more refined
- Value: ⭐⭐⭐⭐⭐ Critically important warning for the AI text detection field, pushing the shift from binary classification to fine-grained detection

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ICLR 2026\] DMAP: A Distribution Map for Text](../../ICLR2026/aigc_detection/dmap_a_distribution_map_for_text.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)

</div>

<!-- RELATED:END -->
