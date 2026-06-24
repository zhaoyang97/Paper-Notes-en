---
title: >-
  [Paper Note] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning
description: >-
  [ICML 2026][Audio & Speech][Text Embeddings] This is a position paper: the authors argue that current text embedding research focuses excessively on "surface semantics" (morphology / syntax / topical similarity) while systematically ignoring "implicit semantics" such as pragmatics, stance, and social context. Empirical evidence from 7 implicit semantic datasets shows that even SOTA embeddings offer only marginal improvements over Bag-of-Tokens…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Text Embeddings"
  - "Implicit Semantics"
  - "Pragmatics"
  - "Stance Detection"
  - "MTEB"
date: 2026-05-08
content_hash: 79a82d86c258456a
---

# Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning

**Conference**: ICML 2026  
**arXiv**: [2506.08354](https://arxiv.org/abs/2506.08354)  
**Code**: https://github.com/dukesun99/Implicit-Embeddings  
**Area**: NLP / Text Embeddings  
**Keywords**: Text Embeddings, Implicit Semantics, Pragmatics, Stance Detection, MTEB

## TL;DR
This is a position paper: the authors argue that current text embedding research focuses excessively on "surface semantics" (morphology / syntax / topical similarity) while systematically ignoring "implicit semantics" such as pragmatics, stance, and social context. Empirical evidence from 7 implicit semantic datasets shows that even SOTA embeddings offer only marginal improvements over Bag-of-Tokens, advocating for implicit semantics as a first-class modeling objective in embedding research.

## Background & Motivation

**Background**: Text embeddings (Sentence-BERT, SimCSE, E5, BGE, GTE, LLM2Vec, OpenAI embeddings, etc.) have become foundational components in modern NLP and IR. They are widely deployed in downstream tasks like RAG, retrieval, clustering, and classification as "out-of-the-box universal semantic interfaces." Architectures, training objectives, and benchmarks (MTEB, BEIR) are evolving rapidly, making models appear increasingly "powerful, stable, and universal" from an external perspective.

**Limitations of Prior Work**: The authors observe a neglected dimension—*Implicit Semantics*. Decades of linguistic research show that much of human meaning is conveyed indirectly, relying on pragmatic inference (implicature, presupposition), speaker stance, and socio-cultural context (dialect, style-shifting, ideology). These are not edge cases but are core to real-world scenarios like irony, persuasion, politeness, and safety filtering. Current embeddings are almost "blind" to these tasks, as their training focuses solely on the surface level.

**Key Challenge**: The limitations are **structural** rather than incidental.
- Training side: Mainstream supervision comes from MS MARCO / NQ / STS / NLI, all of which reward "lexical relevance" or "literal equivalence" and fail to teach the model to distinguish "meaning beyond the words."
- Evaluation side: MTEB / BEIR almost exclusively measure surface similarity, leading models to be optimized for benchmarking artifacts rather than true semantic capabilities.
- Consequence: Embeddings "improve rapidly in directions that are easy to measure, while remaining stagnant in directions that are linguistically significant."

**Goal**: (a) Explicitly propose "implicit semantics" as a modeling target; (b) Quantify the gap in current embeddings regarding implicit semantics through a pilot study; (c) Provide a research agenda across training data, benchmarks, and modeling objectives.

**Key Insight**: Organize NLP tasks using a three-tier linguistic framework (utterance pragmatics / speaker stance / society sociolinguistics) to concretize abstract "implicit meaning" into 7 evaluable datasets.

**Core Idea**: Embeddings should not only encode "what was said," but also preserve "what was implied." Since embeddings are often the first-level representation for downstream systems, if stance, intent, or social framing is discarded here, even the most powerful LLMs downstream cannot retrieve the necessary evidence.

## Method

As a position paper, this work does not introduce new models or loss functions. Instead, the "Method" constitutes a chain of argumentation: first dissecting "implicit semantics" into an evaluable three-tier linguistic framework, then reviewing current embedding research to prove its focus on surface levels, followed by a dual-axis attribution explaining why even SOTA models fail, and finally quantifying the gap through a pilot study on 7 datasets to establish a research agenda for the community. The four core propositions are outlined below.

**1. Implicit semantics can be dissected into Utterance, Speaker, and Society levels, making them evaluable objects.** To address the ambiguity of "implicit meaning," the authors utilize a three-tier linguistic framework. *Utterance Level* draws from pragmatics (Gricean maxims, implicature, presupposition) to focus on "meaning between the lines"; e.g., "Bart managed to pass the test" implies that passing was unexpected. *Speaker Level* utilizes stance theory (evaluation / alignment / investment) to focus on the speaker's affective and social orientation toward a topic. *Society Level* leverages sociolinguistics (Silverstein’s indexicality, Bourdieu’s linguistic ideology) to focus on how dialects, registers, and style-shifting encode identity and power. This decomposition allows Section 6 to directly select datasets for quantification.

**2. The inability of current embeddings to capture implicit semantics is structural, rooted in the misalignment of both training signals and evaluation objectives.** The authors attribute the failure not to "model capacity" but to "incorrect signals and targets." Along the training axis, self-supervised methods (SimCSE dropout, DenoSent) only reinforce invariance to surface perturbations. Supervised datasets (STS / NLI / IR) essentially reward lexical overlap or literal equivalence. On the evaluation axis, MTEB / BEIR score almost exclusively on surface similarity, combined with data leakage and score inflation, causing leaderboards to diverge from true generalization.

**3. The solution is a three-axis agenda: diversified contrastive training data, implicit semantic benchmarks, and redefined modeling objectives.** For data, the authors advocate for **contrastive supervision** specifically for embeddings—constructing samples that are surface-similar but differ in implied meaning (implicature, stance, irony, dialect) using LLM synthesis distilled via strong cross-encoder teachers. For benchmarks, tasks must be designed to directly test pragmatic inference and stance recognition with anti-leakage protocols. Regarding objectives, "distinguishing surface-equivalent but implicitly-distinct text" should be an explicit goal, with instruction-following retrieval (Su et al. 2023) serving as a transitional path.

**4. Future training objectives should utilize an implicit-semantics-sensitive contrastive loss.** In Section 7.3, the authors conceptualize a contrastive objective: pulling together samples with different surface forms but identical implicit intent, while pushing apart samples that are surface-identical but have opposite implied meanings (e.g., irony vs. literal meaning). This provides a clear entry point for subsequent work and addresses the critique of the "surface invariance hypothesis" by actively creating discriminative pressure where surface forms are equivalent.

## Key Experimental Results

### Main Results: 7 Implicit Semantic Datasets × 14 Embedding Models
The authors restructured 7 datasets into classification, pairwise, and zero-shot similarity tasks across three levels: utterance (PUB's P-IMP / P-PRE / P-R&D), speaker (P-Stance), and society (IHS / SBIC / Political Bias).

| Model Category | Representative Model | Utterance Avg | Speaker (P-Stance) | Society Avg | Total Avg |
|:---|:---|:---|:---|:---|:---|
| Bag-of-words | Bag-of-Tokens | 60.0 (56.5/75.3/48.2) | 73.4 | 60.6 | **62.2** |
| Encoder-only | S-BERT | 63.4 | 72.9 | 63.5 | 64.8 |
| Encoder-only | BGE-Large | 67.3 | 76.0 | 66.1 | 68.0 |
| LLM-based | Linq-Mistral | **79.5** | 75.8 | 66.7 | 73.5 |
| LLM-based | E5-Mistral | 74.4 | 81.1 | 73.4 | 74.9 |
| LLM-based | GTE-Qwen | 76.3 | 80.9 | 72.3 | **75.2** |
| Proprietary | OpenAI-Large | 74.2 | **83.7** | **72.9** | 75.0 |

The most striking comparison: S-BERT improved by only ~3.4 points over Bag-of-Tokens in utterance average and ~2.9 points in society average, whereas contemporary SOTAs show double-digit gains over bag-of-words on MTEB.

### Horizontal Analysis (Performance Split)

| Phenomenon | Data Support | Insight |
|:---|:---|:---|
| Encoder-only near BoT | S-BERT is only +2.6 over BoT; BGE-Large +5.8 | Surface-oriented signals severely bottle-neck encoder capabilities in implicit semantics. |
| LLM-based/OpenAI Advantage | Linq/E5/GTE-Qwen/OpenAI scores 73–75 | World and social knowledge from LLM pre-training "leaks" into the embedding space. |
| Uneven Expertise | Linq leads in utterance, OpenAI in society, E5 in political bias | Models are "fragmented experts" in sub-dimensions without a unified representation. |
| MTEB Score $\neq$ Implicit Strength | OpenAI leads in implicit tasks despite not being top-tier on MTEB | Benchmarks are decoupled from true semantic capability. |

### Key Findings
- **Uneven Generalization**: Models fit "highly lexicalized/strong label clue" phenomena (like standardized stance markers) but are weak in contextual inference, speaker modeling, and social context.
- **Metric Misalignment**: MTEB scores are decoupled or even negatively correlated with implicit semantic capability.
- **LLM-based Advantage**: The advantage of LLM-based embeddings stems primarily from base knowledge rather than the contrastive training objectives.

## Highlights & Insights
- **Framework-based Dissection**: Using the utterance/speaker/society tiers to transform "deep meaning" into evaluable subsets is a model for position papers.
- **Bag-of-Tokens as Sanity Baseline**: In tasks like P-IMP, BGE-Large is only ~8 points ahead of BoT, a much smaller gap than on MTEB, proving that "benchmarks dictate what the model learns."
- **Embeddings as "Semantic Interfaces"**: The authors emphasize that while embeddings need not replace LLM reasoning, a failure to capture stance or intent at the retrieval stage cannot be corrected downstream.
- **Actionatble Data Synthesis**: The suggestion to use LLMs to synthesize "surface-similar but implicitly-opposite" pairs provides a well-defined entry point for future research.

## Limitations & Future Work
- **Admitted Limitations**: (a) The tiers are analytical perspectives rather than a strict ontology; (b) Pilot datasets were not designed for embeddings and required restructuring, which may introduce noise; (c) Instruction-following is a "bridge" but does not guarantee pragmatic sensitivity.
- **Observer Observations**: (a) No new loss or model is proposed, leaving its adoption by the community uncertain; (b) Reliance on accuracy may mask biases in subjective tasks like stance detection; (c) Aggregate averages might hide critical trade-offs between dimensions.
- **Future Directions**: (a) Develop dedicated implicit-semantics benchmarks with anti-leakage protocols; (b) Release contrastive synthesis pipelines and reference embedders; (c) Incorporate robustness metrics like demographic subgroup accuracy.

## Related Work & Insights
- **vs. MTEB / BEIR**: Argues that breadth across domains cannot compensate for blindness to implicit semantics; serves as a methodological critique of the MTEB paradigm.
- **vs. SimCSE / E5 / BGE**: These improve contrastive negatives but maintain the "surface invariance" hypothesis; this paper calls for distinguishing implicit intent at the loss level.
- **vs. Instruction-following retrieval**: Viewed as a "bridge" to breaking surface paradigms, though instruction-following $\neq$ pragmatic sensitivity.
- **vs. LLM-as-embedder**: Confirms their strength in implicit tasks but attributes it to base knowledge, cautioning the community against over-crediting contrastive objectives.
- **vs. Social/Pragmatic NLP**: Translates insights from sociolinguistics (Hovy & Yang 2021, etc.) into actionable issues for embedding research.

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[ECCV 2024\] Latent-INR: A Flexible Framework for Implicit Representations of Videos with Discriminative Semantics](../../ECCV2024/audio_speech/latent-inr_a_flexible_framework_for_implicit_representations_of_videos_with_disc.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](../../ACL2025/audio_speech/its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ICML 2026\] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](../../ICML2025/audio_speech/do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)

</div>

<!-- RELATED:END -->
