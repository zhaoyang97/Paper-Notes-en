---
title: >-
  [Paper Note] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning
description: >-
  [ICML 2026][Audio & Speech][text embeddings] This is a position paper: the authors argue that current text embedding research focuses excessively on "surface semantics" (morphology / syntax / thematic similarity) while s…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "text embeddings"
  - "implicit semantics"
  - "pragmatics"
  - "stance detection"
  - "MTEB"
date: 2026-05-08
content_hash: bfe7d345ab05557d
---

# Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning

**Conference**: ICML 2026  
**arXiv**: [2506.08354](https://arxiv.org/abs/2506.08354)  
**Code**: https://github.com/dukesun99/Implicit-Embeddings  
**Area**: NLP / Text Embeddings  
**Keywords**: text embeddings, implicit semantics, pragmatics, stance detection, MTEB

## TL;DR
This is a position paper: the authors argue that current text embedding research focuses excessively on "surface semantics" (morphology / syntax / thematic similarity) while systematically ignoring "implicit semantics" such as pragmatics, stance, and social context. Empirical evidence from 7 implicit semantic datasets shows that even SOTA embeddings provide only marginal gains over Bag-of-Tokens, calling for implicit semantics to be treated as a first-class modeling objective in embedding research.

## Background & Motivation

**Background**: Text embeddings (Sentence-BERT, SimCSE, E5, BGE, GTE, LLM2Vec, OpenAI embeddings, etc.) have become foundational components of modern NLP and IR, widely deployed in downstream tasks like RAG, retrieval, clustering, and classification as "out-of-the-box general semantic interfaces." Architectures, training objectives, and benchmarks (MTEB, BEIR) are evolving rapidly, making models appear increasingly "strong, stable, and general" from an external perspective.

**Limitations of Prior Work**: The authors observe a neglected dimension—*Implicit Semantics*. Decades of linguistic research show that most human meaning is conveyed indirectly, relying on pragmatic inference (implicature, presupposition), speaker stance, and socio-cultural context (dialect, style-shifting, ideology). These are not edge cases but are central to real-world scenarios like irony, persuasion, politeness, and safety filtering. Current embeddings are almost "blind" to these tasks as their training only captures surface patterns.

**Key Challenge**: The limitations are **structural** rather than accidental.
- Training side: Mainstream supervision comes from MS MARCO / NQ / STS / NLI, which all reward "lexical relevance" or "literal equivalence," failing to teach the model to distinguish "what is implied."
- Evaluation side: MTEB / BEIR almost exclusively measure surface similarity, causing models to be optimized to fit benchmark artifacts rather than true semantic capabilities.
- Consequence: Embeddings "improve increasingly in easily measurable directions but remain stagnant in linguistically significant ones."

**Goal**: (a) Explicitly propose "implicit semantics" as a modeling objective; (b) Quantify the gap of current embeddings in implicit semantics through a pilot study; (c) Provide a research agenda across training data, benchmarks, and modeling objectives.

**Key Insight**: Organize NLP tasks using a three-level linguistic framework (utterance pragmatics / speaker stance / society sociolinguistics), grounding the abstract concept of "implicit meaning" into 7 evaluable datasets.

**Core Idea**: Embeddings should not only encode "what was said" but also preserve "what was implied"—because embeddings are often the first-level representation of downstream systems. If stance, intent, or social frameworks are lost here, even the most powerful LLMs downstream cannot retrieve that evidence.

## Method

As a position paper, no new model is proposed; the "method" consists of an **analytical framework + empirical process + research agenda**.

### Overall Architecture
The authors' chain of argument: Establish a three-level linguistic framework → Review current embedding research to prove its surface focus → Deconstruct "why this is the case" (training + evaluation) → Conduct a pilot study on 7 implicit semantic datasets to quantify the gap → Provide three research directions.

### Key Designs

1.  **Three-level Implicit Semantic Framework (Utterance / Speaker / Society)**:
    - **Function**: Deconstruct the vague concept of "implicit semantics" into three levels directly evaluable by NLP.
    - **Mechanism**: (a) *Utterance Level* draws from pragmatics (Grice's Cooperative Principle, implicature, presupposition), focusing on "meaning beyond words," such as "Bart managed to pass the test" implying success was unexpected; (b) *Speaker Level* draws from stance theory (evaluation / alignment / investment), focusing on the speaker's affective and social orientation toward topics; (c) *Society Level* draws from sociolinguistics (Silverstein’s indexicality, Bourdieu’s linguistic ideologies), focusing on how dialect, register, and style-shifting encode identity and power relations.
    - **Design Motivation**: Ground "deep meaning" as measurable objects, allowing for direct dataset selection and quantification in Section 6.

2.  **Structural Failure Analysis (Training × Evaluation Dual Attribution)**:
    - **Function**: Explain why advanced embeddings fail at implicit semantics.
    - **Mechanism**: On the training axis, self-supervision (SimCSE dropout / DenoSent) only reinforces "invariance to surface perturbations," while supervised STS / NLI / IR datasets essentially reward lexical overlap or literal equivalence. Multi-task training (mGTE, Jina, E5) expands domains but does not change the supervisory signal form. On the evaluation axis, MTEB / BEIR score almost entirely on surface similarity, where leakage and score inflation further decouple leaderboards from true generalization.
    - **Design Motivation**: Shift the responsibility from "models aren't strong enough" to "misalignment of training signals and evaluation goals," providing actionable directions for improvement.

3.  **Three-axis Agenda: Diversified Training Data + Implicit Semantics Benchmarks + Redefining Modeling Objectives**:
    - **Function**: Translate "what should be done" into three executable tasks.
    - **Mechanism**: (a) Data: Move beyond scaling web text to create **contrastive supervision**—samples with surface similarity but different implied meanings (implicature / stance / sarcasm / dialect), synthesized via LLMs and distilled through strong cross-encoder teachers; (b) Benchmarks: Design tasks that directly measure pragmatic inference, stance recognition, and social context while preventing leakage; (c) Objectives: Use the "differentiation of surface-equivalent but implicitly-different text" as an explicit loss, potentially using instruction-following retrieval (conditioning embedding space on query instructions, Su et al. 2023) as a transition path.
    - **Design Motivation**: A position paper must provide a checklist of work for the community rather than just pointing out problems.

### Loss & Training
The authors do not propose a new loss but suggest in §7.3 that future work could design contrastive objectives: pulling together *surface-different but intent-same* samples and pushing apart *surface-similar but implied-opposite* samples (e.g., irony vs. literal meaning), combined with multi-task training (pragmatics + stance + social meaning) and LLM teacher distillation.

## Key Experimental Results

### Main Results: 7 Implicit Semantic Datasets × 14 Embedding Models
The authors refactored 7 datasets (following MTEB protocols) into classification / pairwise / zero-shot similarity tasks across three levels: utterance (P-IMP / P-PRE / P-R&D from PUB), speaker (P-Stance), and society (IHS / SBIC / Political Bias).

| Model Category | Representative Model | Utterance Avg | Speaker (P-Stance) | Society Avg | Grand Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BoW Baseline | Bag-of-Tokens | 60.0 (56.5/75.3/48.2) | 73.4 | 60.6 | **62.2** |
| Encoder-only | S-BERT | 63.4 | 72.9 | 63.5 | 64.8 |
| Encoder-only | BGE-Large | 67.3 | 76.0 | 66.1 | 68.0 |
| LLM-based | Linq-Mistral | **79.5** | 75.8 | 66.7 | 73.5 |
| LLM-based | E5-Mistral | 74.4 | 81.1 | 73.4 | 74.9 |
| LLM-based | GTE-Qwen | 76.3 | 80.9 | 72.3 | **75.2** |
| Proprietary | OpenAI-Large | 74.2 | **83.7** | **72.9** | 75.0 |

The most pointed comparison: S-BERT improved by only ~3.4 points over Bag-of-Tokens on the utterance average and ~2.9 points on the society average—nearly "stagnant"—whereas SOTA models on MTEB show improvements of tens of points over BoW.

### Key Findings
- **Inconsistent Generalization, Not Total Failure**: Models can fit implicit phenomena with "high lexicalization / strong label clues" (like certain formulaic stance markers) but are significantly weaker in cases requiring contextual reasoning / speaker modeling / social context.
- **Mismatch or Inverse Correlation Between MTEB Scores and Implicit Semantic Capability**: The authors cite recent research (Chung et al. 2025; Sancheti et al. 2025) noting that high MTEB performance does not predict downstream robustness.
- **Pros of LLM-based Embeddings Stem from Base Knowledge, Not Training Goals**: Transforming generative LLMs into embedders via contrastive objectives preserves some implicit signals, but the training objectives remain surface-oriented, leaving "explicit modeling of implicit semantics" far from achieved.

## Highlights & Insights
- **"Linguistic Slicing Before Measuring" is a paradigm for position papers**: Using the utterance/speaker/society framework to divide "deep meaning" into evaluable subsets avoids empty rhetoric.
- **Bag-of-Tokens as a sanity baseline is highly effective**: On tasks like P-IMP, BGE-Large is only ~8 points ahead of BoT, which is much smaller than its advantage over BoT on MTEB, clearly demonstrating that "what you choose as a benchmark is what you teach the model."
- **Positioning embeddings as "Semantic Interfaces" rather than "Reasoning Replacements"**: The authors do not advocate for embeddings to replace LLMs for complex reasoning but emphasize that "if first-level retrieval loses stance/intent, no matter how strong the downstream system is, it cannot recover"—placing implicit-semantics-aware embeddings correctly within the modern RAG/agent stack.
- **Actionable path for data synthesis**: Suggesting the use of LLMs to synthesize "surface-similar but implied-opposite" contrastive pairs, followed by soft-label distillation from strong cross-encoder teachers, leaves a well-defined entry point for subsequent work.

## Limitations & Future Work
- **Acknowledged Limitations**: (a) The three-level framework is an analytical perspective rather than a strict ontology; (b) The 7 pilot datasets were not designed for embeddings and required refactoring, potentially introducing evaluation noise; (c) Instruction-following retrieval is a "bridge" but does not inherently guarantee capturing stance / presupposition / social semantics.
- **Additional Observations**: (a) No new method or loss is actually proposed, only directions; (b) Evaluation relies entirely on accuracy, which might mask biases in tasks like P-Stance or SBIC that are imbalanced or subjective; (c) Distinguishing between the three levels of "implicit semantics" shows significant variance; aggregating them into a total average might hide key trade-offs (e.g., OpenAI is strong in society but weaker in utterance than Linq).
- **Future Directions**: (a) Develop dedicated benchmarks for implicit semantics with anti-leakage protocols; (b) Design a contrastive data synthesis pipeline and release a trained reference embedder for community reproducibility; (c) Introduce calibration / robustness metrics (e.g., demographic subgroup accuracy) rather than just simple accuracy.

## Related Work & Insights
- **vs MTEB / BEIR**: MTEB / BEIR use "coverage" as a selling point; this paper argues that breadth cannot compensate for the blind spot in implicit semantics, acting as a methodological critique of the MTEB paradigm.
- **vs SimCSE / E5 / BGE**: These improve contrastive objectives and negatives but still operate under the implicit assumption of "surface invariance." This paper calls for including "differentiation of implicit intent despite surface equivalence" in the loss.
- **vs Instruction-following Retrieval (INSTRUCTOR / Promptriever)**: Viewed as the closest "bridge"—conditioning embedding spaces has begun to break the pure surface paradigm, but the authors emphasize that instruction following $\neq$ pragmatic / stance sensitivity.
- **vs LLM2Vec / GritLM / NV-Embed**: Experiments confirm LLM-based embedders are stronger in implicit tasks, but the authors attribute this to base world knowledge rather than contrastive training objectives.
- **vs Social NLP Classics**: This paper "translates" insights from social NLP (Hovy & Yang 2021; Sap et al. 2020) into actionable research agendas for the embedding community.

## Rating
- Novelty: ⭐⭐⭐⭐ (Does not propose a new model, but introduces "implicit semantics" as a first-class citizen with a three-level framework and pilot evidence.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (A pilot of 14 models × 7 datasets is solid for a position paper; more metrics beyond accuracy would be better.)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear chain of argument, user-friendly QUESTION/TAKEAWAY structure, and well-cited linguistic grounding.)
- Value: ⭐⭐⭐⭐ (Provides a three-direction executable agenda; if adopted, it could trigger a new round of restructuring for embedding benchmarks and training paradigms.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[ICML 2026\] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[ICLR 2026\] Efficient Audio-Visual Speech Separation with Discrete Lip Semantics and Multi-Scale Global-Local Attention](../../ICLR2026/audio_speech/efficient_audio-visual_speech_separation_with_discrete_lip_semantics_and_multi-s.md)
- [\[CVPR 2026\] SAVE: Speech-Aware Video Representation Learning for Video-Text Retrieval](../../CVPR2026/audio_speech/save_speech-aware_video_representation_learning_for_video-text_retrieval.md)

</div>

<!-- RELATED:END -->
