---
title: >-
  [Paper Note] CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering
description: >-
  [ACL2026][Video Understanding][Multi-video QA] CRAFT is a claim-centric pipeline for multi-video QA of news events. By combining dynamic key-frame selection, ASR transcription…
tags:
  - "ACL2026"
  - "Video Understanding"
  - "Multi-video QA"
  - "Evidence Attribution"
  - "Key-frame Selection"
  - "ASR"
  - "critic refinement"
date: 2026-05-08
content_hash: e3839ad6d5918581
---

# CRAFT: Critic-Refined Adaptive Key-Frame Targeting for Multimodal Video Question Answering

**Conference**: ACL2026  
**arXiv**: [2605.19075](https://arxiv.org/abs/2605.19075)  
**Code**: https://github.com/bhosalems/CRAFT  
**Area**: Video Understanding / Multi-video Question Answering  
**Keywords**: Multi-video QA, Evidence Attribution, Key-frame Selection, ASR, critic refinement  

## TL;DR
CRAFT is a claim-centric pipeline for multi-video QA of news events. By combining dynamic key-frame selection, ASR transcription, iterative refinement via UNLI/MNLI/LLM critics, and citation merging, it achieves 0.739 macro average, 0.810 reference recall, and 0.635 citation F1 on MAGMaR-Test.

## Background & Motivation
**Background**: Multi-video question answering and grounded generation require systems to extract facts from a set of related videos and provide traceable video sources for each conclusion. News event scenarios are particularly typical: answers may be scattered across multiple clips, reports in different languages, interview audio, and on-screen text.

**Limitations of Prior Work**: Long videos impose severe token/frame budget pressure on VLMs. Uniform sampling often misses sparse but critical frames; focusing only on visual content loses audio evidence such as interviews, broadcasts, and official statements. Even if relevant frames are fed into the model, the VLM may still generate details without visual or audio support.

**Key Challenge**: A high-scoring system must simultaneously "cover more facts" and ensure "every fact has a correct citation." Simply improving recall introduces unsupported claims, while being overly conservative misses key information in the reference answers.

**Goal**: The authors aim to build a multi-video QA system for the MAGMaR 2026 oracle task, using atomic claims as an intermediate layer to first extract, verify, and rank evidence before generating a final report with citations.

**Key Insight**: Instead of treating the initial VLM response as the final answer, CRAFT introduces a critic loop at the claim level. It uses UNLI for video-claim temporal entailment, DeBERTa-v3 MNLI to screen for contradictions between claims, and Llama-3.2-3B to adjudicate and provide repair feedback.

**Core Idea**: Decompose multi-video evidence into verifiable atomic claims, use specialized critics to remove weakly supported and contradictory claims, and finally merge duplicate facts into a report with multi-source citations.

## Method
The CRAFT pipeline can be understood as "Video evidence stream construction → Atomic claim extraction → Iterative critic refinement → Claim scoring and selection → Citation-preserving generation." It runs for each query and related video set, maintaining mapping from chunks to parent videos to ensure citations can trace back to the original videos.

### Overall Architecture
Input consists of a persona, query, and a list of related videos. The system first segments long videos into chunks of up to 120 seconds and caches ASR and translations for each video. For each query-video pair, dynamic key-frame selection provides a compact visual input, followed by Qwen3.5-9B/VL extracting atomic claims with sources, timestamps, and evidence types. The critic loop refines claims for up to 4 rounds. Finally, claims are ranked by UNLI support scores, and a text LLM generates the final report.

### Key Designs
1. **Query-conditioned Multi-modal Evidence Stream**:

	- **Function**: Provides the most relevant visual and audio evidence for the current query within long videos.
	- **Mechanism**: Videos are processed in 120-second chunks; each unique video is transcribed once using Qwen3-ASR-1.7B, with Whisper-large-v3 as a fallback for low-resource languages, followed by English translation for non-English transcripts. On the visual side, CLIP-based image-text similarity scores candidate frames, and key-frames are selected to balance relevance and temporal coverage.
	- **Design Motivation**: In multi-video news QA, critical evidence may appear in audio, on-screen text, or just a few frames. Explicit ASR and DKS reduce evidence loss inherent in uniform sampling and visual-only pipelines.

2. **Atomic Claim Extraction and Critic Loop**:

	- **Function**: Restricts VLM output to fact units that can be individually verified and repaired.
	- **Mechanism**: Each claim must be a single statement with an associated evidence modality. The critic loop uses three types of checks: UNLI scores the cited video segment, where claims with scores below 0.05 are considered unsupported and 0.05 to 0.5 are considered weakly supported; DeBERTa-v3 MNLI identifies contradiction candidates where the probability exceeds 0.5; and Llama-3.2-3B confirms contradictions and returns a repair hint.
	- **Design Motivation**: Checking only at the final report level is too coarse and fails to locate specific errors. Claim-level critics clean up hallucinations, temporal mismatches, and cross-claim contradictions early on.

3. **Citation-preserving Evidence Pooling and Consolidation**:

	- **Function**: Retains sources when merging facts across videos, ensuring citations are not lost during deduplication.
	- **Mechanism**: All refined claims for a query enter an evidence pool, with each record maintaining its video ID, timestamp, modality, and claim ID. Top claims are selected as a claim packet after UNLI re-scoring. The final text LLM is restricted to using information within this packet and merges multiple source identifiers supporting the same fact into a single sentence.
	- **Design Motivation**: MAGMaR evaluates both information quality and citation correctness. Simple deduplication improves conciseness but hurts citation recall; citation merging balances both.

### Loss & Training
CRAFT is a system pipeline without end-to-end training loss. The key strategy involves inference-time constraints and validation: DKS uses image-text similarity to select frames, UNLI support scores are used for temporal grounding and claim ranking, MNLI screens for high-recall contradiction candidates, and a Llama adjudicator handles secondary confirmation. The critic loop runs for a maximum of $R=4$ rounds and terminates early if the claim set no longer changes.

## Key Experimental Results

### Main Results
| System | MAGMaR Ref-P | MAGMaR Ref-R | MAGMaR Cite-F1 | MAGMaR Avg | WikiVideo Ref-F1 | WikiVideo Cite-F1 | WikiVideo Avg |
|------|--------------|--------------|----------------|------------|------------------|-------------------|---------------|
| Molmo2-8B | 0.623 | 0.541 | 0.457 | 0.518 | 0.661 | 0.552 | 0.607 |
| InternVL-3.5-30B + ASR | 0.761 | 0.722 | 0.600 | 0.672 | 0.831 | 0.727 | 0.779 |
| Gemma-4-31B + ASR | 0.712 | 0.701 | 0.580 | 0.644 | 0.754 | 0.640 | 0.697 |
| CRAFT Baseline | 0.437 | 0.756 | 0.359 | 0.518 | 0.834 | 0.764 | 0.814 |
| + Critic Loop | 0.491 | 0.766 | 0.360 | 0.535 | 0.842 | 0.773 | 0.822 |
| + Atomic Claims | 0.808 | 0.762 | 0.426 | 0.673 | 0.735 | 0.848 | 0.809 |
| + ASR / Full CRAFT | 0.760 | 0.810 | 0.635 | 0.739 | 0.854 | 0.762 | 0.823 |

Full CRAFT achieves the highest overall average of 0.739 on MAGMaR-Test, with a reference recall of 0.810 and a citation F1 of 0.635. On WikiVideo, the average is 0.823, slightly higher than the baseline's 0.814 and stronger than the visual+ASR variants of InternVL/Gemma.

### Ablation Study
| Ablation | Ref-P | Ref-R | Ref-F1 | Cite-P | Cite-R | Cite-F1 | Avg | Insight |
|------|-------|-------|--------|--------|--------|---------|-----|------|
| CRAFT full | 0.760 | 0.810 | 0.783 | 0.935 | 0.512 | 0.635 | 0.739 | Complete system |
| Replace ASR backbone with Qwen3-Omni-30B-A3B | 0.745 | 0.761 | 0.735 | 0.878 | 0.346 | 0.471 | 0.656 | Direct audio input is inferior to explicit ASR transcriptions |
| Replace UNLI with Qwen | 0.732 | 0.788 | 0.759 | 0.874 | 0.469 | 0.601 | 0.704 | Specialized temporal entailment is vital for citations |
| Replace Llama-3.2-3B adjudicator with Qwen | 0.763 | 0.812 | 0.787 | 0.937 | 0.516 | 0.619 | 0.732 | 3B adjudicator is sufficient |
| Qwen unified critic without MNLI screen | 0.743 | 0.798 | 0.770 | 0.909 | 0.493 | 0.619 | 0.722 | NLI pre-screening provides signals hard to replace with prompt generalization |

### Key Findings
- **Atomic claim formatting** is critical for MAGMaR precision, leaping from a search baseline Ref-P of 0.437 to 0.808.
- **ASR** is the core source for supplementing recall and citations. After adding ASR, Ref-R reached 0.810 and Cite-F1 improved from 0.426 to 0.635.
- **Explicit ASR transcripts** are more suitable for claim-centric verification than direct audio conditioning because names, dates, and numbers can be directly checked by subsequent text verifiers.
- Under **low frame budgets**, DKS improves precision. For instance, MAGMaR Ref-P with reduced-frame DKS was 0.822, higher than the uniform sampling score of 0.775, though recall may drop, indicating coverage issues still exist in key-frame selection.
- In terms of **auxiliary generation quality**, CRAFT's ROUGE-L/BERTScore/AnsRel were 0.1839/0.1709/0.6504 on MAGMaR and 0.3014/0.2683/0.6664 on WikiVideo, all higher than the compared VLMs.

## Highlights & Insights
- The most valuable contribution of this paper is transforming the intermediate representation of grounded video QA into claims rather than direct paragraph generation. Claims are small enough to be verified one by one by models, NLI, and entailment scorers.
- The role of ASR is proven to be very strong. A large number of facts in news videos come from spoken content, which visual-only VLMs easily miss.
- The design of the critic loop is pragmatic. UNLI, MNLI, and small LLMs each perform their specific roles, avoiding the need for a single large model to handle all verification.
- Citation merging is an engineering design well-aligned with task metrics. It avoids redundant writing of the same fact while preserving multiple supporting sources, benefiting citation recall.

## Limitations & Future Work
- **Recall and citation recall** remain challenging. Full CRAFT's Cite-R on MAGMaR is only 0.512, indicating that finding correct facts and attributing them to the correct video is not yet fully solved.
- **Low-resource language ASR** is still a weak point. The system filters out transcripts with low lexical diversity or high repetition; while this reduces noise, it may also discard useful information in resource-scarce languages.
- **DKS** improves precision under low frame budgets but may sacrifice recall; future work needs to better balance query relevance and broad coverage.
- **Future directions** include stronger cross-video retrieval, more robust multilingual ASR, more precise claim-to-video attribution, and incorporating human evaluation feedback into system tuning.

## Related Work & Insights
- **vs uniform frame sampling VLM**: Uniform sampling is simple, but key frames in long videos are sparse and easily missed; CRAFT uses DKS to select frames based on the query.
- **vs Video-RAG pipeline**: Many Video-RAG systems rely on a single visual stream or final answer aggregation; CRAFT advances verification to the claim extraction stage.
- **vs critic-driven video QA**: Previous verifiers often acted as a final role to check the answer; CRAFT iteratively refines each query-video claim set at a finer granularity.
- **Inspiration for follow-up systems**: If a multimodal generation task requires citations, the system should prioritize designing verifiable intermediate objects and restrict the final generator to using a verified evidence packet.

## Rating
- **Novelty**: ⭐⭐⭐⭐ While individual components have precedents, the combination of ASR, DKS, claim critic, and citation merging into a complete system is very solid.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers MAGMaR and WikiVideo with multiple system ablations; human evaluation is included in the appendix, while the main text focuses on automatic metrics.
- **Writing Quality**: ⭐⭐⭐⭐ Methodological procedures are clear, and metrics are fully explained; tables are dense, requiring attention to the differences between MAGMaR and WikiVideo settings.
- **Value**: ⭐⭐⭐⭐ Highly valuable for video RAG or news analysis systems requiring "answers + citations," particularly the claim-centric evidence pipeline.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](../../CVPR2026/video_understanding/movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)
- [\[CVPR 2026\] HERBench: A Benchmark for Multi-Evidence Integration in Video Question Answering](../../CVPR2026/video_understanding/herbench_a_benchmark_for_multi-evidence_integration_in_video_question_answering.md)
- [\[CVPR 2026\] Do You See What I Am Pointing At? Gesture-Based Egocentric Video Question Answering](../../CVPR2026/video_understanding/do_you_see_what_i_am_pointing_at_gesture-based_egocentric_video_question_answeri.md)
- [\[ACL 2026\] DualFact: A Multimodal Fact Verification Framework for Procedural Video Understanding](dualfact_a_multimodal_fact_verification_framework_for_procedural_video_understan.md)

</div>

<!-- RELATED:END -->
