---
title: >-
  [Paper Note] Multimodal Fact-Level Attribution for Verifiable Reasoning
description: >-
  [ICML 2026][LLM Reasoning][Multimodal Attribution] MURGAT is the first benchmark to evaluate MLLMs’ ability to provide "fact-level, modality+timestamp precise citations" in multimodal reasoning outputs. It introduces a t…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Multimodal Attribution"
  - "Citation Quality Evaluation"
  - "Atomic Fact Decomposition"
  - "MURGAT-SCORE"
  - "Reasoning-Attribution Decoupling"
date: 2026-05-08
content_hash: 2f555bcbf6ea1e33
---

# Multimodal Fact-Level Attribution for Verifiable Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.11509](https://arxiv.org/abs/2602.11509)  
**Code**: [github.com/meetdavidwan/murgat](https://github.com/meetdavidwan/murgat)  
**Area**: Multimodal VLM / Verifiable Reasoning / Evaluation  
**Keywords**: Multimodal Attribution, Citation Quality Evaluation, Atomic Fact Decomposition, MURGAT-SCORE, Reasoning-Attribution Decoupling

## TL;DR
MURGAT is the first benchmark to evaluate MLLMs’ ability to provide "fact-level, modality+timestamp precise citations" in multimodal reasoning outputs. It introduces a three-step evaluation protocol (verifiable claim identification → atomic fact decomposition → attribution quality) and a highly human-aligned automatic evaluator, MURGAT-SCORE (Pearson 0.84). The study reveals that even strong models often cite incorrectly despite correct answers, and that strong reasoning often comes at the expense of verifiable citation.

## Background & Motivation

**Background**: MLLMs are increasingly used for real-world tasks involving multi-step reasoning and long-form answers (e.g., video QA, medical reports, educational demonstrations). Reliable deployment requires outputs to be "traceable"—each factual claim must map back to a specific modality and timestamp in the input. Existing work on textual attribution (Gao 2023b) and video temporal localization (Hendricks 2017, Lei 2021) mostly focuses on simple, observational or retrieval-based scenarios (directly locating "which frame" something appears in).

**Limitations of Prior Work**: (1) Existing evaluations either focus on a single modality (vision) or only assess at the whole-source (whole-video) level, failing to distinguish "observable" from "reasoning" sentences, which allows models to score highly even with incorrect timestamps; (2) Real tasks require joint attribution across heterogeneous modalities (video + audio + charts) and fine-grained evaluation at the "atomic fact" level; (3) Mainstream "generate-then-attribute" pipelines often sacrifice reasoning quality for citation quality.

**Key Challenge**: The internal latent reasoning process and the verifiable surface citation are decoupled in MLLMs—longer reasoning chains make final citations harder to trace, while stricter citation requirements suppress complex reasoning.

**Goal**: (1) Construct a fine-grained multimodal attribution benchmark that distinguishes "observation" from "reasoning"; (2) Provide a highly human-aligned automatic evaluator to enable large-scale benchmarking; (3) Systematically characterize the relationship between reasoning effort, model scale, attribution strategy, and final attribution quality.

**Key Insight**: The response is processed in three layers—only observable sentences require citation, sentences are decomposed into atomic facts for precision/recall evaluation, and modality and timestamp are explicitly distinguished. This fully decouples "reasoning quality" from "citation quality" in evaluation, exposing their trade-off.

**Core Idea**: Reconstruct verifiable multimodal attribution evaluation as a three-stage pipeline: sentence-level filtering → atomic fact decomposition + citation propagation → set-based precision/recall entailment verification. Use MLLM-as-judge to select and calibrate the optimal automatic evaluator to human judgments.

## Method

### Overall Architecture
Task: Given multimodal input $I$ (video/audio/chart) and question $Q$, the MLLM generates a response $R=\{r_i\}$. For each verifiable sentence $r_i$, a citation set $C_i = \{c_i^j\}$ is also required, where each $c_i^j$ specifies modality + timestamp (e.g., `(audio, 0:42-0:46)`). Evaluation proceeds in three steps: (1) **Verifiable Claim Identification**—an LLM verifier determines whether $r_i$ is observable, filtering out reasoning sentences; (2) **Atomic Fact Decomposition**—each verifiable sentence is decomposed into a set of atomic facts $A_i = \{a_i^1, \ldots, a_i^n\}$, with decontextualization replacing pronouns with concrete entities; the sentence-level $C_i$ is propagated to each atomic fact; (3) **Attribution Quality**—for each $(a_i^j, C_i)$, bidirectional entailment is checked to compute recall (whether the joint citation fully supports the fact) and precision (whether each citation is strictly necessary).

### Key Designs

1. **Verifiable Claim Identification: Separating "Observation" from "Reasoning"**:

    - **Function**: Prevents forcing citations for reasoning sentences and stops models from gaming scores by omitting citations in reasoning sentences.
    - **Mechanism**: The LLM verifier determines whether each sentence $r_i$ is directly observable from $I$, yielding $R_v = \{r_i \in R \mid \text{Verifier}(r_i, I) = \text{True}\}$. For example, "The video explicitly defines thrust as positive (audio 0:42-0:46, visual 0:45)" is a verifiable sentence and should be retained; "Therefore, this statement is incorrect" is a reasoning sentence and should be discarded. Subsequent evaluation is performed only on the set of verifiable sentences with citations $R_{vc} = \{r_i \in R_v \mid C_i \neq \emptyset\}$.
    - **Design Motivation**: Traditional attribution evaluation treats all sentences equally, either forcing models to insert citations into reasoning sentences (hurting reasoning quality) or penalizing reasoning sentences as "unattributable" (unfair). This filtering ensures precision/recall is computed only where citations are appropriate, which is the most critical engineering choice in this protocol.

2. **Atomic Fact Decomposition + Citation Propagation + Decontextualization**:

    - **Function**: Eliminates confusion from sentences containing multiple facts, enabling precision/recall computation at the finest granularity.
    - **Mechanism**: For each $r_i \in R_{vc}$, an LLM decomposer splits it into atomic facts $\{a_i^1, \ldots, a_i^n\}$, each being the smallest independently verifiable claim; decontextualization resolves pronouns to concrete entities; the sentence-level citation set $C_i$ is copied to all atomic facts, yielding pairs $\{(a_i^j, C_i)\}$.
    - **Design Motivation**: Sentence-level evaluation can give inaccurate scores to compound sentences with mixed correctness; citation propagation preserves the original citation context without requiring MLLMs to cite at atomic granularity during generation (unrealistic); decontextualization, as validated in FActScore (Min 2023), is extended here to multimodal settings.

3. **Set-Based Bidirectional Entailment + MURGAT-SCORE Calibration**:

    - **Function**: Covers both "whether citations sufficiently support the fact" and "whether each citation is necessary," and selects the MLLM judge most aligned with human judgment.
    - **Mechanism**: For each $(a_i^j, C_i)$, an MLLM determines whether $C_i$ jointly entails $a_i^j$ (recall); if entailed, each $c_i^k$ is tested for strict necessity (precision, akin to leave-one-out). The overall metric MURGAT-S aggregates coverage = $|R_{vc}|/|R_v|$ plus precision/recall/F1. The authors collected full human annotations for all three tasks on WorldSense and Video-MMMU, scanned multiple MLLMs as judges (Gemini-2.5-Flash, Gemini-3-Flash/Pro, Qwen3-Omni-Instruct/Thinking), and selected the optimal judge combination with Pearson r=0.84, significantly outperforming the next-best LLM-as-judge (r=0.59).
    - **Design Motivation**: Bidirectional verification prevents gaming recall by adding redundant citations; aligning citations by "modality+timestamp" is the core difference in multimodal attribution; calibrating the judge to human annotations is essential to make MLLM-as-judge a trustworthy evaluation proxy.

### Loss & Training
No model is trained in this work; only the evaluation protocol is constructed. MURGAT-SCORE is the evaluation metric. A possible research direction is programmatic inference-time decoupling of reasoning and citation (reason first, then extract citations), which the paper shows can improve MURGAT-S by +9.6 at the cost of reduced answer accuracy.

## Key Experimental Results

### Main Results
Evaluation of various strong MLLMs on WorldSense and Video-MMMU.

| Model | QA Accuracy | MURGAT-S | Phenomenon |
|-------|-------------|----------|------------|
| Gemini-3-Pro | High | High | Larger model + more reasoning → more accurate citations |
| Gemini-2.5-Flash | Medium | Medium | Correct answers but frequent citation errors or omissions |
| Qwen3-Omni-Instruct | Medium | Low | Single-step instruction version has mediocre citation quality |
| Qwen3-Omni-Thinking | Slightly higher | Actually lower | Small model with more reasoning → citations become messier |
| Decoupled "reason first → then extract citation" pipeline | Slightly lower answers | +9.6 | Systematic trade-off |

### Ablation Study

| Configuration | Key Phenomenon | Description |
|---------------|---------------|-------------|
| No Verifiable Claim Identification | Reasoning sentences penalized | Distorted precision/recall |
| No atomic decomposition | Sentence-level evaluation, unfair to compound sentences | Mixed correctness gets inflated scores |
| No citation leave-one-out | Precision fails | Model inflates scores with redundant citations |
| Judge uses GPT-4o-mini single model | r=0.59 | Significantly worse than optimal combination |
| Judge uses Gemini-3-Pro + calibration | r=0.84 | Final MURGAT-S setting |

### Key Findings
- "Reasoning tax" phenomenon: Adding citation requirements to simple recognition tasks reduces QA accuracy (reasoning tax), but in complex reasoning tasks, it acts as a scaffold—structured citation forces the model to break down reasoning chains.
- Interaction of model scale and effort: Gemini-3-Pro improves MURGAT-S with increased reasoning budget; smaller models (Qwen3-Omni-Thinking) perform worse with more reasoning, possibly due to decoupling of latent reasoning and surface citation.
- Even strong models with correct QA have high citation error rates (hallucinated grounding), indicating that "knowing the answer" and "knowing where the answer was seen" are distinct abilities in MLLMs.

## Highlights & Insights
- **Explicit distinction between "verifiable" and "reasoning" sentences**: Redefining attribution evaluation from "demanding citations for all sentences" to "evaluating only verifiable sentences" is the key protocol design, establishing a paradigm for future multimodal attribution research.
- **Atomic facts + modality+timestamp citation**: Strictly extends the FActScore approach from textual attribution to multimodal (must specify "in video at 1:16" or "in audio 0:42-0:46"), with leave-one-out verification to prevent redundant citations, making the evaluation much more robust than previous source-level attribution.
- **MURGAT-SCORE highly consistent with human judgment**: LLM-as-judge with r=0.84 enables large-scale automated evaluation; the multi-judge calibration method is transferable to any setting requiring MLLM as evaluator.

## Limitations & Future Work
- Evaluation relies on LLM verifier/decomposer/entailment judge, which may introduce bias; despite human calibration, cross-domain generalization remains a risk.
- Datasets are mainly from WorldSense and Video-MMMU; scalability to other modality combinations (e.g., medical imaging + records + experimental charts) needs validation.
- No training-side solution is proposed—how to train MLLMs to learn accurate citation without sacrificing reasoning remains open; the paper only demonstrates the "decoupled pipeline" trade-off without systematic training.
- Citation granularity (timestamp) depends on manual segmentation accuracy; fuzzy fact boundaries may introduce noise into precision.

## Related Work & Insights
- **vs MCiteBench / MAVIS**: These focus on image-level VQA and document-level evidence, with single modality and coarse granularity; MURGAT enforces dual modality+timestamp labels and includes audio/charts.
- **vs MIRAGE**: MIRAGE uses atomic decomposition and VLM verification for multimodal RAG, finding that strong models also often cite incorrectly; MURGAT’s protocol is finer-grained (distinguishing verifiable vs reasoning) and citation granularity extends to timestamps.
- **vs Video Temporal Localization (Hendricks 2017, Lei 2021)**: Traditional video localization assumes the target segment is specified in the prompt; this work requires the model to select evidence itself, closer to real reasoning tasks.
- **vs FActScore (Min 2023)**: The inspiration for atomic fact decomposition comes directly from FActScore, but is extended here to multimodal and set-based bidirectional verification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Explicit separation of "verifiable vs reasoning" and multimodal timestamp-level citation is the first complete solution in this area.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple strong models, decoupled pipeline, and reasoning effort scan, but only two datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Figure 1 intuitively presents the entire protocol; definitions and examples are very clear.
- Value: ⭐⭐⭐⭐⭐ Provides infrastructure for verifiability research in trustworthy MLLM deployment and will be widely cited by future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Segment-Level Attribution for Selective Learning of Long Reasoning Traces](../../ICLR2026/llm_reasoning/segment-level_attribution_for_selective_learning_of_long_reasoning_traces.md)
- [\[ICML 2026\] Diagnosing Multi-step Reasoning Failures in Black-box LLMs via Stepwise Confidence Attribution](diagnosing_multi-step_reasoning_failures_in_black-box_llms_via_stepwise_confiden.md)
- [\[ICML 2026\] Are Tools Always Beneficial? Learning to Invoke Tools Adaptively for Dual-Mode Multimodal LLM Reasoning](are_tools_always_beneficial_learning_to_invoke_tools_adaptively_for_dual-mode_mu.md)
- [\[ACL 2026\] Stabilizing Efficient Reasoning with Step-Level Advantage Selection](../../ACL2026/llm_reasoning/stabilizing_efficient_reasoning_with_step-level_advantage_selection.md)
- [\[ACL 2026\] SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks](../../ACL2026/llm_reasoning/sppo_sequence-level_ppo_for_long-horizon_reasoning_tasks.md)

</div>

<!-- RELATED:END -->
