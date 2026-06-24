---
title: >-
  [Paper Note] Unlocking Speech Instruction Data Potential with Query Rewriting
description: >-
  [ACL 2025][Speech instruction data] This work proposes a query rewriting framework based on multi-LLM knowledge fusion and a multi-agent annotation verification method. It rewrites text instructions that exceed the TTS vocabulary into formats suitable for speech synthesis, increasing the usability rate of speech instruction data from 72% to 93% to construct high-quality speech instruction datasets for end-to-end Large Speech Language Models (LSLMs).
tags:
  - "ACL 2025"
  - "Speech instruction data"
  - "Query Rewriting"
  - "TTS"
  - "Multi-agent annotation"
  - "Knowledge fusion"
date: 2026-05-08
content_hash: 0aee98287765dd82
---

# Unlocking Speech Instruction Data Potential with Query Rewriting

**Conference**: ACL 2025  
**arXiv**: [2507.08603](https://arxiv.org/abs/2507.08603)  
**Code**: Not released  
**Area**: Other  
**Keywords**: Speech instruction data, Query Rewriting, TTS, Multi-agent annotation, Knowledge fusion  

## TL;DR

This work proposes a query rewriting framework based on multi-LLM knowledge fusion and a multi-agent annotation verification method. It rewrites text instructions that exceed the TTS vocabulary into formats suitable for speech synthesis, increasing the usability rate of speech instruction data from 72% to 93% to construct high-quality speech instruction datasets for end-to-end Large Speech Language Models (LSLMs).

## Background & Motivation

**Background**: End-to-end Large Speech Language Models (LSLMs) demonstrate significant potential in response latency and speech understanding. However, their ability to follow speech instructions is not fully unleashed due to the lack of high-quality datasets and a heavy training task bias (e.g., ASR repetition tasks).

**Limitations of Prior Work**: (1) Hand-crafted collection and annotation of speech instruction data are extremely costly, making large-scale construction difficult; (2) Prior methods utilize LLMs to continue ASR data semantically to construct speech instructions, but there is a gap between LLM-generated results and real human responses, which continuation methods further amplify; (3) When using TTS for speech synthesis, TTS models have limited vocabularies and cannot correctly convert Out-Of-Distribution (OOD) texts such as abbreviations, compound words, and mathematical formulas, leading to loss of linguistic information.

**Key Challenge**: Constructing high-quality speech instruction datasets requires balancing low-cost automation with high linguistic fidelity—direct TTS synthesis leads to information loss due to OOD texts, while manual annotation is not scalable.

**Goal**: To automatically transform text instructions into high-quality synthetic speech with equivalent linguistic information at low cost, thereby constructing large-scale speech instruction datasets.

**Key Insight**: Convert OOD texts into TTS-friendly formats through zero-shot rewriting with multiple LLMs, and utilize a "multi-agent" setup comprising multiple ASR models and multiple embedding models to automatically annotate and verify the quality of synthetic speech.

**Core Idea**: Use multiple LLMs to rewrite text so that TTS can "read accurately," use multiple ASR and embedding models to verify if it "reads correctly," and handle challenging rewrite samples through knowledge fusion.

## Method

### Overall Architecture

Given an original text instruction $c_o$, multiple LLMs (Llama3, Phi3, Qwen2) are first used to rewrite it into multiple candidate texts, forming a candidate set $C = \{c_o, c_l, c_p, c_q\}$ along with the original text. Then, a TTS model (Parler-TTS) is used to synthesize the corresponding speech set $S$. Next, 3 ASR models (Whisper-large-v3, Canary-1b, Parakeet-tdt-1.1b) are used to recognize the linguistic information in the speech, and 3 Embedding models are used to calculate semantic similarity with the original text, selecting the optimal synthesis result. Finally, knowledge fusion training is performed on the remaining failed samples to fine-tune a stronger rewriting model for challenging cases.

### Key Designs

1. **Multi-agent Annotation & Verification**
    - **Function**: Accurate evaluation of the linguistic information fidelity of synthetic speech.
    - **Mechanism**: Transcribe the speech using 3 different ASR architectures (which have similar but complementary performance), compute the average semantic similarity using 3 embedding models, and select the similarity of the best ASR result as the quality score $q = \max_j F(c_o, \bar{c}_{o,j})$.
    - **Design Motivation**: Recognition errors from a single ASR model can lead to improper data filtering. Analogous to the human annotation practice of aggregating opinions from multiple annotators, utilizing the orthogonality between models reduces consensus errors.

2. **Query Rewriting with Multi-LLM**
    - **Function**: Rewrite text that TTS cannot correctly synthesize (abbreviations, formulas, compound words) into a format that TTS can handle.
    - **Mechanism**: Separately perform zero-shot rewriting on the original text using Llama3-8B, Phi3-small, and Qwen2-7B, selecting the one with the highest synthesis quality from the 4 candidates (including the original) as the TTS input.
    - **Design Motivation**: Rule-based rewriting methods are manually designed and difficult to scale. The performance of different LLMs on zero-shot rewriting is orthogonal, and having multiple candidates can cover more cases.

3. **Knowledge Fusion for Challenging Rewriting**
    - **Function**: Solve hard samples that cannot be successfully rewritten even with multiple LLMs combined.
    - **Mechanism**: Collect successfully rewritten $\langle c_i, \hat{c}_i \rangle$ sample pairs as training data to fine-tune Llama-3-8B-Instruct using LoRA. This fuses rewriting knowledge from multiple LLMs to train a stronger rewriting model specialized in handling failed samples.
    - **Design Motivation**: Difficult rewriting tasks require multi-perspective capabilities (such as simultaneously understanding context and domain knowledge). Knowledge fusion can integrate the complementary strengths of multiple models into a single model.

### Loss & Training

- **Knowledge Fusion Training**: Use standard autoregressive language model loss:
  $$\mathcal{L} = -\sum_{i=0}^{M} \log P(y_i | x, c, y_{<i})$$
  where $\langle c, y \rangle$ is a successfully rewritten sample pair.
- **LoRA Configuration**: In the knowledge fusion stage, $r=8$, $\alpha=16$, learning rate is 3e-4; in the LSLM fine-tuning stage, $r=8$, $\alpha=32$, learning rate is 3e-5.
- **Quality Threshold**: $\alpha = 0.9$ is used to distinguish between successful/failed rewrite samples.

## Key Experimental Results

### Main Results

Evaluated on 7 QA datasets, using Parler-TTS-Large-v1 under the Multi-Speaker Setting:

| Method | SIM (Avg, ↑) | PASS (Avg, ↑) |
|------|-------------|--------------|
| Original (No Rewriting) | 93.14 | 72.19 |
| Text Normalization | 95.34 | 82.05 |
| Phi3 Individual Rewriting | 97.05 | 88.48 |
| Llama3 Individual Rewriting | 96.94 | 88.36 |
| Ours w/o KF | 97.91 | 92.52 |
| **Ours** | **97.99** | **93.07** |

### Ablation Study

LSLM (Qwen2-Audio-7B-Instruct) training performance comparison:

| Training Target | Threshold | Data Method | DROP | Quoref | ROPES | NarrativeQA | Avg |
|---------|------|---------|------|--------|-------|-------------|-----|
| No Fine-tuning | - | - | 17.40 | 55.98 | 42.69 | 43.02 | 39.77 |
| Golden | 0 | Original | 29.25 | 76.01 | 55.42 | 48.34 | 52.26 |
| LLM Continue | 0 | Ours | 30.08 | 75.05 | 57.15 | 47.88 | 52.54 |
| Golden | 0.90 | Ours | **44.35** | **86.81** | **64.24** | **56.76** | **63.04** |

### Key Findings

1. Combined multi-LLM rewriting yields approximately 4.5% higher PASS than any single LLM, validating the orthogonality hypothesis.
2. Combined multi-ASR annotation reduces Word Error Rate (WER) by an average of about 1.7 percentage points compared to individual ASR models.
3. Golden (human-annotated answers) as the alignment target is significantly superior to LLM continuation (Avg 60.50 vs. 52.54), indicating that real human responses are irreplaceable.
4. The quality threshold of $t=0.90$ is optimal; setting it too high ($0.95$) degrades performance due to excessive data discarding.

## Highlights & Insights

- The proposed method is fully automatic and requires no manual annotation, performing end-to-end processing from data construction to quality evaluation.
- The multi-agent annotation and verification concept is elegant—drawing inspiration from the crowdsourced annotation idea of "seeking consensus among multiple annotators."
- The experiments reveal a key insight: in speech instruction alignment, LLM continuation data cannot replace human-annotated data.
- The method exhibits strong generalization across different TTS models, narrowing the quality gap between various TTS models from 13.95% to 0.9%.

## Limitations & Future Work

- The method is only validated on English QA datasets; its effectiveness in multilingual scenarios remains unknown.
- It relies on GPT-4 to generate speaker descriptions with only 192 variations, potentially limiting diversity.
- Knowledge fusion is only based on LoRA fine-tuning of a single backbone; multi-model ensemble training might perform better.
- Fidelity aspects of paralinguistic information such as prosody and emotion in synthetic speech have not been explored.

## Related Work & Insights

- The idea of using natural language descriptions to control voice styles in Parler-TTS is worthy of reference.
- The multi-agent annotation concept can be extended to other automatic data construction scenarios (multimodal, code, etc.).
- The knowledge fusion approach inspires research directions focusing on complementary capabilities across different models.

## Rating

- **Novelty**: 3/5 — Though the individual components are combinations of existing technologies, the overall scheme design is highly reasonable.
- **Technical Depth**: 3/5 — Primarily focused on engineering design.
- **Experimental Thoroughness**: 4/5 — Validated across multiple settings and extensive ablation studies.
- **Practicality**: 4/5 — Direct reference value for constructing speech instruction datasets.
- **Overall Score**: 3.5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Instruction-Tuning Data Synthesis from Scratch via Web Reconstruction](instruction-tuning_data_synthesis_from_scratch_via_web_reconstruction.md)
- [\[ACL 2025\] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns](genre_a_french_gender-neutral_rewriting_system_using_collective_nouns.md)
- [\[ACL 2025\] Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)
- [\[CVPR 2025\] Potential Field Based Deep Metric Learning](../../CVPR2025/others/potential_field_based_deep_metric_learning.md)
- [\[ACL 2025\] CoachMe: Decoding Sport Elements with a Reference-Based Coaching Instruction Generation Model](coachme_sport_instruction.md)

</div>

<!-- RELATED:END -->
