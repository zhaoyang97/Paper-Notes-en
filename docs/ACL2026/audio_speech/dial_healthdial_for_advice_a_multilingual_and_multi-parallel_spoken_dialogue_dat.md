---
title: >-
  [Paper Note] Dial HEALTHDIAL for Advice: A Multilingual and Multi-Parallel Spoken Dialogue Dataset for Knowledge-Grounded Information Seeking
description: >-
  [ACL2026][Audio & Speech][spoken dialogue] HEALTHDIAL constructs a dataset comprising 4 official WHO languages, 6,000 multi-parallel health information-seeking dialogues…
tags:
  - "ACL2026"
  - "Audio & Speech"
  - "spoken dialogue"
  - "multilingual benchmark"
  - "health RAG"
  - "WHO knowledge"
  - "ASR"
date: 2026-05-08
content_hash: 5a4dfba219181ccb
---

# Dial HEALTHDIAL for Advice: A Multilingual and Multi-Parallel Spoken Dialogue Dataset for Knowledge-Grounded Information Seeking

**Conference**: ACL2026 Findings  
**arXiv**: [2605.30107](https://arxiv.org/abs/2605.30107)  
**Code**: https://github.com/cambridgeltl/healthdial  
**Area**: Spoken Dialogue / Multilingual RAG  
**Keywords**: spoken dialogue, multilingual benchmark, health RAG, WHO knowledge, ASR

## TL;DR
HEALTHDIAL constructs a dataset comprising 4 official WHO languages, 6,000 multi-parallel health information-seeking dialogues, and 163 hours of real user speech. It establishes a multilingual spoken RAG benchmark based on ASR, TTS, retrieval, knowledge filtering, and user research.

## Background & Motivation
**Background**: Most dialogue system research remains text-centric. Even when speech is supported, modular pipelines (ASR → Text Dialogue Model → TTS) are typically adopted. Existing multilingual dialogue datasets usually cover tourism, daily conversations, or task-oriented text, but lack support for real speech, knowledge grounding, multi-parallel structures, and speaker metadata.

**Limitations of Prior Work**: Speech is the most natural form of human communication, but constructing speech-first dialogue datasets involves high costs, privacy risks, and difficulties in naturally collecting cross-lingual parallel data. The health domain is particularly sensitive: real patient consultations contain personal health information, posing high risks for direct collection. However, without high-quality spoken dialogue data, it is difficult to evaluate future speech-native or multilingual RAG systems.

**Key Challenge**: The research community requires authentic and natural multilingual spoken dialogue, but real health consultation data is difficult to release publicly. Purely machine-generated dialogues tend to be repetitive and lack natural oral variation; purely translated data produces "translationese," weakening the naturalness of each language.

**Goal**: The authors aim to construct a dataset that is multilingual, multi-parallel, and knowledge-grounded, while including real user speech and speaker sociolinguistic variables. Furthermore, they provide baselines, a prototype system, and a reusable data collection toolkit.

**Key Insight**: Content control is decoupled from linguistic naturalness. LLMs and schemas are used to control dialogue structure and knowledge grounding, while native speakers complete the natural oral expressions and recordings.

**Core Idea**: Separate content control and linguistic naturalness: use LLMs and schemas to control dialogue structure and knowledge grounding, and use native speakers to perform natural spoken expressions and recordings.

## Method
HEALTHDIAL serves as both a dataset and a benchmark. The dataset includes Arabic, Chinese, English, and Spanish, with 1,500 dialogues per language, totaling 6,000 dialogues, 41,988 dialogue turns, approximately 163 hours of user speech, and 208 hours of machine-generated system speech. Each system turn is explicitly associated with WHO knowledge snippets, with labels indicating whether retrieval is required.

### Overall Architecture
Data collection proceeds in four steps. Step 1: Knowledge base construction, where snippets are extracted from WHO Questions and Answers and Fact Sheets, and parallel identifiers are assigned to alignable snippets across the four languages. Step 2: Pilot experiment, involving the collection of text consultations between 20 users and a GPT-4o prototype health advisor, with 11 dialogue act categories induced using dialogue act theory. Step 3: A first-order Markov chain is used to sample 1,500 dialogue schemas from the pilot dialogue structures, combined with WHO snippets of the same topic to generate English hypothetical dialogues via GPT-4o. Step 4: User utterances are converted into improvisational prompts, allowing native speakers of the four languages to naturally record and transcribe based on context.

For the benchmark, system input consists of the current user speech with history. The pipeline includes ASR, retrieval turn classification, knowledge retrieval, knowledge filtering, response generation, and TTS. The authors establish baselines for each component rather than providing only an end-to-end score.

### Key Designs
1. **Multi-parallel WHO knowledge grounding**:
    - **Function**: Enables each system response to be traced back to trusted health knowledge sources.
    - **Mechanism**: Knowledge snippets are derived from WHO Q&A and Fact Sheets, totaling 12,045 entries (Arabic: 2,317; Chinese: 2,431; English: 4,785; Spanish: 2,512). 1,618 snippets are fully parallel across all four languages, corresponding to 6,472 parallel snippet instances.
    - **Design Motivation**: Health dialogues cannot rely on unconstrained parametric knowledge. Restricting responses to a knowledge base allows responses not supported by the KB to be defined as extrinsic hallucinations, facilitating explicit modeling of Out-of-Knowledge (OOK) scenarios.

2. **Schema-guided outline-based data generation**:
    - **Function**: Generates structurally diverse and linguistically natural dialogues while avoiding real-world privacy risks.
    - **Mechanism**: Dialogue acts are first summarized from 20 pilot dialogues, then used to sample dialogue act sequences via a Markov chain. The LLM generates English hypothetical dialogues and improvisational prompts; final user utterances are naturally expressed, recorded, ASR-transcribed, and manually revised by native speakers based on the prompt and context.
    - **Design Motivation**: Direct LLM-generated dialogues are prone to templating, and machine translation suffers from translationese. An outline-based method maintains cross-lingual content comparability while allowing each language its own natural surface realization.

3. **Component-level spoken RAG benchmark**:
    - **Function**: Identifies bottlenecks within different modules of the multilingual spoken dialogue pipeline.
    - **Mechanism**: ASR is evaluated using WER/CER; TTS uses MCD and ASR-based CER; retrieval turn classification uses accuracy; knowledge retrieval uses text-to-text and speech-to-text retrieval; knowledge filtering uses Exact Match (EM) and OOK Recall. During knowledge filtering, top-5 retrieved snippets are input, and the model determines which snippets support the answer.
    - **Design Motivation**: Current speech-native models are not yet stable; direct end-to-end evaluation makes it difficult to interpret failure sources. A component-level benchmark clarifies whether ASR, cross-modal retrieval, or deductive filtering is the primary bottleneck.

### Loss & Training
The focus of this paper is the dataset and benchmark; no new model losses are proposed. In the baselines, retrieval turn classification compares fine-tuned XLM-R$_large$ and LLaMA3.1-8B-Inst with 10 in-context examples. Knowledge retrieval compares text-embedding-3L, gte-multilingual-B, MiniLM-L12-v2, NV-Embed-v2, BM25, and speech-to-text encoders like CLAP and SpeechT5. Knowledge filtering compares threshold-based methods, gpt-4.1-nano, LLaMA3.1-8B-Inst, and OpenAI GPT family models. TTS uses gpt-4o-mini-tts, conditioned on speaker variables such as age, primary language, origin, residential region, and education level.

## Key Experimental Results

### Main Results
| Language | ASR WER ↓ | ASR CER ↓ | TTS MCD ↓ | TTS CER ↓ | Turn Cls. Acc. ↑ | R@10 (Text) ↑ | R@10 (Speech) ↑ | Filtering EM ↑ | OOK Recall ↑ |
|--------|------|------|------|------|------|------|------|------|------|
| Arabic | 0.23 | 0.07 | 12.08 | 0.10 | 95.39 | 65.88 | 0.20 | 34.27 | 0.00 |
| Chinese | 0.24 | 0.14 | 11.46 | 0.17 | 95.23 | 70.63 | 0.23 | 39.19 | 14.29 |
| English | 0.03 | 0.01 | 11.44 | 0.06 | 96.30 | 75.72 | 0.52 | 44.29 | 42.86 |
| Spanish | 0.02 | 0.01 | 10.84 | 0.07 | 95.93 | 71.82 | 0.42 | 39.54 | 14.29 |
| Average | 0.13 | 0.06 | 11.46 | 0.10 | 95.71 | 71.01 | 0.34 | 39.32 | 17.36 |

### Ablation Study
| Knowledge Filtering Method | Arabic EM | Chinese EM | English EM | Spanish EM | Average EM | Notes |
|------|---------|------|------|------|------|------|
| Threshold | 6.26 | 6.61 | 6.88 | 6.46 | 6.55 | Fixed similarity threshold; very low performance |
| LLM @ Top-5 | 19.96 | 19.86 | 23.02 | 21.09 | 21.05 | gpt-4.1-nano filtering from top-5; best on average |
| LLM @ Top-10 | 12.58 | 17.15 | 23.33 | 19.55 | 18.15 | More distractors as candidates increase; performance drops |
| LLM @ Top-50 | 10.85 | 12.28 | 18.72 | 11.03 | 13.72 | Long candidate lists significantly hurt filtering accuracy |

### Key Findings
- ASR is strong for English and Spanish, but significantly more difficult for Arabic and Chinese. WER values are Arabic 0.23, Chinese 0.24, English 0.03, and Spanish 0.02.
- Text-to-text retrieval far outperforms speech-to-text retrieval. Average R@10(Text) is 71.01, while R@10(Speech) is only 0.34, indicating that current cross-modal speech-to-text retrieval is essentially unusable.
- Retrieval turn classification is relatively simple. Accuracy across all four languages is approximately 95%, partly because 75.5% of dialogue turns require knowledge retrieval.
- Knowledge filtering is a high-value challenge. Even using gpt-4.1-nano to filter top-5 candidates, the average EM is only 21.05; expanding to top-50 causes it to drop to 13.72.
- English performs best across multiple components, while Arabic is the weakest; this gap persists even in the fully parallel setting, suggesting the issue is systemic model capability and representation differences rather than data inconsistency.

## Highlights & Insights
- The data design of HEALTHDIAL is highly granular. It is not just "health QA + speech" but simultaneously addresses multilinguality, multi-parallelism, knowledge grounding, real user speech, speaker metadata, and OOK scenarios.
- Outline-based collection is the methodological highlight. It bypasses real patient privacy concerns while avoiding the formulaic nature of pure LLM dialogues, making the data more suitable as a spoken dialogue benchmark.
- The component-level benchmark is pragmatic. Instead of forcing a unified end-to-end score, the authors acknowledge that current speech-native models are unstable and decompose failures into ASR, retrieval, filtering, and TTS.
- The knowledge filtering table provides a crucial conclusion: providing more retrieved snippets to the LLM is not necessarily better. A top-50 list introduces numerous distracting snippets, causing EM to drop from 21.05 (top-5) to 13.72.

## Limitations & Future Work
- The authors explicitly state that HEALTHDIAL content is assisted by LLM generation and has not been verified by medical experts. It should be treated as a linguistic resource for researching multilingual knowledge-grounded spoken dialogue, not as clinical advice data.
- The WHO knowledge base ensures parallelism and authoritative sourcing but limits local cultural adaptation. Differences in health practices, traditional medicine, and public health needs across regions will require future collaboration with medical and cultural experts.
- The current benchmark still uses a pipeline architecture. While component-level evaluation is useful for diagnostics, it does not fully represent future speech-native end-to-end systems.
- The user study was conducted only with 25 English-fluent participants, primarily to demonstrate the TAM2 evaluation process; large-scale cross-lingual assessment of usability, trust, and satisfaction remains future work.

## Related Work & Insights
- **vs MultiWOZ / Multi3WOZ**: These multilingual task-oriented dialogue datasets are primarily text-based; HEALTHDIAL further provides real user speech, health knowledge grounding, and speaker sociolinguistic variables.
- **vs MedDialog / Health Forum Data**: Health forums are more authentic but involve complex privacy and noise issues, and are mostly Chinese/English text. HEALTHDIAL trades clinical authenticity for public availability, multi-parallelism, and controllable grounding.
- **vs Common Voice / Switchboard**: These speech datasets contain speaker metadata or rich audio but do not provide multi-turn knowledge-grounded health dialogues. HEALTHDIAL integrates speech with dialogue tasks.
- **vs RAG benchmark**: Traditional RAG benchmarks focus on text-based retrieval QA; HEALTHDIAL incorporates spoken input, turn classification, knowledge filtering, TTS, and user experience into a single system evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of multilinguality, multi-parallelism, real user speech, health RAG, and speaker metadata is rare; the data collection method is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Component benchmarks are comprehensive, but end-to-end speech-native evaluation and cross-lingual user studies are still insufficient.
- Writing Quality: ⭐⭐⭐⭐⭐ Data workflows, ethical constraints, and benchmark task definitions are clearly articulated, with transparent numerical results.
- Value: ⭐⭐⭐⭐⭐ High long-term value for research in multilingual spoken dialogue, medical RAG, ASR/TTS fairness, and cross-modal retrieval.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SDiaReward: Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness](sdiareward_modeling_and_benchmarking_spoken_dialogue_rewards_with_modality_and_c.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](../../ICML2026/audio_speech/the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)

</div>

<!-- RELATED:END -->
