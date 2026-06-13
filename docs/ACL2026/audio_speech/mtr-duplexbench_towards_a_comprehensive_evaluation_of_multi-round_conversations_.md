---
title: >-
  [Paper Note] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
description: >-
  [ACL 2026][Audio & Speech][Full-duplex speech models] MTR-DuplexBench is proposed as a comprehensive multi-round evaluation benchmark for Full-Duplex Speech Language Models (FD-SLMs). By introducing an innovative turn se…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Full-duplex speech models"
  - "Multi-round conversation evaluation"
  - "Turn segmentation"
  - "Dialogue quality"
  - "Safety evaluation"
date: 2026-05-08
content_hash: dfc4c15a494a0754
---

# MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models

**Conference**: ACL 2026  
**arXiv**: [2511.10262](https://arxiv.org/abs/2511.10262)  
**Code**: [https://github.com/ZhangHe0918/MTR-DuplexBench](https://github.com/ZhangHe0918/MTR-DuplexBench)  
**Area**: Speech Language Models / Evaluation Benchmarks  
**Keywords**: Full-duplex speech models, Multi-round conversation evaluation, Turn segmentation, Dialogue quality, Safety evaluation

## TL;DR

MTR-DuplexBench is proposed as a comprehensive multi-round evaluation benchmark for Full-Duplex Speech Language Models (FD-SLMs). By introducing an innovative turn segmentation method, it addresses challenges such as blurred turn boundaries and context inconsistency in full-duplex dialogues. The benchmark covers four dimensions—dialogue features, dialogue quality, instruction following, and safety—revealing a performance decay in existing FD-SLMs during multi-round interactions.

## Background & Motivation

**Background**: Full-Duplex Speech Language Models (FD-SLMs) enable real-time "listen-while-speak" interactions, supporting complex dialogue features like interruptions and backchanneling, representing the future of voice interaction. Moshi and Freeze-Omni are currently the only two open-source FD-SLMs.

**Limitations of Prior Work**: Existing benchmarks (e.g., Full-Duplex-Bench, Full-Duplex-Bench v1.5) primarily focus on single-round evaluations, whereas real-world conversations are typically multi-round. Furthermore, most current benchmarks only evaluate dialogue features (e.g., interruptions, backchanneling) while ignoring critical capabilities like instruction following and safety. While FD-Bench supports multiple rounds, it only focuses on interruption scenarios; Talking Turns requires expensive manual data collection.

**Key Challenge**: Evaluation of full-duplex dialogue faces two technical challenges: (1) Blurred turn boundaries: unlike half-duplex, full-duplex communication is spontaneous without explicit start/end markers; (2) Context inconsistency: in multi-round evaluations, model responses from previous rounds may deviate significantly from ground truth, causing subsequent user inputs to become detached from the actual scenario, thereby reducing evaluation reliability.

**Goal**: To construct a comprehensive evaluation benchmark for full-duplex SLMs that supports round-by-round assessment and covers four dimensions: dialogue features, quality, instruction following, and safety.

**Key Insight**: By designing a full-duplex turn segmentation algorithm, continuous full-duplex dialogues are partitioned into discrete rounds. By filling the assistant channel of historical rounds with ground truth responses during each evaluation step, both turn boundary and context inconsistency issues are resolved simultaneously.

**Core Idea**: Utilize GPT-4o for multiple segmentations followed by majority voting and cluster filtering to determine turn boundaries. A strategy of "ground truth for previous rounds + model inference for current round" is employed to eliminate context drift, establishing a four-dimensional comprehensive evaluation framework.

## Method

### Overall Architecture

The MTR-DuplexBench pipeline consists of two parts: (1) A full-duplex turn segmentation method, which partitions continuous dual-channel audio into discrete user turns and assigns assistant response periods to each; (2) A four-dimensional evaluation framework designing data, processes, and metrics for dialogue features (200 items x 10 rounds), dialogue quality (200 natural dialogue segments), instruction following (300 items x 10 rounds), and safety (520 items x 10 rounds).

### Key Designs

1.  **Full-Duplex Turn Segmentation Algorithm**:
    - **Function**: Identifies start and end timestamps of user turns within continuous full-duplex dialogue.
    - **Mechanism**: A four-step process: (a) Use Whisper + Silero VAD to extract dual-channel transcriptions and timestamps; (b) Sort user and assistant VAD segments chronologically and feed them into GPT-4o for turn segmentation; (c) Repeat GPT segmentation 6 times and aggregate via majority voting—new turns are merged if they have $\ge 30\%$ time overlap with existing candidates, with start/end times taking the median; (d) Final overlap resolution merges segments into definitive turns. The assistant response period is set from the start of the current user turn to the end of the next user turn.
    - **Design Motivation**: Single GPT segmentation results are unstable; majority voting and clustering ensure robust turn boundaries. The response period design ensures the assistant has sufficient time to complete its response.

2.  **Context Consistency Maintenance Strategy**:
    - **Function**: Ensures the input context for each round in a multi-round evaluation remains consistent with the ground truth scenario.
    - **Mechanism**: When evaluating round $k$, all historical responses in the assistant channel (rounds 1 to $k-1$) use ground truth audio, while only the current round is generated by the model. This ensures the model faces a "correct" context in every round, avoiding unreliable evaluations caused by error accumulation.
    - **Design Motivation**: If model responses from previous rounds are propagated, deviations would amplify, leading to evaluation scenarios that would "never occur in real conversation."

3.  **Four-Dimensional Evaluation System**:
    - **Function**: Comprehensively evaluates multiple capabilities of FD-SLMs.
    - **Mechanism**: (a) Dialogue Features: Use GPT-4o to generate 200 items of 10-round synthetic dialogues to evaluate success rates and latency for smooth turn-taking, interruptions, pause handling, background speech, and backchanneling; (b) Dialogue Quality: Use 200 segments of 120-second segments from the Candor real dialogue dataset, evaluating GPT-score (0-5) round-by-round after turn segmentation; (c) Instruction Following: Evaluate success rates using 300 voice queries from the Llama Question dataset; (d) Safety: Evaluate multi-round refusal rates using 520 harmful queries from AdvBench.
    - **Design Motivation**: Practical deployment of FD-SLMs requires ensuring instruction following and safe outputs, especially in multi-round interruption scenarios.

### Loss & Training

This paper presents an evaluation benchmark and does not involve model training. Evaluation metrics include: success rate for dialogue features (binary), GPT-score (0-5) for dialogue quality, success rate for instruction following, and refusal rate for safety, all automatically determined by GPT-4o.

## Key Experimental Results

### Main Results

Success rates of dialogue features decrease as the number of rounds increases (Round 1 vs. Average of Rounds 1-10):

| Model | Turn-taking | Interruption | Pause Handling | Background Speech |
| :--- | :--- | :--- | :--- | :--- |
| Moshi | 73.0 $\rightarrow$ 57.4% | 72.5 $\rightarrow$ 54.2% | 93.5 $\rightarrow$ 84.8% | 53.0 $\rightarrow$ 25.7% |
| Freeze-Omni | 69.0 $\rightarrow$ 36.4% | 76.0 $\rightarrow$ 56.6% | 89.0 $\rightarrow$ 68.5% | 0.5 $\rightarrow$ 1.1% |
| VocalNet (HD) | 100 $\rightarrow$ 100% | 100 $\rightarrow$ 100% | 100 $\rightarrow$ 100% | 0 $\rightarrow$ 0% |
| Cascaded | 98.5 $\rightarrow$ 99.0% | 99.5 $\rightarrow$ 96.3% | 100 $\rightarrow$ 100% | 0 $\rightarrow$ 0% |

### Ablation Study

Performance comparison of multi-feature combinations vs. single features (using Moshi as an example, average success rate for rounds 1-10):

| Configuration | Success Rate | Description |
| :--- | :--- | :--- |
| Smooth Turn-taking (S) only | 57.4% | Single feature baseline |
| S + Interruption (I) | 54.5% | Two features alternating |
| S + I + Pause (P) | 54.3% | Three features alternating |
| S + I + P + Background (B) | 37.6% | Four features alternating; significant performance drop |

### Key Findings
- FD-SLMs exhibit continuous degradation in multi-round scenarios: Freeze-Omni's turn-taking success dropped from 69% to 36%, showing the most severe decline.
- HD (Half-Duplex) models surprisingly perform perfectly on dialogue features (100% for turn-taking/interruption/pause) but fail completely to handle background speech.
- Moshi is the only model capable of handling background speech, though its success rate dropped from 53% to 25.7%.
- Moshi has the lowest latency (~0.6-0.9s), while Cascaded systems are the highest (~9-12s).
- Performance drops further when multiple features are combined, indicating that handling various dialogue features simultaneously is a major challenge for FD-SLMs.
- Multi-round interruptions may lead to a decline in FD-SLM safety refusal capabilities, posing security risks.

## Highlights & Insights
- **Engineering Design of Turn Segmentation**: The scheme using multiple GPT segmentations + majority voting + clustering is highly practical, transforming uncertain full-duplex audio streams into deterministic discrete turns. This methodology is transferable to any scenario requiring the extraction of structured events from continuous interactions.
- **Ingenuity of the Context Consistency Strategy**: Using ground truth audio to fill historical rounds eliminates error accumulation. While this evaluates "single-round capability under ideal conditions" rather than true multi-round accumulation, it is a reasonable compromise given current model levels.
- **Comprehensiveness of the Four Dimensions**: This is the first work to introduce instruction following and safety into full-duplex evaluation. The research question "safety under multi-round interruptions" is highly forward-looking.

## Limitations & Future Work
- Only two open-source FD-SLMs (Moshi and Freeze-Omni) were evaluated; the small sample size makes it difficult to draw generalized conclusions.
- While the context consistency strategy ensures evaluation reliability, it cannot assess model performance under real-world multi-round error accumulation.
- Dialogue feature synthetic data is generated by GPT-4o and may not perfectly reflect real human dialogue patterns.
- Safety evaluation only uses known harmful queries from AdvBench, lacking coverage of more complex jailbreak attacks.
- Future work could introduce more FD-SLMs and explore how to leverage benchmark signals during the training phase to improve multi-round consistency.

## Related Work & Insights
- **vs Full-Duplex-Bench**: Only supports single-round evaluation and lacks round-by-round analysis. MTR-DuplexBench significantly surpasses it in multi-round support and evaluation dimensions.
- **vs FD-Bench**: Although it supports multiple rounds (up to 5), it only focuses on interruption scenarios and lacks round-by-round evaluation.
- **vs Talking Turns**: Requires actual human-model interaction to collect data, which is difficult to scale; MTR-DuplexBench uses an automated pipeline with better reproducibility.

## Rating
- Novelty: ⭐⭐⭐⭐ First full-duplex benchmark supporting round-by-round evaluation and four-dimensional comprehensive assessment. The turn segmentation method is innovative.
- Experimental Thoroughness: ⭐⭐⭐ Only two true FD-SLMs were available for evaluation; experiment scale is limited by model availability.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and detailed methodology.
- Value: ⭐⭐⭐⭐ Fills the gap in multi-round evaluation for FD-SLMs and holds significant value for advancing full-duplex speech interaction research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)
- [\[ICML 2026\] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models](../../ICML2026/audio_speech/moshirag_asynchronous_knowledge_retrieval_for_full-duplex_speech_language_models.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ACL 2026\] Closing the Modality Reasoning Gap for Speech Large Language Models](closing_the_modality_reasoning_gap_for_speech_large_language_models.md)

</div>

<!-- RELATED:END -->
