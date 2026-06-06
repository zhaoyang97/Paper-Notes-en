---
title: >-
  [Paper Note] Multimodal Fact-Level Attribution for Verifiable Reasoning
description: >-
  [ICML 2026][Audio & Speech][Multimodal Attribution] MURGAT is the first benchmark to evaluate MLLMs' ability to provide "fine-grained modality + timestamp citations" for multimodal reasoning outputs. It employs a three-s…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Multimodal Attribution"
  - "Citation Quality Evaluation"
  - "Atomic Fact Decomposition"
  - "MURGAT-SCORE"
  - "Reasoning-Attribution Decoupling"
date: 2026-05-08
content_hash: 01f2c87dcdd8e4b1
---

# Multimodal Fact-Level Attribution for Verifiable Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.11509](https://arxiv.org/abs/2602.11509)  
**Code**: [github.com/meetdavidwan/murgat](https://github.com/meetdavidwan/murgat)  
**Area**: Multimodal VLM / Verifiable Reasoning / Evaluation  
**Keywords**: Multimodal Attribution, Citation Quality Evaluation, Atomic Fact Decomposition, MURGAT-SCORE, Reasoning-Attribution Decoupling

## TL;DR
MURGAT is the first benchmark to evaluate MLLMs' ability to provide "fine-grained modality + timestamp citations" for multimodal reasoning outputs. It employs a three-step evaluation protocol (Verifiable Claim Identification → Atomic Fact Decomposition → Attribution Quality) and an automated evaluator, MURGAT-SCORE, which shows high human alignment (Pearson 0.84). The study reveals that strong models often produce hallucinated citations even when answers are correct, and robust reasoning often comes at the cost of verifiable attribution.

## Background & Motivation

**Background**: MLLMs are increasingly used for multi-step reasoning and long-form responses in real-world tasks (video QA, medical reports, educational demos). Reliable deployment requires "traceable" outputs—meaning every factual claim must map back to a specific modality and timestamp in the input. While work exists in text attribution (Gao 2023b) and video temporal grounding (Hendricks 2017, Lei 2021), these focus on simple observational or retrieval scenarios.

**Limitations of Prior Work**: (1) Existing evaluations typically measure only one modality or whole-video granularity, failing to distinguish between "observable claims" and "reasoning claims," which allows models to score high despite incorrect timestamps. (2) Real-world tasks require joint attribution across heterogeneous modalities (video + audio + charts) and fine-grained evaluation at the "atomic fact" level. (3) Mainstream "generate-then-attribute" pipelines often sacrifice reasoning quality for citation accuracy.

**Key Challenge**: There is a disconnect between internal latent reasoning processes and verifiable surface citations in MLLMs—longer chain-of-thought often makes final citations harder to track, while strict citation requirements can stifle complex reasoning capabilities.

**Goal**: (1) Construct a fine-grained multimodal attribution benchmark that distinguishes "observation vs. reasoning." (2) Propose an automated evaluator with high human alignment to make large-scale benchmarking affordable. (3) Systematically characterize the relationships between reasoning effort, model scale, attribution strategies, and final attribution quality.

**Key Insight**: Processes responses across three layers—requiring citations only for observable claims, decomposing sentences into atomic facts for precision/recall evaluation, and explicitly distinguishing modalities and timestamps. This decouples "reasoning quality" from "citation quality" during evaluation, exposing the trade-offs between them.

**Core Idea**: Reconstruct verifiable multimodal attribution evaluation as a three-stage pipeline (sentence screening → atomic fact decomposition + citation propagation → set-based precision/recall entailment verification), utilizing an MLLM-as-judge calibrated to human judgment.

## Method

### Overall Architecture
Task: Given multimodal input $I$ (video/audio/charts) and a question $Q$, an MLLM generates a response $R=\{r_i\}$. For each verifiable sentence $r_i$, the model must provide a citation set $C_i = \{c_i^j\}$, where each $c_i^j$ specifies a modality and timestamp (e.g., `(audio, 0:42-0:46)`). Evaluation proceeds in three steps: (1) **Verifiable Claim Identification**—Use an LLM verifier to determine if $r_i$ is observable, filtering out reasoning sentences. (2) **Atomic Fact Decomposition**—Decompose verifiable sentences into atomic fact sets $A_i = \{a_i^1, \ldots, a_i^n\}$ and resolve pronouns via decontextualization; propagate sentence-level $C_i$ to each atomic fact. (3) **Attribution Quality**—Perform bidirectional entailment verification for each $(a_i^j, C_i)$ to calculate recall (whether the joint citations support the fact) and precision (whether each citation is strictly necessary).

### Key Designs

1.  **Verifiable Claim Identification: Separating Observation from Reasoning**:
    - **Function**: Avoids forcing citations on reasoning sentences and prevents models from gaining unfair scores by omitting citations where they are not applicable.
    - **Mechanism**: An LLM verifier determines if each sentence $r_i$ can be directly observed from $I$, resulting in $R_v = \{r_i \in R \mid \text{Verifier}(r_i, I) = \text{True}\}$. For example, "The video explicitly defines thrust as forward (audio 0:42-0:46, vision 0:45)" is a verifiable claim, while "Therefore, this statement is incorrect" is a reasoning sentence to be discarded. Subsequent evaluation is performed only on the set of cited verifiable sentences $R_{vc} = \{r_i \in R_v \mid C_i \neq \emptyset\}$.
    - **Design Motivation**: Traditional attribution evaluations treat all sentences equally, either forcing models to hallucinate citations for reasoning steps or penalizing them for "unattributed" reasoning. This filtering step ensures precision/recall are calculated only where citations are expected.

2.  **Atomic Fact Decomposition + Citation Propagation + Decontextualization**:
    - **Function**: Eliminates ambiguity in complex sentences containing multiple facts, allowing for evaluation at the finest granularity.
    - **Mechanism**: For each $r_i \in R_{vc}$, an LLM decomposer breaks it into atomic facts $\{a_i^1, \ldots, a_i^n\}$, where each is a "minimal, independently verifiable" claim. Decontextualization resolves pronouns to specific entities. Sentence-level citations $C_i$ are then mapped to all derived atomic facts, forming pairs $\{(a_i^j, C_i)\}$.
    - **Design Motivation**: Sentence-level evaluation fails for compound sentences that are partially correct. Propagation retains the original citation context without requiring MLLMs to cite at an impractical atomic level during generation.

3.  **Set-based Bidirectional Entailment + MURGAT-SCORE Calibration**:
    - **Function**: Covers both citation sufficiency (recall) and necessity (precision) while selecting the most human-aligned MLLM judge.
    - **Mechanism**: For each $(a_i^j, C_i)$, an MLLM determines if $C_i$ jointly entails $a_i^j$ (recall). If it does, each $c_i^k$ is tested for necessity (precision, via leave-one-out testing). The MURGAT-S metric integrates coverage ($|R_{vc}|/|R_v|$) with precision, recall, and F1. The authors collected human annotations on WorldSense and Video-MMMU to calibrate various MLLM judges (e.g., Gemini, Qwen), selecting an optimal combination with Pearson $r=0.84$.
    - **Design Motivation**: Bidirectional verification prevents "citation stuffing" to inflate recall. Modality+timestamp alignment is the core differentiator of multimodal attribution.

### Loss & Training
This work focuses on the evaluation protocol rather than training a specific model. MURGAT-SCORE serves as the metric. An inference-time method for decoupling reasoning and attribution (reasoning followed by independent citation extraction) was tested, showing a $+9.6$ MURGAT-S gain at the cost of answer accuracy.

## Key Experimental Results

### Main Results
Evaluated on WorldSense and Video-MMMU across multiple MLLMs.

| Model | QA Accuracy | MURGAT-S | Observations |
|------|----------|----------|------|
| Gemini-3-Pro | High | High | Large model + more thinking → more accurate citations |
| Gemini-2.5-Flash | Medium | Medium | Correct answers but citations often wrong or missing |
| Qwen3-Omni-Instruct | Medium | Lower | Average citation quality for instruction-only version |
| Qwen3-Omni-Thinking | Slight Incr. | Lowered | Small model + more thinking → messier citations |
| Decoupled "Reason → Attribute" | Slight Decr. | +9.6 | Systematic trade-off observed |

### Ablation Study

| Configuration | Key Observation | Explanation |
|------|----------|----------|
| W/O Verifiable Claim ID | Reasoning sentences penalized | Distorted precision/recall |
| W/O Atomic Decomposition | Unfair for compound sentences | Partially correct sentences get inaccurate scores |
| W/O Leave-one-out | Precision becomes useless | Models game scores via redundant citations |
| Judge using GPT-4o-mini | $r=0.59$ | Significantly worse than the optimal combo |
| Judge using Gemini-3-Pro + Calib. | $r=0.84$ | Final setting for MURGAT-S |

### Key Findings
- **"Reasoning Tax"**: Requiring citations in simple tasks reduces QA accuracy, but in complex reasoning tasks, citations can act as a scaffold, forcing the model to decompose the reasoning chain.
- **Scale and Effort Interaction**: Gemini-3-Pro sees MURGAT-S gains with higher thinking budgets, while smaller models like Qwen3-Omni-Thinking may deviate further, suggesting a disconnect between latent reasoning and surface citation.
- **Hallucinated Grounding**: Even when QA is correct, citation error rates remain high, indicating that "knowing the answer" and "knowing where the evidence is" are distinct capabilities in MLLMs.

## Highlights & Insights
- **Explicit Separation of "Verifiable vs. Reasoning"**: Redefining attribution evaluation to target only sentences that *should* be verified establishes a new paradigm for multimodal research.
- **Atomic Facts + Modality/Timestamp Citations**: Extending the FActScore concept to multimodal data with leave-one-out verification ensures robustness against redundant citations.
- **High Human Alignment of MURGAT-SCORE**: The $r=0.84$ correlation makes large-scale automated evaluation credible and provides a transferable method for multi-judge calibration.

## Limitations & Future Work
- Dependency on LLM verifiers/decomposers/judges may introduce inherent biases; cross-domain generalization requires further validation.
- The dataset focuses on WorldSense and Video-MMMU; scalability to other modality combinations (e.g., medical imaging + charts) remains to be tested.
- No training-side solution was proposed for balancing reasoning and citation accuracy beyond inference-time decoupling.
- Citation granularity (timestamp accuracy) depends on the precision of human segmentation, which may introduce noise for facts with fuzzy boundaries.

## Related Work & Insights
- **vs. MCiteBench / MAVIS**: These focus on image-level VQA or document-level evidence; MURGAT enforces dual modality+timestamp labels and includes audio/charts.
- **vs. MIRAGE**: MIRAGE uses VLM verification for multimodal RAG; MURGAT provides a finer protocol by distinguishing verifiable claims and using timestamp-level granularity.
- **vs. Temporal Grounding**: Traditional grounding assumes target segments are specified in the prompt; MURGAT requires the model to select evidence autonomously during reasoning.
- **vs. FActScore (Min 2023)**: Directly inspired by atomic decomposition but extended to multimodal bidirectional set verification.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Explicit verifiable/reasoning separation + multimodal timestamp-level attribution creates a complete loop.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers multiple strong models and reasoning effort scans, though limited to two primary datasets.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Figure 1 clearly presents the protocol; definitions and examples are lucid.
- **Value**: ⭐⭐⭐⭐⭐ Provides critical infrastructure for verifiability research in trustworthy MLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PSA-MF: Personality-Sentiment Aligned Multi-Level Fusion for Multimodal Sentiment Analysis](../../AAAI2026/audio_speech/psa-mf_personality-sentiment_aligned_multi-level_fusion_for_multimodal_sentiment.md)
- [\[ACL 2026\] When Misinformation Speaks and Converses: Rethinking Fact-Checking in Audio Platforms](../../ACL2026/audio_speech/when_misinformation_speaks_and_converses_rethinking_fact-checking_in_audio_platf.md)
- [\[ICML 2026\] JAEGER: Joint 3D Audio-Visual Grounding and Reasoning in Simulated Physical Environments](jaeger_joint_3d_audio-visual_grounding_and_reasoning_in_simulated_physical_envir.md)
- [\[ICML 2026\] The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning](the_silent_thought_modeling_internal_cognition_in_full-duplex_spoken_dialogue_mo.md)
- [\[ICML 2026\] Multimodal Fusion via Self-Consistent Task-Gradient Fields](multimodal_fusion_via_self-consistent_task-gradient_fields.md)

</div>

<!-- RELATED:END -->
