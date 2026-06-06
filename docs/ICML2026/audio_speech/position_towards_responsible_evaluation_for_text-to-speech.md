---
title: >-
  [Paper Note] Position: Towards Responsible Evaluation for Text-to-Speech
description: >-
  [ICML 2026][Audio & Speech][TTS Evaluation] This is a position paper proposing that TTS evaluation should evolve from "technical metrics only" to a three-layer hierarchical **Responsible Evaluation** framework—Fidelity &…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "TTS Evaluation"
  - "Responsible AI"
  - "MOS"
  - "Fairness"
  - "Traceability"
date: 2026-05-08
content_hash: 70b8344f473059ff
---

# Position: Towards Responsible Evaluation for Text-to-Speech

**Conference**: ICML 2026  
**arXiv**: [2510.06927](https://arxiv.org/abs/2510.06927)  
**Code**: None (Position paper)  
**Area**: Speech Synthesis / TTS Evaluation / Responsible AI  
**Keywords**: TTS Evaluation, Responsible AI, MOS, Fairness, Traceability

## TL;DR
This is a position paper proposing that TTS evaluation should evolve from "technical metrics only" to a three-layer hierarchical **Responsible Evaluation** framework—Fidelity & Accuracy, Comparability & Standardization, and Governance-Fairness-Security. It systematically diagnoses the failure modes of current metrics like WER, SIM, MOS, and RTF, and provides 13 actionable recommendations.

## Background & Motivation
**Background**: Recent TTS systems such as VALL-E, NaturalSpeech 3, F5-TTS, CosyVoice, and Qwen3-TTS can synthesize high-fidelity speech that is "nearly indistinguishable from real humans," entering a new stage based on diffusion models and foundation models (LLM-based TTS).

**Limitations of Prior Work**: Evaluation methods have not kept pace. Mainstream evaluation still revolves around technical dimensions like naturalness, intelligibility, speaker similarity, and efficiency, relying on the "four-piece set" of WER + SIM + MOS + RTF. The authors point out that this metric system is **failing simultaneously**: WER is contaminated by ASR errors, SIM saturates after a threshold, MOS has reached a ceiling where it cannot distinguish between strong models, and RTF reporting is inconsistent and incomparable.

**Key Challenge**: The exponential growth in technical capabilities versus the stagnation of evaluation protocols. This asymmetry creates two types of problems: (1) figures between papers are actually incomparable, leading to self-deception in the field; (2) evaluation completely ignores social dimensions—whether training data sources are legal, whether different accents/groups are treated fairly, and whether synthesized speech can be traced—which are precisely the ethical risks (telecom fraud, voice identity theft, deepfake forensics) triggered by high-fidelity TTS.

**Goal**: To expand the "object" of TTS evaluation from the model itself to the entire pipeline of **Model + Data + Deployment Consequences**; simultaneously providing a progressive evaluation framework to offer an actionable path for academia and industry.

**Key Insight**: Instead of inventing new metrics or benchmarks, the authors adopt a "position paper + diagnostic review" approach—deconstructing each commonly used metric to explain "what is broken, why it is broken, and how to fix it," then organizing all diagnoses using a three-layer pyramid.

**Core Idea**: In short, "evaluation must be split into three layers; if the lower layer is unreliable, the upper layer is meaningless"—only when metrics truthfully reflect model capabilities (Level 1) can cross-system comparability (Level 2) be discussed, and finally, governance and security (Level 3) can be addressed.

## Method
As a position paper, there is no "model pipeline." The method section corresponds to the **Responsible Evaluation Three-Layer Framework** and the specific diagnoses and suggestions under each layer. Three "Key Designs" are used below to represent these layers.

### Overall Architecture
The authors reconstruct TTS evaluation into a **pyramid**:

- **Level 1 — Fidelity & Accuracy**: Does a single metric truly reflect model capability? This is the foundation; if it is unstable, all upper layers fail.
- **Level 2 — Comparability, Standardization, Transferability**: Can numbers be compared directly across papers, systems, and years?
- **Level 3 — Governance, Fairness, Security**: Evaluation must incorporate data legality, fairness across populations, and the security and traceability of synthesized speech.

Input: The current set of TTS evaluation practices (metric definitions, dataset usage, reporting habits, ethical gaps).
Output: 13 prioritized actionable recommendations, plus a checklist of failure modes for each commonly used metric.

### Key Designs

1.  **Level 1: Itemized Diagnosis of Metric Failures**:
    - **Function**: Deconstructs WER, SIM, Predicted MOS, $F_0$ RMSE, and subjective MOS one by one to explain "under what conditions each metric is no longer credible."
    - **Mechanism**: The authors summarize two fundamental flaws of objective metrics: first, the relationship between metric values and human perception is **non-linear or even non-monotonic**, meaning score improvements do not equal perceptual improvements (typical example: SIM has a saturated interval $\text{SIM} > \tau$ where further increases contribute almost nothing to perception); second, neural-based metrics inherit **their own biases and uncertainties** (e.g., DNSMOS is trained on speech enhancement data but misused for synthesized speech; ASR model errors contaminate WER). For subjective MOS, the "ceiling effect" is noted—as synthesis quality approaches human levels, the 5-point MOS lacks resolution, and more discriminative protocols like the audio Turing test should be used. Another ignored dimension is **insufficient coverage**: there are few specialized benchmarks for real-world scenarios like reading mathematical formulas, long-form synthesis, emotional expression, or punctuation sensitivity.
    - **Design Motivation**: To expose "foundational" problems first, without which the next two layers cannot be discussed. This diagnosis itself has independent value—even if readers do not accept the framework, they can correct their experimental designs using this "metric blacklist."

2.  **Level 2: Systemic Failure of Comparability**:
    - **Function**: Reveals that even when everyone uses LibriSpeech test-clean + WER + SIM + MOS, numbers between papers are almost incomparable.
    - **Mechanism**: The authors provide specific counter-examples—VALL-E evaluates on 1234 samples of LibriSpeech test-clean, while NaturalSpeech 3 and MaskGCT use only a 40-item subset, and F5-TTS uses 1127 samples with punctuation and casing; E2-TTS redefined the VALL-E Continuation task (using the last 3s instead of the first 3s as a prompt); SIM is split into SIM-o/SIM-r and papers vary on whether to include the prompt segment; MOS reports generally do not follow ITU-T P.808, often lacking scale definitions, rater calibration, or playback conditions; RTF often omits whether it includes the vocoder, the batch size, or if it is streaming. The authors advocate for distinguishing between **comparable** and **incomparable** results and encourage "human-aligned automatic metrics" like LLM-as-a-Judge as transferable alternatives—so new papers do not need to re-run MOS for all baselines.
    - **Design Motivation**: This layer reveals an embarrassing fact—a large part of "SOTA improvements" in TTS comes from differences in evaluation protocols rather than real capabilities. Highlighting this provides momentum for standardization.

3.  **Level 3: Integrating Governance, Fairness, and Security into Evaluation**:
    - **Function**: Expands the evaluation object from "model output" to "training data source + cross-population performance + post-deployment traceability."
    - **Mechanism**: Diagnosis and recommendations are provided for three sub-problems. **Governance**—technical reports often use "in-house data" vaguely; the authors demand disclosure of sources, licenses, and consent terms; voice is biometric data, and unauthorized use leads to legal risks. **Fairness**—aggregate scores hide the degradation of quality for minority groups (specific accents, dialects, vulnerable populations); ASR-based WER misjudges compliant pronunciations of minority languages as errors, and ASV-based SIM similarly inherits baseline model bias. The authors suggest **group-disaggregated reporting** and bias audits for evaluators. **Security**—high-fidelity TTS is already used for fraud and bypassing voice authentication, but standard evaluations lack a spoofing/deepfake detection column; the authors advocate for including **traceability** (e.g., imperceptible watermarking) as a standard metric to answer "is this speech synthesized, and by whose model."
    - **Design Motivation**: As TTS enters large-scale deployment, purely technical metrics are insufficient to measure whether a system "should be released." This layer instantiates general responsible AI principles into actionable TTS evaluation items.

### Loss & Training
Not applicable (position paper, no training process). The authors provide 13 actionable recommendations organized by the three layers.

## Key Experimental Results

As a position paper, there are no experiments, but the authors use two sets of "illustrative measurements" to support their arguments.

### Main Results: How Evaluation Dataset Size Distorts WER (Core conclusion from Appendix A)

| Evaluation Protocol | Number of Samples | Impact on WER |
|---------------------|-------------------|---------------|
| VALL-E Original     | LibriSpeech test-clean, 1234 items | Baseline |
| NaturalSpeech 3 / MaskGCT | Only 40-item subset | Large variance at small samples; scores not comparable to baseline |
| F5-TTS              | 1127 items + preserved punctuation/casing | Text normalization differences further contaminate WER |

Key implication: The same "LibriSpeech test-clean WER" is completely different across papers, making cross-paper WER comparisons statistically invalid.

### Key Findings
- **SIM Threshold Saturation**: When SIM exceeds a certain empirical threshold, further increases contribute almost nothing to the perceived speaker similarity (evidence from Wester et al., 2016 is cited as key support).
- **MOS Ceiling**: Modern TTS generally approaches human recordings in MOS; the 5-point scale can hardly distinguish between strong models, requiring a switch to audio Turing tests or CMOS.
- **Optimizing WER Backfires on Prosody**: Using WER as an RL reward signal causes models to collapse prosodic variance into monotonic output—the metric improves, but naturalness decreases. This is the most dramatic case of metric failure.
- **DNSMOS Cross-Domain Misuse**: DNSMOS is trained on speech enhancement data but is widely used to evaluate synthesized speech; this is a classic example of "metrics inheriting training distribution bias."
- **Demographic Bias in ASR/ASV Evaluators**: WER based on pre-trained ASR and SIM based on ECAPA-TDNN have been independently proven to perform worse on minority groups (Koenecke et al., 2020; Hutiri & Ding, 2022); this bias is systematically inherited by TTS evaluation.

## Highlights & Insights
- **The "Three-Layer Pyramid" is the most transferable structure of this paper**: Level 1 unreliable $\rightarrow$ Level 2 meaningless $\rightarrow$ Level 3 impossible. This hierarchical structure can be directly transferred to evaluation reviews of other generative tasks (image/video/code generation) by replacing specific metrics.
- **The "Metric Blacklist" is highly valuable**: Every item among WER, SIM, Predicted MOS, $F_0$ RMSE, and MOS is specifically dissected for "when it is untrustworthy," making it immediately useful for writing experiments or acting as a reviewer.
- **Using "metric optimization backfiring on perceived quality" as a case study** (WER as RL reward collapsing prosody) is highly persuasive, debunking the myth that "higher metrics mean a better model."
- **Including data legality in evaluation** is a long-neglected blind spot. Elevating vague "in-house data" descriptions to problems of both reproducibility and legal risk has significant cautionary value, especially for industry.
- **Traceability (watermarking) as an evaluation dimension** is a rare but forward-looking proposal—evaluating a TTS system not just on how well it synthesizes, but on whether the synthesized speech can subsequently be detected and attributed.

## Limitations & Future Work
- **Acknowledged Limitations**: In the "Alternative Views" section, the authors admit that for low-resource languages and domains, strict data governance might inhibit research—a real tension between Level 3 and technical progress.
- **Lack of Actionable Implementation Details**: The 13 recommendations are somewhat principled; for instance, how specifically to build "representation-aware benchmarks," who should build them, and who pays for them is not addressed. This is a common limitation of position papers.
- **No New Unified Metric Proposed**: The authors are mostly "deconstructing"—pointing out failure modes of current metrics without providing a unified new protocol to replace MOS. LLM-as-a-Judge is mentioned but not expanded upon as the core contribution.
- **Boundaries of the Three-Layer Framework**: For example, should "hallucinatory reading" (errors in math formulas) belong to Level 1 or a new evaluation dimension? It is placed in "underexplored dimensions" in Level 1 but overlaps with "evaluation coverage" in Level 2.
- **Possible Extensions**: Turning the three-layer framework into an **evaluation self-check checklist** (similar to an ML reproducibility checklist) and making it mandatory for TTS paper submissions—this would be the true path to implementing the position.

## Related Work & Insights
- **vs. EmergentTTS-Eval (Manku et al., 2025)**: EmergentTTS-Eval built a new benchmark covering emails, phone numbers, URLs, STEM formulas, etc. This paper cites it as a positive example of "underexplored dimensions" in Level 1 and advocates for more such multi-domain benchmarks. They are complementary—one builds the dataset, the other the evaluation philosophy.
- **vs. ITU-T P.808 / King 2014 (MOS Standardization)**: P.808 already provides detailed listening test protocols for MOS, but this paper points out that the vast majority of papers do not follow them. This paper essentially calls for "please actually execute it" and elevates non-compliance to a systemic Level 2 problem.
- **vs. audio Turing test (Wang et al., 2025f)**: Wang et al. proposed the audio Turing test to alleviate MOS saturation; this paper cites it as a recommended protocol for Level 1 within the broader context of upgrading subjective evaluation.
- **vs. LLM-as-a-Judge Paradigm (Wang et al., 2025e/d/2026a; Zhang et al., 2025b)**: This paper views LLM-as-a-Judge as a potential path for Level 2 "transferable metrics"—interpretable, reproducible, and removing the need for every new paper to redo subjective tests.
- **vs. General Responsible AI Principles (e.g., sociotechnical framing by Selbst et al., 2019)**: This paper instantiates general responsible AI principles into actionable evaluation items specifically for the TTS domain.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Not a new metric or benchmark, but a new framework; the hierarchical pyramid is a rare and clear organization for such position papers.
- **Experimental Thoroughness**: ⭐⭐⭐ Position papers do not require heavy experiments, but the use of specific cases like LibriSpeech sample size variance and WER as an RL reward backfiring is sufficient to support the arguments.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The structure is very clear, with each layer organized by "Challenge $\rightarrow$ Sub-problem $\rightarrow$ Recommendation," making it highly readable and friendly to reviewers and industry practitioners.
- **Value**: ⭐⭐⭐⭐⭐ The TTS community's current protocols are indeed failing systematically; this paper provides an "evaluation self-check list" that is immediately useful, regardless of whether one accepts the entire three-layer framework.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](position_text_embeddings_should_capture_implicit_semantics_not_just_surface_mean.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](../../AAAI2026/audio_speech/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ICML 2026\] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)
- [\[ACL 2026\] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](../../ACL2026/audio_speech/speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)

</div>

<!-- RELATED:END -->
