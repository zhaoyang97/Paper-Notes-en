---
title: >-
  [Paper Note] Position: Towards Responsible Evaluation for Text-to-Speech
description: >-
  [ICML 2026][Audio & Speech][TTS evaluation] This is a position paper proposing that TTS evaluation should evolve from "technical metrics only" to a three-layer hierarchical **Responsible Evaluation** framework—Fidelity & Accuracy, Comparability & Standardization, and Governance-Fairness-Security. It systematically diagnoses the failure modes of current metrics such as WER, SIM, MOS, and RTF, and provides 13 actionable recommendations.
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "TTS evaluation"
  - "Responsible AI"
  - "MOS"
  - "Fairness"
  - "Traceability"
date: 2026-05-08
content_hash: 41f7513444bb78fd
---

# Position: Towards Responsible Evaluation for Text-to-Speech

**Conference**: ICML 2026  
**arXiv**: [2510.06927](https://arxiv.org/abs/2510.06927)  
**Code**: None (Position paper)  
**Area**: Speech Synthesis / TTS Evaluation / Responsible AI  
**Keywords**: TTS evaluation, Responsible AI, MOS, Fairness, Traceability

## TL;DR
This is a position paper proposing that TTS evaluation should evolve from "technical metrics only" to a three-layer hierarchical **Responsible Evaluation** framework—Fidelity & Accuracy, Comparability & Standardization, and Governance-Fairness-Security. It systematically diagnoses the failure modes of current metrics such as WER, SIM, MOS, and RTF, and provides 13 actionable recommendations.

## Background & Motivation
**Background**: Recent TTS systems such as VALL-E, NaturalSpeech 3, F5-TTS, CosyVoice, and Qwen3-TTS can synthesize high-fidelity speech that is "nearly indistinguishable from human speech," marking a new phase based on diffusion models and Foundation Models (LLM-based TTS).

**Limitations of Prior Work**: Evaluation methods have failed to keep pace. Mainstream evaluation still centers on technical dimensions like naturalness, intelligibility, speaker similarity, and efficiency, primarily relying on the "four-piece set" of WER + SIM + MOS + RTF. The authors argue that this metric system is **failing simultaneously**: WER is contaminated by ASR errors, SIM reaches saturation after a certain threshold, MOS has hit a ceiling and cannot distinguish between strong models, and RTF reporting is too inconsistent to be comparable.

**Key Challenge**: The exponential growth in technical capabilities vs. the stagnation of evaluation protocols. This asymmetry creates two sets of issues: (1) figures across papers are not comparable, leading to self-deception in the field; (2) evaluation ignores social aspects—legitimacy of training data, fairness across different accents/groups, and traceability of synthesized speech—which are precisely the ethical risks (fraud, identity theft, deepfake forensics) triggered by high-fidelity TTS.

**Goal**: To extend the "object" of TTS evaluation from the model itself to the entire chain of **Model + Data + Deployment Consequences**, while providing a progressive evaluation framework to offer actionable paths for academia and industry.

**Key Insight**: Rather than inventing new metrics or benchmarks, this work adopts a "position paper + diagnostic review" approach—dissecting "what is broken, why, and how to fix it" for each common metric and organizing these diagnoses into a three-layer pyramid.

**Core Idea**: Evaluation must be divided into three layers; the upper layers are meaningless if the lower layers are unreliable. Only when metrics accurately reflect model capabilities (Level 1) can cross-system comparability (Level 2) be established, followed by governance and security (Level 3).

## Method
As a position paper, this work does not present a model pipeline. Instead, it reconstructs "how to do TTS evaluation correctly" into a three-layer hierarchical pyramid, exposing failure modes in current practices and offering 13 actionable recommendations. The key constraint of this pyramid is that **lower layers must be reliable for upper layers to be meaningful**.

### Overall Architecture
The input to the framework consists of current TTS evaluation practices—metric definitions, dataset usage, reporting habits, and neglected ethical gaps. The output is a hierarchically organized rectification list. The three layers are divided as follows: **Level 1 (Fidelity & Accuracy)** asks whether individual metrics truly reflect model capabilities; this is the foundation. **Level 2 (Comparability / Standardization / Transferability)** asks whether figures across papers, systems, and years can be compared directly. **Level 3 (Governance / Fairness / Security)** expands the evaluation object to training data sources, cross-group performance, and deployment consequences.

### Key Designs

**1. Level 1—Common metrics have boundaries where they are no longer credible:** The authors provide individual diagnoses for WER, SIM, Predicted MOS, $F_0$ RMSE, and subjective MOS. Two fundamental flaws in objective metrics are summarized. First, the relationship between metric values and human perception is **non-linear or even non-monotonic**; improved scores do not always equate to improved perception. For instance, when $\text{SIM}>\tau$ exceeds an empirical threshold, further increases provide no perceptible gain to human ears. Second, neural-model-based metrics **inherit their own biases and uncertainties**: DNSMOS was trained on speech enhancement data but is misused for synthesis; ASR errors directly contaminate WER. Regarding subjective MOS, a "ceiling effect" is noted where 5-point scales lack the resolution to distinguish between top-tier models, suggesting a transition to more discriminative protocols like the audio Turing test.

**2. Level 2—Metric figures are incomparable across papers even when using the same metrics:** This layer reveals that SOTA gains often stem from protocol differences rather than real capability. For example, while VALL-E used 1234 samples for "LibriSpeech test-clean WER," NaturalSpeech 3 and MaskGCT used a subset of only 40 samples, and F5-TTS used 1127 samples while preserving punctuation and case. Such differences in text normalization further contaminate WER. On SIM, papers vary on whether prompt segments are included in the similarity calculation. Furthermore, MOS reporting frequently fails to comply with ITU-T P.808, and RTF often omits details regarding vocoders, batch size, or streaming status. To address this, the authors advocate for distinguishing between **comparable** and **incomparable** results and encourage "LLM-as-a-Judge" as a transferable alternative.

**3. Level 3—Incorporating governance, fairness, and security into evaluation:** As TTS enters large-scale deployment, pure technical metrics are insufficient. **Governance** requires disclosing training data sources, licenses, and consent clauses to address legal risks. **Fairness** calls for **group-disaggregated reporting** to ensure that high overall scores do not mask quality degradation for minority groups (accents, dialects). **Security** emphasizes **traceability** (e.g., imperceptible watermarking) to identify whether audio is synthesized and which model generated it.

## Key Experimental Results

This position paper uses demonstrative measurements to support its arguments.

### Main Results: How evaluation protocol variations distort WER (Core conclusions from Appendix A)

| Evaluation Protocol | Evaluation Samples | Impact on WER |
| :--- | :--- | :--- |
| Original VALL-E | LibriSpeech test-clean, 1234 samples | Baseline |
| NaturalSpeech 3 / MaskGCT | Only 40-sample subset | Large variance in small samples; results incomparable to baseline |
| F5-TTS | 1127 samples + punctuation/case preserved | Text normalization differences further contaminate WER |

Key implication: The same "LibriSpeech test-clean WER" metric represents entirely different things across papers, making cross-paper comparisons statistically invalid.

### Key Findings
- **SIM Threshold Saturation**: Beyond an empirical threshold, further increases in SIM contribute almost nothing to perceptible speaker similarity.
- **MOS Ceiling**: Modern TTS MOS scores are so close to ground truth that 5-point scales cannot distinguish strong models; audio Turing tests or CMOS are required.
- **WER Optimization Backfiring**: Using WER as an RL reward signal causes prosody to collapse into monotonic output—improving the metric while decreasing naturalness.
- **DNSMOS Domain Misuse**: DNSMOS is widely misused for TTS evaluation despite being trained on speech enhancement data.
- **Evaluator Bias**: ASR/ASV-based metrics inherit the demographic biases of their pre-trained backbones, performing worse on minority groups.

## Highlights & Insights
- **The "Three-layer Pyramid" structure**: Level 1 unreliable $\rightarrow$ Level 2 meaningless $\rightarrow$ Level 3 impossible. This structure can be applied to other generative tasks like image or video generation.
- **"Metric Blacklist"**: Detailing exactly when WER / SIM / Predicted MOS / $F_0$ RMSE / MOS become untrustworthy is immediately useful for researchers and reviewers.
- **The Case of RL Reward Collapse**: Proving that optimizing WER can degrade prosody effectively challenges the myth that "higher metrics always mean better models."
- **Data Legality**: Raising "in-house data" issues to the level of reproducibility and legal risk provides a necessary warning to the industry.
- **Traceability as a Metric**: Incorporating watermarking and detection into standard evaluation is a forward-looking proposal for responsible AI.

## Limitations & Future Work
- **Acknowledged Limitations**: The authors admit that strict data governance may inhibit research in low-resource languages—a tension between Level 3 and technical progress.
- **Lack of Implementation Details**: The 13 recommendations are principled; specific details on how to build "representation-aware benchmarks" are not fully provided.
- **No Single Replacement Metric**: The paper is more "deconstructive," pointing out flaws without offering a single unified metric to replace MOS.
- **Framework Boundaries**: Some issues, such as formula reading errors, oscillate between Level 1 (fidelity) and Level 2 (coverage).
- **Future Extension**: Proposing a mandatory **evaluation checklist** (similar to ML reproducibility checklists) for TTS paper submissions.

## Related Work & Insights
- **vs EmergentTTS-Eval (2025)**: EmergentTTS-Eval provides benchmarks for specific domains; Ours incorporates this into Level 1 as "underexplored dimensions."
- **vs ITU-T P.808**: Ours calls for the actual enforcement of these neglected subjective testing standards.
- **vs audio Turing test (2025f)**: Ours recommends this protocol to mitigate MOS saturation within a broader hierarchical framework.
- **vs LLM-as-a-Judge (2025/2026)**: Ours views this paradigm as a potential path to achieving "transferable metrics" for Level 2.
- **vs General Responsible AI Principles**: Ours instantiates sociotechnical framing specifically for the TTS modality.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning](position_text_embeddings_should_capture_implicit_semantics_not_just_surface_mean.md)
- [\[AAAI 2026\] CCFQA: A Benchmark for Cross-Lingual and Cross-Modal Speech and Text Factuality Evaluation](../../AAAI2026/audio_speech/ccfqa_a_benchmark_for_cross-lingual_and_cross-modal_speech_and_text_factuality_e.md)
- [\[ICML 2026\] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)
- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](../../ACL2026/audio_speech/unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ACL 2026\] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation](../../ACL2026/audio_speech/speechllm-as-judges_towards_general_and_interpretable_speech_quality_evaluation.md)

</div>

<!-- RELATED:END -->
