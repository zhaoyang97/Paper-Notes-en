---
title: >-
  [Paper Note] Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs
description: >-
  [ACL 2026][Audio & Speech][disfluency correction] The authors propose a multilingual disfluency correction pipeline: first, MuRIL is used for token-level fluent/disfluent labeling…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "disfluency correction"
  - "contrastive loss"
  - "MuRIL"
  - "instruction tuning"
  - "Hindi/Bengali/Marathi"
date: 2026-05-08
content_hash: 38825139c27a373c
---

# Mind the Pause: Disfluency-Aware Objective Tuning for Multilingual Speech Correction with LLMs

**Conference**: ACL 2026  
**arXiv**: [2605.12242](https://arxiv.org/abs/2605.12242)  
**Code**: https://github.com/deepak-kumar-98/Mind-the-Pause (Available)  
**Area**: Speech / Multilingual NLP / Indic Languages / LLM fine-tuning  
**Keywords**: disfluency correction, contrastive loss, MuRIL, instruction tuning, Hindi/Bengali/Marathi

## TL;DR
The authors propose a multilingual disfluency correction pipeline: first, MuRIL is used for token-level fluent/disfluent labeling, then the "original transcript + token labels" are fed into Llama-3.2-3B / Qwen2.5-3B for instruction fine-tuning. The key innovation is an **anti-disfluency contrastive loss term** that explicitly penalizes the probability of generating disfluent tokens ($-\log(1-\sum_v w_v P_\theta(v))$). On real Hindi/Bengali/Marathi ASR data, this approach achieves a +1.97 BLEU gain over non-contrastive baselines and +8.54 BLEU over mBART. The 3B models match or surpass GPT-4o in most settings.

## Background & Motivation

**Background**: Spontaneous speech almost inevitably contains disfluencies (fillers like "uh/um", repetitions, false starts, self-repairs), which ASR systems do not automatically remove. The authors' tests show that Whisper v3 Large and AI4Bharat Indic Conformer leave at least one disfluency in approximately 30% of sentences in real Indic language dialogues. This noise causes drops of 0.5–1.6 points in QA (on a 5-point scale), 2–4.7 BLEU in MT, and ~2 points in TTS naturalness MOS.

**Limitations of Prior Work**: (i) Traditional pipelines use "detect-then-delete," where sequence taggers like MuRIL mark disfluent tokens for direct removal, leading to grammatical fractures and semantic incompleteness. (ii) Research on Indic languages (Hindi/Bengali/Marathi) has mostly stopped at the detection stage (Bhat 2023, Kundu 2022), lacking full-sentence correction solutions. (iii) Existing LLM-based work either uses LLMs as data generators to train small taggers (Cheng 2024) or prompts GPT-4 directly to delete disfluencies (Lima & Campelo 2024 for Portuguese), lacking an integrated end-to-end correction pipeline combining token-level detection and LLM rewriting.

**Key Challenge**: Cross-entropy (CE) fine-tuning provides LLMs with a positive signal to "look like the fluent reference," but **lacks a mechanism to explicitly instruct the model "not to copy disfluent tokens."** The authors observed that even when MuRIL tags are provided, models trained only with cross-entropy occasionally copy fillers into the output. Thus, positive-only supervision is insufficient.

**Goal**: (a) Perform end-to-end disfluency correction for Hindi/Bengali/Marathi rather than just detection; (b) Design a training objective that directly suppresses the generation probability of disfluent tokens to fill the gap in cross-entropy; (c) Verify if 3B-scale open-source LLMs with this training strategy can rival GPT-4o / Gemini 2.5 Pro.

**Key Insight**: Treat the output of the token-level detector as a "negative sample indicator" for contrastive learning. Since MuRIL already identifies disfluent tokens, pulling down their generation probability provides targeted negative supervision.

**Core Idea**: CE loss learns "what should be generated" (push), while contrastive loss learns "what should not be generated" (pull). These push-pull signals collaborate to separate fluent targets from disfluent tokens in the representation space.

## Method

### Overall Architecture
A two-stage process with dual losses. **Stage 1**: MuRIL (Multilingual BERT pre-trained on 17 Indic languages) performs token-level binary classification (0=fluent, 1=disfluent), fine-tuned on a merged dataset of three languages. **Stage 2**: The "instruction + disfluent sentence + MuRIL predicted token labels" are concatenated in Alpaca-style and fed to the LLM (Llama-3.2-3B-Instruct or Qwen2.5-3B-Instruct). The target output is the fluent reference. Training Objective = CE loss + $\lambda$ × Contrastive loss. During inference, the LLM directly generates the fluent transcript using the same input format.

### Key Designs

1.  **MuRIL token-tagging as Auxiliary Supervision**:
    *   **Function**: Downgrades disfluency detection from "deciding which tokens to delete" to "providing a hint to the LLM," avoiding grammatical damage caused by hard deletion.
    *   **Mechanism**: MuRIL achieves a token-level F1 of 0.987 on manually edited data but only ~85% sentence-level accuracy (falling to 33–63% on real data). The authors allow the LLM to "reference but not blindly trust" tags. The input $x_i$ includes the instruction, disfluent sentence, and labels. The CE loss is $L_{CE} = -\sum_i \sum_t \log P_\theta(y^t_i \mid y^{<t}_i, x_i)$.
    *   **Design Motivation**: Detection-only methods fail because they separate recognition from rewriting, losing grammatical context. Providing tags to the LLM allows it to use its language modeling capabilities to decide whether a disfluent token should be deleted or rewritten. Imperfact MuRIL sentence-level accuracy actually aids robustness training.

2.  **Anti-Disfluency Contrastive Loss**:
    *   **Function**: Explicitly penalizes the LLM for assigning probability mass to identified disfluent tokens at generation step $t$.
    *   **Mechanism**: For sample $i$, the disfluent token set $D_i$ is pre-calculated. At step $t$, the disfluent probability mass is $s_{i,t} = \sum_{v \in D_i} w_v P_\theta(v \mid y^{<t}_i, x_i)$, where $w_v \in (0, 1]$ follows a geometric decay weight (1, 0.5, 0.25, ...) based on subword position. The contrastive loss is $L_{\text{contrastive}} = \frac{1}{N}\sum_i \frac{1}{T_i} \sum_{t=r_i}^{T_i} -\log(1 - s_{i,t})$, where $r_i$ is the response start. The final loss is $L_{\text{total}} = L_{CE} + \lambda \cdot L_{\text{contrastive}}$ with a warm-up schedule for $\lambda$.
    *   **Design Motivation**: Unlike traditional representation-level InfoNCE, this is a **hard constraint at the token-distribution level**. $-\log(1-s_{i,t})$ explodes as $s \to 1$, providing a strong back-gradient if the model favors disfluent tokens. Geometric decay weights target the most identifiable first subword after BPE tokenization. The warm-up ensures CE establishes basic generation capability before the contrastive penalty begins.

3.  **Merged Multilingual Instruction Tuning**:
    *   **Function**: Handles Hindi/Bengali/Marathi with a single model using instructions for fluent rewriting.
    *   **Mechanism**: 120k parallel disfluent-fluent pairs (40k per language). Instructions follow the Alpaca format. Zero-shot cross-lingual transfer experiments show that migrating from Hindi-only to Bengali maintains 87.1 BLEU, demonstrating strong shared representations.
    *   **Design Motivation**: High lexical/syntactic similarity among Indic languages allows one checkpoint to cover three languages, reducing deployment costs.

### Loss & Training
$L_{\text{total}} = L_{CE} + \lambda \cdot L_{\text{contrastive}}$; $\lambda$ uses a warm-up schedule. Geometric decay weights $w_v$ for disfluent subwords follow $1, 0.5, 0.25, \dots$. Backbones used: Llama-3.2-3B-Instruct and Qwen2.5-3B-Instruct.

## Key Experimental Results

### Main Results
Performance of Llama-3.2-3B-Instruct on real ASR data (BLEU / chrF2 / TER):

| Language | Data | mBART | Multilingual Instruction FT | w/o Contrastive | **With Contrastive** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hindi | Real | 71.4 / 85.5 / 15.1 | 64.8 / 81.7 / 23.4 | 87.4 / 93.3 / 9.2 | **90.4 / 95.6 / 5.8** |
| Bengali | Real | 73.5 / 87.9 / 13.0 | 69.6 / 89.0 / 21.6 | 70.7 / 90.5 / 20.8 | **74.4 / 93.8 / 17.9** |
| Marathi | Real | 82.6 / 93.1 / 8.2 | 80.0 / 94.3 / 11.8 | 83.2 / 95.5 / 9.3 | **83.6 / 96.6 / 9.2** |

Qwen2.5-3B-Instruct showed even higher gains (Hindi real: 91.1 BLEU vs 84.2 w/o contrastive, +6.9). Average gains: +4.68 BLEU / +2.37 chrF2 / −3.22 TER compared to the non-contrastive baseline.

### Ablation Study
Average gains attributed to contrastive loss on Llama-3.2-3B-Instruct:

| Configuration | ΔBLEU | ΔchrF2 | ΔTER |
| :--- | :--- | :--- | :--- |
| Multilingual instruction FT (no MuRIL tags) | baseline | baseline | baseline |
| + MuRIL tag conditioning (w/o contrastive) | +6.16 | — | — |
| **+ MuRIL tag + Contrastive loss (Ours)** | **+1.97 over above** | +1.53 | −1.65 |
| **Total vs mBART** | **+8.54** | — | — |

LLM-as-Judge results (using Qwen2.5-3B to avoid self-preference, bidirectional pairwise):

| Language | Data | Proposed Win | Parallel FT Win | Draw |
| :--- | :--- | :--- | :--- | :--- |
| Hindi | Real | 28.0% | 9.3% | 62.7% |
| Marathi | Real | 30.0% | 8.0% | 62.0% |
| Bengali | Real | 18.0% | 27.0% | 55.0% |

### Key Findings
*   **Contrastive loss benefits Qwen significantly more than Llama**: Qwen improved by 4.68 BLEU vs Llama's 1.97. This is attributed to Qwen's superior multilingual grounding; contrastive loss is most effective for models that "understand the language but occasionally slip."
*   **3B models rival GPT-4o**: They match or exceed GPT-4o in 4 out of 6 evaluation conditions and beat Gemini 2.5 Pro across all three languages.
*   **Significant cross-lingual zero-shot transfer**: Models fine-tuned on one language keep BLEU scores in the mid-60s to low-80s on others, proving strong representation transfer in the Indic family.
*   **Disfluencies significantly harm downstream tasks**: In QA, LLaMA Hindi dropped from 1.70 to 1.18; in MT (Hindi→Bengali), BLEU dropped 3.9.

## Highlights & Insights
*   **"Negative sample indicators" is a portable concept**: This hard-constraint contrastive loss can be applied to any task where an external tagger identifies negatives—such as hallucination suppression, toxicity removal, or deprecated API usage in code generation.
*   **Geometric weights for BPE is a clever detail**: Assigning higher weights to the first subword concentrates the penalty where the disfluency signal is strongest.
*   **Small models beating GPT-4o**: This provides strong evidence for industrial deployment that task-specific contrastive training is a viable alternative to scaling and prompting.
*   **Rigorous LLM-as-Judge**: Using a different model as judge (Qwen to judge Llama/Qwen) and bidirectional pairwise testing minimizes self-preference and position bias.

## Limitations & Future Work
*   Experiments were limited to 3B models; the effect at 70B+ scales remains unverified.
*   The dataset relies on one public parallel dataset (Kundu 2022) and synthetic rules, potentially missing complex real-world disfluencies like code-mixing or accent-specific issues.
*   The gain from contrastive loss versus traditional instruction tuning is not fully disentangled from potential data distribution factors in Bengali.
*   The $\lambda$ parameter and warm-up schedule require empirical tuning with no theoretical guiding framework presented yet.

## Related Work & Insights
*   **vs Bhat et al. 2023a**: Bhat uses detection-only + hard deletion; Ours uses detection + LLM rewriting + contrastive suppression, raising BLEU from the 60s to the 90s.
*   **vs Smooth-LLaMa (2025)**: Smooth-LLaMa uses end-to-end audio-to-text; Ours is ASR-agnostic and modular, making it easier to deploy but losing raw audio cues.
*   **vs Lima & Campelo (2024)**: They find GPT-4 zero-shot sufficient for Portuguese; Ours proves zero-shot performance is poor for Indic languages, necessitating task-specific training.

## Rating
*   Novelty: ⭐⭐⭐⭐ The anti-disfluency contrastive loss is a novel design for token-level suppression.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across models, languages, and downstream tasks.
*   Writing Quality: ⭐⭐⭐⭐ Formulas and logic are clear, though some failure modes in Bengali could be explored further.
*   Value: ⭐⭐⭐⭐ High utility for Indic ASR applications with a portable contrastive learning idea.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)
- [\[ACL 2026\] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)
- [\[AAAI 2026\] A Mind Cannot Be Smeared Across Time](../../AAAI2026/audio_speech/a_mind_cannot_be_smeared_across_time.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition](pseudo2real_task_arithmetic_for_pseudo-label_correction_in_automatic_speech_reco.md)
- [\[ACL 2026\] SEPT: Semantically Expanded Prompt Tuning for Audio-Language Models](generalizable_prompt_tuning_for_audio-language_models_via_semantic_expansion.md)
- [\[ACL 2026\] From Flat Language Labels to Typological Priors: Structured Language Conditioning for Multilingual Speech-to-Speech Translation](from_flat_language_labels_to_typological_priors_structured_language_conditioning.md)
- [\[NeurIPS 2025\] EuroSpeech: A Multilingual Speech Corpus](../../NeurIPS2025/audio_speech/eurospeech_a_multilingual_speech_corpus.md)
- [\[AAAI 2026\] A Mind Cannot Be Smeared Across Time](../../AAAI2026/audio_speech/a_mind_cannot_be_smeared_across_time.md)

</div>

<!-- RELATED:END -->
