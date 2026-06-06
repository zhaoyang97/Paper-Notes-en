---
title: >-
  [Paper Note] SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation
description: >-
  [ACL 2026][Audio & Speech][Speech Quality Assessment] This paper extends speech quality assessment from "assigning a score" to "interpretable speech judging" by constructing the SpeechEval dataset—containing 32…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Speech Quality Assessment"
  - "Multi-task Evaluation"
  - "SpeechEval"
  - "CoT Reasoning"
  - "GRPO"
date: 2026-05-08
content_hash: c6f9451d1958e279
---

# SpeechLLM-as-Judges: Towards General and Interpretable Speech Quality Evaluation

**Conference**: ACL 2026  
**arXiv**: [2510.14664](https://arxiv.org/abs/2510.14664)  
**Code**: https://github.com/NKU-HLT/SpeechLLM-as-Judges  
**Area**: Speech Quality Assessment / Multi-modal LLMs / LLM Safety  
**Keywords**: Speech Quality Assessment, Multi-task Evaluation, SpeechEval, CoT Reasoning, GRPO

## TL;DR
This paper extends speech quality assessment from "assigning a score" to "interpretable speech judging" by constructing the SpeechEval dataset—containing 32,207 multi-lingual audios and 128,754 annotations. By employing CoT instruction fine-tuning and GRPO, the authors trained SQ-LLM, which outperforms existing speech LLMs and expert models across four task categories: quality scoring, pairwise comparison, improvement suggestions, and deepfake detection.

## Background & Motivation
**Background**: Generative speech systems now cover scenarios such as TTS, speech translation, voice dialogue, and singing voice generation. Evaluation typically relies on objective metrics like MOS, AB preference tests, MCD/STOI, or specialized quality predictors like MOSNet, UTMOS, and Audiobox Aesthetics.

**Limitations of Prior Work**: Most existing methods provide only scalar scores or binary classifications, making it difficult to explain "why this audio is poor." They also struggle to simultaneously handle real-world evaluation needs across multiple languages, sources, and tasks. For speech system developers, a 3.7 MOS score does not indicate whether they should fix pronunciation clarity, dynamic range, emotional expression, or distorted segments.

**Key Challenge**: Speech quality is inherently multi-dimensional, subjective, and task-dependent, yet existing evaluation protocols tend to compress it into a single number. Conversely, while general-purpose speech LLMs possess multi-modal input capabilities, they lack fine-grained quality supervision, leading to weak alignment with human judgment when used directly as judges.

**Goal**: The authors aim to establish a unified framework where a model can listen to audio, understand task instructions, provide quality judgments, explain the underlying reasons, compare two samples, offer improvement suggestions, and provide reliable classification results in deepfake detection.

**Key Insight**: A critical observation is that speech quality assessment is not merely a signal processing problem but can also be modeled as a multi-modal judging task with structured reasoning. Given annotations covering multiple tasks, languages, and fine-grained dimensions, speech LLMs can learn judgment processes similar to those of human reviewers.

**Core Idea**: Use SpeechEval to provide multi-dimensional quality supervision, then transform Qwen2.5-Omni into an interpretable LLM judge for speech quality through CoT instruction fine-tuning and preference-based reward optimization (GRPO).

## Method
The methodology consists of two parts: constructing the SpeechEval dataset to support "judging-style output" and training SQ-LLM based on this dataset. The approach does not create a traditional MOS predictor but reformulates evaluation tasks into natural language instructions to output quality dimensions, rationales, suggestions, or detection conclusions.

### Overall Architecture
The input can be a single audio, a pair of audios, or an audio for authenticity detection; text input consists of task instructions. The model extracts acoustic representations using a speech encoder, which are then fed alongside task prompts into a speech-aware language decoder to generate structured natural language answers. Four task types share the same model and instructional interface: Single-sample quality assessment, double-sample quality comparison, quality improvement suggestions, and deepfake detection.

### Key Designs
1.  **SpeechEval Multi-task Dataset**:
    - **Function**: Provides supervision data covering multiple languages, quality dimensions, and task formats for speech LLMs.
    - **Mechanism**: The dataset includes 32,207 unique audios and 128,754 annotations across Chinese, English, Japanese, and French. The annotation protocol decomposes quality into three high-level aspects (Overall Quality, Production Quality, Content Enjoyment) and 8 sub-dimensions (Intelligibility, Distortion, Rate, Dynamic Range, Emotional Impact, Artistic Expression, Subjective Experience, etc.). It also collects distortion types, emotion types, speaker gender, and open-ended descriptions.
    - **Design Motivation**: Traditional datasets usually cover only single languages or limited scalar scores. SpeechEval uses natural language explanations and structured labels for joint supervision, enabling the model to learn the "reasons behind the scores."

2.  **Dimension-level CoT Instruction Fine-tuning**:
    - **Function**: Forces the model to explicitly consider multiple quality dimensions before providing a final answer, improving interpretability and consistency.
    - **Mechanism**: SQ-LLM is based on Qwen2.5-Omni-7B, with the speech encoder frozen and the LLM fine-tuned using LoRA. The training objective includes both intermediate predictions for 8 quality dimensions and the final answer, summarized as $L=\lambda\sum_i L_{dim}^{(i)}+L_{ans}$, where $\lambda=0.3$.
    - **Design Motivation**: Without explicit dimension supervision, models often generate plausible but inconsistent explanations. Dimension-level CoT injects structural information from human annotations into the training process, aligning judging rationales with human logic.

3.  **Reward Optimization via GRPO**:
    - **Function**: Further aligns model outputs with human preferences for "helpful, relevant, accurate, and detailed" answers.
    - **Mechanism**: A frozen Qwen3 acts as an automatic evaluator to score Assessment, Comparison, and Suggestion outputs across four dimensions: Helpfulness, Relevance, Accuracy, and Detail. These are aggregated into a reward using weights of 1, 1, 2, and 0.5. For detection tasks, a reward based on label correctness is used.
    - **Design Motivation**: While SFT teaches format and basic capabilities, open-ended suggestion tasks require higher quality. GRPO allows the model to improve generation quality based on multi-dimensional feedback without collecting additional preference pairs.

### Loss & Training
SQ-LLM training involves two stages. Stage one is instruction fine-tuning with dimension-level CoT (8 epochs, batch size 4, learning rate 1e-4). Stage two is GRPO (batch size 1, 4 candidates sampled per prompt, LoRA learning rate 1e-6). Total training cost is approximately 43 A100 GPU hours (12 for SFT, 31 for GRPO).

## Key Experimental Results

### Main Results

| Task / Metric | Strongest Direct SpeechLLM | Strongest Trained Baseline | SQ-LLM | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| Quality Assessment LScore | MiDashengLM 5.536 | Qwen2.5-7B + AES-E 6.533 | 6.833 | SQ-LLM is highest |
| Quality Comparison LScore | Qwen2-Audio 4.591 | FT Qwen2-Audio 5.648 | 6.434 | SQ-LLM leads significantly |
| Suggestions SBERT / FENSE | MiDashengLM 0.600 / 0.490 | FT Qwen2-Audio 0.708 / 0.708 | 0.735 / 0.735 | Suggestions are more consistent |
| Overall Quality PCC | Audiobox Aesthetics 0.464 | Qwen2.5-7B + AES-E 0.457 | 0.520 | Best alignment with human scores |
| Comparison Overall ACC | UTMOS 0.741 | FT Qwen2-Audio 0.587 | 0.751 | Exceeds expert and LLM baselines |
| Deepfake EER / ACC | MiDashengLM ACC 67.480 | FT Qwen2-Audio 8.593 / 89.312 | 6.249 / 89.358 | Lowest EER, slightly better ACC |

### Ablation Study

| Configuration | SQA LScore ↑ | SQC LScore ↑ | SQI LScore ↑ | DSD EER ↓ | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| SQ-LLM Full | 6.833 | 6.434 | 7.420 | 6.249 | CoT + GRPO |
| w/o GRPO | 6.804 | 6.420 | 7.018 | 6.264 | Suggestion task drops most |
| Weakened CoT/Reward | 6.657 | 6.316 | 6.733 | 8.574 | Detection robustness drops |

### Key Findings
- Direct calls to multi-modal speech LLMs do not reliably perform speech quality judging; Qwen2-Audio, Qwen2.5-Omni, and MiDashengLM lag significantly behind the specialized SQ-LLM in PCC, ACC, and generation quality.
- CoT contributions are mainly seen in structured judgment and deepfake detection robustness; GRPO provides the greatest gain in improvement suggestion tasks, where open-ended answers benefit from preference-based feedback.
- The dataset itself is a key asset: SpeechEval unifies quality scoring, comparison, suggestions, and deepfake detection into interpretable natural language outputs.
- Expert models remain competitive in specific single metrics (e.g., UTMOS for comparison accuracy), but they lack natural language explanations and multi-task versatility.
- The improvement suggestion task reveals a new use case: evaluation results can be converted into actionable development feedback rather than just scores.

## Highlights & Insights
- The paper transforms speech quality assessment from "predicting MOS" to "LLM-as-Judge." This paradigm aligns better with actual development workflows where developers need to know specific issues and how to fix them.
- SpeechEval’s annotation design is practical: structured labels provide trainable intermediate supervision, while natural language explanations ensure human-readable outputs.
- Using GRPO with a single frozen LLM evaluator for multi-dimensional rewards offers a scalable post-training path for speech quality assessment, even without human preference pairs.

## Limitations & Future Work
- SpeechEval currently covers 4 languages and 4 fixed task types. Coverage for low-resource languages, code-switching, emotional expression, and speaker consistency is still insufficient.
- Although deepfake detection achieves the lowest EER, performance is not balanced across languages; detection for Chinese and French is relatively weaker, suggesting domain shifts in multilingual forgery artifacts.
- The automatic reward model depends on Qwen3; reward quality may be affected by the evaluator's bias. Future work could include human preference calibration or multi-evaluator ensembles.
- Training requires multi-modal LLMs and A100 resources, posing cost challenges for small teams.
- The paper primarily reports offline benchmark results and has not yet demonstrated closed-loop evaluation gains in real-world TTS/dialogue system iterations.

## Related Work & Insights
- **vs MOSNet / UTMOS / Audiobox Aesthetics**: These expert models excel at single-quality prediction, whereas SQ-LLM covers scoring, comparison, suggestions, and detection with higher interpretability but greater training/inference costs.
- **vs QualiSpeech / ALLD-dataset**: Existing natural language quality data focuses on English or few tasks. SpeechEval expands this to multiple languages and 4 task types with complete dimension annotations.
- **vs General Speech LLMs**: Qwen2-Audio and MiDashengLM have auditory input capabilities but lack specialized supervision for quality judging. This work demonstrates that "hearing speech" is not identical to "evaluating speech quality."
- **Insight**: Multi-modal evaluation tasks can be constructed via the combination of "structured dimensions + natural language rationales + preference optimization." Similar logic could be applied to image, video, and robot trajectory quality assessment.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically introduces LLM-as-Judge to speech evaluation with a large-scale multi-task dataset.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers four tasks and multiple languages against various baselines; more real-world deployment experiments would be beneficial.
- Writing Quality: ⭐⭐⭐⭐☆ Complete structure; tables are dense but the logic is clear.
- Value: ⭐⭐⭐⭐⭐ Direct reference value for automatic evaluation, quality diagnosis, and safety detection in speech generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] UniSRM: A Unified Speech Reward Model for Fine-Grained Speech Evaluation](unisrm_a_unified_speech_reward_model_for_reasoning-based_fine-grained_assessment.md)
- [\[ICML 2026\] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech](../../ICML2026/audio_speech/sparse_autoencoders_for_interpretable_emotion_control_in_text-to-speech.md)
- [\[ICML 2026\] Position: Towards Responsible Evaluation for Text-to-Speech](../../ICML2026/audio_speech/position_towards_responsible_evaluation_for_text-to-speech.md)
- [\[ACL 2026\] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models](mtr-duplexbench_towards_a_comprehensive_evaluation_of_multi-round_conversations_.md)
- [\[ACL 2026\] Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner](full-duplex-bench-v2_a_multi-turn_evaluation_framework_for_duplex_dialogue_syste.md)

</div>

<!-- RELATED:END -->
