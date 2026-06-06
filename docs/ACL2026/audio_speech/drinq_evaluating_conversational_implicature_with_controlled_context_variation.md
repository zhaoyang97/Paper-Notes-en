---
title: >-
  [Paper Note] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation
description: >-
  [ACL 2026][Audio & Speech][Conversational Implicature] DRInQ constructs a conversational implicature evaluation set by fixing the surface form of questions and systematically varying the context. It finds that while LLMs…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "Conversational Implicature"
  - "Pragmatic Reasoning"
  - "Context Control"
  - "Speech Acts"
  - "LLM Evaluation"
date: 2026-05-08
content_hash: f19c7a6debeb255b
---

# DRInQ: Evaluating Conversational Implicature with Controlled Context Variation

**Conference**: ACL 2026  
**arXiv**: [2605.24267](https://arxiv.org/abs/2605.24267)  
**Code**: https://github.com/hjarai/drinq  
**Area**: Pragmatic Reasoning / LLM Evaluation  
**Keywords**: Conversational Implicature, Pragmatic Reasoning, Context Control, Speech Acts, LLM Evaluation

## TL;DR
DRInQ constructs a conversational implicature evaluation set by fixing the surface form of questions and systematically varying the context. It finds that while LLMs can generate seemingly plausible pragmatic scenarios, they often over-interpret context during reasoning and fall below human judgment consistency.

## Background & Motivation
**Background**: Human dialogue relies heavily on conversational implicature—meanings not explicitly stated but triggered by context, politeness principles, social relationships, and common knowledge. Existing LLMs are already strong in surface semantics, social common sense, and fluent dialogue, yet they remain unstable regarding "what exactly is implied by this sentence in this specific scenario."

**Limitations of Prior Work**: Existing pragmatic benchmarks often use coarse-grained labels, such as literal/non-literal, or focus on well-defined phenomena like irony, metaphor, presupposition, and scalar implicature. These resources struggle to isolate variations where "the same question yields different meanings due to different contexts," making it difficult to determine whether model errors stem from misunderstanding the question, ignoring the context, or over-extending contextual details.

**Key Challenge**: Conversational implicature depends on context, yet the context must not explicitly state the answer. Data construction must simultaneously satisfy three conditions: the context provides enough support for a unique interpretation, distractors are plausible in other contexts, and the varying factors are pragmatically relevant rather than random paraphrasing. This makes large-scale human construction very costly.

**Goal**: The authors propose DRInQ, a multiple-choice task using question-context-interpretation to evaluate whether models can recover the implied intent of a question utterance from context. It also compares model-generated data with human-authored data to analyze the differing capabilities of LLMs in pragmatic scenario construction versus pragmatic reasoning.

**Key Insight**: The paper focuses on question utterances because question forms frequently serve non-literal functions, such as requests, reproaches, invitations, comfort, or sarcasm. The authors use speech acts as intent labels and organize context variations into controllable dimensions rather than relying solely on free generation.

**Core Idea**: By keeping the question $Q$ fixed and only varying the context $C$, each candidate interpretation is designed to be a plausible meaning of the same question in a different context. This specifically tests whether the model can calibrate "which specific implied meaning the context actually supports."

## Method
The key to DRInQ is not asking models to answer commonsense questions, but controlling pragmatic variables. Each sample contains a question, a context, and 5 candidate implied comments, only one of which is supported by the current context. The other options are rational interpretations of the same question in different contexts. Thus, the model cannot rely on the question itself or the surface wording of the options but must judge the strength of the contextual evidence.

### Overall Architecture
The data construction pipeline begins with 30 hand-written everyday questions, expanded to 300 base questions using GPT-4o. For each question, the authors first obtain a default intent and a default implied comment, then select semantically distant intents from 23 speech act intent labels to generate multiple context-interpretation pairs. These pairs are converted into multiple-choice tasks and verified by Prolific annotators. Samples with at least 4/5 agreement are retained, from which 400 difficult samples are extracted as a benchmark.

### Key Designs
1. **Minimal Contrast Task with Fixed Question and Varying Context**:
    - **Function**: Isolates the impact of context variation on conversational implicature.
    - **Mechanism**: Each instance is represented as $(Q,C,A)$, where $Q$ is a fixed question, $C$ is the current context, and $A$ consists of 5 candidate interpretations. Incorrect options are not random distractors but implied comments that could hold true for the same question in other contexts.
    - **Design Motivation**: If standard multiple-choice questions were used, models might rely on option saliency or question templates. Fixing $Q$ while ensuring all candidates remain pragmatically plausible more closely mirrors real-world pragmatic disambiguation.

2. **Organizing Pragmatic Variation via Speech Act Intents**:
    - **Function**: Ensures data generation covers diverse communicative functions like requests, reproaches, invitations, warnings, thanks, and complaints.
    - **Mechanism**: The authors extract four major categories (Directive, Assertive, Commissive, Expressive) from Searle's speech act theory, totaling 23 representative act verbs. For each question, they rank the embedding distance between the default implied comment and other intents, then select intents with large semantic differences to generate new contexts.
    - **Design Motivation**: Variations in conversational implicature are not arbitrary paraphrases but reflect speakers performing different communicative acts. Intent labels make the generation process more controllable and easier to cover fine-grained pragmatic functions.

3. **Dual Assessment via Human Verification and Model Reasoning**:
    - **Function**: Ensures the data is not merely a model's internal loop while observing systematic differences between LLM and human judgment.
    - **Mechanism**: 62 pre-screened Prolific annotators participated in verification, retaining 819 samples with at least 4/5 agreement and constructing a 400-sample hard subset. Evaluation covers 12 SOTA models, comparing vanilla few-shot with explanation prompting. An additional human-authoring study compared the quality of contexts generated by 16 humans versus GPT-4o.
    - **Design Motivation**: Pragmatic meaning is naturally ambiguous; a single gold label can be overly deterministic. Human consistency filters unreliable samples and exposes the asymmetry between a model's ability to "generate a scenario" and "identify a meaning."

### Loss & Training
The paper does not train new models, focusing instead on data generation, human verification, and prompting evaluation. During the generation phase, GPT-4o is used to produce context-interpretation pairs, with an instruction to abstain on unreasonable question-intent combinations. Reasoning evaluation uses few-shot prompts: the vanilla condition provides 3 in-context examples, while the explanation condition requires the model to provide a brief rationale before selecting an answer. Subsequent prompt interventions include conservative, charitable, reasoning, and all, designed to suppress over-inference and malicious intent attribution.

## Key Experimental Results

### Main Results
| Dataset / Setting | Metric | Ours | Comparison | Description |
|--------|------|------|----------|------|
| DRInQ Construction | Base questions / intents | 300 / 23 | 30 hand-written seed questions | Each question linked to at least 5 different intents |
| Human Verification | Retained samples | 819 | 4/5 annotator agreement | Forms the validated pool |
| Benchmark | Hard subset | 400 | Low model consistency/disputed samples | Used for main model evaluation |
| Hard subset | Human Avg | 0.88 ± 0.10 | SOTA LLM ~0.56-0.67 | Humans maintain a significant lead |
| Hard subset | Best Model | OpenAI-o3: 0.67 ± 0.02/0.03 | GPT-4o: 0.62/0.63 | Explanation yields limited gains for large models |

### Ablation Study
| Configuration | Key Metric | Description |
|------|---------|------|
| GPT-5-Nano prompting | 41% -> 73% | Structured prompts help small models most |
| GPT-5-Mini prompting | 71% -> 81% | Reasoning scaffold narrow gap with strong models |
| GPT-4o prompting | ~82% ceiling | Large models are less sensitive to prompt intervention |
| LLM Gen vs Human Writing | LLM novelty 37%, human novelty 22%, tie 40% | LLMs generate more novel scenarios; humans are more conservative |
| Human consensus vs generated label | Standard sampling 81%, validated overall 67%, hard baseline 27% | Hard samples specifically retain model/human disagreements |

### Key Findings
- The primary errors of LLMs are not a complete lack of semantic understanding but rather poorly calibrated inference strength. They often amplify negative details in the context into malicious intent or treat a possible interpretation as the only one.
- Human annotators tend toward more charitable interpretations unless the context explicitly supports malice or reproach; models are more likely to select overly strong or negative options.
- Prompt intervention is effective for small models, suggesting some errors can be mitigated by reasoning process constraints. However, gains for strong models are limited, indicating that pragmatic calibration is more than just a prompt formatting issue.
- Regarding data generation, LLM-generated scenarios exhibit more variation and novelty, but they sometimes make implied comments too explicit or exceed the support of the context. Human contexts are safer and more predictable but can be underspecified.

## Highlights & Insights
- The task design is clever: by fixing the same question and making the context the sole source of variation, it locates whether a model truly understands the scenario more clearly than standard pragmatic multiple-choice tests.
- The paper translates the abstract linguistic concept of "conversational implicature" into a scalable generation pipeline. Speech act labels serve as practical tools for controlling data diversity rather than theoretical ornaments.
- The "Generation-Inference asymmetry" is a critical observation. The fact that a model can generate a plausible pragmatic scenario does not mean it can recover the appropriate meaning in someone else's scenario like a human.
- Error analysis provides insights for safety evaluation: model over-attribution of malice or hidden intents could impact scenarios requiring careful intent understanding, such as customer service, psychological support, or content moderation.

## Limitations & Future Work
- The multiple-choice format serves only as a diagnostic proxy and cannot fully measure real-world dialogue capability. True pragmatic understanding should involve the ability to generate appropriate responses, ask follow-ups, or maintain uncertainty.
- Fixed candidate interpretations may overshadow other plausible meanings. Even with 4/5 agreement, it does not mean the remaining interpretations are strictly incorrect.
- The data is English-centric and reflects the cultural and linguistic backgrounds of Prolific annotators. As conversational implicature is highly dependent on cultural norms, generalization to low-context/high-context cultures or non-English communities is limited.
- Intent-conditioned generation places GPT-4o in the data production chain, which may introduce model-specific stylistic biases. Future work could introduce stronger uncertainty modeling, open-ended generation evaluation, and cross-cultural annotation.

## Related Work & Insights
- **vs IMPRES / GRICE**: These datasets also focus on implicature or presupposition but are more oriented toward linguistic phenomena and rule control; DRInQ emphasizes fine-grained pragmatic differences under context variation for the same question.
- **vs FLUTE**: FLUTE covers irony, metaphor, and idioms; DRInQ focuses on the communicative functions of interrogative sentences, making the task closer to indirect expressions in daily conversation.
- **vs Social Commonsense Benchmarks**: Social commonsense tasks often ask "what happens next" or "how does the character feel," whereas DRInQ requires judging what communicative act the speaker is performing through a question.
- **Insights for LLM Eval**: Future evaluations should look beyond whether a model can provide a plausible interpretation and instead check if it recognizes when interpretations are "insufficiently evidenced."

## Rating
- Novelty: ⭐⭐⭐⭐☆ The pragmatic evaluation design of fixing questions while controlling context is highly distinctive, and the speech act-based generation is practical.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes data validation, 12 models, prompt intervention, human-AI writing comparisons, and error analysis, though limited by the multiple-choice format.
- Writing Quality: ⭐⭐⭐⭐☆ Motivation and error patterns are clearly explained; some summary statistics and main table splits may require careful reading to align.
- Value: ⭐⭐⭐⭐☆ Directly valuable for pragmatic reasoning, LLM dialogue evaluation, and intent calibration in safety-critical scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)
- [\[ICML 2026\] Algorithmic Recourse of In-Context Learning for Tabular Data](../../ICML2026/audio_speech/algorithmic_recourse_of_in-context_learning_for_tabular_data.md)
- [\[ACL 2026\] S2S-Arena: Evaluating Paralinguistic Instruction Following in Speech-to-Speech Models](s2s-arena_evaluating_paralinguistic_instruction_following_in_speech-to-speech_mo.md)
- [\[AAAI 2026\] Hearing More with Less: Multi-Modal Retrieval-and-Selection Augmented Conversational LLM-Based ASR](../../AAAI2026/audio_speech/hearing_more_with_less_multi-modal_retrieval-and-selection_augmented_conversatio.md)

</div>

<!-- RELATED:END -->
