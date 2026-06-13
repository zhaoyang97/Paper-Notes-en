---
title: >-
  [Paper Note] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge
description: >-
  [ACL2026][Multilingual & Machine Translation][activation steering] This SemEval system paper utilizes FLORES parallel corpora to extract language directions and injects a language steering vector into the residual stream…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "activation steering"
  - "cultural awareness"
  - "FLORES"
  - "BLEnD"
  - "SemEval"
date: 2026-05-08
content_hash: 1ff39bb660e363ac
---

# DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge

**Conference**: ACL2026  
**arXiv**: [2605.23069](https://arxiv.org/abs/2605.23069)  
**Code**: https://github.com/Yusser96/SemEval-2026-Track7  
**Area**: Multilingual Models / Cultural Knowledge Evaluation  
**Keywords**: activation steering, cultural awareness, FLORES, BLEnD, SemEval

## TL;DR
This SemEval system paper utilizes FLORES parallel corpora to extract language directions and injects a language steering vector into the residual stream of a multilingual LLM during inference. The system achieved matching official MCQ results of 86.96% accuracy (ranking 7th out of 17 teams), though post-hoc analysis reveals that gains are highly sensitive to layers, prompts, models, and locales.

## Background & Motivation
**Background**: Multilingual LLMs can process multiple languages fluently, but linguistic fluency does not equate to the reliability of cultural knowledge. Benchmarks such as BLEnD and SemEval-2026 Task 7 focus on whether models can answer questions specific to certain languages, regions, and cultural backgrounds rather than merely producing grammatically correct text.

**Limitations of Prior Work**: Many cultural knowledge gaps cannot be resolved through simple translation or general multilingual instruction tuning. A model may understand a language yet lack knowledge regarding the daily culture, food, festivals, social habits, or local common sense of a specific region. While fine-tuning can improve specific tasks, the SemEval shared task does not provide BLEnD training data, and fine-tuning is costly and prone to overfitting.

**Key Challenge**: Cultural knowledge and linguistic representations may overlap within the model's internal states, but how to leverage this overlap without updating parameters remains unclear. Activation steering provides a lightweight solution, but its stability in improving cultural reasoning across multiple languages, locales, prompts, and models requires verification.

**Goal**: The DFKI-MLT system aims to use language vectors for inference-time adaptation to participate in the SAQ and MCQ tracks of SemEval-2026 Task 7, while analyzing the actual gains and failure modes of steering on cultural topics.

**Key Insight**: The authors hypothesize that language identity forms stable directions within the residual stream, and accessing cultural knowledge partially depends on these language/region-related directions. Consequently, they extract language vectors from FLORES parallel sentences and add the target language vector to the residual stream of specific transformer layers during generation.

**Core Idea**: Instead of fine-tuning model parameters, the model's internal representations are "nudged" along the target language direction during inference, making it easier for the model to access corresponding linguistic and cultural contexts.

## Method
The system consists of three parts: task setup, language vector extraction, and inference-time steering. The tasks include Track 1 (SAQ) and Track 2 (MCQ). SAQ requires generating short answers in the input language, matched against an acceptable answer set; MCQ involves an English question with four regional cultural options, where the system must select the correct option corresponding to the target locale. Official metrics for both are accuracy.

### Overall Architecture
The authors map BLEnD language-region pairs to FLORES language/script identifiers. For each mappable language, the first 1,000 sentences from the FLORES dev set are tokenized and fed into a multilingual instruction-tuned LLM to collect post-normalization residual-stream activations at specified layers. Language vectors are constructed using DiffMean, defined as the difference between the mean activation of the target language and the mean activation of a reference set.

During inference, $\beta v_{lang}$ is added to the hidden states of a selected transformer layer, where $v_{lang}$ is the normalized language direction and $\beta$ is the steering strength. The final submission utilized Qwen2.5-72B-Instruct, Layer 26, $\beta=1$, and a cultural prompt. All tracks used greedy decoding (temperature=0) to reduce sampling noise when evaluating steering effects.

### Key Designs
1.  **FLORES DiffMean language vectors**:
    - **Function**: Constructs injectable internal directions for each target language.
    - **Mechanism**: Calculates the mean residual-stream activation for FLORES parallel sentences and takes the difference between the target language and a control language or set. Each mapped language uses 1,000 FLORES dev sentences without additional preprocessing, relying solely on the model's own tokenizer.
    - **Design Motivation**: Since FLORES consists of parallel multilingual data, content variance is controlled; thus, the mean difference is more likely to capture language identity directions rather than differences in topic or sentence content.

2.  **Inference-time activation steering**:
    - **Function**: Shifts the model's tendency to access cultural knowledge without fine-tuning parameters.
    - **Mechanism**: Applies additive intervention to the residual stream of a selected transformer layer, formulated as $h' = h + \beta v_{lang}$. During development, the team searched across $\beta \in \{1, 3, 5\}$ and candidate layers, ultimately selecting $\beta=1$ and Layer 26 for the final submission.
    - **Design Motivation**: Compared to full fine-tuning, steering is low-cost, allows for rapid switching between languages, and is better suited for shared task settings without training data.

3.  **Post-hoc sensitivity analysis of prompts, layers, and models**:
    - **Function**: Explains why a single steering configuration fails to provide stable global improvements.
    - **Mechanism**: After the official submission, the authors performed layer sweeps, prompt comparisons, and steering strength comparisons on Qwen2.5-72B/7B, Aya Expanse 8B/32B, and Qwen3 8B/32B. Analysis included generic vs. cultural prompts and random Gaussian vectors vs. language vectors.
    - **Design Motivation**: Official results only reflect a single locked configuration. Because the steering effect for cultural reasoning is highly localized, it is essential to examine the interactions between layers, prompts, locales, and models.

### Loss & Training
The system has no training loss as it does not update model parameters. The development strategy involved selecting models, layers, and $\beta$ based on the SemEval development phase. Candidate models included the Qwen2.5, Aya Expanse, and Qwen3 series. The official submission used Qwen2.5-72B-Instruct + Layer 26 + $\beta=1$. SAQ generation was capped at 32 tokens with lightweight normalization, while MCQ output the selected option.

## Key Experimental Results

### Main Results
| Track | Metric | DFKI-MLT | Rank | Description |
|-------|--------|----------|------|-------------|
| Track 1 (SAQ) | Acc. | N/A | - / 10 | Official submission file was incorrect/corrupted; not successfully evaluated. |
| Track 2 (MCQ) | Acc. | 86.96 | 7 / 17 | Official score using cultural prompt and activation steering. |
| Track 2 best system | Acc. | 96.78 | 1 / 17 | Best system, leading DFKI-MLT by 9.82 percentage points. |

### Ablation Study
| Locale | DFKI-MLT (%) | Ours Locale Rank | Prev. SOTA (%) | Gain |
|--------|--------------|------------------|----------------|------|
| es-EC  | 97.54        | 7                | 98.67          | -1.13|
| en-GB  | 96.12        | 6                | 99.17          | -3.05|
| es-MX  | 94.94        | 4                | 99.32          | -4.38|
| ar-EG  | 94.84        | 2                | 91.03          | +3.81|
| bg-BG  | 94.60        | 8                | 99.54          | -4.94|

### Key Findings
- The best performing locale does not represent global optimality. For `ar-EG`, DFKI-MLT reached 94.84%, which is 3.81 percentage points higher than the 91.03% of the overall leaderboard champion for the same locale, but it lagged by 4.94 points on `bg-BG`.
- The average gain from steering is small and unstable. The abstract notes that individual locales saw up to +1.5% absolute accuracy, but other configurations degraded performance, and gains did not generalize consistently across language-region pairs.
- Layer selection is highly sensitive. In the Qwen2.5-72B post-hoc sweep, the optimal layer for MCQ changed to Layer 2 or 3 depending on the prompt, while the optimal layer for SAQ shifted to Layer 8 or 7. The Layer 26 used in the official submission was a compromise based on the dev split.
- $\beta=1$ is a safe default. Larger steering strengths are prone to causing instability in earlier layers, though some Qwen3/Aya configurations tolerated stronger steering, suggesting strength shouldn't be determined solely by model scale.
- FLORES sample size is not a major source of instability. For the DiffMean vectors across all six models, the joint median cosine similarity was at least 0.99 at $N=100$ and 0.999 at $N=500$; the authors consider 1,000 sentences a conservative choice.

## Highlights & Insights
- The system paper honestly demonstrates the boundaries of activation steering. Instead of framing steering as a stable, universal method for cultural alignment, it points out that gains are highly dependent on locale, layer, and prompt.
- Using FLORES parallel corpora to extract language directions is lightweight. This method requires no cultural question training sets and no parameter updates, making it suitable for shared tasks and rapid prototyping.
- Random vector controls are essential. The authors found that random-vector effects concentrated around 0, whereas language-vector effects were more dispersed with negative outliers, indicating that language vectors are not merely random perturbations but do not guarantee positive gains.
- SAQ and MCQ have different prompt requirements. Cultural prompts might help option probabilities in MCQ but can lead to overly long or explanatory answers in SAQ, which hinders short-answer matching.

## Limitations & Future Work
- The absence of official SAQ results is the biggest experimental regret. Due to a submission file error, Track 1 only had post-hoc offline re-evaluation, which cannot be directly compared to the official leaderboard.
- Limited scope of comparison. While the paper analyzes layers, $\beta$, prompts, models, and locales, it does not systematically compare alternative methods like prompt-only, fine-tuning, CAA, ReFT, or SAE-based steering.
- A single global configuration is not suitable for all locales. The official submission used a single $(\beta, layer)$ pair, but post-hoc analysis shows optimal settings vary by language, region, task, and prompt. Future work should explore per-locale or per-prompt adaptive steering.
- Language vectors are only approximate proxies for cultural knowledge. Language and culture are related but not equivalent; many cultural differences exist at regional, ethnic, and social context levels that cannot be fully represented by FLORES language directions.

## Related Work & Insights
- **vs BLEnD / SemEval cultural awareness evaluation**: BLEnD provides the cultural knowledge evaluation framework; this paper is a participant system focusing on lightweight model adaptation without training data.
- **vs activation steering / CAA**: While general steering methods regulate model behavior via internal directions, this paper defines the directions as language identity directions and tests their transfer to cultural tasks.
- **vs fine-tuning**: Fine-tuning directly learns cultural knowledge but requires data and high costs. The proposed method is non-parametric and suitable for rapid experimentation during development, though it is less stable than an ideal task-specific fine-tuning solution.
- **vs prompt-only methods**: Prompts can explicitly request the model to consider cultural context, while steering acts on internal representations. The results suggest these should be optimized jointly rather than treated as alternatives.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Using language-vector steering for a cultural knowledge shared task is interesting, though the technology builds on existing activation steering concepts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ The official MCQ and extensive post-hoc sweeps are valuable, but the official SAQ gap and lack of strong baseline comparisons are missing.
- **Writing Quality**: ⭐⭐⭐⭐☆ The reporting on negative results and sensitivity is honest, and the system description is clear, though some appendix charts are somewhat fragmented.
- **Value**: ⭐⭐⭐⭐☆ Provides significant insights into multilingual cultural evaluation and inference-time intervention, particularly as a reminder not to equate linguistic fluency with cultural competence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Lingo_Research_Group at SemEval-2026 Task 9: Evaluating Prompt Variants for Polarization Detection](lingo_research_group_at_semeval-2026_task_9_evaluating_prompt_variants_for_polar.md)
- [\[ACL 2026\] Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection](multilingual_steering_by_design_multilingual_sparse_autoencoders_and_principled_.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[AAAI 2026\] Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering](../../AAAI2026/multilingual_mt/mitigating_content_effects_on_reasoning_in_language_models_through_fine-grained_.md)
- [\[ACL 2026\] SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](steereval_inference-time_interventions_strengthen_multilingual_generalization_in.md)

</div>

<!-- RELATED:END -->
