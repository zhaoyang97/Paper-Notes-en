---
title: >-
  [Paper Note] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings
description: >-
  [ACL 2025][Social Computing][Chinese toxicity detection] This work systematizes "mixed glyph, phonetic, and semantic perturbations" in Chinese toxic texts into 3 categories and 8 strategies, constructs a large-scale perturbation benchmark named CNTP, and demonstrates that current mainstream LLMs from both China and the US are significantly unstable under such Chinese multimodal toxicity detection. While few-shot ICL / SFT can raise the detection rate…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Chinese toxicity detection"
  - "multimodal perturbation"
  - "robustness evaluation"
  - "LLM safety"
  - "benchmark dataset"
date: 2026-05-08
content_hash: f8f2854594ee7931
---

# Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings

**Conference**: ACL 2025  
**arXiv**: [2505.24341](https://arxiv.org/abs/2505.24341)  
**Code**: [https://github.com/thomasyyyoung/ToxiBenchCN](https://github.com/thomasyyyoung/ToxiBenchCN)  
**Area**: Social Computing  
**Keywords**: Chinese toxicity detection, multimodal perturbation, robustness evaluation, LLM safety, benchmark dataset

## TL;DR

This work systematizes "mixed glyph, phonetic, and semantic perturbations" in Chinese toxic texts into 3 categories and 8 strategies, constructs a large-scale perturbation benchmark named CNTP, and demonstrates that current mainstream LLMs from both China and the US are significantly unstable under such Chinese multimodal toxicity detection. While few-shot ICL / SFT can raise the detection rate, they easily lead to false positives on benign content.

## Background & Motivation

Toxic content detection has become a fundamental problem in content moderation and model safety.

However, the Chinese scenario is much more complex than the English one.

The reason is not merely the scarcity of Chinese data, but that the Chinese language naturally possesses multimodal attributes.

An individual Chinese character consists of a glyph structure, a pinyin pronunciation, and carries semantics as well as metaphoric expressions established by internet culture.

This means attackers do not need complex adversarial optimization; they can easily bypass model detection simply by modifying several glyphs, homophones, pinyin initials, character orders, or even by inserting emojis.

The paper points out that existing research typically focuses on only a few evasion techniques, such as homophone or emoji substitution.

The limitations of this approach include:

1. An incomplete understanding of the Chinese perturbation space.
2. Inability to systematically evaluate which types of perturbations models are vulnerable to.
3. Difficulty in designing truly robust mitigation methods.

The authors therefore reformulate the core problem as:

Do LLMs truly understand perturbed Chinese toxic content, or do they merely memorize partial explicit patterns?

To address this question, this paper accomplishes three things:

1. Formulates a taxonomy for Chinese toxic perturbations.
2. Constructs a large-scale benchmark, CNTP, based on the taxonomy.
3. Systematically evaluates nine mainstream LLMs and analyzes whether low-cost enhancement strategies like ICL/SFT are truly effective.

## Method

### Overall Architecture

The methodology pipeline of this paper is highly clear:

1. Sample toxic and non-toxic texts from the Chinese toxicity dataset Toxi_CN.
2. Employ GPT-4o-mini for toxic entity extraction to locate the actual offensive spans in the sentences.
3. Rewrite the extracted toxic spans according to the eight proposed perturbation methods.
4. Control the perturbation ratio and conduct human validation to ensure readability and semantics remain understandable to native Chinese speakers.
5. Benchmark the detection capabilities of nine SOTA LLMs using the constructed CNTP.
6. Further investigate whether in-context learning and few-shot fine-tuning can mitigate the issues.

This is not a paper proposing a new model, but rather a "task definition + dataset + benchmark analysis" work.

Its value lies in transforming the robustness problem in Chinese toxicity detection from scattered case studies into a systematic evaluation problem.

### Key Designs

The authors propose three major categories and eight specific perturbation methods.

The first category consists of glyph-based visual perturbations.

1. VSim: Visually similar substitution.  
   Replace original characters with visually similar characters or characters with radicals removed.
2. Split: Character splitting.  
   Split a single Chinese character into two or more consecutive components.
3. Trad: Simplified and Traditional Chinese mixture.  
   Replace some simplified Chinese characters with traditional characters.

The second category consists of pronunciation-based phonetic perturbations.

4. PY_Init: Pinyin acronyms.
5. PY_Full: Full pinyin conversion.
6. Homo: Homonic substitution.

The third category consists of semantic-based flexible perturbations.

7. Shuff: Character shuffling.
8. Emoji: Substituting key words or morphemes with emojis.

The critical point of this taxonomy is not its simple enumeration but its emphasis on the multimodal nature of Chinese, where glyphic, phonetic, and semantic channels coexist simultaneously.

Consequently, Chinese toxicity detection cannot be treated purely as a standard text classification problem.

### Dataset Construction Pipeline

The construction process of CNTP consists of two steps.

The first step is base dataset sampling.

The authors sampled from Toxi_CN:

1. 2,533 toxic texts.
2. 2,696 non-toxic texts.

The second step is toxic entity extraction and perturbation embedding.

Specifically:

1. GPT-4o-mini is first utilized to extract the actually harmful toxic words or phrases from the original sentence.
2. Perturbations are then applied to these localized spans rather than randomly altering the entire sentence.
3. The average perturbation ratio is controlled to remain below 30%.

This design is crucial.

If the perturbation is too strong, neither models nor humans can understand it, rendering the evaluation meaningless.

If the perturbation is too weak, it fails to reflect the realistic difficulty of bypassing detection.

Thus, referencing settings from RoCBert, ToxiCloakCN, and Adversarial GLUE, the authors control the average perturbation rate between 0.27 and 0.29.

### Human Validation Mechanism

The paper does not rely solely on automated generation.

They introduce four native Chinese speakers as annotators for human validation.

The validation metrics include:

1. Extraction Accuracy: Whether the toxic entities extracted by GPT-4o-mini are accurate.
2. Human Readability: Whether the perturbed texts remain readable and understandable.

Results indicate that:

1. The accuracy rate of toxic entity extraction reaches 98.6%.
2. The average readability of perturbed text is 3.94 / 5.
3. A total of 20,087 perturbed toxic texts were finally obtained.

This indicates that the dataset is not a mere collection of garbled characters, but rather realistic evasion samples that can still be comprehended by Chinese users.

## Overall Architecture

Summarized from a task perspective, the overall framework of this paper can be illustrated as:

Original Chinese toxic text  
$\rightarrow$ Toxic span extraction  
$\rightarrow$ Localized multimodal perturbation injection  
$\rightarrow$ Human validation & filtering  
$\rightarrow$ Construction of CNTP benchmark  
$\rightarrow$ Evaluation of LLMs' toxic/non-toxic recognition ability  
$\rightarrow$ Analysis of prompt, ICL, and SFT impact on robustness  

Here, "multimodal" does not refer to vision-language fusion in the sense of image-text VLMs.

Instead, it refers to the multi-channel representation within Chinese text containing visual glyphs, pronunciation cues, and semantic cues.

This is the most notable conceptual transfer in this paper.

## Key Designs

### Design 1: Structuring Chinese Perturbations into an Interpretable Taxonomy

Many instances of toxic content bypassing were previously scattered empirical observations.

The authors' contribution lies in organizing them into a structured taxonomy.

This brings three direct benefits:

1. Model weaknesses can be analyzed statistically by category.
2. It allows for an analysis of which perturbations are most likely to deceive models.
3. Future defense mechanisms can be designed targeting specific categories rather than individual cases.

### Design 2: Localized Toxic Span Perturbation Instead of Whole-Sentence Perturbation

Directly altering the entire sentence can easily distort the text.

The authors first extract the toxic span and then manipulate it.

This keeps the evaluation closer to the realistic evasive writing used on social platforms.

### Design 3: Controlling Perturbation Ratio and Readability

The paper consistently emphasizes human readability.

This is because the real target of bypass attacks is "machine-unreadable, yet human-readable."

Hence, the average perturbation rate is capped around 28%, and samples with a readability score below 3 are filtered out.

### Design 4: Jointly Analyzing Detection Rate and F1

Evaluating solely the toxic detection rate is insufficient.

A model could easily maximize recall by classifying all sentences as toxic.

Therefore, the authors report F1 simultaneously, ensuring that false positives on non-toxic texts are taken into account.

### Design 5: Dedicated Analysis on "Over-Correction"

This aspect serves as the second core theme of the paper.

The authors discover that few-shot ICL or SFT causes models to learn a highly dangerous shortcut:

"Whenever strange Chinese variants are observed, tend to classify them as toxic."

This does not indicate a deeper understanding of Chinese, but rather heightened sensitivity and conservatism.

## Loss & Training

This paper does not propose a new training objective, so there is no complex mathematical formulation for loss functions.

The summary of the training/inference strategies consists of three components.

The first component is the benchmarking strategy.

1. Perform binary toxicity classification on nine LLMs.
2. Set the inference temperature to 0 to minimize randomness.
3. Employ both Chinese and English prompt templates to compare the impact of instructions.

The second component is ICL (In-Context Learning).

1. Include 10 examples for each perturbation category in the original prompt.
2. Examples contain the perturbed text, binary labels, and brief human rationales.

The third component is SFT (Supervised Fine-Tuning).

1. Fine-tune GPT-4o-mini using the OpenAI fine-tuning playground.
2. Set training sample sizes to 10, 20, and 40, respectively.
3. Training samples are drawn from a few perturbed cases in CNTP.

The paper does not detail underlying optimizers, learning rates, or epoch configurations.

Thus, from an analysis perspective, it can be understood as:

The authors focus on "whether few-shot enhancement can improve robustness" rather than designing new loss functions.

## Key Experimental Results

### Dataset Validation Results

| Metric | VSim | Split | Trad | PY_Init | PY_Full | Homo | Shuff | Emoji | Avg. |
|------|------|-------|------|---------|---------|------|-------|-------|------|
| Readability Score | 3.7 | 3.5 | 4.5 | 3.5 | 4.4 | 4.2 | 3.8 | 3.9 | 3.94 |
| Perturbation Ratio | 0.29 | 0.27 | 0.27 | 0.28 | 0.29 | 0.28 | 0.27 | 0.29 | 0.28 |

This table reveals two insights:

1. The dataset does not create difficulties by extremely degrading readability.
2. Trad and PY_Full present the highest readability, indicating almost zero barrier to human comprehension.

### Main Results Table

The following results represent performance under the Chinese prompts, which are most representative.

| Model | Base | VSim | Split | Trad | PY_Init | PY_Full | Homo | Shuff | Emoji | Avg. | F1 |
|------|------|------|-------|------|---------|---------|------|-------|-------|------|----|
| o3-mini | 91.78 | 70.10 | 67.68 | 67.31 | 92.08 | 57.09 | 80.72 | 48.56 | 76.35 | 70.98 | 0.65 |
| GPT-4o | 81.29 | 72.55 | 66.51 | 74.20 | 93.68 | 55.73 | 88.55 | 48.99 | 79.45 | 73.26 | 0.58 |
| GPT-4o-mini | 85.51 | 66.95 | 61.79 | 59.01 | 94.16 | 50.53 | 75.82 | 44.20 | 76.62 | 73.49 | 0.60 |
| DeepSeek-V3 | 83.05 | 59.53 | 59.59 | 56.00 | 82.35 | 41.68 | 74.45 | 38.95 | 63.81 | 59.42 | 0.59 |
| GLM-4-Air | 89.48 | 73.72 | 69.58 | 73.19 | 93.09 | 54.62 | 86.60 | 53.19 | 82.92 | 76.60 | 0.63 |
| Qwen-turbo | 90.63 | 85.63 | 85.86 | 83.04 | 94.86 | 79.11 | 93.96 | 68.10 | 89.93 | 90.20 | 0.64 |
| Qwen2.5-7B | 90.92 | 70.25 | 74.99 | 75.46 | 84.72 | 53.10 | 72.71 | 53.49 | 76.98 | 70.53 | 0.65 |
| Yi-1.5-9B | 90.58 | 78.86 | 77.10 | 79.05 | 91.13 | 69.16 | 82.08 | 60.65 | 82.39 | 89.34 | 0.65 |

Note that Avg. represents the mean detection rate across the 8 perturbation categories, excluding the base version.

The following trends are visible in this table:

1. The hardest categories typically involve subclasses of PY_Full, Shuff, Split, VSim, and Homo.
2. DeepSeek-V3's overall performance on Chinese perturbations is sub-optimal.
3. Qwen-turbo is among the most stable models overall.
4. All models perform significantly higher on the base scenario, indicating that the core vulnerability stems from perturbations rather than general toxic semantic understanding.

### ICL / SFT Analysis Table

| Strategy | Model | Split | PY_Init | Emoji | Benign Error Rate (ER) | Misinterpretation Rate (MR) |
|------|------|-------|---------|-------|------------------|----------|
| No ICL | DeepSeek-V3 | 56.00 | 41.68 | 59.42 | 2.24 | - |
| ICL | DeepSeek-V3 | 81.83 | 86.38 | 79.02 | 2.47 | Split 70.00 / PY_Init 67.67 / Emoji 46.67 |
| No ICL | GPT-4o-mini | 59.01 | 50.53 | 73.49 | 2.71 | - |
| ICL | GPT-4o-mini | 87.13 | 92.46 | 88.36 | 3.99 | Split 73.33 / PY_Init 60.00 / Emoji 30.00 |
| No FT | GPT-4o-mini | 59.01 | 50.53 | 73.49 | 2.71 | - |
| FT-10 | GPT-4o-mini | 98.13 | 98.64 | 95.07 | 30.59 | Split 74.07 / PY_Init 62.96 / Emoji 42.86 |
| FT-20 | GPT-4o-mini | 97.90 | 98.81 | 97.03 | 32.80 | Not specified |
| FT-40 | GPT-4o-mini | 99.40 | 99.24 | 96.67 | 31.33 | Not specified |

This table constitutes a critical piece of empirical evidence in the paper.

It demonstrates that ICL and few-shot SFT can indeed substantially boost detection rates.

However, the error rates and misinterpretation rates rise markedly at the same time.

Particularly from FT-10 to FT-40, although toxic detection rates are elevated to over 95%, benign texts are falsely penalized at rates exceeding 30%.

This is no longer "safety alignment" but rather a distinct behavioral shift (or over-conservatism).

## Main Results

### Comparison Across Different Perturbation Types

The paper points out that the most confounding perturbations are Homo and PY_Init.

Many models show consistent detection rates below 60% on these two categories.

This is reasonable, since these two categories heavily rely on implicit phonetics and internet expression patterns unique to native Chinese speakers.

In contrast, Trad and occasionally PY_Full exhibit higher detection rates, sometimes even exceeding the base performance.

This indicates that while some perturbations alter surface forms, they do not disrupt the triggers of toxic semantics in models.

### Comparison Across Different Models

Under unperturbed settings, most models perform well.

However, upon introducing the perturbations within the taxonomy, all models experience a sharp decline.

The authors highlight two main points:

1. Chinese native models do not exhibit an inherent advantage.
2. Even state-of-the-art reasoning models do not fully resolve this issue.

For example, the average detection rate of o3-mini drops significantly compared to its base performance.

DeepSeek-V3 and DeepSeek-R1-Llama drop to incredibly low performance levels under certain settings.

This showcases that "understanding standard Chinese" and "understanding perturbed Chinese" are two entirely different problems.

### Influence of Prompt Language

The paper also compares Chinese prompts with English prompts.

The results show that all LLMs perform better using Chinese prompts than English ones.

This indicates that keeping the prompt language aligned with the input text helps models more stably recognize Chinese toxicity.

However, this is merely a marginal improvement rather than a fundamental solution.

## Ablation Study

Strictly speaking, this paper does not contain block-level ablation studies common in traditional model-centric papers.

Its "ablation/analytical experiments" are mainly represented through three types of comparisons:

1. Chinese prompt vs English prompt.
2. No enhancement vs ICL.
3. No enhancement vs different scales of SFT.

The latter two are the most valuable because they directly answer the question: "Can a few demonstrative samples teach models to resolve these perturbations?"

The answer is: they capture surface behaviors but fail to grasp the actual underlying semantics.

The cases listed in the paper are highly convincing.

For instance, a model would classify a benign sentence containing emojis or split characters as toxic simply because it has seen visually similar toxic variants during training.

This reflects shortcut learning rather than robust understanding.

## Key Findings

### Finding 1: Chinese Toxicity Detection Inherently Presents Multimodal Challenges

Multimodality here does not refer to image-text inputs, but rather to how Chinese glyph, phonology, and semantic channels collectively dictate readability.

This makes text toxicity detection inherently harder in Chinese than in English.

### Finding 2: SOTA LLMs are Unrobust Against Perturbed Chinese Toxic Content

Even powerful models like GPT-4o, o3-mini, DeepSeek-V3, and the Qwen series suffer distinct performance drops in perturbed scenarios.

### Finding 3: ICL and Few-Shot SFT Cause Over-Correction

Models appear "stricter" on the surface but lack genuine understanding of the perturbed semantics.

Consequently, false positive rates on benign texts rise drastically.

### Finding 4: Chinese Native Models Show No Overwhelming Superiority

This possesses high practical implications.

Many naturally assume that Chinese LLMs are better suited for Chinese content moderation, but the empirical results do not support this simplification.

## Highlights & Insights

### Highlight 1: A Highly Practical Taxonomy

The proposed taxonomy is designed not for theoretical completeness, but for direct utility in data construction and model evaluation.

This carries strong engineering-research value.

### Highlight 2: Capturing the Narrative of "Chinese Multimodality"

The authors re-interpret the inner connection among Chinese glyphs, pronunciations, and semantics as a multimodal challenge.

This perspective is highly inspiring for Chinese NLP safety tasks.

### Highlight 3: Unveiling the Side Effects of Few-Shot Safety Enhancements

Many works show only post-enhancement recall improvements.

This paper digs deeper by asking:

Does the performance gain arise because the model understands better, or simply because it has become more radical?

This is an extremely pertinent question.

And the results carry significant warning signs.

### Highlight 4: Direct Practical Value for Commercial Moderation Engines

Real-world evasion methods on social platforms heavily rely on homophones, character splitting, emojis, and acronyms.

The benchmark and subsequent analyses provided in this paper are highly aligned with real-world scenarios.

## Limitations & Future Work

1. Although the taxonomy is more comprehensive than prior work, the authors acknowledge that internet culture evolves rapidly, and new perturbations will inevitably emerge.
2. The work focuses on the "intrinsic multimodality" of text and does not yet cover standard visual-text multimodal toxic content.
3. ICL and SFT are evaluated only on a small-sample scale, which may not fully represent the upper bound of large-scale fine-tuning.
4. The paper lacks detailed fine-tuning configurations, making it challenging to precisely replicate the origins of the behavioral shift during adaptation.
5. Main evaluation metrics are limited to detection rate, F1, ER, and MR, without further analysis on calibration or threshold sensitivity.

## Related Work & Insights

This paper is closely related to several strands of literature.

The first is Chinese offensive language detection datasets, such as Toxi_CN and COLD.

These focus more on coverage of toxicity types and standard classification tasks.

The second is Chinese adversarial attacks or robustness, such as ToxiCloakCN, RoCBert, and various sound-shape substitution attacks.

These typically only cover a single kind of perturbation.

The third is LLM safety and moderation works.

Most of these assume formal surface text formats, whereas this paper highlights that real-world Chinese online communities differ drastically.

Three key takeaways emerge for personal research:

1. Chinese safety tasks must explicitly model glyph and phonological characteristics rather than relying purely on text tokens.
2. Evaluations ought to clearly distinguish between gains from "genuine understanding" and those from "excessive conservatism."
3. To build stronger future systems, joint modeling of span-level grounding, glyph encoding, pinyin encoding, and context alignment might be required.

## Rating

**Novelty**: 8.5/10  
Systematizing Chinese toxicity evasions into a multimodal taxonomy and constructing a large-scale benchmark represents a solid task formulation.

**Methodology**: 8.0/10  
The method itself is not complicated, but the dataset construction pipeline, human validation, and evaluation are meticulously designed.

**Experimental Thoroughness**: 8.5/10  
Encompassing 9 models, 8 perturbation classes, bilingual prompts, along with ICL/SFT analysis, forming a complete chain of evidence.

**Value**: 9.0/10  
Offers direct reference value for Chinese content moderation, platform security, and LLM safety evaluation.

**Overall Rating**: 8.5/10

## Additional Remarks

The most crucial takeaway of this paper is not "which model won," but rather its explicit demonstration that:

The core difficulty in Chinese toxicity detection lies not merely in dataset size or model parameters, but inside the highly deformable, multi-channel, and culturally dependent nature of Chinese online expression itself.

If future research solely stacks larger general-purpose LLMs without addressing structural issues like glyphs, homophones, emojis, and internet metaphors, robustness will remain highly fragile.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] STATE ToxiCN: A Benchmark for Span-level Target-Aware Toxicity Extraction in Chinese Hate Speech Detection](state_toxicn_a_benchmark_for_span-level_target-aware_toxicity_extraction_in_chin.md)
- [\[ACL 2025\] Culture Matters in Toxic Language Detection in Persian](culture_matters_in_toxic_language_detection_in_persian.md)
- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](llm_label_propagation.md)
- [\[ACL 2025\] Exploring the Impact of Instruction-Tuning on LLMs' Susceptibility to Misinformation](exploring_the_impact_of_instruction-tuning_on_llms_susceptibility_to_misinformat.md)
- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](../../ACL2026/social_computing/toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)

</div>

<!-- RELATED:END -->
