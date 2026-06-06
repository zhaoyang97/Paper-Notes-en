---
title: >-
  [Paper Note] LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models
description: >-
  [ACL 2026][Multilingual & Machine Translation][Lao] LaoBench is the first large-scale, multi-dimensional Lao LLM evaluation benchmark. It comprises 17,000+ expert-curated samples covering cultural-knowledge application…
tags:
  - "ACL 2026"
  - "Multilingual & Machine Translation"
  - "Lao"
  - "Low-resource NLP"
  - "Cultural Reasoning"
  - "Black-box Evaluation"
  - "Expert+Agent Collaborative Construction"
date: 2026-05-08
content_hash: 4f54ab6f256dc320
---

# LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2511.11334](https://arxiv.org/abs/2511.11334)  
**Code**: https://huggingface.co/datasets/BAAI/LaoBench  
**Area**: Multilingual Evaluation / Low-resource Languages / SE Asian Languages / Dataset  
**Keywords**: Lao, Low-resource NLP, Cultural Reasoning, Black-box Evaluation, Expert+Agent Collaborative Construction

## TL;DR
LaoBench is the first large-scale, multi-dimensional Lao LLM evaluation benchmark. It comprises 17,000+ expert-curated samples covering cultural-knowledge application, the Lao K12 curriculum, and Lao-Chinese-English trilingual translation. It adopts a unique three-stage design: "Open 7k + Black-box 10k + Open-ended 500." The 10k black-box set prevents contamination via a controlled scoring service. Mainstream closed-source models (GPT-5-High, Gemini-2.5-Pro, etc.) still lag behind human experts by approximately 10-20 percentage points, demonstrating that Lao cultural reasoning and translation fidelity remain largely unsolved challenges.

## Background & Motivation

**Background**: LLM evaluation is heavily biased toward high-resource languages. Although Southeast Asian benchmarks like SeaEval, SEA-HELM, and SeaExam exist, Lao is nearly absent. Existing limited Lao resources are task-specific (morphology, bilingual MT) and lack a systematic, reproducible "general LLM capability evaluation."

**Limitations of Prior Work**: (1) Most SEA benchmarks are translated from English (losing local cultural anchoring) or only test high-level multilingual reasoning while skipping native language proficiency aligned with curricula. (2) Lao script is *scriptio continua* (continuous writing without clear word boundaries), making traditional BLEU/Tokenizers inaccurate. (3) Public benchmarks increasingly suffer from contamination and leaderboard overfitting, especially since no black-box evaluation services are available for low-resource languages like Lao.

**Key Challenge**: To evaluate the true capability of LLMs in low-resource languages, one must simultaneously provide: native expert-written content, multi-dimensional coverage (knowledge/education/translation), black-box mechanisms against data contamination, and reproducible statistical protocols. No existing Lao resource combines all these elements.

**Goal**: (1) Construct the first large-scale Lao benchmark written by local native speakers; (2) Cover cultural-knowledge application, K12 curriculum, and Lao↔Zh↔En trilingual translation; (3) Design open + black-box dual subsets to counter contamination; (4) Use an expert + agent collaborative pipeline to balance quality and scale; (5) Systematically evaluate mainstream open/closed-source LLMs to quantify the gap with human experts.

**Key Insight**: Benchmark construction is treated as a holistic engineering problem involving software, processes, and evaluation protocols. Instead of just providing data, this work standardizes the entire process with Lao-aware SacreBLEU configurations, Arena-style open-ended evaluation, bootstrap Confidence Intervals (CI), multi-judge aggregation, and a black-box service API.

**Core Idea**: Utilize a structure of three dimensions × three subsets (Lao-7k Open MCQ / Lao-10k Black-box MCQ / Lao-500 Open-ended prompt) with an expert + agent dual-loop construction, supported by an Arena dual-judge + bootstrap CI evaluation protocol.

## Method

### Overall Architecture
The LaoBench construction pipeline consists of three stages: (A) **Raw Material Collection**—collecting K12 textbooks, government/legal documents, encyclopedic educational publications, and local cultural articles from authoritative Lao sources; (B) **Dataset Construction**—MCQ subsets (Lao-7k Open + Lao-10k Black-box) are manually written by 11 Lao native experts, including question stems, 4-way choices, and difficulty calibration. The Lao-500 open-ended prompts are selected from a large candidate pool using a BenchBuilder-style pipeline (LLM scoring for specificity/clarity/domain depth + thematic clustering + diversity sampling); (C) **Multi-stage Verification**—expert reviews + automated agent checks (duplicate detection, semantic consistency, context independence, and sensitive content screening). The 17,000+ samples are organized into Knowledge Application, K12, and Translation dimensions. Evaluation uses accuracy for MCQs, SacreBLEU + chrF++ (standardized via LaoNLP word segmentation) for translation, and Arena pairwise evaluation with dual judges for Lao-500.

### Key Designs

1.  **Open + Black-box Dual Subsets Against Data Contamination**:
    - **Function**: The Lao-7k open MCQ subset is provided for reproducibility research, while the Lao-10k closed MCQ subset is managed by a controlled service that only returns aggregate scores.
    - **Mechanism**: The authors specify that Lao-10k test items will never be released. Evaluators must either submit an item ID-to-answer dictionary or provide an inference API endpoint for the service to run standardized prompts. This prevents leaderboard overfitting via submission rate limits. The open subset underwent web overlap retrieval and n-gram overlap checks (6.2% of candidates had suspected overlap, mostly common-sense statements rather than direct leaks).
    - **Design Motivation**: Released benchmarks for low-resource languages are almost inevitably "consumed" by pre-training corpora; black-box services are the only realistic path for long-term benchmark viability.

2.  **Expert + Agent Dual-loop Construction Pipeline**:
    - **Function**: Balances "native expert fidelity" with "scalable productivity," allowing 17k+ samples to be high-quality and completed within a reasonable timeframe.
    - **Mechanism**: 55 contributors were divided by roles (25 domain experts writing questions, 11 translation experts for bilingual alignment, 10 senior reviewers for final audit, and 9 NLP data curators). Each question was reviewed by at least two independent experts, with disputes adjudicated by senior reviewers. Agents performed duplicate detection (character n-gram + embedding retrieval), semantic consistency (verifying unique correct answers), context independence (removing questions dependent on external info), and sensitivity screening. A sample of 500 questions showed an inter-annotator agreement of Fleiss $\kappa=0.87$.
    - **Design Motivation**: Pure manual labor is prohibitively expensive, while pure agent generation has a high failure rate. This Hendrycks-style hybrid pipeline outsets mechanical steps to agents while reserving value judgments and cultural accuracy for humans.

3.  **Lao-aware Translation Evaluation + Arena Dual Judge Open-ended Evaluation**:
    - **Function**: Addresses specific pain points: inaccurate BLEU due to lack of word boundaries and the inability of MCQs to measure generation quality.
    - **Mechanism**: (i) Translation evaluation uses SacreBLEU with Lao-aware LaoNLP v0.7 word segmentation + chrF++ (character n-gram, insensitive to segmentation); (ii) Lao-500 follows Arena pairwise evaluation: using GPT-5-High as a fixed baseline $B$, for each prompt $x_i$, candidate $M$ and baseline $B$ generate responses $y_i^M, y_i^B$. Judge models (Gemini-2.5-Pro + Qwen3-Max) determine the winner based on correctness, completeness, reasoning, clarity, and Lao fluency. To eliminate position bias, each pair is evaluated twice (swapping A/B). Bootstrap resampling provides 95% CI, and the final score $S(M)$ is the average across dual judges.
    - **Design Motivation**: Raw BLEU for Lao translation is uninterpretable; the Arena mode standardizes "generation quality" in a way both humans and LLMs can judge.

### Loss & Training
LaoBench is a dataset and evaluation protocol, not a trained model. All evaluated LLMs are tested in a zero-shot setting with a decoding temperature of 0 (where supported). MCQ outputs undergo A/B/C/D post-processing. CoT variants (Thinking) and direct answers (Non-Thinking) are evaluated separately. Judge models for Lao-500 are Gemini-2.5-Pro and Qwen3-Max. To avoid self-preference, candidate models do not judge their own responses.

## Key Experimental Results

### Main Results
Comparison across three major dimensions on Lao-7k (K12 Avg / Translation BLEU for Social & Law / Knowledge Application Avg):

| Model | K12 Avg ↑ | Translation Soc.&Law BLEU ↑ | Knowledge App Avg ↑ |
| :--- | :--- | :--- | :--- |
| Random Choice | 25.00 | – | 25.00 |
| Ministral-8B-Instruct | 28.29 | 0.83 | 24.15 |
| Ling-mini-2.0 | 36.91 | 0.69 | 30.25 |
| Qwen3-Next-80B-A3B-Instruct | 79.80 | 16.03 | 63.05 |
| DeepSeek-V3.2-Exp (Thinking) | 85.12 | 20.57 | 69.11 |
| Qwen3-235B-A22B-Instruct-2507 | 86.18 | 21.81 | 67.42 |
| Qwen3-Max (Closed) | 86.78 | 21.70 | 69.06 |
| Gemini-2.5-Pro | 89.56 | **26.22** | 73.68 |
| Claude-Opus-4.1 | 87.95 | 24.78 | 73.40 |
| **GPT-5-High** | **89.46** | 20.96 | **74.89** |
| **Human Experts** | **98.52** | – | **98.74** |

GPT-5-High still has a gap of ~9 points in K12 and ~24 points in Knowledge Application compared to humans. The strongest translation BLEU remains in the mid-30s.

### Ablation Study
**Lao-500 Dual Judge Arena Cross-Judge Bias Analysis**:

| Model | Gemini Win Rate | Qwen3-Max Win Rate | Δ(G−Q) | Gap |
| :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | 54.22 | 48.85 | +5.37 | 5.37 |
| Qwen3-Max | 45.16 | 52.80 | −7.64 | 7.64 |
| Qwen3-235B-A22B-Instruct-2507 | 45.53 | 51.75 | −6.22 | 6.22 |
| Claude-Sonnet-4.5 thinking | 50.50 | 50.08 | +0.42 | 0.42 |
| GPT-5-High (baseline) | 49.94 | 49.94 | 0.00 | 0.00 |

**Annotator Consistency**: Fleiss $\kappa=0.87$ for 500 samples; Spearman $\rho=0.83$ between Lao-500 Arena judges; human sanity check agreement with LLM judges is 84%.

### Key Findings
- **Closed > Open Source holds**: GPT-5-High, Gemini-2.5-Pro, and Claude-Opus lead the strongest open-source models (Qwen3-235B / DeepSeek-V3.2) in almost all subdomains, though the gap has narrowed to 1-3 points.
- **K12 is significantly easier than Knowledge Application**: Curriculum-aligned structured content is easier (strong models score 90%+), yet culturally anchored reasoning is the true differentiator—GPT-5 drops from 89.5 in K12 to 74.9 in Knowledge App.
- **Translation BLEU is generally stuck below mid-30s**: Culture & History and Society & Law are the hardest subdomains due to specialized terminology and cultural expressions.
- **CoT (Thinking) primarily aids cultural reasoning**: Gains in factual subdomains like K12 are minimal, while Knowledge Application and Translation show steady improvement.
- **Judges prefer models from their own family**: Qwen3-Max judge favors the Qwen family (Δ −6 to −8); dual judge averaging is necessary to mitigate bias.
- **Human-Machine Gap is significant**: Humans (97%+) vs. strongest models (89% in K12 / 75% in Knowledge App) indicates substantial headroom for improvement.

## Highlights & Insights
- Implementing "benchmark anti-contamination" as a concrete engineering solution using "open + black-box services" serves as an urgent template for the community.
- Utilizing LaoNLP for Lao-aware tokenization before calculating SacreBLEU + chrF++, and providing translation prompts, judge protocols, and bootstrap processes in the appendix, lowers the replication threshold.
- The Arena dual-judge + bootstrap CI approach for Lao-500 serves as a statistically rigorous model for low-resource open-ended evaluation.
- Detailed disclosure of the 55-contributor role distribution (PhD/Master/Bachelor), double-blind review, and senior final audit in the appendix provides a rare "evaluation of the evaluation" transparency.

## Limitations & Future Work
- The focus is still on MCQs, which can be partially gamed by test-taking tricks and may not fully reflect open-ended reasoning.
- Translation evaluation relies heavily on reference-based metrics (BLEU/chrF++), which penalize valid paraphrasing.
- Arena evaluation depends on LLM judges and a fixed baseline, introducing self-preference and anchoring biases.
- The black-box service was not yet officially live at the time of writing (stated "upon publication").
- Task coverage is limited to three types, missing code, agents, and long-context tasks; fewer comparable models exist specifically for the Lao domain.

## Related Work & Insights
- **vs SeaEval / SEA-HELM / SeaExam**: These provide broad SEA multilingual coverage, but Lao is missing or marginal. LaoBench provides monolingual depth.
- **vs VMLU (Vietnamese) / LoRaXBench (Indonesian)**: Similar philosophy (native K12 + culture), but LaoBench adds translation and black-box services, serving as a template for future benchmarks.
- **vs M3Exam / BenchBuilder**: Lao-500 adopts BenchBuilder-style LLM scoring + thematic clustering pipeline, successfully porting English community best practices to Lao.

## Rating
- Novelty: ⭐⭐⭐⭐ First truly multidimensional + black-box benchmark for Lao; substantial innovation in engineering combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ 14 SOTA models across 13 subdomains plus translation and open-ended protocols.
- Writing Quality: ⭐⭐⭐⭐ Clear pipeline diagrams and comparison tables; excellent reproducibility with detailed appendices.
- Value: ⭐⭐⭐⭐⭐ Vital infrastructure for the Lao NLP community from 0 to 1; provides a reusable engineering template for other low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models](the_gaoyao_benchmark_a_comprehensive_framework_for_evaluating_multilingual_and_m.md)
- [\[ACL 2026\] LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models](llm-xtm_enhancing_cross-lingual_topic_models_with_large_language_models.md)
- [\[ACL 2026\] Evaluating Robustness of Large Language Models Against Multilingual Typographical Errors](evaluating_robustness_of_large_language_models_against_multilingual_typographica.md)
- [\[ACL 2026\] TransLaw: A Large-Scale Dataset and Multi-Agent Benchmark Simulating Professional Translation of Hong Kong Case Law](translaw_a_large-scale_dataset_and_multi-agent_benchmark_simulating_professional.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)

</div>

<!-- RELATED:END -->
