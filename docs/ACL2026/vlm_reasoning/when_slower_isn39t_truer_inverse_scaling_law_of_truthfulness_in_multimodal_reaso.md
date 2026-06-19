---
title: >-
  [Paper Note] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Paper Note] This paper identifies an "inverse scaling law" in multimodal reasoning—reasoning (slow-thinking) models are more prone to generating untruthful outputs than chat (fast-thinking) models when faced with misleading visual inputs. It constructs the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-tier tiered pro
tags:
  - ACL 2026
  - Multimodal VLM
date: 2026-05-08
content_hash: 5b04757d50d7432f
---
# When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**Conference**: ACL 2026  
**arXiv**: [2505.20214](https://arxiv.org/abs/2505.20214)  
**Code**: [https://truthfulvqa.github.io](https://truthfulvqa.github.io)  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: Multimodal Reasoning, Truthfulness Evaluation, Inverse Scaling Law, Depth-First Reasoning, Hallucination Detection

## TL;DR

This paper identifies an "inverse scaling law" in multimodal reasoning—reasoning (slow-thinking) models are more prone to generating untruthful outputs than chat (fast-thinking) models when faced with misleading visual inputs. It constructs the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-tier tiered prompting) and the TruthfulJudge evaluation model (88.4% accuracy) to systematically diagnose this phenomenon.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in visual understanding tasks. Reasoning models (e.g., QVQ, Mulberry) have achieved breakthroughs in structured tasks like mathematics and code via longer reasoning chains. Hallucination has been widely studied, but primarily focuses on unintentional errors by models on benign inputs.

**Limitations of Prior Work**: (1) Truthfulness and hallucination are related but distinct concepts—the former emphasizes robustness in maintaining factual loyalty under adversarial or misleading inputs, while the latter focuses on unintentional fabrication under benign inputs; (2) Existing benchmarks primarily use binary or multiple-choice tests, failing to probe the link between deep reasoning and truthfulness; (3) AI-as-Judge evaluation exhibits systematic biases, potentially allowing untruthfulness to escape detection.

**Key Challenge**: Reasoning models are designed to "think" deeper to improve accuracy, yet deeper reasoning leads to more untruthful outputs when facing ambiguous or misleading multimodal inputs. The cause is that reasoning models tend toward Depth-First Search (DFS)—once an initial interpretation is selected, they dig deeper rather than exploring alternative explanations.

**Goal**: (1) Build the first benchmark for systematically evaluating multimodal truthfulness with human-in-the-loop verification; (2) Reveal systematic differences in truthfulness between reasoning and chat models; (3) Develop a reliable automated truthfulness evaluator.

**Key Insight**: Design a three-tier tiered prompting system (Basic Perception $\rightarrow$ Inductive Misleading $\rightarrow$ False Premise Reasoning) to progressively increase reasoning complexity and misleading intensity, thereby finely diagnosing the truthfulness performance of models at different depths.

**Core Idea**: The DFS-style reasoning topological structure itself (rather than model capacity or training data) is the structural cause for the decline in truthfulness—applying CoT prompting to chat models can reproduce similar degradation.

## Method

### Overall Architecture

TruthfulVQA consists of three parts: (1) 5,000+ visually misleading images (annotated by 50 annotators) organized into 8 categories and 21 sub-categories based on Whaley’s deception taxonomy; (2) A three-tier tiered prompting system to systematically probe truthfulness under perception, inductive reasoning, and false premise reasoning; (3) The TruthfulJudge evaluation model, fine-tuned on Qwen2.5-VL-7B using the Critique-Label paradigm.

### Key Designs

**1. Three-tier tiered prompting: Manifesting truthfulness degradation level by level with reasoning depth**

Binary and multiple-choice tests only capture surface correctness and fail to detect truthfulness vulnerabilities hidden within deep reasoning. The authors pair each misleading image with three questions of increasing difficulty. Level 1 (Basic Perception) tests direct vision-semantic recognition, such as "How many people are in the image?"; Level 2 (Inductive Misleading) introduces subtle deceptive contextual cues to challenge hypothetical reasoning, such as "How far is the sun from the person's feet?", forcing the model to induce based on seemingly plausible premises; Level 3 (False Premise Reasoning) uses statements that seem factual but are false to build a wrong narrative, such as "Horses have intelligence equivalent to a 5-year-old... so can horses play the accordion while sitting?", requiring the model to see through invalid logic. Consequently, where a model begins to be misled and the magnitude of degradation can be finely diagnosed, rather than relying on a single aggregate accuracy score.

**2. Logit Advantage Loss (LAL) Metric: Quantifying the erosion of decision-making by misleading prompts at the confidence level**

Accuracy alone cannot distinguish between "high-confidence correct" and "barely correct," and thus cannot measure how much the misleading input shifts the model's decision boundary. LAL first defines the logit advantage of the correct answer as $A_i = \ell_i(o^*) - \max_{o \neq o^*} \ell_i(o)$, which represents the leading margin of the correct option relative to the strongest distractor. It then calculates the difference between levels LAL $= A_i - A_j$, which can be decomposed into "depression of the correct option" and "elevation of the incorrect option" to see which side the degradation stems from. To eliminate arbitrary logit scaling factors between different models, the authors also use a normalized version $A_i^{\text{norm}}$ for cross-model comparison. This indicator turns the intuition that "reasoning models are more confidently wrong under misleading input" into a measurable numerical value—the LAL of reasoning models is systematically higher than that of chat models from the same series.

**3. TruthfulJudge: Replacing expensive and biased human evaluation with a specialized trained judge**

General MLLMs are unreliable as judges, with accuracies of only 52–64%, and they systematically accept approximately one-third of hallucinated answers, allowing untruthful outputs to escape detection. The authors fine-tuned Qwen2.5-VL-7B on 7.1k human-annotated QA pairs (including explanatory critiques and preference labels) using the Critique-Label paradigm: the model first generates a critical analysis and then provides a preference label, rather than scoring directly. The authors compared Bradley-Terry, Critique-Score, and Pure-Label paradigms, finding Critique-Label to be significantly superior (88.4% vs 57.5%), with Cohen's $\kappa = 0.79$ (approaching "almost perfect agreement"), FPR $= 0.12$, and ECE $= 0.11$. The reason "providing reasons before making a judgment" is more stable is that it forces the judge to lay out the evidence, making it difficult to overlook a seemingly fluent but untruthful answer based on impression alone.

### Loss & Training

TruthfulJudge uses supervised fine-tuning (SFT). The training data consists of 7.1k high-quality critique-label pairs generated by GPT-4o through prompt engineering, validated by human-annotated truthfulness labels and preference labels. The test set comprises 812 samples.

## Key Experimental Results

### Main Results

**Average Accuracy of 50+ Models on TruthfulVQA**

| Level | Average Accuracy | Description |
|------|-----------|------|
| Level 1 (Basic Perception) | 81.85% | Direct visual recognition |
| Level 2 (Inductive Misleading) | 55.37% | Decrease of 26.5 percentage points |
| Level 3 (False Premise) | 44.96% | Further decrease of 10.4 percentage points |

**LAL Comparison: Reasoning Models vs. Chat Models**

| Model Pair | Chat LAL | Reasoning LAL |
|--------|---------|--------------|
| Qwen2.5-VL vs QVQ-72B | Lower | 0.89 (Significantly higher) |
| Qwen2-VL vs Mulberry-7B | Lower | 0.71 |
| Kimi-VL-A3B vs Thinking | Lower | 0.53 |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| Chat Model + CoT Prompting | Decrease of 2.8-8.3 pp | Proves DFS topology itself is the cause |
| Chat Model ECE | 0.16-0.25 | Better calibration |
| Reasoning Model ECE | >0.25 | Overconfident |
| Qwen2.5-VL-72B ECE | 0.188 | Chat version |
| QVQ-72B ECE | 0.325 | Reasoning version, significantly worse calibration |

### Key Findings

- Inverse Scaling Law: Reasoning models consistently perform lower than their corresponding chat models in the same series; larger reasoning models do not guarantee better performance.
- DFS vs BFS: Reasoning models tend toward DFS (digging deep once an initial explanation is chosen), whereas chat models are closer to BFS (exploring multiple paths before concluding).
- Causal Validation: After applying CoT prompting (forcing serialized reasoning) to chat models, all five models degraded by 2.8-8.3 percentage points, with failure modes identical to reasoning models. This proves the vulnerability stems from the reasoning topology rather than the model itself.
- General judge models (GPT-4o, Gemini, etc.) perform poorly in truthfulness evaluation (52-64% accuracy), while TruthfulJudge reaches 88.4%.

## Highlights & Insights

- The discovery of the "Inverse Scaling Law" serves as an important warning—reasoning models may be more dangerous in safety-critical scenarios than simpler models because they confidently fabricate details to support erroneous reasoning.
- The DFS vs BFS analysis provides a clear mechanistic explanation rather than just empirical observation. Causal experiments (CoT $\rightarrow$ degradation) further exclude confounding factors.
- The Critique-Label paradigm of TruthfulJudge is transferable—generating an analysis before making a judgment is more reliable than direct scoring.

## Limitations & Future Work

- The dataset scale (5,000+) is still relatively small compared to commercial benchmarks.
- Cultural homogeneity of the annotation team may introduce bias.
- The 8-category untruthfulness taxonomy may not cover the full spectrum of vision-semantic deception.
- Future work should develop BFS-inspired reasoning mechanisms to balance reasoning depth with truthfulness.

## Related Work & Insights

- **vs CHAIR/MME-Hallucination**: These focus on hallucinations under benign inputs, while TruthfulVQA focuses on truthfulness under adversarial inputs.
- **vs MultiTrust**: MultiTrust provides a unified seven-stage evaluation but still relies mainly on multiple-choice questions; TruthfulVQA provides deeper tiered prompting probes.
- **vs LLM-as-Judge**: This paper empirically proves that general MLLMs are unreliable as truthfulness judges and require specialized training.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the inverse scaling law of truthfulness in reasoning models; the DFS/BFS analysis framework has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 50+ models, causal validation, specialized judge model; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Significant warning for safety research on reasoning models; both the benchmark and evaluator have lasting utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] Vocabulary Scaling Law: Tuning Open-vocabulary Predictors for Their Openness](../../CVPR2026/multimodal_vlm/vocabulary_scaling_law_tuning_open-vocabulary_predictors_for_their_openness.md)
- [\[CVPR 2026\] When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought](../../CVPR2026/multimodal_vlm/when_visualizing_is_the_first_step_to_reasoning_mira_a_benchmark_for_visual_chai.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)

</div>

<!-- RELATED:END -->
