---
title: >-
  [Paper Note] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning
description: >-
  [ACL 2026][Multimodal VLM][multimodal reasoning] This paper identifies an "inverse scaling law" in multimodal reasoning models — slow-thinking (reasoning) models are more prone to producing untruthful outputs than fast-t…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "multimodal reasoning"
  - "truthfulness evaluation"
  - "inverse scaling law"
  - "depth-first reasoning"
  - "hallucination detection"
date: 2026-05-08
content_hash: b84d1162d5884c83
---

# When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**Conference**: ACL 2026
**arXiv**: [2505.20214](https://arxiv.org/abs/2505.20214)
**Code**: [https://truthfulvqa.github.io](https://truthfulvqa.github.io)
**Area**: Multimodal VLM / AI Safety
**Keywords**: multimodal reasoning, truthfulness evaluation, inverse scaling law, depth-first reasoning, hallucination detection

## TL;DR

This paper identifies an "inverse scaling law" in multimodal reasoning models — slow-thinking (reasoning) models are more prone to producing untruthful outputs than fast-thinking (chat) models when faced with misleading visual inputs. To systematically diagnose this phenomenon, the authors construct the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-tier graded prompts) and the TruthfulJudge evaluation model (88.4% accuracy).

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have achieved significant progress on visual understanding tasks. Reasoning models (e.g., QVQ, Mulberry) have demonstrated breakthroughs on structured tasks such as mathematics and coding through extended reasoning chains. The hallucination problem has been extensively studied, but primarily focused on unintentional errors on benign inputs.

**Limitations of Prior Work**: (1) Truthfulness and hallucination are related but distinct concepts — the former emphasizes robustness in maintaining factual fidelity under adversarial or misleading inputs, while the latter concerns unintentional fabrications on benign inputs. (2) Existing benchmarks predominantly employ binary or multiple-choice formats, which cannot probe the relationship between deep reasoning and truthfulness. (3) AI-as-Judge evaluation suffers from systematic biases that may allow untruthfulness to evade detection.

**Key Challenge**: Reasoning models are designed to "think" more deeply to improve accuracy, yet when confronted with ambiguous or misleading multimodal inputs, deeper reasoning leads to more untruthful outputs. This occurs because reasoning models tend toward depth-first search (DFS) — once an initial interpretation is selected, they continue to elaborate on it rather than exploring alternative interpretations.

**Goal**: (1) Construct the first benchmark for systematically evaluating multimodal truthfulness with human-in-the-loop validation. (2) Reveal systematic differences in truthfulness between reasoning and chat models. (3) Develop a reliable automated truthfulness evaluator.

**Key Insight**: A three-tier graded prompt design (basic perception → inductive misleading → false-premise reasoning) is employed to progressively increase reasoning complexity and misleading intensity, enabling fine-grained diagnosis of model truthfulness at different reasoning depths.

**Core Idea**: The DFS-style reasoning topology of reasoning models — rather than model capacity or training data — is the structural cause of truthfulness degradation. Applying chain-of-thought (CoT) prompting to chat models reproduces similar degradation, supporting this causal claim.

## Method

### Overall Architecture

TruthfulVQA consists of three components: (1) 5,000+ visually misleading images annotated by 50 annotators, organized into 8 major categories and 21 subcategories according to Whaley's taxonomy of deception; (2) a three-tier graded prompt system that systematically probes truthfulness across perception, inductive reasoning, and false-premise reasoning; and (3) TruthfulJudge, an evaluation model fine-tuned on Qwen2.5-VL-7B using a Critique-Label paradigm.

### Key Designs

1. **Three-Tier Graded Prompt System**:

    - **Function**: Systematically evaluates model truthfulness under varying reasoning depths and misleading intensities.
    - **Mechanism**: Level 1 (basic perception) tests direct visual-semantic recognition, e.g., "How many people are in the image?"; Level 2 (inductive misleading) introduces subtle deceptive contextual cues to challenge hypothetical reasoning, e.g., "How far is the sun from the person's feet?"; Level 3 (false-premise reasoning) constructs erroneous narratives using false but seemingly factual statements, requiring models to identify invalid logic, e.g., "A horse has the intelligence equivalent to a 5-year-old child... so can a horse sit and play an accordion?"
    - **Design Motivation**: Binary and multiple-choice formats capture only surface-level correctness and cannot probe truthfulness vulnerabilities in deep reasoning. The graded design enables more precise diagnosis.

2. **Logit Advantage Loss (LAL) Metric**:

    - **Function**: Quantifies the impact of misleading prompts on model decision-making at the confidence level.
    - **Mechanism**: The logit advantage of the correct answer is defined as $A_i = \ell_i(o^*) - \max_{o \neq o^*} \ell_i(o)$; inter-level LAL $= A_i - A_j$ is then computed and decomposed into the degradation of the correct option and the amplification of incorrect options. A normalized variant $A_i^{\text{norm}}$ eliminates arbitrary scaling factors across models.
    - **Design Motivation**: Accuracy alone cannot distinguish between "high-confidence correct" and "low-confidence correct." LAL reveals the effect of misleading inputs on a model's internal decision boundary, providing more fine-grained behavioral diagnosis.

3. **TruthfulJudge (Reliable Evaluation Model)**:

    - **Function**: Replaces costly human evaluation for truthfulness judgment.
    - **Mechanism**: Fine-tuned on Qwen2.5-VL-7B using 7.1k human-annotated question-answer pairs with explanatory critiques and preference labels. The Critique-Label paradigm is adopted: the model first generates a critique analysis, then produces a preference label. Comparisons with Bradley-Terry, Critique-Score, and Pure-Label paradigms show that Critique-Label is significantly superior (88.4% vs. 57.5%). Cohen's $\kappa = 0.79$ (approaching "almost perfect agreement"), FPR = 0.12, ECE = 0.11.
    - **Design Motivation**: General-purpose MLLMs used as judges achieve only 52–64% accuracy and systematically accept approximately one-third of hallucinated responses. A specially trained evaluator is required to reliably detect truthfulness issues.

### Loss & Training

TruthfulJudge is trained via supervised fine-tuning (SFT). Training data consists of 7.1k high-quality critique-label pairs generated by GPT-4o through prompt engineering, validated against human-annotated truthfulness and preference labels. The test set contains 812 samples.

## Key Experimental Results

### Main Results

**Average accuracy of 50+ models on TruthfulVQA**

| Level | Average Accuracy | Description |
|-------|-----------------|-------------|
| Level 1 (Basic Perception) | 81.85% | Direct visual recognition |
| Level 2 (Inductive Misleading) | 55.37% | Drop of 26.5 percentage points |
| Level 3 (False Premise) | 44.96% | Further drop of 10.4 percentage points |

**LAL comparison: Reasoning vs. Chat models**

| Model Pair | Chat LAL | Reasoning LAL |
|------------|---------|--------------|
| Qwen2.5-VL vs. QVQ-72B | Lower | 0.89 (significantly higher) |
| Qwen2-VL vs. Mulberry-7B | Lower | 0.71 |
| Kimi-VL-A3B vs. Thinking | Lower | 0.53 |

### Ablation Study

| Configuration | Effect | Description |
|---------------|--------|-------------|
| Chat model + CoT prompting | Drop of 2.8–8.3 percentage points | Confirms DFS topology as the cause of degradation |
| Chat model ECE | 0.16–0.25 | Reasonably calibrated |
| Reasoning model ECE | >0.25 | Overconfident |
| Qwen2.5-VL-72B ECE | 0.188 | Chat version |
| QVQ-72B ECE | 0.325 | Reasoning version; significantly worse calibration |

### Key Findings

- **Inverse scaling law**: Reasoning models consistently underperform their chat counterparts within the same model family; larger reasoning models do not guarantee better truthfulness performance.
- **DFS vs. BFS**: Reasoning models tend toward DFS (once an initial interpretation is selected, they continue to elaborate on it), while chat models more closely resemble BFS (exploring multiple paths before drawing conclusions).
- **Causal validation**: Applying CoT prompting to chat models (enforcing serialized reasoning) causes degradation of 2.8–8.3 percentage points across five models, with failure modes consistent with those of reasoning models. This confirms that the vulnerability originates from the reasoning topology rather than the models themselves.
- General-purpose judge models (GPT-4o, Gemini, etc.) perform poorly on truthfulness evaluation (52–64% accuracy), while TruthfulJudge achieves 88.4%.

## Highlights & Insights

- The discovery of the "inverse scaling law" carries important cautionary implications — reasoning models may be more dangerous than simpler models in safety-critical scenarios, as they fabricate details more confidently to support erroneous reasoning.
- The DFS vs. BFS analysis provides a clear mechanistic explanation beyond mere empirical observation. The causal experiment (CoT → degradation) further rules out confounding factors.
- The Critique-Label paradigm of TruthfulJudge is transferable — generating analysis prior to judgment is more reliable than direct scoring.

## Limitations & Future Work

- The dataset scale (5,000+) remains relatively small compared to commercial benchmarks.
- Cultural homogeneity among the annotation team may introduce biases.
- The 8-category untruthfulness taxonomy may not cover the full spectrum of visual-semantic deception.
- Future work should develop BFS-inspired reasoning mechanisms to balance reasoning depth with truthfulness.

## Related Work & Insights

- **vs. CHAIR/MME-Hallucination**: These focus on hallucinations under benign inputs; TruthfulVQA targets truthfulness under adversarial inputs.
- **vs. MultiTrust**: MultiTrust unifies evaluation across seven stages but remains primarily multiple-choice; TruthfulVQA provides deeper graded prompt probing.
- **vs. LLM-as-Judge**: This paper empirically demonstrates that general-purpose MLLMs are unreliable as truthfulness judges, necessitating specially trained evaluators.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic revelation of the inverse scaling law of truthfulness in reasoning models; the DFS/BFS analytical framework has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluation of 50+ models, causal validation, dedicated judge model — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and in-depth analysis, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Carries significant cautionary implications for the safety of reasoning models; both the benchmark and the evaluator have lasting utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] When Helpers Become Hazards: A Benchmark for Analyzing Multimodal LLM-Powered Safety in Daily Life](when_helpers_become_hazards_a_benchmark_for_analyzing_multimodal_llm-powered_saf.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[NeurIPS 2025\] When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](../../NeurIPS2025/multimodal_vlm/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ACL 2026\] When Vision-Language Models Judge Without Seeing: Exposing Informativeness Bias](when_vision-language_models_judge_without_seeing_exposing_informativeness_bias.md)

</div>

<!-- RELATED:END -->
