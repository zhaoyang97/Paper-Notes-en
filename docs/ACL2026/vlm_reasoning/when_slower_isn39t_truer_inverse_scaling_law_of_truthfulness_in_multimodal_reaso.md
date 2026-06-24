---
title: >-
  [Paper Note] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning
description: >-
  [ACL 2026][VLM Reasoning][Multimodal Reasoning] This paper identifies an "Inverse Scaling Law" in multimodal reasoning—reasoning (slow-thinking) models are more prone to generating untruthful outputs than chat (fast-thinking) models when facing misleading visual inputs. To systematically diagnose this, the authors constructed the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-level hierarchical prompts) and the TruthfulJudge evaluation model (88.4% accuracy).
tags:
  - "ACL 2026"
  - "VLM Reasoning"
  - "Multimodal Reasoning"
  - "Truthfulness Evaluation"
  - "Inverse Scaling Law"
  - "Depth-First Reasoning"
  - "Hallucination Detection"
date: 2026-05-08
content_hash: b07a3a40699b24f1
---

# When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**Conference**: ACL 2026  
**arXiv**: [2505.20214](https://arxiv.org/abs/2505.20214)  
**Code**: [https://truthfulvqa.github.io](https://truthfulvqa.github.io)  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: Multimodal Reasoning, Truthfulness Evaluation, Inverse Scaling Law, Depth-First Reasoning, Hallucination Detection

## TL;DR

This paper identifies an "Inverse Scaling Law" in multimodal reasoning—reasoning (slow-thinking) models are more prone to generating untruthful outputs than chat (fast-thinking) models when facing misleading visual inputs. To systematically diagnose this, the authors constructed the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-level hierarchical prompts) and the TruthfulJudge evaluation model (88.4% accuracy).

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in visual understanding. Reasoning models (e.g., QVQ, Mulberry) have achieved breakthroughs in structured tasks like mathematics and coding through longer chains of thought. While hallucinations have been widely studied, research has primarily focused on unintentional errors on benign inputs.

**Limitations of Prior Work**: (1) Truthfulness and hallucination are related but distinct concepts—the former emphasizes robustness in maintaining factual loyalty under adversarial or misleading inputs, while the latter focuses on unintentional fabrication under benign inputs; (2) existing benchmarks mostly use binary or multiple-choice tests, failing to probe the link between deep reasoning and truthfulness; (3) AI-as-Judge evaluations suffer from systematic biases and may allow untruthfulness to escape detection.

**Key Challenge**: Reasoning models are designed to "think" deeper to improve accuracy; however, when facing ambiguous or misleading multimodal inputs, deeper reasoning instead leads to more untruthful outputs. This occurs because reasoning models tend to follow Depth-First Search (DFS)—once an initial interpretation is selected, they continue to dig deeper rather than exploring alternative explanations.

**Goal**: (1) Construct the first benchmark for systematically evaluating multimodal truthfulness with human-in-the-loop verification; (2) reveal systematic differences in truthfulness between reasoning models and chat models; (3) develop a reliable automated truthfulness evaluator.

**Key Insight**: Design a three-level hierarchical prompting system (Basic Perception → Inductive Misleading → False Premise Reasoning) to gradually increase reasoning complexity and misleading intensity, thereby finely diagnosing truthfulness performance at different depths.

**Core Idea**: The DFS-style reasoning topology of reasoning models itself (rather than model capacity or training data) is the structural cause of truthfulness degradation. Similar degradation can be reproduced by applying CoT prompting to chat models.

## Method

### Overall Architecture

TruthfulVQA consists of three components: (1) 5,000+ visually misleading images (annotated by 50 contributors), organized into 8 major and 21 sub-categories based on Whaley's deception taxonomy; (2) a three-level hierarchical prompting system to probe truthfulness across perception, inductive reasoning, and false-premise reasoning; (3) the TruthfulJudge evaluation model, fine-tuned on Qwen2.5-VL-7B using a Critique-Label paradigm.

### Key Designs

**1. Three-level Hierarchical Prompting: Identifying Truthfulness Degradation across Reasoning Depths**

Binary and multiple-choice tests only capture surface correctness and fail to reveal truthfulness vulnerabilities hidden in deep reasoning. The authors pair each misleading image with three questions of increasing difficulty. Level 1 (Basic Perception) tests direct visual-semantic recognition, e.g., "How many people are in the image?"; Level 2 (Inductive Misleading) introduces subtle deceptive contextual cues to challenge hypothetical reasoning, e.g., "Approximately how far is the sun from the person's feet?", forcing the model to induce based on seemingly plausible premises; Level 3 (False Premise Reasoning) uses statements that appear factual but are false to build an incorrect narrative, e.g., "Horses have intelligence equivalent to a 5-year-old child... so can a horse play the accordion while sitting?", requiring the model to identify invalid logic. This allows for precise diagnosis of where a model begins to be misled and the magnitude of its degradation.

**2. Logit Advantage Loss (LAL) Metric: Quantifying Decision Erosion by Misleading Prompts via Confidence**

Accuracy alone cannot distinguish between "confident correct answers" and "barely correct answers," nor can it measure how much misleading input shifts a model's decision boundary. LAL first defines the logit advantage of the correct answer $A_i = \ell_i(o^*) - \max_{o \neq o^*} \ell_i(o)$, representing the lead of the correct option over the strongest distractor. It then calculates the difference between levels LAL $= A_i - A_j$, which can be decomposed into "depression of the correct option" and "elevation of the incorrect option." To eliminate arbitrary logit scaling factors across different models, a normalized version $A_i^{\text{norm}}$ is used for cross-model comparison. This metric transforms the intuition that "reasoning models are more confidently wrong under misleading input" into measurable data—reasoning models systematically show higher LAL than their chat counterparts.

**3. TruthfulJudge Evaluation Model: Specialized Evaluator over Biased Manual Assessment**

General-purpose MLLMs are unreliable as judges, with accuracies of only 52–64% and a tendency to accept approximately one-third of hallucinated responses. The authors fine-tuned Qwen2.5-VL-7B on 7.1k human-annotated QA pairs (including explanatory critiques and preference labels) using a Critique-Label paradigm: the model first generates a critical analysis and then provides a preference label instead of a direct score. Comparing Bradley-Terry, Critique-Score, and Pure-Label paradigms, Critique-Label performed significantly better (88.4% vs 57.5%), with Cohen's $\kappa = 0.79$ (near perfect agreement), FPR $= 0.12$, and ECE $= 0.11$. Providing reasons before making a judgment forces the judge to lay out evidence, making it harder to overlook a fluent but untruthful response.

### Loss & Training

TruthfulJudge uses Supervised Fine-Tuning (SFT). The training data consists of 7.1k high-quality critique-label pairs generated by GPT-4o via prompt engineering, validated by human-annotated truthfulness and preference labels. The test set contains 812 samples.

## Key Experimental Results

### Main Results

**Average Accuracy of 50+ Models on TruthfulVQA**

| Level | Avg. Accuracy | Description |
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
| Chat Model + CoT Prompt | Decrease of 2.8-8.3 pp | Proves DFS topology is the cause of degradation |
| Chat Model ECE | 0.16-0.25 | Better calibration |
| Reasoning Model ECE | >0.25 | Overconfident |
| Qwen2.5-VL-72B ECE | 0.188 | Chat version |
| QVQ-72B ECE | 0.325 | Reasoning version, significantly worse calibration |

### Key Findings

- Inverse Scaling Law: Reasoning models consistently underperform their corresponding chat versions within the same series; larger reasoning models do not guarantee better performance.
- DFS vs BFS: Reasoning models tend toward DFS (digging deep once an initial interpretation is selected), while chat models are closer to BFS (exploring multiple paths before concluding).
- Causal Verification: Applying CoT prompts (forcing serialized reasoning) to chat models led to a 2.8-8.3 percentage point degradation across 5 models, with failure modes matching reasoning models. This proves the vulnerability stems from reasoning topology rather than the models themselves.
- General judge models (GPT-4o, Gemini, etc.) perform poorly in truthfulness evaluation (52-64% accuracy), while TruthfulJudge reaches 88.4%.

## Highlights & Insights

- The discovery of the "Inverse Scaling Law" serves as a critical warning—reasoning models may be more dangerous in safety-critical scenarios than simpler models because they confidently fabricate details to support incorrect reasoning.
- The DFS vs. BFS analysis provides a clear mechanistic explanation rather than just empirical observation. Causal experiments (CoT → degradation) further exclude confounding factors.
- The Critique-Label paradigm of TruthfulJudge is transferable—generating analysis before judgment is more reliable than direct scoring.

## Limitations & Future Work

- The dataset size (5,000+) is relatively small compared to commercial benchmarks.
- Cultural homogeneity of the annotation team may introduce bias.
- The 8-category untruthfulness taxonomy may not cover the full spectrum of visual-semantic deception.
- Future work should develop BFS-inspired reasoning mechanisms to balance reasoning depth with truthfulness.

## Related Work & Insights

- **vs CHAIR/MME-Hallucination**: These focus on hallucinations under benign inputs, whereas TruthfulVQA focuses on truthfulness under adversarial inputs.
- **vs MultiTrust**: MultiTrust unifies seven-stage evaluation but remains primarily multiple-choice; TruthfulVQA provides deeper probing via hierarchical prompts.
- **vs LLM-as-Judge**: This paper provides empirical evidence that general MLLMs are unreliable as truthfulness judges and require specialized training.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to systematically reveal the inverse scaling law of truthfulness in reasoning models; the DFS/BFS analysis framework offers theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 50+ model evaluations, causal verification, and a specialized judge model; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis, though some notation is dense.
- Value: ⭐⭐⭐⭐⭐ Significant warning for safety research on reasoning models; both the benchmark and evaluator have lasting utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/vlm_reasoning/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[ICLR 2026\] Unleashing Perception-Time Scaling to Multimodal Reasoning Models](../../ICLR2026/vlm_reasoning/unleashing_perception-time_scaling_to_multimodal_reasoning_models.md)
- [\[CVPR 2026\] UniT: Unified Multimodal Chain-of-Thought Test-time Scaling](../../CVPR2026/vlm_reasoning/unit_unified_multimodal_chain-of-thought_test-time_scaling.md)
- [\[CVPR 2026\] When Visualizing is the First Step to Reasoning: MIRA, a Benchmark for Visual Chain-of-Thought](../../CVPR2026/vlm_reasoning/when_visualizing_is_the_first_step_to_reasoning_mira_a_benchmark_for_visual_chai.md)
- [\[NeurIPS 2025\] When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](../../NeurIPS2025/vlm_reasoning/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)

</div>

<!-- RELATED:END -->
