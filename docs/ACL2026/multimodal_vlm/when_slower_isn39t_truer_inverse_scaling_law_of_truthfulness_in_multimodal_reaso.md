---
title: >-
  [Paper Note] When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Reasoning] This paper discovers an "inverse scaling law" in multimodal reasoning models: slow-thinking (reasoning) models are more prone to generating untruthful outputs than fast-th…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Reasoning"
  - "Truthfulness Evaluation"
  - "Inverse Scaling Law"
  - "Depth-First Reasoning"
  - "Hallucination Detection"
date: 2026-05-08
content_hash: 433d4f4637bc369f
---

# When Slower Isn't Truer: Inverse Scaling Law of Truthfulness in Multimodal Reasoning

**Conference**: ACL 2026  
**arXiv**: [2505.20214](https://arxiv.org/abs/2505.20214)  
**Code**: [https://truthfulvqa.github.io](https://truthfulvqa.github.io)  
**Area**: Multimodal VLM / AI Safety  
**Keywords**: Multimodal Reasoning, Truthfulness Evaluation, Inverse Scaling Law, Depth-First Reasoning, Hallucination Detection

## TL;DR

This paper discovers an "inverse scaling law" in multimodal reasoning models: slow-thinking (reasoning) models are more prone to generating untruthful outputs than fast-thinking (chat) models when faced with misleading visual inputs. The authors construct the TruthfulVQA benchmark (5,000+ samples, 50 annotators, three-level hierarchical prompting) and the TruthfulJudge evaluation model (88.4% accuracy) to systematically diagnose this phenomenon.

## Background & Motivation

**Background**: Multimodal Large Language Models (MLLMs) have made significant progress in visual understanding tasks. Reasoning models (e.g., QVQ, Mulberry) have achieved breakthroughs in structured tasks like mathematics and coding through longer reasoning chains. While hallucination issues have been widely studied, research primarily focuses on unintentional errors under benign inputs.

**Limitations of Prior Work**: (1) Truthfulness and hallucination are related but distinct concepts—the former emphasizes robustness in maintaining factual loyalty under adversarial or misleading inputs, while the latter focuses on unintentional fabrication under benign inputs; (2) Existing benchmarks primarily use binary or multiple-choice tests, which fail to probe the connection between deep reasoning and truthfulness; (3) AI-as-Judge evaluations suffer from systematic biases, potentially allowing untruthfulness to escape detection.

**Key Challenge**: Reasoning models are designed to "think" more deeply to improve accuracy, but when faced with ambiguous or misleading multimodal inputs, deeper reasoning leads to more untruthful outputs. This occurs because reasoning models tend to follow Depth-First Search (DFS)—once an initial interpretation is selected, they continue to dig deeper rather than exploring alternative explanations.

**Goal**: (1) Construct the first benchmark for systematically evaluating multimodal truthfulness with human-in-the-loop verification; (2) Reveal systemic differences in truthfulness between reasoning models and chat models; (3) Develop a reliable automated truthfulness evaluator.

**Key Insight**: A three-level hierarchical prompting system (Basic Perception $\rightarrow$ Inductive Misleading $\rightarrow$ False Premise Reasoning) is designed to gradually increase reasoning complexity and misleading intensity, enabling a fine-grained diagnosis of model truthfulness at different depths.

**Core Idea**: The DFS-style reasoning topology of reasoning models itself (rather than model capacity or training data) is the structural cause of truthfulness degradation—a similar decline can be reproduced by applying CoT prompting to chat models.

## Method

### Overall Architecture

TruthfulVQA consists of three components: (1) 5,000+ visually misleading images (annotated by 50 contributors), organized into 8 major categories and 21 subcategories based on Whaley’s deception taxonomy; (2) A three-level hierarchical prompting system to systematically probe truthfulness under perception, inductive reasoning, and false-premise reasoning; (3) The TruthfulJudge evaluation model, fine-tuned on Qwen2.5-VL-7B using the Critique-Label paradigm.

### Key Designs

1.  **Three-level Hierarchical Prompting System**:
    - **Function**: Systematically evaluate model truthfulness across different reasoning depths and misleading intensities.
    - **Mechanism**: Level 1 (Basic Perception) tests direct visual-semantic recognition, e.g., "How many people are in the image?"; Level 2 (Inductive Misleading) introduces subtle deceptive contextual cues to challenge hypothetical reasoning, e.g., "Approximately how far is the sun from the person's feet?"; Level 3 (False Premise Reasoning) uses false but fact-like statements to construct an erroneous narrative, requiring the model to identify invalid logic, e.g., "Horses have intelligence equivalent to a 5-year-old... so can a horse sit and play the accordion?"
    - **Design Motivation**: Binary and multiple-choice tests only capture surface-level correctness. Hierarchical design provides a more granular diagnosis of truthfulness vulnerabilities in deep reasoning.

2.  **Logit Advantage Loss (LAL) Metric**:
    - **Function**: Quantify the impact of misleading prompts on model decision-making at the confidence level.
    - **Mechanism**: The logit advantage of the correct answer is defined as $A_i = \ell_i(o^*) - \max_{o \neq o^*} \ell_i(o)$. Inter-level LAL is then calculated as $A_i - A_j$, which can be decomposed into the degradation of the correct option and the amplification of incorrect options. A normalized version $A_i^{\text{norm}}$ eliminates arbitrary scaling factors across different models.
    - **Design Motivation**: Accuracy alone cannot distinguish between "high-confidence correct" and "low-confidence correct." LAL reveals how misleading inputs shift internal decision boundaries.

3.  **TruthfulJudge (Reliable Evaluator)**:
    - **Function**: Replace expensive human evaluation for truthfulness assessment.
    - **Mechanism**: Fine-tuned on Qwen2.5-VL-7B using 7.1k human-annotated question-answer pairs (including explanatory critiques and preference labels). It employs the Critique-Label paradigm: the model first generates a critical analysis before providing a preference label. Compared to Bradley-Terry, Critique-Score, and Pure-Label paradigms, Critique-Label is significantly superior (88.4% vs. 57.5%). Cohen's $\kappa=0.79$ (approaching "almost perfect agreement"), with FPR=0.12 and ECE=0.11.
    - **Design Motivation**: General MLLMs as judges achieve only 52-64% accuracy and systematically accept about 1/3 of hallucinated answers. A specialized evaluator is needed.

### Loss & Training

TruthfulJudge uses Supervised Fine-Tuning (SFT) on 7.1k high-quality critique-label pairs generated by GPT-4o via prompt engineering and verified through human-labeled truthfulness and preference tags. The test set contains 812 samples.

## Key Experimental Results

### Main Results

**Average Accuracy of 50+ Models on TruthfulVQA**

| Level | Average Accuracy | Description |
| :--- | :--- | :--- |
| Level 1 (Basic Perception) | 81.85% | Direct visual recognition |
| Level 2 (Inductive Misleading) | 55.37% | 26.5 percentage point drop |
| Level 3 (False Premise) | 44.96% | Further 10.4 percentage point drop |

**LAL Comparison: Reasoning Models vs. Chat Models**

| Model Pair | Chat LAL | Reasoning LAL |
| :--- | :--- | :--- |
| Qwen2.5-VL vs. QVQ-72B | Lower | 0.89 (significantly higher) |
| Qwen2-VL vs. Mulberry-7B | Lower | 0.71 |
| Kimi-VL-A3B vs. Thinking | Lower | 0.53 |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| Chat Model + CoT Prompt | 2.8-8.3% decrease | Proves DFS topology causes degradation |
| Chat Model ECE | 0.16-0.25 | Better calibration |
| Reasoning Model ECE | >0.25 | Overconfident |
| Qwen2.5-VL-72B ECE | 0.188 | Chat version |
| QVQ-72B ECE | 0.325 | Reasoning version, significantly worse calibration |

### Key Findings

- **Inverse Scaling Law**: Reasoning models consistently underperform their corresponding chat equivalents within the same series; larger reasoning models do not guarantee better truthfulness.
- **DFS vs. BFS**: Reasoning models favor DFS (digging deep once an initial explanation is chosen), while chat models behave more like BFS (exploring multiple paths before concluding).
- **Causal Verification**: When CoT prompting (forced serialized reasoning) is applied to chat models, 5 models showed a 2.8-8.3% performance drop, with failure modes matching reasoning models. This proves the vulnerability stems from the reasoning topology, not the model weights themselves.
- General judge models (GPT-4o, Gemini, etc.) perform poorly in truthfulness evaluation (52-64% accuracy), whereas TruthfulJudge reaches 88.4%.

## Highlights & Insights

- The discovery of the "Inverse Scaling Law" serves as a critical warning—reasoning models can be more dangerous in safety-critical scenarios than simpler models because they more confidently fabricate details to support erroneous reasoning.
- The DFS vs. BFS analysis provides a clear mechanistic explanation beyond empirical observation. Causal experiments (CoT $\rightarrow$ degradation) further rule out confounding factors.
- The Critique-Label paradigm of TruthfulJudge is transferable—generating analysis before judgment is more reliable than direct scoring.

## Limitations & Future Work

- The dataset size (5,000+) is relatively small compared to commercial benchmarks.
- Homogeneity in the annotation team's culture may introduce bias.
- The 8-category untruthfulness taxonomy may not cover the full spectrum of visual-semantic deception.
- Future work should develop BFS-inspired reasoning mechanisms to balance reasoning depth with truthfulness.

## Related Work & Insights

- **vs. CHAIR/MME-Hallucination**: These focus on hallucinations under benign inputs; TruthfulVQA focuses on truthfulness under adversarial inputs.
- **vs. MultiTrust**: While MultiTrust unifies seven-stage evaluation, it remains primarily multiple-choice. TruthfulVQA provides deeper hierarchical probing.
- **vs. LLM-as-Judge**: This work empirically proves that general MLLMs are unreliable as truthfulness judges, necessitating specialized training.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First systematic revelation of the inverse scaling law in reasoning models; DFS/BFS framework offers theoretical depth.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Assessment of 50+ models, causal verification, and a specialized judge model.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and deep analysis, though some notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ Critical implications for the safety of reasoning models; the benchmark and evaluator possess long-term utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[NeurIPS 2025\] When One Modality Sabotages the Others: A Diagnostic Lens on Multimodal Reasoning](../../NeurIPS2025/multimodal_vlm/when_one_modality_sabotages_the_others_a_diagnostic_lens_on_multimodal_reasoning.md)
- [\[CVPR 2026\] Scaling Spatial Intelligence with Multimodal Foundation Models](../../CVPR2026/multimodal_vlm/scaling_spatial_intelligence_with_multimodal_foundation_models.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[ICCV 2025\] Scaling Laws for Native Multimodal Models](../../ICCV2025/multimodal_vlm/scaling_laws_for_native_multimodal_models.md)

</div>

<!-- RELATED:END -->
