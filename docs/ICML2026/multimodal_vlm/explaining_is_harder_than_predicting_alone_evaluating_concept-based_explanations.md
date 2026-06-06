---
title: >-
  [Paper Note] Explaining Is Harder than Predicting Alone: Evaluating Concept-Based Explanations of MLLMs as ICL Visual Classifiers
description: >-
  [ICML 2026][Multimodal VLM][Concept-based explanation] The authors conducted 2,080 ICL classification experiments on four SOTA MLLMs using a five-level ladder of increasingly formalized explanation conditions (from bare…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Concept-based explanation"
  - "Description Logics"
  - "LLM-as-a-judge"
  - "Few-shot ICL"
  - "XAI evaluation"
date: 2026-05-08
content_hash: a9960ae64e57fe32
---

# Explaining Is Harder than Predicting Alone: Evaluating Concept-Based Explanations of MLLMs as ICL Visual Classifiers

**Conference**: ICML 2026  
**arXiv**: [2605.28215](https://arxiv.org/abs/2605.28215)  
**Code**: To be confirmed  
**Area**: Interpretability / Multimodal VLM / In-Context Learning  
**Keywords**: Concept-based explanation, Description Logics, LLM-as-a-judge, Few-shot ICL, XAI evaluation

## TL;DR
The authors conducted 2,080 ICL classification experiments on four SOTA MLLMs using a five-level ladder of increasingly formalized explanation conditions (from bare classification to Description Logic axioms) and an LLM-as-a-judge pipeline evaluating 9 XAI metrics. The study reveals that "forcing the model to generate more formal concept explanations leads to a monotonic decline in classification accuracy (93.8% → 90.1%)", and "local discriminativeness" is identified as the only explanation quality dimension significantly correlated with accuracy.

## Background & Motivation

**Background**: MLLMs paired with few-shot In-Context Learning (ICL) can perform image classification without weight updates. The predominant "explanation" method is Chain-of-Thought (CoT) prompting, where the model vocalizes its reasoning steps.

**Limitations of Prior Work**: (1) CoT text does not equal true internal reasoning—Barez et al. (2025) proved CoT trajectories may not reflect internal computations, and Turpin et al. (2023) noted models often provide "plausible but misleading" post-hoc rationales. (2) ICL literature focuses almost exclusively on classification accuracy, lacking formalized, machine-verifiable evaluation of explanation quality. (3) Neuro-symbolic approaches (e.g., logic-explained networks) rely on supervised training and cannot evaluate whether "frozen MLLMs themselves can generate symbolic explanations."

**Key Challenge**: There is a mismatch between "natural language fluency" and "concept verifiability" in explanations—the latter is what XAI truly requires, yet the former dominates current evaluations.

**Goal**: Under a well-controlled few-shot image classification setting, this paper systematically addresses two questions: (1) Can frozen MLLMs spontaneously produce explanations following conceptual and formal requirements? (2) Does the requirement for explanations conversely damage the classification itself?

**Key Insight**: Image classification is used as a "concept anchor"—visual features can be directly verified against the query image, bringing "concepts" back from linguistic abstractions to visual evidence. Designing explanation requirements as a "five-level formalization ladder" allows for isolating the marginal effects of "incrementally increasing complexity" on the same dataset.

**Core Idea**: Concept explanations are re-formalized as machine-verifiable artifacts like Description Logics (DL) axioms. These are quantified using an independent judge and 9 XAI dimensions to answer whether demanding an explanation drags down prediction performance.

## Method

### Overall Architecture
Task Setting: $N$-way $K$-shot image classification. Given a support set $\mathcal{S}=\{(x_i, y_i)\}_{i=1}^{N\times K}$ and a query image $x_q$, the frozen MLLM observes examples in context and outputs a predicted class $\hat{y}_q \in \mathcal{Y}$. Simultaneously, it produces a structured explanation based on a specific "explanation condition." The system prompt enforces three constraints: (i) use only observable visual evidence from the query image; (ii) prohibit external world knowledge/hypotheses; (iii) the final class label must be placed within `<response>` XML tags and copied verbatim from the candidate list for deterministic parsing. The judge (gpt-5-thinking-mini) receives the query image, candidate labels, model output, and explanation condition descriptions to score 9 metrics on a 1–5 scale. The judge is **not shown the support set images**, ensuring a zero-shot evaluation.

### Key Designs

1.  **Five-Level Formalization Ladder (E1–E5)**:
    -   **Function**: Uses a set of increasingly strict explanation requirements to treat "explanation complexity" as an isolatable experimental variable.
    -   **Mechanism**: E1 outputs only the label in `<response>` (accuracy baseline); E2 adds an `<explanation>` tag with a short natural language explanation (standard CoT style); E3 lists minimal sufficient observable visual features in `<features>` (bulleted noun phrases); E4 lists features, induces IF-THEN rules from examples in `<kb>`, and identifies the best-matching rule in `<rule_check>`; E5 uses DL form: `<tbox>` defines concept axioms with `hasVisualFeature` roles, `<abox>` contains property assertions for the query, and `<dl_explanation>` derives the predicted class.
    -   **Design Motivation**: Previous XAI evaluations were either incomparable across prompt styles or only measured free-form text. Making five conditions a "monotonic ladder of complexity" directly measures the marginal cost of "formalization."

2.  **LLM-as-a-judge + 9-dimensional XAI Metrics**:
    -   **Function**: Expands "explanation quality" from "readability" to 9 independently scorable conceptual dimensions.
    -   **Mechanism**: The judge scores (1–5): Textual Groundedness (TG), Hallucination Free (HF), Concept Counting (CC), Comprehensibility (CP), Conciseness (Cn), Specificity (S), Local Discriminativeness (LD), Instruction Following (IF), and Logical Coherence (LC). The judge relies on its prior knowledge of candidate classes for the LD metric without seeing support images.
    -   **Design Motivation**: A single quality score cannot distinguish between "verbose but accurate" and "concise but irrelevant" failures. This 9-dimensional breakdown identifies specific failure modes in DL axioms and enables correlation analysis with accuracy.

3.  **Reproducible Experimental Grid + Balanced Statistical Design**:
    -   **Function**: Ensures cross-model and cross-condition comparisons satisfy non-parametric significance testing requirements.
    -   **Mechanism**: 4 datasets (CIFAR-10 / DTD / Flowers102 / Pets) × 4 models (Gemini 2.5 Flash, Gemma 4 26B, Qwen3 VL 8B, LLaMA 4 Scout) × 5 conditions (E1–E5) × 6 $(N,K)$ configurations = 2,080 runs. Query size $Q=1$ ensures independent samples for McNemar / Wilcoxon / Friedman tests. Repetitions are balanced such that $\text{Reps}\times N=12$ to avoid oversampling small $N$ configurations. All models/conditions share a fixed seed for support set sampling to eliminate noise.
    -   **Design Motivation**: Previous work often suffered from small sample sizes or inconsistent support sets across models. This work elevates "statistical design" to the same importance as the methodology.

### Loss & Training
Frozen MLLMs with no gradient updates. All models accessed via OpenRouter API with temperature $T=0$ to ensure deterministic output.

## Key Experimental Results

### Main Results
Average accuracy (%) aggregated by explanation condition × model (104 observations per cell):

| XAI Condition | Gemini 2.5F | Gemma 4 | Qwen3 VL | LLaMA 4 |
| :--- | :--- | :--- | :--- | :--- |
| E1 — Classification only | **96.9** | 94.4 | 95.1 | 88.5 |
| E2 — NLE | 97.2 | 94.1 | 92.7 | 90.3 |
| E3 — Features | 96.9 | 93.1 | 93.8 | 88.5 |
| E4 — Feature-value pairs | 95.8 | 94.4 | 92.4 | 86.5 |
| E5 — DL Axioms | 96.2 | 92.4 | **83.0** | 88.9 |

The overall mean is 92.6%. Accuracy monotonically decreases from E1 to E5 (93.8% → 90.1%), with Qwen3 VL 8B showing the sharpest drop (−12.1 pp), while Gemini 2.5 Flash remains stable.

### Ablation Study
Judge scores for 9 explanation quality metrics across 4 explanation conditions (Mean, best in bold):

| Condition | TG | HF | CC | CP | Cn | S | LD | IF | LC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NLE | 3.62 | 4.46 | **4.68** | 4.95 | 4.81 | 3.73 | 3.69 | 4.70 | 4.84 |
| Features | 3.62 | **4.81** | **4.68** | **4.99** | **4.97** | 3.81 | 3.62 | **4.82** | **4.89** |
| Feature-value pairs | **3.70** | 4.77 | 4.37 | 4.92 | 4.95 | **4.14** | **3.91** | 4.43 | 4.72 |
| DL Axioms | 2.31 | 4.40 | 4.20 | 3.97 | 4.94 | 2.85 | 3.10 | 3.05 | 2.97 |

### Key Findings
-   **Stronger formalization degrades classification**: E5 (DL axioms) pulled the overall mean down from 93.8% to 90.1%, contradicting the default assumption that explicit reasoning is generally beneficial.
-   **DL axioms collapse on 5 dimensions**: TG (2.31), Specificity (2.85), LD (3.10), IF (3.05), and LC (2.97) are significantly lower than other conditions—MLLMs can write syntactically correct axioms (HF=4.40, Cn=4.94) but struggle to anchor them to visual evidence that actually discriminates classes.
-   **Increasing support shots from $K=1$ to $K=5$ improves accuracy by 7.0 pp ($p=2.0\times 10^{-13}$)**. Among the 9 metrics, only LD increases significantly ($\Delta=+0.26$), suggesting more examples primarily help find discriminative features.
-   **Increasing class count $N$ monotonically decreases accuracy**, with LD being the only metric to drop significantly (3.86 at $N=2$ → 3.40 at $N=4$), identifying LD as the most sensitive dimension under pressure.
-   **LD is the only explanation quality metric significantly correlated with accuracy** (Spearman, after Bonferroni correction for 36 tests)—the other 8 dimensions do not predict classification correctness.

## Highlights & Insights
-   **Quantification of "Explanation Cost"**: While CoT/explanations are often assumed to be "free" or beneficial, this paper provides a reproducible cost (E1→E5 accuracy drop of 3.7 pp), warning against sacrificing predictive power for formal explanations.
-   **Diagnosis of DL Axiom Failure**: The failure is not "syntactic" (IF=3.05 is not a complete collapse; HF/Cn are high) but "semantic"—models can write structures but lack *discriminative* content. This points towards instruction tuning specifically for DL axioms rather than just increasing compute.
-   **LD as a Proxy for "Explanation Utility"**: Since LD is the only dimension correlated with accuracy, future XAI evaluations should prioritize LD as a core KPI rather than accumulating unrelated metrics.
-   **Transferable Task Design**: The paradigm of using ICL and multi-prompting to probe frozen model capabilities is a structural contribution that can be extended to other verifiable tasks like counting or spatial relations.

## Limitations & Future Work
-   The judge (gpt-5-thinking-mini) does not see support images; its LD judgment depends on its own priors. For fine-grained categories unfamiliar to the judge, LD scores might be noisy.
-   The study covers 4 standard visual datasets (CIFAR-10/DTD/Flowers/Pets) but does not explore high-stakes domains like medicine or satellite imagery where formal explanations are critical.
-   Prompts for the 5 conditions are fixed templates; low E5 scores might partially result from insufficient prompt engineering. Prompt robustness remains to be tested.
-   All experiments use $Q=1$ for statistical independence, leaving the question of explanation consistency across multiple queries for the same support set unanswered.
-   Lack of human evaluation makes it difficult to determine if the judge's 9-dimensional scores align with human intuition, especially for subjective dimensions like Comprehensibility.

## Related Work & Insights
-   **vs. Barez et al. (2025) on CoT Unreliability**: This paper provides quantitative support—standard NLE (E2) is not poor in LD (3.69), but more formalized structures (E5) are where the collapse occurs.
-   **vs. Neuro-symbolic / Logic-explained Networks**: Those rely on supervised learning of axioms. Ours proves that frozen MLLMs produce syntactically valid but semantically weak axioms, suggesting fine-tuning is required.
-   **vs. Liu et al. (2025) "CoT Decreases Accuracy"**: This paper provides stronger evidence in a multimodal + formal ladder context—not just CoT, but increasingly formal requirements cause larger drops.
-   **vs. Polignano et al. (2024) XAI Survey**: Ours responds to the call for systematic evaluation frameworks, offering the 9-metric + judge pipeline as a reusable template.

## Rating
-   Novelty: ⭐⭐⭐⭐ The "cost of explanation" is systematically quantified for the first time in multimodal ICL; the 5-level ladder and 9-metric combination is a structural contribution.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 2,080 runs, 4 models × 4 datasets × 5 conditions × 6 configurations, with balanced statistical design and non-parametric tests—rarely seen rigor.
-   Writing Quality: ⭐⭐⭐⭐ Task definitions, conditions, and metrics are clearly explained.
-   Value: ⭐⭐⭐⭐ Challenges the "explanation = good" default in the XAI community and identifies LD as the core KPI for future evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyper-ICL: Attention Calibration with Hyperbolic Anchor Distillation for Multimodal ICL](hyper-icl_attention_calibration_with_hyperbolic_anchor_distillation_for_multimod.md)
- [\[CVPR 2026\] Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](../../CVPR2026/multimodal_vlm/where_mllms_attend_and_what_they_rely_on_explaining_autoregressive_token_generat.md)
- [\[ICCV 2025\] SparseMM: Head Sparsity Emerges from Visual Concept Responses in MLLMs](../../ICCV2025/multimodal_vlm/sparsemm_head_sparsity_emerges_from_visual_concept_responses_in_mllms.md)
- [\[CVPR 2026\] When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs](../../CVPR2026/multimodal_vlm/when_token_pruning_is_worse_than_random_understanding_visual_token_information_i.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)

</div>

<!-- RELATED:END -->
