---
title: >-
  [Paper Note] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects
description: >-
  [ACL 2026][Interpretability][Content effects] Through representational analysis, this study reveals that the concepts of "logical validity" and "plausibility" are highly aligned in the latent space of LLMs. This alignmen…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Content effects"
  - "logical validity"
  - "plausibility"
  - "linear representations"
  - "steering vectors"
date: 2026-05-08
content_hash: 88d6ff0c995b4287
---

# How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects

**Conference**: ACL 2026 Findings  
**arXiv**: [2510.06700](https://arxiv.org/abs/2510.06700)  
**Code**: [https://github.com/leobertolazzi/content-effect-interpretability](https://github.com/leobertolazzi/content-effect-interpretability)  
**Area**: Social Computing  
**Keywords**: Content effects, logical validity, plausibility, linear representations, steering vectors

## TL;DR
Through representational analysis, this study reveals that the concepts of "logical validity" and "plausibility" are highly aligned in the latent space of LLMs. This alignment leads models to conflate plausibility with validity (content effect). By constructing debiasing steering vectors, these two concepts can be effectively decoupled, reducing content effects while improving reasoning accuracy.

## Background & Motivation

**Background**: Humans exhibit a "content effect" in logical tasks like syllogistic reasoning, where the plausibility of semantic content influences the judgment of logical validity (e.g., invalid arguments with plausible conclusions are often misjudged as valid). This human phenomenon is explained by dual-process theories (fast intuitive system vs. slow analytical system). Recent research indicates that LLMs also exhibit similar content effects.

**Limitations of Prior Work**: While the behavioral content effects of LLMs have been well-documented, the underlying mechanisms remain unclear. Existing studies have focused on behavioral observations without in-depth analysis of the internal representations within LLMs.

**Key Challenge**: Logical validity should depend solely on argument structure rather than content; however, LLMs may entangle these two independent concepts within their representation space.

**Goal**: (1) Verify the presence of content effects in LLMs; (2) Analyze how validity and plausibility are encoded in internal representations; (3) Investigate whether representational entanglement predicts behavioral content effects; (4) Design interventions to decouple these concepts.

**Key Insight**: Based on the linear representation hypothesis—which suggests high-level concepts are linearly encoded in LLM latent spaces—this study tests whether the linear directions of validity and plausibility are highly similar.

**Core Idea**: The root of the content effect in LLMs is the entangled alignment of validity and plausibility directions in the representational geometry, which can be addressed by constructing debiasing steering vectors to decouple them.

## Method

### Overall Architecture
The study evaluates 10 LLMs (Qwen-2.5, Qwen-3, Gemma-3 series) on 1,280 syllogisms. Linear directions for validity and plausibility are extracted using the Difference-in-Means method. Their similarity is analyzed, causal relationships are verified through cross-task steering experiments, and debiasing vectors are finally constructed to reduce content effects.

### Key Designs

1. **Concept Direction Extraction (Difference-in-Means)**:
    - **Function**: Represents a binary concept as a single direction in the latent space.
    - **Mechanism**: For each layer $l$, the difference between the average activation vectors at the last token position for samples predicted as positive (e.g., "valid") and negative (e.g., "invalid") is calculated: $v_{\text{concept}}^l = \mu_{\text{positive}}^l - \mu_{\text{negative}}^l$. The model's own predicted labels are used instead of ground truth to focus on the model's internal "belief" encoding.
    - **Design Motivation**: The Difference-in-Means method is concise, effective, and directly aligns with the linear representation hypothesis.

2. **Cross-Task Steering Experiments**:
    - **Function**: Verifies the causal interaction between validity and plausibility—determining if plausibility vectors can influence validity judgments and vice versa.
    - **Mechanism**: Steering vectors $v_{\text{plausibility}}^l$ extracted from plausibility tasks are applied to logical validity classification tasks (and vice versa) to measure steering strength (label flip ratio). Steering is always performed against the model's original prediction: adding the vector if predicted as negative, and subtracting it if predicted as positive.
    - **Design Motivation**: If plausibility vectors can effectively alter validity judgments, it suggests causal entanglement in the representation space rather than mere correlation.

3. **Debiasing Steering Vector Construction**:
    - **Function**: Decouples representations of validity and plausibility to mitigate content effects.
    - **Mechanism**: Construct debiasing vectors to ensure the model's evaluation of logical validity is independent of plausibility. These are applied to effective layers (steering strength $>0.75$) to decrease the content effect metric (CE) while simultaneously increasing reasoning accuracy.
    - **Design Motivation**: If entanglement is the source of the content effect, then decoupling should simultaneously reduce bias and enhance reasoning capabilities.

### Indicator Design
Content Effect CE = $\frac{1}{2}(\Delta_{v^+} + \Delta_{v^-})$, where $\Delta_{v^+}$ measures the accuracy advantage of valid arguments when the conclusion is plausible. CE=0 represents independence of validity and plausibility, while CE=1 indicates that predictions are entirely driven by plausibility.

## Key Experimental Results

### Main Results
Behavioral Content Effects:

| Model | Setting | $D_{v^+,p^+}$ Acc | $D_{v^-,p^+}$ Acc | $D_{v^+,p^-}$ Acc | CE |
|------|------|---------------------|---------------------|---------------------|-----|
| Qwen2.5-32B | 0-shot | 100.00 | 67.50 | 60.92 | 0.348 |
| Qwen2.5-32B | CoT | 98.67 | 86.64 | 93.10 | 0.096 |
| Qwen3-14B | 0-shot | 97.33 | 90.83 | 60.92 | 0.213 |
| Qwen3-14B | CoT | 95.31 | 99.10 | 92.50 | 0.014 |

### Representational Analysis

| Concept Pair | Avg Cosine Similarity | Description |
|--------|--------------|------|
| Validity - Plausibility | 0.48-0.64 | Highly aligned |
| Validity - Harmlessness | 0.10-0.13 | Low similarity (Control) |
| Validity - Hypernymy | -0.12 to -0.17 | Low similarity (Control) |

### Key Findings
- All tested models exhibit content effects; CoT prompting significantly reduces CE (from 0.213-0.348 to 0.014-0.096).
- The cosine similarity between validity and plausibility vectors (0.48-0.64) is significantly higher than that of control concepts (0.10-0.13), confirming specific entanglement.
- Cross-task steering was successful: plausibility vectors can effectively flip validity judgments and vice versa.
- The degree of validity-plausibility alignment positively correlates with the strength of the behavioral CE.
- Debiasing vectors simultaneously reduce CE and improve reasoning accuracy, proving that decoupling is effective.
- While CoT reduces behavioral CE, the degree of alignment at the representational level does not change significantly ($p=0.625$).

## Highlights & Insights
- Provides the first representational level explanation for content effects in LLMs—it is not a behavioral "bug" but a structural issue in representational geometry. This finding is much deeper than purely behavioral research.
- The observation that CoT reduces behavioral CE without changing representation alignment is particularly interesting—CoT may "bypass" rather than "solve" the entanglement issue during the reasoning process.
- The use of debiasing steering vectors as an intervention method demonstrates a complete closed loop from representational analysis to practical improvement.

## Limitations & Future Work
- Validation was only performed on syllogistic reasoning; content effect mechanisms in other reasoning forms (conditional, probabilistic) may differ.
- The dataset size is relatively small (1,280 syllogisms); while it covers all 64 types, semantic variation is limited.
- The effectiveness of debiasing vectors depends on layer selection; a validation set is needed to identify optimal layers.
- Future work could explore whether similar entanglement phenomena exist for concepts encoded non-linearly.

## Related Work & Insights
- **vs Lampinen et al.**: While they documented behavioral content effects in LLMs, this work reveals the mechanism from a representational perspective.
- **vs Marks & Tegmark (Truth Directions)**: They found that truth values are encoded linearly; this work further discovers that validity directions are entangled with plausibility directions.
- **vs Arditi et al. (Refusal Directions)**: Uses a similar methodology but applies it to different concepts; the innovation here lies in analyzing the interaction between two concepts rather than a single one.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to explain LLM content effects via representational geometry, providing deep insights.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ten models, control experiments, and causal validation are used, though the dataset type remains narrow.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear RQ-driven structure with progressive analysis.
- Value: ⭐⭐⭐⭐⭐ Offers important insights into understanding and improving logical reasoning in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)
- [\[ICML 2026\] How Language Models Process Negation](../../ICML2026/interpretability/how_language_models_process_negation.md)
- [\[ICML 2026\] A Behavioural and Representational Evaluation of Goal-Directedness in Language Model Agents](../../ICML2026/interpretability/a_behavioural_and_representational_evaluation_of_goal-directedness_in_language_m.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](../../ICML2026/interpretability/query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
