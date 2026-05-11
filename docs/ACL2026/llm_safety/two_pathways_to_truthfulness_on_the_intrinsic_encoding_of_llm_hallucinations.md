---
title: >-
  [Paper Note] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations
description: >-
  [ACL 2026][LLM Safety][Hallucination Detection] This paper identifies two distinct information pathways through which LLMs internally encode truthfulness signals: Question-Anchored (relying on information flow from quest…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Hallucination Detection"
  - "Truthfulness Encoding"
  - "Attention Mechanism"
  - "Information Pathways"
  - "Knowledge Boundary"
date: 2026-05-08
content_hash: 607775f3595d14de
---

# Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations

**Conference**: ACL 2026
**arXiv**: [2601.07422](https://arxiv.org/abs/2601.07422)
**Code**: [https://github.com/RowanWenLuo/llm-truthfulness-pathways](https://github.com/RowanWenLuo/llm-truthfulness-pathways)
**Area**: LLM Safety
**Keywords**: Hallucination Detection, Truthfulness Encoding, Attention Mechanism, Information Pathways, Knowledge Boundary

## TL;DR

This paper identifies two distinct information pathways through which LLMs internally encode truthfulness signals: Question-Anchored (relying on information flow from question to answer) and Answer-Anchored (extracting self-contained evidence from the generated answer itself). Both pathways are closely associated with knowledge boundaries. Building on this finding, the paper proposes two pathway-aware hallucination detection methods—Mixture-of-Probes and Pathway Reweighting—achieving AUC improvements of up to 10%.

## Background & Motivation

**Background**: LLMs frequently produce hallucinations—outputs that appear plausible but are factually incorrect. Prior work has demonstrated that LLM internal representations encode rich truthfulness signals detectable via linear probes. However, the origins and underlying mechanisms of these signals remain poorly understood.

**Limitations of Prior Work**: Existing internal probing methods treat all samples as homogeneous and apply a single probe to detect all hallucinations. In practice, truthfulness signals may arise through different mechanisms for different samples, and a unified approach yields suboptimal performance.

**Key Challenge**: Saliency analysis reveals a bimodal distribution in the importance of question-to-answer information flow: a subset of samples relies heavily on question information, while another subset does not. This suggests the existence of two fundamentally distinct truthfulness encoding mechanisms.

**Goal**: (1) Validate and disentangle the two truthfulness pathways; (2) characterize their emergent properties; (3) leverage pathway distinction to improve hallucination detection performance.

**Key Insight**: Two causal intervention techniques—attention knockout and token patching—are employed to disentangle and validate the two pathways.

**Core Idea**: Truthfulness signals arise through two independent pathways. The Q-Anchored pathway relies on information flow from the question to the answer and is associated with facts within the model's knowledge scope, while the A-Anchored pathway extracts self-contained evidence from the generated text itself and is associated with long-tail facts beyond the model's knowledge boundary.

## Method

### Overall Architecture

The approach proceeds in three stages: (1) saliency analysis reveals the bimodal distribution and motivates the two-pathway hypothesis; (2) attention knockout and token patching are used to validate the hypothesis; (3) the properties of each pathway (knowledge boundary association, self-awareness capacity) are characterized, and pathway-aware hallucination detection methods are designed. Experiments span 12 models (base/instruct/reasoning) and 4 QA datasets.

### Key Designs

1. **Attention Knockout for Pathway Disentanglement**:

    - **Function**: Validate the existence and independence of the two pathways.
    - **Mechanism**: For a probe trained at layer $k$, the attention weights from exact question tokens to subsequent positions across layers 1 through $k$ are zeroed out, blocking information flow from the question to the answer. Samples are then divided into Q-Anchored (probe prediction flips) and A-Anchored (prediction unchanged). Across all models and datasets, the two groups exhibit clearly divergent behavior—one group shows large probability shifts, the other remains nearly invariant.
    - **Design Motivation**: If truthfulness signals were homogeneous, blocking question information flow should affect all samples uniformly. The bimodal behavior directly evidences the existence of two distinct mechanisms.

2. **Pathway Association with Knowledge Boundaries**:

    - **Function**: Reveal the cognitive significance of the two pathways.
    - **Mechanism**: Three metrics—answer accuracy, I-don't-know rate, and entity popularity—are used to characterize knowledge boundaries. Q-Anchored samples exhibit significantly higher accuracy and involve more popular entities (within the model's knowledge scope), while A-Anchored samples show lower accuracy and involve long-tail entities (beyond the knowledge boundary). This indicates that the model encodes truthfulness primarily via the question-answer information flow when it possesses relevant knowledge, and falls back to extracting cues from the intrinsic patterns of the generated text when knowledge is insufficient.
    - **Design Motivation**: Understanding the cognitive basis of each pathway facilitates the design of more targeted detection strategies.

3. **Pathway-Aware Hallucination Detection (MoP + PR)**:

    - **Function**: Exploit pathway distinction to improve detection performance.
    - **Mechanism**: (a) *Mixture-of-Probes (MoP)*: Multiple expert probes are trained, each specializing in a particular truthfulness encoding mechanism; the model's pathway self-awareness (>87% classification accuracy) is leveraged to automatically route each sample to the appropriate expert. (b) *Pathway Reweighting (PR)*: Based on which pathway a given sample belongs to, the pathway-relevant internal signals are selectively amplified, enhancing the most informative activation dimensions. Both methods consistently outperform single-probe baselines across multiple datasets and models.
    - **Design Motivation**: Given that the two pathways have fundamentally different signal origins, pathway-specialized detectors are more effective than a generic detector.

### Loss & Training

Probes are trained as linear classifiers using binary cross-entropy loss. The pathway classifier is likewise a linear probe trained on raw internal representations, validating the model's self-awareness of pathway membership.

## Key Experimental Results

### Main Results

| Method | PopQA AUC | TriviaQA AUC | HotpotQA AUC | NQ AUC |
|--------|-----------|--------------|--------------|--------|
| Standard Probing | Baseline | Baseline | Baseline | Baseline |
| MoP (Ours) | +5–10% | +3–8% | +2–5% | +3–7% |
| PR (Ours) | Similar gain | Similar gain | Similar gain | Similar gain |

### Ablation Study

| Analysis | Result | Note |
|----------|--------|------|
| Pathway self-awareness accuracy | 75–93% | Models can distinguish pathways from raw representations |
| Q-Anchored accuracy | Significantly higher than A-Anchored | In-scope facts use Q-Anchored |
| Entity popularity | Q-Anchored >> A-Anchored | Q-Anchored involves high-frequency entities |
| Random token knockout | No significant effect | Confirms specificity to exact question tokens |

### Key Findings

- **Two pathways are robust across models and datasets**: The bimodal pattern appears consistently across all 12 models (1B to 70B; base, instruct, and reasoning variants) and all 4 datasets.
- **Knowledge boundary determines pathway selection**: When the model "knows the answer," it adopts the Q-Anchored pathway (assessing truthfulness through question understanding); when it does not, it adopts the A-Anchored pathway (relying on statistical patterns in the answer itself).
- **Models possess pathway self-awareness**: Internal representations contain sufficient information to distinguish the two pathways, with classification accuracy of 75–93%, forming the basis of the MoP approach.
- **Self-contained property of the A-Anchored pathway**: When the question is removed and only the answer is passed through the forward pass, predictions for A-Anchored samples remain nearly unchanged, whereas those for Q-Anchored samples shift substantially.

## Highlights & Insights

- **Depth of mechanistic understanding**: The paper not only demonstrates the existence of two pathways but also reveals their association with knowledge boundaries, providing a cognitively grounded interpretation.
- **Clear path from discovery to application**: The connection between findings and methods is direct—MoP and PR exploit mechanistic insights to improve detection performance, making this more than a purely analytical study.
- **Experimental scale**: Comprehensive validation across 12 models (including the recent Qwen3) and 4 datasets ensures high credibility.

## Limitations & Future Work

- The current study focuses on factual QA settings; pathway patterns in open-ended generation, multi-turn dialogue, and other scenarios remain unexplored.
- Pathway self-awareness accuracy is not 100%, and misrouting degrades MoP performance.
- The possibility of enhancing the reliability of specific pathways through training interventions is not investigated.
- The definition of exact tokens relies on semantic frame theory, and automated extraction may introduce noise.

## Related Work & Insights

- **vs. Burns et al. (2023)**: CCS identifies linear truthfulness directions in LLMs but does not distinguish signal origins. This paper reveals the dual-pathway structure underlying those signals.
- **vs. Orgad et al. (2025)**: Their work shows that probing on exact answer tokens yields the best performance; this paper further explains why—the Q-Anchored pathway concentrates its signals in the information flow through exact tokens.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reveal the dual-pathway structure of truthfulness encoding in LLMs; a profound finding.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models and 4 datasets with rigorous causal intervention validation.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative arc from hypothesis to validation to application is clear and well-structured.
- Value: ⭐⭐⭐⭐⭐ Contributes meaningfully to both mechanistic understanding and practical improvement of hallucination detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/llm_safety/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[NeurIPS 2025\] SWE-SQL: Illuminating LLM Pathways to Solve User SQL Issues in Real-World Applications](../../NeurIPS2025/llm_safety/swe-sql_illuminating_llm_pathways_to_solve_user_sql_issues_in_real-world_applica.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/llm_safety/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/llm_safety/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)

</div>

<!-- RELATED:END -->
