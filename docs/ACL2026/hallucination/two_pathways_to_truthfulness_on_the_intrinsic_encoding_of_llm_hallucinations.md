---
title: >-
  [Paper Note] Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations
description: >-
  [ACL 2026][Hallucination Detection][Truthfulness Encoding] This paper discovers two distinct information pathways for truthfulness signals within LLMs: Question-Anchored (relying on the flow from question to answer) and…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "Truthfulness Encoding"
  - "Attention Mechanism"
  - "Information Pathways"
  - "Knowledge Boundary"
date: 2026-05-08
content_hash: a81140d4d3583b7e
---

# Two Pathways to Truthfulness: On the Intrinsic Encoding of LLM Hallucinations

**Conference**: ACL 2026  
**arXiv**: [2601.07422](https://arxiv.org/abs/2601.07422)  
**Code**: [https://github.com/RowanWenLuo/llm-truthfulness-pathways](https://github.com/RowanWenLuo/llm-truthfulness-pathways)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Detection, Truthfulness Encoding, Attention Mechanism, Information Pathways, Knowledge Boundary

## TL;DR

This paper discovers two distinct information pathways for truthfulness signals within LLMs: Question-Anchored (relying on the flow from question to answer) and Answer-Anchored (extracting self-contained evidence from the generated answer itself). These pathways are closely linked to knowledge boundaries. Based on this, two pathway-aware hallucination detection methods, Mixture-of-Probes and Pathway Reweighting, are proposed, achieving AUC improvements of up to 10%.

## Background & Motivation

**Background**: LLMs frequently produce hallucinations—plausible but factually incorrect outputs. Previous work has demonstrated that internal representations encode rich truthfulness signals, which can be detected via linear probes. However, the sources and mechanisms of these signals remain unclear.

**Limitations of Prior Work**: Existing internal probing methods treat all samples as homogeneous, using a single probe for all hallucinations. However, truthfulness signals in different samples may arise from different mechanisms; using a unified approach leads to suboptimal performance.

**Key Challenge**: Saliency analysis reveals a bimodal distribution in the importance of information flow from the question to the answer—some samples rely heavily on question information, while others do not. This suggests the existence of two fundamentally different truthfulness encoding mechanisms.

**Goal**: (1) Validate and decouple the two truthfulness pathways; (2) Reveal their emergent properties; (3) Leverage pathway distinction to improve hallucination detection performance.

**Key Insight**: Decouple and verify the two pathways through causal intervention experiments, specifically attention knockout and token patching.

**Core Idea**: Truthfulness signals are generated via two independent pathways—Q-Anchored relies on question-to-answer information flow (applicable to facts within the model's knowledge), while A-Anchored extracts self-contained evidence from the generated text (applicable to long-tail facts outside the knowledge boundary).

## Method

### Overall Architecture

The study consists of three stages: (1) Discovery of the bimodal distribution via saliency analysis and proposal of the two-pathway hypothesis; (2) Verification of the hypothesis using attention knockout and token patching; (3) Exploration of pathway characteristics (knowledge boundary correlation, self-perception ability) and design of pathway-aware hallucination detection methods. Experiments cover 12 models (base/instruct/reasoning) and 4 QA datasets.

### Key Designs

1. **Attention Knockout Decoupling Experiment**:
    - **Function**: Validates the existence and independence of the two pathways.
    - **Mechanism**: For a probe trained at layer k, attention weights from exact question tokens to subsequent positions in layers 1 to k are set to 0, blocking the question-to-answer information flow. Samples are classified as Q-Anchored if the probe prediction flips, and A-Anchored if it does not. Across all models and datasets, behaviors bifurcate—one group shows significant probability shifts, while the other remains nearly unchanged.
    - **Design Motivation**: If truthfulness signals were homogeneous, blocking the question flow should affect all samples uniformly. The bimodal behavior directly proves the existence of two distinct mechanisms.

2. **Correlation between Pathways and Knowledge Boundaries**:
    - **Function**: Reveals the cognitive significance of the two pathways.
    - **Mechanism**: Knowledge boundaries are measured using three metrics (answer accuracy, I-don't-know rate, entity popularity). Q-Anchored samples show significantly higher accuracy and involve more popular entities (within knowledge). A-Anchored samples show low accuracy and involve long-tail entities (outside knowledge). This indicates that the model encodes truthfulness through the QA flow when it possesses knowledge, but shifts to extracting clues from intrinsic patterns of the generated text when knowledge is lacking.
    - **Design Motivation**: Understanding the cognitive basis of the pathways helps in designing targeted detection strategies.

3. **Pathway-Aware Hallucination Detection (MoP + PR)**:
    - **Function**: Leverages pathway distinction to enhance detection performance.
    - **Mechanism**: (a) Mixture-of-Probes (MoP): Multiple expert probes are trained, each focusing on a specific encoding mechanism. Samples are automatically routed to the appropriate expert using the model's pathway self-perception capability (>87% classification accuracy). (b) Pathway Reweighting (PR): Internal signal strength related to a specific pathway is selectively enhanced based on the sample's category, magnifying the most informative activation dimensions. Both methods consistently outperform single-probe baselines.
    - **Design Motivation**: Since the two pathways have fundamentally different signal sources, pathway-specific detectors are more effective than a general-purpose one.

### Loss & Training

Probes are trained as linear classifiers using binary cross-entropy loss. Pathway classifiers are also trained as linear probes on raw internal representations to verify the model's self-perception ability.

## Key Experimental Results

### Main Results

| Method | PopQA AUC | TriviaQA AUC | HotpotQA AUC | NQ AUC |
|--------|------|------|----------|------|
| Standard Probing | Baseline | Baseline | Baseline | Baseline |
| MoP (Ours) | +5-10% | +3-8% | +2-5% | +3-7% |
| PR (Ours) | Similar Gain | Similar Gain | Similar Gain | Similar Gain |

### Ablation Study

| Analysis | Result | Description |
|------|---------|------|
| Pathway Self-perception Accuracy | 75-93% | Models can distinguish the two pathways from raw representations |
| Q-Anchored Accuracy | Significantly > A-Anchored | Q-Anchored is used for facts within knowledge |
| Entity Popularity | Q-Anchored >> A-Anchored | Q-Anchored involves high-frequency entities |
| Random Token Knockout | No significant effect | Confirms the effect is specific to exact question tokens |

### Key Findings

- **Robust presence across models and datasets**: The bimodal pattern consistently appears across all 12 models (1B to 70B, base to reasoning) and 4 datasets.
- **Knowledge boundaries dictate pathway selection**: Models use Q-Anchored (truthfulness via question understanding) when they "know the answer" and A-Anchored (truthfulness via statistical patterns of the answer) when they "don't know."
- **Model self-perception of pathways**: Internal representations contain sufficient information to distinguish the two pathways with 75-93% accuracy, forming the foundation of the MoP method.
- **Self-contained nature of A-Anchored**: After removing the question and performing a forward pass on the answer alone, predictions for A-Anchored samples remain nearly identical, whereas Q-Anchored samples change drastically.

## Highlights & Insights

- **Depth of mechanistic understanding**: Provides a cognitive explanation by not only proving the pathways' existence but also linking them to knowledge boundaries.
- **Practical application of pathway separation**: Offers a clear path from discovery to application—MoP and PR utilize mechanistic insights to directly improve detection performance.
- **Scale of experiments**: High credibility due to comprehensive validation across 12 models (including the latest Qwen3) and 4 datasets.

## Limitations & Future Work

- Currently focused on factual QA; pathway patterns in open-ended generation or multi-turn dialogues remain unknown.
- Pathway self-perception accuracy is not 100%, and routing errors can degrade MoP performance.
- Training interventions to enhance specific pathway reliability have not been explored.
- The definition of exact tokens relies on semantic frame theory, and automated extraction may introduce noise.

## Related Work & Insights

- **vs Burns et al. (2023)**: CCS identifies linear truthfulness directions in LLMs but does not distinguish signal sources. This paper reveals a dual-pathway structure.
- **vs Orgad et al. (2025)**: While they show that probing works best on exact answer tokens, this paper explains why—signals in the Q-Anchored pathway are concentrated in the information flow of exact tokens.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to reveal the dual-pathway structure of truthfulness encoding in LLMs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Rigorous causal intervention across 12 models and 4 datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear narrative logic from hypothesis to validation to application.
- Value: ⭐⭐⭐⭐⭐ Significant contribution to both mechanistic understanding and practical improvement of hallucination detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](../../ICML2026/hallucination/realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)
- [\[NeurIPS 2025\] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations](../../NeurIPS2025/hallucination/seca_semantically_equivalent_and_coherent_attacks_for_eliciting_llm_hallucinatio.md)
- [\[ACL 2026\] Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)

</div>

<!-- RELATED:END -->
