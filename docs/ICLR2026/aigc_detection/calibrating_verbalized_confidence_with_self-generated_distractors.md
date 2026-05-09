---
title: >-
  [Paper Note] Calibrating Verbalized Confidence with Self-Generated Distractors
description: >-
  [ICLR 2026][AIGC Detection][confidence calibration] This paper proposes DiNCo, a method that exposes the "suggestibility bias" of LLMs by having them **independently** evaluate automatically generated distractor options (plausible but incorrect alternative answers). It normalizes confidence using the total confidence assigned to distractors, and integrates two complementary dimensions—generation consistency and verification consistency—to significantly improve confidence calibration on both short-form QA and long-form generation tasks.
tags:
  - ICLR 2026
  - AIGC Detection
  - confidence calibration
  - verbalized probability
  - distractor generation
  - NLI reweighting
  - generation-verification consistency
date: 2026-05-08
content_hash: 3df80bfcff96544f
---

# Calibrating Verbalized Confidence with Self-Generated Distractors

**Conference**: ICLR 2026
**arXiv**: [2509.25532](https://arxiv.org/abs/2509.25532)
**Code**: [victorwang37/dinco](https://github.com/victorwang37/dinco)
**Area**: AIGC Detection
**Keywords**: confidence calibration, verbalized probability, distractor generation, NLI reweighting, generation-verification consistency

## TL;DR

This paper proposes DiNCo, a method that exposes the "suggestibility bias" of LLMs by having them **independently** evaluate automatically generated distractor options (plausible but incorrect alternative answers). It normalizes confidence using the total confidence assigned to distractors, and integrates two complementary dimensions—generation consistency and verification consistency—to significantly improve confidence calibration on both short-form QA and long-form generation tasks.

## Background & Motivation

**Background**: LLMs can directly express their certainty via "verbalized confidence"—either by self-reporting a numeric value (e.g., "80%") or via the $P(\text{True})$ paradigm. This single-call approach is far more efficient than multi-sample methods, but its calibration quality is poor.

**Limitations of Prior Work**:
- **Overconfidence**: Verbalized confidence is systematically inflated; models frequently report confidence scores above 0.8 even for incorrect answers.
- **Confidence saturation**: Scores concentrate in a narrow range (e.g., 0.9–1.0), making it impossible to effectively distinguish correct from incorrect answers regardless of threshold choice.
- **Cross-difficulty incomparability**: Incorrect answers to easy questions and correct answers to hard questions may receive identical scores.

**Key Challenge**: The authors propose a "suggestibility" hypothesis—when an LLM has limited knowledge about a topic, **the mere presence of a claim in context inflates the model's confidence in that claim**. Empirical validation shows that the total confidence $\beta(C)$ on incorrectly answered questions is significantly higher than on correctly answered ones, confirming that models are more prone to uncritical acceptance under epistemic uncertainty.

## Method

### Modeling Suggestibility Bias

Verbalized confidence is modeled as the product of latent true confidence and a suggestibility bias scalar:

$$f^{\text{VC}}(c) = \beta(c) \cdot f^{\text{lat}}(c)$$

where $\beta(c)$ is the suggestibility bias of claim $c$. The key assumption is that for a set $C$ of logically related, mutually exclusive claims, the bias is approximately equal: $\beta(c) \approx \beta(C)$. Since latent confidence should satisfy probabilistic normalization $\sum_{c \in C} f^{\text{lat}}(c) = 1$, it follows that:

$$\beta(C) \approx \sum_{c \in C} f^{\text{VC}}(c), \quad f^{\text{NVC}}(c_0) = \frac{f^{\text{VC}}(c_0)}{\beta(C)}$$

In practice, $\beta(C) \leftarrow \max(1, \beta(C))$ is applied to prevent over-scaling when the claim set is incomplete.

### Distractor Generation Strategy

The goal is to find a set of mutually exclusive alternative claims with high generation probability, maximizing $\sum_{c \in C} f^{\text{VC}}(c)$ subject to $|C| \leq K$:

| Scenario | Distractor Generation | Characteristics |
|---|---|---|
| Open-source models (logits available) | Beam search to generate high-probability alternatives | Efficiently covers probability mass; avoids repetition from independent sampling |
| API models (top-token probabilities available) | Pseudo beam search (using top token probabilities) | Approximates beam search behavior |
| Black-box models (no probability access) | Prompt model to produce a list of candidate answers | Requires no probability access |
| Long-form generation | Decompose into atomic claims, then generate distractors per claim | Compatible with the FactScore evaluation framework |

### NLI Reweighting Mechanism

Since generated distractors cannot be guaranteed to be strictly mutually exclusive, an NLI model (DeBERTa-v3-base) is used to compute two weights:

- **Uniqueness weight**: $w_{\text{unique}}(c) = \frac{1}{\sum_{c' \in C} P(\text{entail} \mid c', c)}$, downweighting items entailed by other claims in $C$.
- **Contradiction weight**: $w_{\text{contra}}(c) = \frac{P(\text{contra} \mid c_0, c) + P(\text{contra} \mid c, c_0)}{2}$, downweighting items that do not contradict the original claim.

The normalization factor becomes:

$$\beta(C) = \max\left(1, f^{\text{VC}}(c_0) + \sum_{c \in C} f^{\text{VC}}(c) \cdot w_{\text{unique}}(c) \cdot w_{\text{contra}}(c)\right)$$

### Generation–Verification Consistency Fusion

The authors find that the highest-probability answer from beam search and the highest-confidence answer from the verification stage agree on only 59.2% of questions, indicating a systematic discrepancy between the generator and verifier. DiNCo integrates these two complementary dimensions:

$$f^{\text{DiNCo}}(c) = \frac{1}{2} f^{\text{SC}}(c) + \frac{1}{2} f^{\text{NVC}}(c)$$

where $f^{\text{SC}}$ is the generation confidence estimated via self-consistency, and $f^{\text{NVC}}$ is the normalized verification confidence. Under an inference budget of $K=10$, 5 samples are allocated to SC and 5 distractors to NVC.

## Key Designs

1. **Independent evaluation rather than joint prompting**: Each distractor is evaluated independently rather than presenting all candidates simultaneously—joint presentation would allow the model to satisfy probabilistic normalization through simple arithmetic, thereby concealing inconsistencies.
2. **NLI reweighting ensures normalization quality**: Continuous weights based on entailment and contradiction relations handle partially equivalent or partially contradictory claims, eliminating the bias introduced by simple counting.
3. **Dual-dimension consistency fusion**: Self-consistency from sampling (SC) and normalized verification consistency (NVC) are integrated as complementary signals, compensating for the blind spots of each individual dimension.

## Key Experimental Results

### Short-Form QA Results

| Method | TriviaQA ECE ↓ | TriviaQA AUC ↑ | SimpleQA ECE ↓ | SimpleQA AUC ↑ |
|---|:-:|:-:|:-:|:-:|
| VC | 0.240 | 0.817 | 0.547 | 0.644 |
| K-VC | 0.341 | 0.604 | 0.338 | 0.632 |
| MSP | 0.149 | 0.819 | 0.263 | 0.800 |
| SC | 0.236 | 0.785 | 0.220 | 0.750 |
| NVC | 0.171 | 0.853 | 0.164 | 0.729 |
| **DiNCo** | **0.097** | **0.879** | **0.089** | **0.786** |

> TriviaQA results are reported for Qwen3-8B; SimpleQA results for GPT-4.1. DiNCo outperforms the best baseline (MSP) on ECE by an average margin of 0.077 (TriviaQA) and 0.092 (SimpleQA).

### Long-Form Generation Results (FactScore)

| Method | Qwen3-8B ECE ↓ | Qwen3-8B Pearson $r$ ↑ | Gemma-3-4B ECE ↓ | Gemma-3-4B Pearson $r$ ↑ |
|---|:-:|:-:|:-:|:-:|
| VC | 0.433 | 0.073 | 0.527 | -0.081 |
| SC | 0.162 | 0.468 | 0.197 | 0.629 |
| NVC | 0.191 | 0.444 | 0.123 | 0.695 |
| **DiNCo** | **0.076** | **0.518** | **0.172** | **0.724** |

> DiNCo's passage-level Pearson/Spearman correlation outperforms SC by an average of 0.072/0.074.

### Saturation and Scalability Analysis

- **Saturation**: DiNCo achieves $\Delta_0 = 0.998$ (nearly all sample pairs have distinct confidence scores), compared to 0.670 for VC and 0.832 for SC@100.
- **Scaling SC cannot close the gap**: Expanding SC from 10 to 100 samples (a 7.6× FLOP increase relative to DiNCo) yields negligible ECE improvement and fails to match DiNCo.
- **NLI ablation**: Removing NLI reweighting degrades NVC ECE from 0.171 to 0.358, confirming the critical role of NLI weights.

## Highlights & Insights

**Strengths** ⭐⭐⭐⭐
- The theoretical motivation grounded in "suggestibility bias" is clearly articulated and empirically validated.
- The method is applicable to both open-source and closed-source models, and transfers seamlessly from short-form QA to long-form generation.
- Only a lightweight NLI model is required (184M parameters, <1% of total FLOPs); the approach is training-free and requires no annotated data.
- The proposed saturation metric $\Delta_\epsilon$ quantifies a phenomenon previously discussed only qualitatively.

**Weaknesses** ⭐⭐⭐
- Distractor quality depends on the model's own generative capability, which may limit effectiveness for smaller models.
- The assumption that bias $\beta$ is approximately equal across logically related claims may not hold for claims with large semantic distance.
- The long-form generation setting requires an additional claim decomposition step, increasing pipeline complexity.
- Comparisons with recent post-hoc calibration methods (e.g., temperature scaling) in labeled-data settings are absent.

## Related Work & Insights

| Method Type | Representative Work | Distinction from DiNCo |
|---|---|---|
| Verbalized confidence | P(True), Verbalized Numerical | Single-pass evaluation; susceptible to suggestibility bias; confidence saturation |
| Joint multi-candidate prompting | Top-K-VC, CaCoST | Joint presentation allows the model to satisfy normalization arithmetically, masking inconsistency |
| Self-consistency | SC, SC-VC | Exploits only generation consistency; ignores the verification dimension |
| Sequence probability | MSP | Relies on canonical answer format; does not extend to long-form generation |
| **DiNCo** | Ours | Independent distractor evaluation + NLI reweighting + generation/verification dual-dimension fusion |

## Limitations & Future Work

DiNCo addresses the often-overlooked "suggestibility bias" of LLMs by automatically generating distractors and independently evaluating confidence to estimate and correct the bias, then fusing generation and verification consistency as complementary dimensions. The method achieves cross-task and cross-model calibration improvements under a zero-resource, training-free setting with minimal additional overhead compared to SC (only 32% more FLOPs). Future directions include: generating distractors with smaller models to further reduce cost, extending the method to multi-turn dialogue and agentic decision-making scenarios, and exploring combinations with post-hoc calibration approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](../../ACL2026/aigc_detection/who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](../../ACL2026/aigc_detection/biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)

</div>

<!-- RELATED:END -->
