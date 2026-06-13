---
title: >-
  [Paper Note] LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment
description: >-
  [ACL 2026][LLM Safety][jailbreak] LLM-VA identifies that LLMs encode "whether to answer" (answer vector $v_a$) and "whether the input is safe" (benign vector $v_b$) as nearly orthogonal directions internally…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "jailbreak"
  - "over-refusal"
  - "vector steering"
  - "SVM probe"
  - "closed-form weight update"
date: 2026-05-08
content_hash: c7d2d09588eaab8d
---

# LLM-VA: Resolving the Jailbreak-Overrefusal Trade-off via Vector Alignment

**Conference**: ACL 2026  
**arXiv**: [2601.19487](https://arxiv.org/abs/2601.19487)  
**Code**: https://hotbento.github.io/LLM-VA-Web/  
**Area**: LLM Safety / Alignment / Representation Engineering  
**Keywords**: jailbreak, over-refusal, vector steering, SVM probe, closed-form weight update

## TL;DR
LLM-VA identifies that LLMs encode "whether to answer" (answer vector $v_a$) and "whether the input is safe" (benign vector $v_b$) as nearly orthogonal directions internally, leading to a zero-sum trade-off between jailbreak and over-refusal. By employing closed-form minimum-norm weight updates to align $v_a$ with $v_b$, the model's "willingness to answer" becomes causally dependent on "input safety." Evaluated across 12 LLMs, LLM-VA improves F1 scores by 11.45% over the strongest baseline with only a 4.08% utility loss, requiring no fine-tuning or architectural modifications.

## Background & Motivation
**Background**: Safety alignment in LLMs suffers from two failure modes: jailbreak (providing harmful responses to toxic queries) and over-refusal (refusing benign queries). Prevailing mitigation strategies include RLHF/adversarial training/rule-based filtering (expensive) and vector steering (cost-effective, manipulating latent space directions, e.g., VectorSteer, AlphaSteer, SCANS, CAST).

**Limitations of Prior Work**: Existing vector steering methods primarily manipulate the "magnitude of $v_a$." Decreasing magnitude suppresses jailbreaks but exacerbates over-refusal, while increasing it does the opposite. AlphaSteer utilizes null-space projection to preserve utility but remains magnitude-based; SCANS/CAST incorporate input toxicity information but require architecture hooks and treat the two failure modes as independent targets for hyperparameter tuning.

**Key Challenge**: This study extracts $v_a$ and $v_b$ layer-by-layer across 12 LLMs using SVMs and finds they are **nearly orthogonal** $(\sim 90^\circ)$ across all layers. This implies that the model's internal judgments of "willingness to respond" and "input risk" are entirely independent. Consequently, magnitude adjustments only scale the projection along the $v_a$ direction globally, inevitably affecting benign and toxic inputs in the same way, making it impossible to suppress both error types simultaneously.

**Goal**: Align $v_a$ with $v_b$ so that the projection onto $v_a$ inherently carries "input safety" information. This naturally suppresses toxic inputs while encouraging benign ones, resolving both failure modes **simultaneously**.

**Key Insight**: Use SVMs to identify directions $\rightarrow$ implement alignment via closed-form minimum-norm weight modifications, bypassing gradient optimization and fine-tuning.

**Core Idea**: The root cause of the jailbreak/over-refusal trade-off is $v_a \perp v_b$ (structural decoupling); the solution is geometric alignment rather than magnitude regulation.

## Method

### Overall Architecture
LLM-VA consists of three steps, performed without gradient descent:
1. **Vector Identification**: For each layer, SVMs are trained on 128 toxic (S-Eval) + 128 benign (ORFuzzSet) samples to obtain $v_b$ (benign vs. toxic normal vector) and $v_a$ (answer vs. refuse normal vector).
2. **Layer Selection**: A subset of layers most relevant to "safety decision-making" is selected based on their contribution to the final output and SVM classification accuracy (avoiding ambiguous early layers and task-irrelevant late layers).
3. **Vector Alignment**: A minimum-norm weight update $\Delta W$ is applied to rotate the $v_a$ direction in the MLP/attention output space of the selected layers to align with $v_b$, iterated through all selected layers.

When an input $x$ enters the aligned model: if $x$ is benign $\rightarrow$ the $v_b$ projection is positive $\rightarrow$ the $v_a$ projection is also positive (due to alignment) $\rightarrow$ the model tends to answer. If toxic $\rightarrow$ the reverse occurs $\rightarrow$ the model tends to refuse.

### Key Designs

1. **Dual Vector Extraction via SVM ($v_a$ and $v_b$):**
    - **Function**: Trains two SVMs per transformer layer—one to distinguish benign/toxic samples to obtain $v_b$, and another to distinguish answered/refused samples to obtain $v_a$.
    - **Mechanism**: For layer output $h^{(\ell)}$, activations from positive/negative samples are collected. The normal vector of the SVM maximum-margin hyperplane represents the safety/answer direction for that layer.
    - **Design Motivation**: Compared to logistic regression, SVMs provide geometrically clearer directions (maximum margin) and are insensitive to sample size (achieving high separation with only 256 samples per layer). Extraction from real samples is more accurate than "concept vectors" derived from prompt engineering.

2. **Orthogonality Diagnosis as a Methodological Pillar:**
    - **Function**: Measures $\angle(v_a, v_b) \approx 90^\circ$ across all layers of four model families (Llama-3.1, Gemma-2, Mistral-v0.3, Qwen3) to prove this is a structural phenomenon.
    - **Mechanism**: Calculates cosine similarity $\cos\theta = \frac{v_a^\top v_b}{\|v_a\|\|v_b\|}$ while verifying high classification accuracy for both directions to ensure they are not noise.
    - **Design Motivation**: This is the most critical observation—it elevates the "magnitude-based failure" from an empirical observation to a geometric necessity, providing a theoretical basis for alignment-based design.

3. **Closed-Form Minimum-Norm Weight Alignment:**
    - **Function**: Finds the minimal weight perturbation $\Delta W$ to rotate $v_a$ toward $v_b$ in the output space: $\min \|\Delta W\|_F$ s.t. $(W+\Delta W) v_a \parallel v_b$.
    - **Mechanism**: The optimal solution is obtained via a closed-form approach (similar to Procrustes rotation) using SVD. This is applied iteratively across selected layers without backpropagation.
    - **Design Motivation**: (a) Closed-form solution $\rightarrow$ fast and reproducible; (b) Minimum norm $\rightarrow$ minimal disturbance to general utility; (c) No architecture changes $\rightarrow$ compatible with existing inference frameworks and ship-ready checkpoints.

### Loss & Training
**Training-free**. The "training" phase is limited to: (a) SVM training per layer (standard hinge loss + 256 samples, completing in seconds); (b) Closed-form $\Delta W$ calculation (SVD in one step). The entire process completes in minutes on a single GPU.

## Key Experimental Results

### Main Results
Evaluated across 12 LLMs from 5 families, comparing 5 baselines (None, VectorSteer, AlphaSteer, SCANS, CAST) plus Fine-tuning:

| Method | F1 (Safety Trade-off) | Utility Retention | Requires Fine-tuning | Changes Architecture |
|------|--------------------------|----------------|-------------------|-------------|
| None (Original) | Baseline | 100% | – | – |
| Fine-tuning | High but Costly | Medium | ✗ Yes | ✓ No |
| VectorSteer | Low (Heavy trade-off) | High | ✓ No | ✗ Yes |
| AlphaSteer (Best Baseline) | Medium | 100% | ✓ No | ✗ Yes |
| CAST | Medium | Medium | ✓ No | ✗ Yes |
| SCANS | Medium | Medium | ✓ No | ✗ Yes |
| **LLM-VA (Ours)** | **AlphaSteer + 11.45%** | **95.92%** | ✓ No | ✓ No |

LLM-VA is the only method that mitigates both jailbreak and over-refusal without fine-tuning or architectural changes.

### Ablation Study

| Configuration / Observation | Key Metric | Description |
|-------------|---------|------|
| Full LLM-VA | F1 +11.45% vs AlphaSteer | Complete method. |
| Magnitude-only ($v_a$ tuning) | Significant trade-off | Strong coupling between jailbreak↓ and over-refusal↑. |
| $\angle(v_a,v_b)$ Measurement | $\sim 90°$ across models | Confirms structural orthogonality. |
| Model Bias Adaptation | Automatic correction | Models biased toward jailbreaking primarily reduce jailbreaks; over-conservative models primarily reduce over-refusal. |
| Layer Selection | Optimal vs. All-layer | Early layers have ambiguous directions; alignment there hurts utility. |
| Closed-form vs. Gradient | Stable + Fast | SVD provides a robust solution without hyperparameters. |

### Key Findings
- $v_a \perp v_b$ is a universal phenomenon across 12 models from 5 families, suggesting RLHF systematically optimizes helpfulness and harmlessness into orthogonal objectives.
- LLM-VA automatically adapts to each model's safety bias without manual hyperparameter tuning, a feat magnitude-based methods cannot achieve.
- The trade-off of a 4.08% utility drop for an 11.45% F1 gain is superior to all baselines. Aligned weights can be hot-swapped with original models.
- Layer selection is crucial; "more is not better," as aligning early layers with low separation can degrade general utility.

## Highlights & Insights
- The reasoning chain "Orthogonality $\rightarrow$ Geometric Impossibility $\rightarrow$ Alignment not Scaling" provides a clean geometric explanation for a long-observed empirical trade-off.
- Closed-form minimum-norm alignment is a rare "theoretical + practical" method in representation engineering: fast, low data requirement, and directly combinable with RLHF models.
- Extensive coverage (12 models) and open-sourced weights provide immediate value to the community.
- Implicit critique of RLHF: optimizing helpfulness and harmlessness as independent rewards naturally leads to orthogonalization; future alignment might require explicit encouragement of correlation between these directions.

## Limitations & Future Work
- The method is based on *single-turn* classification; its efficacy against multi-turn jailbreaks or complex attacks in code/agent contexts remains untested.
- $v_a/v_b$ are extracted via linear SVMs, assuming safety/compliance are linear directions. This may not cover safety concepts encoded non-linearly (e.g., role-playing circumvention).
- Cannot be applied to flagship closed-source models (e.g., GPT-4) as it requires hidden state access.
- Since alignment forces the model to follow a binary safety classifier, it might increase false positives for "gray area" queries (e.g., dangerous info for research). Future work could introduce a "neutral direction" for tripartite alignment.

## Related Work & Insights
- **vs. VectorSteer (Zou 2023a)**: Only tunes $v_a$ magnitude; LLM-VA proves this cannot geometrically resolve the trade-off.
- **vs. AlphaSteer (Sheng 2025)**: Uses null-space projection for utility but remains magnitude-based; LLM-VA improves F1 by 11.45%.
- **vs. SCANS / CAST (Cao 2025 / Lee 2024)**: Introduces toxicity info but requires architecture hooks and heavy hyperparameter tuning; LLM-VA is architecture-agnostic and self-adapting.
- **vs. Fine-tuning**: Fine-tuning resolves trade-offs but is expensive and risks catastrophic forgetting; LLM-VA achieves comparable effects at near-zero cost.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Orthogonality diagnosis + Closed-form alignment" is a standout theoretical and practical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage across 12 models, though multi-turn/agent scenarios are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely tight motivation and derivation.
- Value: ⭐⭐⭐⭐⭐ Resolves the twin safety failures without fine-tuning or architecture changes; highly deployment-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hard to Read, Easy to Jailbreak: How Visual Degradation Bypasses MLLM Safety Alignment](hard_to_read_easy_to_jailbreak_how_visual_degradation_bypasses_mllm_safety_align.md)
- [\[ICLR 2026\] Improving the Trade-off Between Watermark Strength and Speculative Sampling Efficiency for Language Models](../../ICLR2026/llm_safety/improving_the_trade-off_between_watermark_strength_and_speculative_sampling_effi.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ACL 2026\] Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring](rethinking_jailbreak_detection_of_large_vision_language_models_with_representati.md)

</div>

<!-- RELATED:END -->
