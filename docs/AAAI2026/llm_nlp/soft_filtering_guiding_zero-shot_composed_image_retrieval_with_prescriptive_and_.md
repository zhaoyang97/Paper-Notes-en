---
title: >-
  [Paper Note] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts
description: >-
  [LLM (Other)] This paper proposes SoFT, a training-free plug-and-play reranking module that leverages a multimodal LLM to extract dual textual constraints — "must include" (prescriptive) and "must avoid" (proscriptive) — from a reference image and modification text, and applies soft-filtering reranking over candidate results in zero-shot composed image retrieval. A multi-target triplet dataset construction pipeline is also introduced to improve evaluation.
tags:
  - "LLM (Other)"
date: 2026-05-08
content_hash: 232c11f15fcca933
---

# Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts

- **Conference**: AAAI 2026
- **arXiv**: [2512.20781](https://arxiv.org/abs/2512.20781)
- **Code**: [https://github.com/jjungyujin/SoFT](https://github.com/jjungyujin/SoFT)
- **Area**: Multimodal Vision-Language Models / Composed Image Retrieval
- **Keywords**: Zero-shot CIR, soft filtering, prescriptive constraints, proscriptive constraints, CLIP, LLM reranking

## TL;DR

This paper proposes SoFT, a training-free plug-and-play reranking module that leverages a multimodal LLM to extract dual textual constraints — "must include" (prescriptive) and "must avoid" (proscriptive) — from a reference image and modification text, and applies soft-filtering reranking over candidate results in zero-shot composed image retrieval. A multi-target triplet dataset construction pipeline is also introduced to improve evaluation.

## Background & Motivation

Composed Image Retrieval (CIR) retrieves a target image given a reference image and a modification text describing the desired change. Zero-shot CIR (ZS-CIR) leverages pretrained vision-language models to eliminate the need for annotated data, but existing methods suffer from three core issues:

1. **Information dilution from single-query fusion**: Compressing all visual and textual cues into a single representation causes critical user requirements to be diluted by irrelevant details.
2. **Neglect of negative constraints**: Existing methods focus solely on satisfying positive cues, without penalizing attributes the user wishes to avoid.
3. **Single-target assumption in evaluation benchmarks**: Current CIR benchmarks assume a single correct target per query, ignoring the inherent ambiguity of modification texts.

## Method

### Design 1: Soft Filtering with Dual Textual Constraints (SoFT)

The core idea of the SoFT module is to decompose user intent into prescriptive and proscriptive aspects, which are used to reward and penalize candidate images respectively.

**Step 1 — Attribute Classification**: A multimodal LLM (GPT-4o) is prompted to identify key attribute-value pairs from the reference image $I_{\text{ref}}$ and modification text $T_{\text{mod}}$, categorized into three types:
- **keep attributes**: attributes to be preserved from the reference image
- **add attributes**: new attributes introduced by the modification text
- **remove attributes**: attributes present in the reference image that should be eliminated in the target

**Step 2 — Constraint Generation**: The above attribute sets are converted into:
- **Prescriptive constraints** = keep + add, describing attributes the target must contain
- **Proscriptive constraints** = remove, describing attributes the target must avoid

**Soft reweighting formula**: For each candidate image $I_c$, three CLIP cosine similarities are computed:
- $s_{\text{base}}$: retrieval score from the base CIR model
- $s_{\text{reward}}$: similarity between $I_c$ and the prescriptive constraints
- $s_{\text{penalty}}$: similarity between $I_c$ and the proscriptive constraints

The SoFT score is defined as:

$$s_{\text{SoFT}} = s_{\text{base}} \odot \frac{s_{\text{reward}} + 1 - s_{\text{penalty}}}{2}$$

The final ranking score is a convex combination of the base score and the SoFT score:

$$s_{\text{final}} = (1 - \lambda) \cdot s_{\text{base}} + \lambda \cdot s_{\text{SoFT}}$$

where $\lambda$ controls the influence of soft filtering ($\lambda = 1.0$ for CIReVL; $\lambda = 0.2$ for SEARLE).

### Design 2: Multi-Target Triplet Dataset Pipeline

To address the limitations of single-target evaluation, a two-stage dataset construction pipeline is proposed:

**Stage 1 — Multi-Target Triplet Construction**: Three CLIP-based retrieval strategies are used to obtain candidate targets:
- Text-only retrieval based on the modification text (top-k)
- Composed retrieval combining the modification text with non-conflicting reference details (top-k)
- Visual similarity retrieval against the original target image (top-k)

Each candidate set is scored by an LLM visual evaluator (0.0–1.0) with threshold $\tau = 0.85$ and $k = 10$.

**Stage 2 — Single-Target Rewriting with Contrastive Prompting**: One target and two contrastive distractors are randomly sampled from the multi-target pool, and an LLM rewrites the modification text to uniquely identify the target while excluding the distractors, producing more challenging and fine-grained triplets.

## Key Experimental Results

### Table 1: Main Results on CIRCO and CIRR (ViT-L/14)

| Method | CIRCO mAP@5 | CIRCO mAP@50 | CIRR R@1 | CIRR R@5 | CIRR R_sub@1 |
|--------|------------|-------------|----------|----------|-------------|
| CIReVL | 18.57 | 21.80 | 24.55 | 52.31 | 59.54 |
| CIReVL + SoFT | **23.90** | **27.93** | **35.54** | **65.25** | **71.59** |
| SEARLE | 11.68 | 15.12 | 24.24 | 52.48 | 53.76 |
| SEARLE + SoFT | 15.72 | 18.93 | 30.29 | 59.74 | 65.47 |
| LDRE | 23.35 | 27.50 | 26.53 | 55.57 | 60.43 |
| OSrCIR | 23.87 | 28.97 | 29.45 | 57.68 | 62.12 |

CIReVL + SoFT achieves R@5 of 65.25 (+12.94) on CIRR and mAP@50 of 27.93 (+6.13) on CIRCO.

### Table 2: Results on FashionIQ (ViT-L/14, Average)

| Method | Avg. R@10 | Avg. R@50 |
|--------|----------|----------|
| CIReVL | 28.55 | 48.57 |
| CIReVL + SoFT | **31.68** | **52.53** |
| SEARLE | 25.56 | 46.23 |
| OSrCIR | 33.26 | 54.37 |

CIReVL + SoFT improves average R@50 to 52.53 (+3.96) on FashionIQ.

### Table 3: Ablation Study — Component Analysis (ViT-L/14)

| Method | CIRCO mAP@5 | CIRR R@1 | FIQ R@10 |
|--------|------------|----------|----------|
| CIReVL | 18.57 | 24.55 | 28.55 |
| + Reward only | 22.19 | 32.80 | 32.57 |
| + Penalty only | 21.83 | 32.92 | 28.77 |
| + SoFT (both) | **23.90** | **35.54** | 31.68 |

The joint use of both constraints yields the best performance, with reward and penalty contributions being largely additive.

## Key Findings

1. **Complementarity of dual constraints**: Prescriptive and proscriptive constraints each provide independent gains, and their combination yields additive improvement.
2. **Differential model sensitivity**: CIReVL adapts more stably to SoFT than SEARLE; SEARLE is highly sensitive to the penalty term.
3. **Effect of $\lambda$**: CIReVL performs best at $\lambda = 1.0$, indicating that full reliance on SoFT reranking is beneficial; SEARLE requires a smaller $\lambda$.
4. **More reliable multi-target evaluation**: On multi-target FashionIQ, SoFT consistently improves all baseline models, demonstrating that standard single-target evaluation underestimates the method's effectiveness.

## Highlights & Insights

- **Plug-and-play, training-free**: SoFT operates purely at inference time via score-level modulation and can be seamlessly integrated into any CIR system.
- **Dual constraint modeling**: This work is the first in ZS-CIR to explicitly model both "must include" and "must avoid" aspects of user intent.
- **Dataset contribution**: The multi-target triplet pipeline meaningfully extends existing CIR benchmarks, yielding an average of 2.89 valid targets per query on CIRR and 4.6–4.9 on FashionIQ.
- **Substantial performance gains**: R@5 on CIRR improves by 12.94 percentage points, representing a significant advancement.

## Limitations & Future Work

1. **Reliance on external LLM APIs**: Constraint generation depends on GPT-4o, introducing additional inference cost (approximately \$15.36 for CIRR) and latency.
2. **Limitations of CLIP representations**: In fine-grained domains such as fashion, CLIP embeddings struggle to capture subtle visual differences, leading to unreliable penalty signals.
3. **$\lambda$ requires tuning**: Different baseline models require different values of $\lambda$, and no adaptive mechanism is provided.
4. **Reranking only**: SoFT cannot improve the recall upper bound of initial retrieval; performance is bounded by the candidate quality of the base retriever.

## Related Work & Insights

- **ZS-CIR methods**: SEARLE (textual inversion), CIReVL (two-step LLM text generation), OSrCIR (direct query generation), LDRE (multi-pseudo-query ensemble).
- **CIR benchmarks**: CIRCO (multi-target), CIRR (fine-grained reasoning on natural scenes), FashionIQ (fashion attribute modification).
- **LLM-augmented retrieval**: CoLLM reduces ambiguity by rewriting modification texts via LLM, but still assumes a single target.

## Rating

⭐⭐⭐⭐ — The motivation is clear and well-grounded; the dual-constraint formulation is natural and effective, and the plug-and-play design offers strong practical utility. The dataset pipeline represents a meaningful methodological contribution to evaluation. Primary limitations include dependence on external LLMs and the representational bottleneck of CLIP in fine-grained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] STEM: Efficient Relative Capability Evaluation of LLMs through Structured Transitive Evaluation Model](stem_efficient_relative_capability_evaluation_of_llms_through_structured_transit.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](../../ACL2025/llm_nlp/explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](../../ACL2025/llm_nlp/culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](../../ACL2025/llm_nlp/plangenllms_planning_survey.md)
- [\[ACL 2025\] Classifying Unreliable Narrators with Large Language Models](../../ACL2025/llm_nlp/classifying_unreliable_narrators.md)

</div>

<!-- RELATED:END -->
