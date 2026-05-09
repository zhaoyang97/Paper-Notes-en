---
title: >-
  [Paper Note] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM hallucination] This paper proposes Topic-level Preference Rewriting (TPR), which systematically optimizes the reward gap configuration in preference data through fine-grained semantic control at the topic level, combined with a curriculum learning strategy that progressively increases the difficulty of negative samples, achieving approximately 93% hallucination reduction across multiple hallucination benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - VLM hallucination
  - DPO
  - preference learning
  - topic-level rewriting
  - curriculum learning
date: 2026-05-08
content_hash: 00913fc273917042
---

# Systematic Reward Gap Optimization for Mitigating VLM Hallucinations

**Conference**: NeurIPS 2025
**arXiv**: [2411.17265](https://arxiv.org/abs/2411.17265)
**Code**: [https://tpr-dpo.github.io](https://tpr-dpo.github.io)
**Area**: Multimodal Large Models / VLM Hallucination Mitigation
**Keywords**: VLM hallucination, DPO, preference learning, topic-level rewriting, curriculum learning

## TL;DR

This paper proposes Topic-level Preference Rewriting (TPR), which systematically optimizes the reward gap configuration in preference data through fine-grained semantic control at the topic level, combined with a curriculum learning strategy that progressively increases the difficulty of negative samples, achieving approximately 93% hallucination reduction across multiple hallucination benchmarks.

## Background & Motivation

VLMs (e.g., GPT-4V, LLaVA) achieve strong performance on multimodal tasks but suffer pervasively from **visual hallucinations**—generating content inconsistent with the input image. Existing DPO-based hallucination mitigation methods exhibit systematic flaws in preference data construction:

**Ranking-based methods** (e.g., RLAIF-V, AMP): directly select $y_w$ and $y_l$ from model outputs without correcting underlying hallucinations; the resulting data carries insufficient learning signal, and the reward gap may be too small.

**Rewriting-based methods** (e.g., HA-DPO, HSA-DPO): rely on external "black-box" models (GPT-4V) for rewriting, making it difficult to precisely control the type and magnitude of modifications, and potentially introducing hallucinations that deviate from the model's intrinsic failure modes.

**Key Challenge**: The effectiveness of DPO depends on the quality and magnitude of the **true reward gap** in preference pairs—i.e., $r(y_w;x) - r(y_l;x)$—yet existing methods lack a **systematic and fine-grained control mechanism** over this gap.

**Starting Point of TPR**: Operate at the **topic level**, leverage the model's own resampled candidates (avoiding external bias), and precisely control the divergence between $y_w$ and $y_l$ on each semantic topic through selective replacement, thereby systematically shaping an optimal reward gap configuration.

## Method

### Overall Architecture

The TPR pipeline consists of three core steps:
1. **Topic-level Alternatives Generation** (§3.2): generate diverse candidates for each semantic topic.
2. **Selective Topic Replacement** (§3.3): strategically replace topics to construct preference pairs.
3. **Curriculum Learning Strategy** (§3.4): progressively adjust the difficulty of negative samples.

### Key Designs

1. **Topic-level Candidate Generation**:

    - **Decomposition**: the reference model $\pi_{ref}$ decomposes multiple candidate responses into fine-grained semantic units $\{u_{m,n}\}$.
    - **Topic Clustering**: clustering is performed based on textual consistency (the model judges whether two units describe the same topic) and visual relevance (CLIP features verify whether two units refer to the same image region).
    - **Intra-topic Self-Resampling**: each semantic unit is converted into a wh-question and $\pi_{ref}$ is queried multiple times to obtain diverse candidates within the same topic. Advantage: more efficient than full-response resampling (only a single topic needs to be correct) and provides fine-grained control at the topic level.

2. **Selective Topic Replacement**:

    - **Intra-topic Ranking**: semantic units are converted into yes-no questions and scored by $\pi_{label}$ as $S(u) = p_Y - p_N$, where high scores indicate factual correctness and low scores indicate hallucination.
    - **Selective Replacement**: a template response $y_k$ is sampled at random; for each of its topic units, candidates from the ranking pool are substituted according to a defined strategy. Greedy strategy: replace with the highest- or lowest-scoring candidate (maximizing the reward gap).
    - **In-context Rewriting**: $\pi_{ref}$ seamlessly integrates the substituted semantic content into the template to preserve linguistic fluency.

3. **Curriculum Learning Strategy (TPR-CL)**:

    - **Warm-Up Phase** (60%): adopts the greedy strategy, using the lowest-scoring candidates in $y_l$ to provide a strong initial learning signal.
    - **Hard-Mining Phase** (40%): uses progressively higher-scoring incorrect candidates in $y_l$ (hard negatives closer to the decision boundary), forcing the model to distinguish subtle hallucinations.
    - This strategy of gradually narrowing the reward gap resembles hard negative mining, transitioning the model from "distinguishing obvious errors" to "distinguishing subtle errors."

### Loss & Training

The standard DPO loss is used: $\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$. A total of 20,000 preference pairs are constructed; training uses the AdamW optimizer with a learning rate of $5 \times 10^{-7}$ and cosine decay.

## Key Experimental Results

### Main Results (Hallucination Benchmarks)

| Method | ObjHal CHs↓ | ObjHal CHi↓ | MMHal Score↑ | MMHal Hall.↓ | AMBER Acc↑ | POPE F1↑ |
|---|---|---|---|---|---|---|
| LLaVA-1.5-7B | 53.6 | 25.2 | 2.36 | 51.0 | 73.5 | 77.6 |
| RLAIF-V-7B | 8.5 | 4.3 | 3.06 | 29.2 | 76.8 | 84.5 |
| HSA-DPO-13B | 5.3 | 3.2 | 2.61 | 48.0 | - | - |
| **TPR-CL-7B** | **3.4** | **1.8** | **3.06** | **30.2** | **82.7** | **87.8** |

TPR-CL reduces hallucinations on ObjHal by ~93% relative to the baseline LLaVA-1.5 (CHs: 53.6→3.4) and by ~41% on MMHal.

### Ablation Study

| Configuration | ObjHal CHs↓ | AMBER Acc↑ | Note |
|---|---|---|---|
| w/o multi-response sampling | 6.8 | 79.1 | Multiple responses increase topic diversity |
| w/o intra-topic resampling | 5.2 | 80.3 | Self-resampling enriches candidate pool |
| Replace preferred only | 5.8 | 80.0 | Bidirectional replacement is more effective |
| w/o in-context rewriting | 5.6 | 79.5 | Rewriting preserves fluency |
| Greedy (TPR) | 4.0 | 82.3 | Greedy strategy is already effective |
| **Curriculum (TPR-CL)** | **3.4** | **82.7** | Curriculum learning yields further gains |

### Key Findings
- Topic-level manipulation is more fine-grained and efficient than response-level manipulation and is central to TPR's success.
- The curriculum learning strategy (easy-to-hard) consistently outperforms the greedy strategy, validating the effectiveness of progressive reward gap optimization.
- TPR exhibits excellent data efficiency: SOTA performance is achieved with only 20K preference pairs, far surpassing methods that require manual annotation.
- Hallucination mitigation does not degrade general capabilities (LLaVA-Bench and MMStar metrics remain comparable or improve).

## Highlights & Insights
- The perspective of **reward gap configuration optimization** targets the core weakness of DPO-based methods—not only must preferences be ranked correctly, but the magnitude and dimensions of the gap must be carefully designed.
- Using the model's own resampled candidates to avoid bias introduced by external models is an elegant design choice.
- The decoupled topic-level operations (different topics are weakly correlated) provide a theoretical basis for fine-grained control.

## Limitations & Future Work
- Relies on LLaVA-NeXT-34B as a labeler model for scoring, which incurs non-trivial computational cost.
- The quality of topic clustering depends on the VLM's topic judgment capability, which may be unreliable in complex scenes.
- The phase split (60/40) and difficulty schedule of the curriculum learning strategy are manually designed; adaptive strategies warrant further exploration.
- Validation is limited to LLaVA-1.5-7B; generalizability to larger models remains to be confirmed.

## Related Work & Insights
- **vs. RLAIF-V**: RLAIF-V ranks model outputs via divide-and-conquer scoring without modifying content; TPR actively rewrites to control the reward gap.
- **vs. HA-DPO/HSA-DPO**: These methods rely on GPT-4V for rewriting; TPR uses model self-resampling to avoid external bias.
- **vs. AMP**: AMP constructs preferences by contrasting models of different scales, yielding coarse granularity; TPR performs precise manipulation at the topic level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The reward gap configuration optimization perspective is novel, and the combination of topic-level manipulation with curriculum learning is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple hallucination benchmarks, general capability benchmarks, detailed ablations, and data efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic with a coherent chain from motivation to method to experiments.
- Value: ⭐⭐⭐⭐⭐ 93% hallucination reduction combined with high data efficiency yields substantial practical value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Mitigating Object Hallucinations via Sentence-Level Early Intervention](../../ICCV2025/multimodal_vlm/mitigating_object_hallucinations_via_sentence-level_early_intervention.md)
- [\[ICCV 2025\] DASH: Detection and Assessment of Systematic Hallucinations of VLMs](../../ICCV2025/multimodal_vlm/dash_detection_and_assessment_of_systematic_hallucinations_of_vlms.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[NeurIPS 2025\] When Semantics Mislead Vision: Mitigating Large Multimodal Models Hallucinations](when_semantics_mislead_vision_mitigating_large_multimodal_models_hallucinations_.md)
- [\[NeurIPS 2025\] HermesFlow: Seamlessly Closing the Gap in Multimodal Understanding and Generation](hermesflow_seamlessly_closing_the_gap_in_multimodal_understanding_and_generation.md)

<!-- RELATED:END -->
