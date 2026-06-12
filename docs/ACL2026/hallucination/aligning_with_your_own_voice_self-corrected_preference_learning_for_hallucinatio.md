---
title: >-
  [Paper Note] Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs
description: >-
  [ACL 2026][Hallucination Detection][AVES-DPO] The AVES-DPO framework is proposed: utilizing consistency-based multi-model verification (YOLO/GroundingDINO/Qwen3-VL) to detect fine-grained hallucinations in an LVLM's own…
tags:
  - "ACL 2026"
  - "Hallucination Detection"
  - "AVES-DPO"
  - "Self-correction"
  - "Preference Learning"
  - "LVLM Hallucination"
  - "Distribution Alignment"
date: 2026-05-08
content_hash: b340803121d55731
---

# Aligning with Your Own Voice: Self-Corrected Preference Learning for Hallucination Mitigation in LVLMs

**Conference**: ACL 2026  
**arXiv**: [2604.24395](https://arxiv.org/abs/2604.24395)  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: AVES-DPO, Self-correction, Preference Learning, LVLM Hallucination, Distribution Alignment

## TL;DR
The AVES-DPO framework is proposed: utilizing consistency-based multi-model verification (YOLO/GroundingDINO/Qwen3-VL) to detect fine-grained hallucinations in an LVLM's own responses across object/attribute/relation levels, followed by self-correction and detail enrichment by the same LVLM. This ensures preference pairs remain within the target model's "internal distribution." With only 5.2K samples, it surpasses SOTA methods dependent on GPT-4V teachers while achieving approximately 25× data efficiency across multiple benchmarks.

## Background & Motivation

**Background**: DPO has become a mainstream approach for mitigating LVLM hallucinations by constructing preference pairs (preferred as correct detailed responses, dispreferred as hallucinated responses) to guide the model.

**Limitations of Prior Work**: (1) Existing methods (POVID, oDPO, mDPO, HALVA, etc.) rely almost entirely on closed-source models like GPT-4V to synthesize positive samples or perturb negative samples, which is costly and difficult to control. (2) A **generation mode distribution mismatch** exists between closed-source teachers and target LVLMs—differences in writing style and vocabulary distribution mean that optimizing for "external styles" may lead to imitating the teacher rather than factual improvement. (3) Existing datasets and training objectives are biased toward object-level hallucinations, with insufficient coverage of fine-grained attribute and relation hallucinations (e.g., color, pose, spatial relations).

**Key Challenge**: The gap between teacher-generated positive samples and the target model’s internal distribution is amplified by the DPO log-likelihood ratio as "style difference > factual difference," diluting alignment efficiency.

**Goal**: To derive preferred responses from the target model itself (in-distribution) while covering object, attribute, and relation hallucinations.

**Key Insight**: Empirical pre-experiments show that preference margins for self-corrected responses are centered at 0 (in-distribution), whereas responses corrected by GPT-4V show large negative margins (out-of-distribution), proving that self-correction is naturally compatible with the target model.

**Core Idea**: Construct entirely in-distribution preference pairs via consistency verification and self-correction, then apply standard DPO.

## Method

### Overall Architecture

The process involves two main phases: (1) Hallucination Verification—Initial responses are parsed into a scene graph $G=(O, A, R)$. Objects are verified using YOLO and Grounding DINO with Qwen3-VL (30B/32B) as an arbitrator. Attributes and relations are verified via consensus between two Qwen3-VL models. A consensus mechanism requires independent models to agree on a diagnosis; otherwise, it is labeled Ambiguous. (2) Self-correction—The LVLM rewrites responses based on the diagnosis: first through Factual Rectification (removing/replacing hallucinations) and then Detailed Enrichment (adding missing visual details while avoiding previously hallucinated objects). Responses are iteratively verified and refined (up to 3-5 rounds). The final enriched response serves as $y^+$, and the initial hallucinated response as $y^-$ for DPO.

### Key Designs

1. **O/A/R Consensus Verification**:
    - **Function**: Provides precise detection across object (existence), attribute (color/material/pose/state), and relation (spatial/action) levels to minimize single-model noise.
    - **Mechanism**: For an element $x$, two verifiers $V_1, V_2$ must yield the same label $l$: $L(x) = l$ if $V_1(x) = V_2(x) = l$, else it is Ambiguous. For objects, YOLOv8x-worldv2 and Grounding DINO are used with cosine similarity grouping. Attributes and relations are verified only for confirmed factual objects using a dictionary of 148 objects, 38 attributes, and 17 relations.
    - **Design Motivation**: Prioritizing reliability through consensus and layered verification allows for more accurate error localization.

2. **Factual Rectification + Detailed Enrichment**:
    - **Function**: Transforms hallucinated responses into correct, detailed versions for DPO $y^+$ while preserving the model’s internal distribution.
    - **Mechanism**: Phase A (Factual Rectification) prompts the LVLM to delete or replace hallucinated elements, maintaining its original linguistic style. Phase B (Detailed Enrichment) prompts the model to add descriptive details, explicitly excluding previously flagged hallucinations. A closed-loop "verify-rectify-enrich" cycle ensures high data quality; 7B models undergo up to 3 rounds and 13B up to 5 rounds.
    - **Design Motivation**: Simple deletion results in low information, while free enrichment can introduce new errors. A structured cycle ensures both factuality and descriptiveness.

3. **In-distribution DPO Training (AVES-DPO Loss)**:
    - **Function**: Trains the model using standard DPO on the self-constructed $\mathcal{D}_{SC} = \{(x, y^+, y^-)\}$.
    - **Mechanism**: The loss is defined as:
      $$\mathcal{L}_{\text{AVES-DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y^+, y^-)}\big[\log \sigma\big(\beta \log \frac{\pi_\theta(y^+ \mid x)}{\pi_{\text{ref}}(y^+ \mid x)} - \beta \log \frac{\pi_\theta(y^- \mid x)}{\pi_{\text{ref}}(y^- \mid x)}\big)\big]$$
      $y^-$ is the initial hallucinated response, while $y^+$ is the final verified enriched response.
    - **Design Motivation**: Since $y^+$ and $y^-$ originate from the same model, the preference margin distribution is positive-centered. Gradients focus on factual accuracy rather than stylistic differences, enhancing data efficiency.

### Loss & Training

LLaVA-1.5-7B and 13B were fine-tuned with LoRA (rank=128, alpha=256) for 1 epoch using $\beta=0.1$ and the AdamW optimizer on an RTX A6000 Ada. Training used 5.2K pairs for 7B and 4.6K for 13B.

## Key Experimental Results

### Main Results

| Method | Data Size | Feedback Source | Obj-Hal CHAIR$_S\downarrow$ | Obj-Hal CHAIR$_I\downarrow$ | AMBER Hal Rate$\downarrow$ | MMHal Score$\uparrow$ |
|------|-------|---------|---------------------------|---------------------------|--------------------------|--------------------|
| LLaVA-1.5-7B | - | - | 51.4 | 14.8 | 32.5 | 2.22 |
| LLaVA-RLHF | 122K | Self-Reward | 53.6 | 14.8 | 42.5 | 2.06 |
| HALVA | 21.5K | GPT-4V | 41.4 | 11.7 | 32.2 | 2.25 |
| POVID | 17K | GPT-4V | 45.8 | 13.9 | 31.5 | 2.18 |
| oDPO | 19K | GPT-4V | 34.3 | 9.5 | 25.1 | 2.50 |
| mDPO | 10K | GPT-4V | 35.7 | 9.8 | 24.5 | 2.39 |
| SENTINEL | 8.6K | Self | 12.6 | 4.0 | 14.6 | 2.04 |
| **AVES-DPO (Ours)** | **5.2K** | **Self** | **12.2** | **3.9** | **12.6** | **2.35** |

On AMBER 7B relations, the model scored 66.6 (vs SENTINEL 48.7 and base 58.5). AMBER Acc (82.3) and F1 (86.9) also achieved top results.

### Ablation Study

| Configuration | Obj-Hal CHAIR$_S$ | AMBER CHAIR | AMBER Hal Rate |
|------|------------------|------------|----------------|
| 7B base | 51.4 | 7.1 | 32.5 |
| + GPT-4V Teacher Supervision (5.2K) | 38.4 | 7.6 | 31.7 |
| **+ AVES-DPO Self-Correction (5.2K)** | **12.2** | **3.3** | **12.6** |
| AVES-DPO w/ Two-Phase (Default) | 12.2 | 3.3 | 12.6 |
| AVES-DPO w/ Phase-1 only (No Enrichment) | 16.6 | 3.5 | 12.8 |
| AVES-DPO w/ Phase-2 only (No Rectification) | 19.8 | 3.9 | 17.0 |
| Data Size 5.2K (Sweet Spot) | 12.2 | 3.3 | 12.6 |
| Data Size 10.9K (Over-correction) | 10.6 | 2.8 | 10.1 (Cover dropped to 37.8) |

### Key Findings
- Self-correction (CHAIR$_S$ 12.2) significantly outperforms GPT-4V teacher supervision (CHAIR$_S$ 38.4) with the same data volume, highlighting the importance of distribution alignment.
- The "Two-Phase" approach is essential; correction alone yields overly brief outputs, while enrichment alone introduces new errors.
- Benefits plateau after 5.2K samples; excessive data leads to "over-correction" which reduces descriptive richness.
- Relation hallucinations are the most challenging, yet AVES-DPO (13B) reached 77.5 on AMBER relations.

## Highlights & Insights
- Empirically validates that in-distribution preference data is critical for DPO based on margin distribution shifts.
- Demonstrates that a consensus mechanism using open-source models can match GPT-4V labeling quality (87.1% win rate).
- Identifies a trade-off in larger models (13B), where suppressing hallucinations slightly reduced MME scores, suggesting a need for scale-aware alignment.
- The automated loop for data construction ensures high quality and efficiency.

## Limitations & Future Work
- Verification is currently limited to single-object levels, lacking coverage for complex multi-object interactions.
- MME degradation in 13B models highlights the need for scale-aware training strategies.
- The system depends on external detectors (YOLO/GDINO), limiting domain adaptability.
- The fixed vocabulary restricted to COCO and GQA limits performance in open-vocabulary scenarios.

## Related Work & Insights
- **vs POVID (Zhou et al., 2024a)**: Unlike POVID, which perturbs images and uses GPT-4V for negatives, this method uses the model's own initial responses, which are more natural.
- **vs SENTINEL (Peng et al., 2025)**: While both use self-feedback, AVES-DPO provides O/A/R level refinement and iterative enrichment, leading to better relation-level improvements.
- **vs mDPO / oDPO**: AVES-DPO eliminates the need for expensive GPT-4V teachers or image masking, reducing data costs by 25×.

## Rating
- Novelty: ⭐⭐⭐⭐ Grounded "in-distribution self-correction" logic with an effective two-phase design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered five major benchmarks and extensive ablation paths.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow from empirical motivation to results.
- Value: ⭐⭐⭐⭐ Highly reproducible for the open-source community, offering significant cost reduction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Causally-Grounded Dual-Path Attention Intervention for Object Hallucination Mitigation in LVLMs](../../AAAI2026/hallucination/causally-grounded_dual-path_attention_intervention_for_objec.md)
- [\[ACL 2026\] Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)
- [\[ACL 2026\] Vocabulary Hijacking in LVLMs: Unveiling Critical Attention Heads by Excluding Inert Tokens to Mitigate Hallucination](vocabulary_hijacking_in_lvlms_unveiling_critical_attention_heads_by_excluding_in.md)
- [\[ACL 2026\] MeasHalu: Mitigation of Scientific Measurement Hallucinations for LLMs](meashalu_mitigation_of_scientific_measurement_hallucinations_for_large_language_.md)
- [\[ACL 2026\] Spotlight and Shadow: Attention-Guided Dual-Anchor Introspective Decoding for MLLM Hallucination Mitigation](spotlight_and_shadow_attention-guided_dual-anchor_introspective_decoding_for_mll.md)

</div>

<!-- RELATED:END -->
