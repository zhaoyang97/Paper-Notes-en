---
title: >-
  [Paper Note] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection
description: >-
  [NeurIPS 2025][Social Computing][HOI Detection] This paper proposes the VDRP framework, which addresses two core challenges in zero-shot HOI detection — intra-class visual diversity and inter-class visual entanglement —…
tags:
  - "NeurIPS 2025"
  - "Social Computing"
  - "HOI Detection"
  - "Zero-shot Learning"
  - "Prompt Learning"
  - "CLIP"
  - "Visual Diversity"
date: 2026-05-08
content_hash: d76322f270f79346
---

# VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection

**Conference**: NeurIPS 2025
**arXiv**: [2510.25094](https://arxiv.org/abs/2510.25094)  
**Code**: [https://github.com/mlvlab/VDRP](https://github.com/mlvlab/VDRP)  
**Area**: Video/Image Understanding
**Keywords**: HOI Detection, Zero-shot Learning, Prompt Learning, CLIP, Visual Diversity

## TL;DR
This paper proposes the VDRP framework, which addresses two core challenges in zero-shot HOI detection — intra-class visual diversity and inter-class visual entanglement — through visual diversity-aware prompt learning (via group-level variance injection and Gaussian perturbation) and region-aware prompt augmentation (via LLM-generated regional concept retrieval).

## Background & Motivation
Human-Object Interaction (HOI) detection requires localizing humans and objects and recognizing the interactions between them. Zero-shot HOI detection demands that models generalize to unseen verb-object combinations at test time, introducing two core visual challenges:

**Intra-class visual diversity**: The same verb (e.g., "holding a baseball glove") exhibits substantial visual variation across different poses, viewpoints, and scenes. The authors quantitatively show that verb classes have a significantly higher diversity score ($0.364 \pm 0.060$) than object classes ($0.274 \pm 0.048$), indicating that a single static prompt cannot capture the visual variation of verbs.

**Inter-class visual entanglement**: Semantically distinct verbs (e.g., "eating," "licking," "sitting next to") produce highly similar visual patterns under global or union-region features. t-SNE visualizations reveal substantial overlap among different verb categories.

**Limitations of Prior Work**: Most CLIP-based prompt methods (GEN-VLKT, ADA-CM) use only a single static prompt per verb; CMMP incorporates spatial cues but the text prompts remain region-agnostic; EZ-HOI leverages LLM descriptions but ignores intra-class variation.

**Core Idea**: Simultaneously encode visual variation statistics (variance injection and perturbation) and region-specific semantics (concept retrieval and augmentation) into the prompt embeddings, with the two components complementarily addressing the two challenges above.

## Method

### Overall Architecture
A two-stage HOI detection pipeline is adopted: (1) a frozen DETR detector localizes humans and objects; (2) a CLIP-based interaction classifier extracts human ($\mathbf{x}_h$), object ($\mathbf{x}_o$), and union-region ($\mathbf{x}_u$) features via lightweight adapters. The key innovation lies in the text prompt side: visual diversity-aware prompts are first generated, then augmented with regional concepts to produce region-aware prompts, and the final verb classification is performed by averaging logits over the three regions.

### Key Designs

1. **Visual Diversity-aware Prompt Learning**:

    - Union-region CLS features are extracted for each verb $v$ from the training set, and the variance $\boldsymbol{\sigma}_v^2$ is computed.
    - Semantically similar verb groups $\mathcal{G}(v)$ are constructed based on cosine similarity in CLIP text embedding space, and the group-level variance is computed as $\bar{\boldsymbol{\sigma}}_v^2 = \frac{1}{|\mathcal{G}(v)|}\sum_{v' \in \mathcal{G}(v)} \boldsymbol{\sigma}_{v'}^2$ (a stabilized estimate, particularly important for rare verbs).
    - An MLP maps the group-level variance to a modulation vector $\mathbf{d}_v$, which is injected into the shared context embedding: $\hat{\mathbf{E}}_v = \mathbf{E} + \mathbf{d}_v \alpha$.
    - After passing through the CLIP text encoder, variance-guided Gaussian perturbation is applied: $\tilde{\mathbf{t}}^v = \mathbf{t}^v + (\epsilon \odot \tilde{\boldsymbol{\sigma}}_v)\beta$.

2. **Region-aware Prompt Augmentation**:

    - An LLM (LLaMA-7B / GPT-4) generates $K$ visual concept descriptions per verb for each region (human / object / union).
    - These are encoded into concept pools $\mathcal{C}_{(\cdot)}^v$ using the CLIP text encoder.
    - Given a region feature $\mathbf{x}_{(\cdot)}$, cosine similarities with the concepts are computed, and Sparsemax (rather than Softmax) is applied to produce sparse weights, retaining only the most relevant concepts.
    - A weighted aggregation yields the regional concept vector $\bar{\mathbf{c}}_{(\cdot)}^v$, which augments the diversity-aware prompt: $\hat{\mathbf{t}}_{(\cdot)}^v = \mathbf{t}^v + \bar{\mathbf{c}}_{(\cdot)}^v \gamma$.

3. **Spatially Augmented Union-region Features**: A SpatialHead fuses union-region features with human and object features along with their bounding boxes, incorporating spatial priors.

### Loss & Training
- Focal Loss is used for multi-label verb classification.
- Lightweight adapters are inserted into multiple Transformer blocks of the CLIP visual encoder; only 4.50M parameters are trained.

## Key Experimental Results

### Main Results

| Zero-shot Setting | Metric | Ours (VDRP) | Prev. SOTA (EZ-HOI) | Gain |
|-----------|------|-----------|-----------------|------|
| NF-UC | HM / Unseen | 33.85 / 36.45 | 31.76 / 33.66 | +2.09 / +2.79 |
| RF-UC | HM / Unseen | 32.77 / 31.29 | 31.18 / 29.02 | +1.59 / +2.27 |
| UO | HM / Unseen | 34.41 / 36.13 | 32.14 / 33.28 | +2.27 / +2.85 |
| UV | HM / Unseen | 29.80 / 26.69 | 29.09 / 25.10 | +0.71 / +1.59 |

### Ablation Study

| Configuration | NF-UC Unseen | RF-UC Unseen | UO Unseen | UV Unseen |
|------|-------------|-------------|-----------|-----------|
| BASE | 28.32 | 25.64 | 28.60 | 22.41 |
| + VDP (diversity prompt) | 32.19 | 29.16 | 33.29 | 23.78 |
| + RAP (region prompt) | 34.93 | 26.46 | 33.90 | 24.53 |
| + VDRP (full) | 36.45 | 31.29 | 36.13 | 26.69 |

### Key Findings
- Both VDP and RAP individually yield significant improvements, and their combination achieves the best performance, demonstrating that intra-class diversity and inter-class discriminability are complementary dimensions.
- Under the NF-UC setting, VDRP improves Unseen class performance by +8.13 (from 28.32 to 36.45), substantially outperforming each individual module.
- Only 4.50M trainable parameters are required, far fewer than CLIP4HOI (56.7M) and HOICLIP (66.18M).

## Highlights & Insights
- **Variance as Signal**: Converting intra-class visual variance from "noise" into "signal" by injecting it into prompts to guide learning is an elegant design principle.
- **Sparsemax for Concept Retrieval**: Compared to Softmax, Sparsemax assigns exact zero weights to irrelevant concepts, preventing noise interference.
- **Quantitative Analysis-Driven Design**: The methodology of first quantitatively characterizing the problem via diversity scores and t-SNE, then designing targeted solutions, is a valuable practice worth emulating.

## Limitations & Future Work
- Regional concepts depend on LLM generation, and their quality is bounded by the LLM's capability.
- The definition of "semantically similar verb groups" for group-level variance relies on CLIP text embedding similarity, which may introduce bias.
- Evaluation is conducted only on HICO-DET; validation on other HOI benchmarks such as V-COCO is lacking.
- Hyperparameters such as perturbation strengths $\alpha, \beta, \gamma$ require careful tuning.

## Related Work & Insights
- **vs. EZ-HOI**: EZ-HOI uses LLM descriptions to distinguish semantic differences between verbs but ignores intra-class variation; VDRP addresses both dimensions simultaneously.
- **vs. CMMP**: CMMP incorporates spatial cues but the text prompts are region-agnostic; VDRP's regional concept retrieval operates at a finer granularity.
- **vs. CoOp/CoCoOp**: This work extends prompt learning from classification tasks to the multi-region setting of HOI detection.

## Rating
- Novelty: ⭐⭐⭐⭐ The combined design of variance injection into prompts and region-aware concept retrieval is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Full coverage of four zero-shot settings with thorough ablations, though cross-dataset validation is absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is analyzed with quantitative clarity and method diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Meaningfully advances zero-shot HOI detection; the variance injection idea is transferable to other visual tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Visual Proxy for Compositional Zero-Shot Learning](../../ICCV2025/social_computing/learning_visual_proxy_for_compositional_zero-shot_learning.md)
- [\[NeurIPS 2025\] Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals](worse_than_zero-shot_a_fact-checking_dataset_for_evaluating_the_robustness_of_ra.md)
- [\[NeurIPS 2025\] Policy-as-Prompt: Turning AI Governance Rules into Guardrails for AI Agents](policy-as-prompt_turning_ai_governance_rules_into_guardrails_for_ai_agents.md)
- [\[NeurIPS 2025\] DeepTraverse: A Depth-First Search Inspired Network for Algorithmic Visual Understanding](deeptraverse_a_depth-first_search_inspired_network_for_algorithmic_visual_unders.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->
