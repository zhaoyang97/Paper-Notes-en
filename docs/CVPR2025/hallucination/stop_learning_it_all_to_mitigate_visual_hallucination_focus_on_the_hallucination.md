---
title: >-
  [Paper Note] Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target
description: >-
  [CVPR 2025][Hallucination Detection][Visual Hallucination] Proposes **TL-DPO** (Target-Learning DPO), which limits traditional full-sentence preference learning to the **target chunk where hallucination occurs** and the **corresponding image region**. By excluding irrelevant signals through target generation loss and target condition loss, it reduces CHAIR_s on LLaVA-1.5 from 66.8 to 20.1, while improving LLaVA-Bench from 63.4 to 71.2.
tags:
  - "CVPR 2025"
  - "Hallucination Detection"
  - "Visual Hallucination"
  - "Target Learning"
  - "DPO"
  - "Preference Optimization"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 68237a61215ddba4
---

# Stop Learning It All to Mitigate Visual Hallucination, Focus on the Hallucination Target

**Conference**: CVPR 2025  
**arXiv**: [2506.11417](https://arxiv.org/abs/2506.11417)  
**Code**: Not publicly available  
**Area**: Hallucination Detection  
**Keywords**: Visual Hallucination, Target Learning, DPO, Preference Optimization, Multimodal Large Language Models

## TL;DR

Proposes **TL-DPO** (Target-Learning DPO), which limits traditional full-sentence preference learning to the **target chunk where hallucination occurs** and the **corresponding image region**. By excluding irrelevant signals through target generation loss and target condition loss, it reduces CHAIR_s on LLaVA-1.5 from 66.8 to 20.1, while improving LLaVA-Bench from 63.4 to 71.2.

## Background & Motivation

Multimodal Large Language Models (MLLMs) excel in vision-language tasks but suffer from severe **hallucination issues**—generating information about objects not present in the image or inaccurate spatial descriptions. Preference learning methods (e.g., RLHF, DPO) are widely used to mitigate hallucinations, but their clinical effectiveness remains limited.

**Limitations of Prior Work**:
1. **Limitations of Global Learning**—Traditional preference learning optimizes at the full response level, whereas hallucinations may only occur in a localized part of the response (e.g., in "the clock shows 11:20," only "11:20" is incorrect). Global optimization tends to learn a large number of signals irrelevant to hallucinations.
2. **Attention Shift**—Analysis of attention maps reveals that after traditional DPO training, the model may shift its attention from the target objects in the image to textual signals, leading to overfitting to text patterns rather than correcting visual understanding.
3. **Interference from Irrelevant Signals**—In preference learning, most content in "good/bad" response pairs is identical, and the true difference lies only in the hallucinated portion. However, the model is forced to learn preferences over the entire response, which is inefficient and may lead to learning incorrect signals.

**Key Challenge**: When humans correct an error, they only modify where it went wrong (e.g., correcting a typo). However, existing preference learning methods require the model to "rewrite the entire content," leading to low learning efficiency and negative side effects.

**Key Insight**: Focusing only on the erroneous parts just like humans do—limiting preference learning to the hallucination target, which includes the erroneous text chunks in the response and the target object regions in the image that cause the hallucinations.

## Method

### Overall Architecture

TL-DPO consists of two complementary loss functions: (1) **Target Generation Loss**—computes DPO loss only on the text chunks where hallucinations occur, filtering out irrelevant text signals; (2) **Target Condition Loss**—by masking the target objects in the image that lead to hallucinations, it trains the model to learn to leverage the visual information of the target region to provide correct answers. The training data is constructed based on the Visual Genome dataset, consisting of hallucinated responses, correct responses, and hallucination target location information (including text chunk locations and image bounding boxes).

### Key Designs

1. **Target Generation Loss**
    - **Function**: Narrows the preference comparison of DPO from the full-sentence level down to the hallucination chunk level, discarding parts of the response irrelevant to the hallucination.
    - **Mechanism**: Assume only a part $y^t$ of the response $y$ contains hallucination information. While the standard DPO loss compares the reward difference of the entire response $(y_r, y_h)$, TL-DPO only compares the reward difference of the target chunks $(y_r^t, y_h^t)$:
     $$\mathcal{L}_t = -\mathbb{E}_{(x, y_r^t, y_h^t) \sim D} [\log \sigma(u(x, y_r^t, y_h^t))]$$
     For example, for "the clock shows about 11:20 (incorrect)" and "the clock shows about 15:26 (correct)", $y_h^t$ = "11:20", $y_r^t$ = "15:26", and preferences are computed only over these two spans.
    - **Design Motivation**: Theoretical proof (Theorem 3.3) shows that under Assumption 3.1 (hallucination-irrelevant signals do not affect reward differences), target-level DPO is equivalent to full DPO, but the former has a much smaller hypothesis space (Proposition 1), requiring fewer samples to achieve the same generalization error.

2. **Target Condition Loss**
    - **Function**: Trains the model to learn to utilize the visual information of the target region in the image, rather than relying on textual priors to answer.
    - **Mechanism**: Given a hallucination-related image region $m_i^t$ (indicated by a bounding box), a masked image $\tilde{m}_i^t$ is constructed (with this region masked out). This forms a preference pair of $(m_i, q, y_r)$ (original image + correct answer) vs $(\tilde{m}_i^t, q, y_r)$ (masked image + correct answer), training the model to prefer using the complete image information:
     $$\mathcal{L}_c = -\mathbb{E}_{(m_i, \tilde{m}_i^t, x, y_r) \sim D} [\log \sigma(u^*(m_i, \tilde{m}_i^t, x, y_r))]$$
     where $u^* = r(m_i, x, y_r) - r(\tilde{m}_i^t, x, y_r)$.
    - **Design Motivation**: Addresses the issue in preference learning where the model might overfit text patterns and neglect image information. Through the masked/unmasked image pairs, it explicitly guides the model to focus on the visual regions relevant to the hallucination.

3. **Final Training Objective**
    - **Function**: Integrates preference learning for both textual and visual targets.
    - **Mechanism**: $\mathcal{L}_{TL-DPO} = \mathcal{L}_t + \mathcal{L}_c$, where the two losses are complementary—target generation loss ensures precise corrections on the textual level, and target condition loss ensures correct focus on the visual level.
    - **Design Motivation**: Using either loss alone is insufficient—solely using target generation loss may still result in the model ignoring the image, whereas solely using target condition loss may not correct textual hallucinations with enough precision.

### Loss & Training

Fine-tuned LLaVA-v1.5-7B using LoRA, with batch size = 32, 3 epochs, learning rate of 1e-5, cosine schedule, warm-up 0.1. DPO $\beta=0.1$, LoRA $\alpha=128$, rank = 64. The training data is constructed based on the VG dataset, where responses are generated using baseline models, and GPT-4 is utilized to determine correctness and generate revisions, constructing a preference dataset that contains target location details.

## Key Experimental Results

### Main Results (LLaVA-1.5 Baseline, Comparison with Other Preference Learning Methods)

| Method | CHAIR_s ↓ | CHAIR_i ↓ | POPE ↑ | MMHal ↑ | MMBench ↑ | LLaVA-Bench ↑ |
|------|-----------|-----------|--------|---------|-----------|---------------|
| LLaVA-1.5 | 66.8 | 12.7 | 85.9 | 2.42 | 63.0 | 63.4 |
| +RLHF-V | 44.6 | 7.9 | 86.2 | 2.59 | 63.6 | 65.4 |
| +HA-DPO | 37.2 | 10.0 | 86.9 | 1.97 | 64.0 | 66.2 |
| +HALVA | 46.6 | 23.1 | 87.0 | 2.25 | 66.1 | 67.2 |
| **+TL-DPO** | **20.1** | **5.2** | **86.95** | **2.72** | **67.8** | **71.2** |

### Cross-Model Generalization

| Model | CHAIR_s (Baseline $\rightarrow$ +TL-DPO) | POPE (Baseline $\rightarrow$ +TL-DPO) | MMBench (Baseline $\rightarrow$ +TL-DPO) |
|------|----------------------|---------------------|----------------------|
| LLaVA-1.5 | 66.8 $\rightarrow$ 20.1 | 85.9 $\rightarrow$ 87.0 | 63.0 $\rightarrow$ 67.8 |
| LLaVA-Next | 29.1 $\rightarrow$ 25.1 | 84.8 $\rightarrow$ 87.1 | 63.0 $\rightarrow$ 63.1 |
| InternVL-2.5(8B) | 18.4 $\rightarrow$ 7.6 | 86.5 $\rightarrow$ 87.0 | 68.6 $\rightarrow$ 80.0 |
| Llama3 | 5.5 $\rightarrow$ 7.1 | 82.8 $\rightarrow$ 87.1 | 85.8 $\rightarrow$ 87.3 |

### Ablation Study

| Configuration | CHAIR_s ↓ | CHAIR_i ↓ | POPE ↑ | MMHal ↑ | LLaVA-Bench ↑ |
|------|-----------|-----------|--------|---------|---------------|
| LLaVA-1.5 Baseline | 66.8 | 12.7 | 85.9 | 2.42 | 63.4 |
| + Target Condition Only | 32.4 | 8.6 | 84.4 | 2.58 | 66.5 |
| + Target Generation Only | 14.6 | 6.1 | **89.6** | 2.70 | 68.7 |
| **TL-DPO (Combination of Both)** | **20.1** | **5.2** | 87.0 | **2.72** | **71.2** |

### Key Findings

1. TL-DPO reduces CHAIR_s by 57% compared to the strongest baseline HALVA (20.1 vs 46.6), while comprehensively improving overall performance.
2. Using target generation loss alone yields an even lower CHAIR_s (14.6) but slightly decreases POPE; the combination of both losses achieves the best trade-off.
3. TL-DPO is effective across multiple models (LLaVA, Qwen, InternVL, etc.), demonstrating good generalization capability.
4. On InternVL-2.5, CHAIR_s drops significantly from 18.4 to 7.6, while MMBench increases from 68.6 to 80.0 (+11.4), which is remarkably effective.

## Highlights & Insights

1. **Intuition of "Correcting Errors Like Humans"**—Analogous to how humans only modify erroneous parts in drawing or writing rather than redrawing or rewriting entirely, this simple intuition inspires the core idea of target learning.
2. **Solid Theoretical Foundation**—The equivalence between target-level DPO and full DPO is proven starting from the Bradley-Terry model (Theorem 3.3), and it is proven that target learning requires fewer samples (Proposition 1), showing consistency between theory and experiment.
3. **Complementary Dual Losses**—Target generation loss addresses "what text to correct," and target condition loss addresses "what image region to look at." The combination of both achieves precise dual corrections on both text and vision levels.
4. **No Drop in Comprehensive Performance**—While most hallucination mitigation methods suffer from a drop in comprehensive benchmark scores, TL-DPO achieves significant improvements on MMBench and LLaVA-Bench. This demonstrates that excluding irrelevant signals not only reduces hallucinations but also improves overall learning.

## Limitations & Future Work

1. The weights of the target generation loss and target condition loss are fixed at 1:1, without exploring the impact of different weightings.
2. The construction of training data relies on GPT-4 to identify hallucinations and generate corrections, meaning data quality is constrained by GPT-4's capabilities.
3. For some models (e.g., Qwen VL Chat), certain comprehensive metrics decreased after adding TL-DPO, showing inconsistent generalization.
4. Assumption 3.1 (hallucination-irrelevant signals do not affect reward differences) may not hold perfectly in practice.

## Related Work & Insights

- **Preference Learning**: The evolution from RLHF to DPO has made preference learning more concised. This paper refines DPO further from sentence-level to target-level, marking a significant advancement in the granularity of preference learning.
- **Visual Hallucination**: Methods such as HA-DPO, POVID, and RLHF-V mitigate hallucinations from data-driven or global optimization perspectives. This work is the first to approach the problem from the angle of "excluding irrelevant signals."
- **Insights**: In preference learning, "learning less" can be more effective than "learning more"—excluding irrelevant signals not only increases efficiency but also enhances performance, a principle that may be applicable to other preference learning scenarios.

## Rating

⭐⭐⭐⭐ — Clear intuition, solid theory, and remarkable results (with CHAIR_s reduced by ~70%). The cross-model generalization is thoroughly validated. However, the practical justification of the assumptions and the heavy reliance of data construction on GPT-4 remain potential limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset](phd_a_chatgpt-prompted_visual_hallucination_evaluation_dataset.md)
- [\[ICML 2026\] Learning from Fine-Grained Visual Discrepancies: Mitigating Multimodal Hallucinations via In-Context Visual Contrastive Optimization](../../ICML2026/hallucination/learning_from_fine-grained_visual_discrepancies_mitigating_multimodal_hallucinat.md)
- [\[ICML 2026\] A Unified Definition of Hallucination: It's The World Model, Stupid!](../../ICML2026/hallucination/a_unified_definition_of_hallucination_its_the_world_model_stupid.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](../../ACL2025/hallucination/retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)
- [\[ICLR 2026\] Visual Multi-Agent System: Mitigating Hallucination Snowballing via Visual Flow](../../ICLR2026/hallucination/visual_multi-agent_system_mitigating_hallucination_snowballing_via_visual_flow.md)

</div>

<!-- RELATED:END -->
