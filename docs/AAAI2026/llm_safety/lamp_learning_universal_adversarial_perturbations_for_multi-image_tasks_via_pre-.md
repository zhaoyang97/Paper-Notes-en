---
title: >-
  [Paper Note] LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models
description: >-
  [AAAI2026][LLM Safety][Universal Adversarial Perturbation] This paper proposes LAMP, a black-box Universal Adversarial Perturbation (UAP) learning method targeting multi-image MLLMs. By incorporating attention constraint…
tags:
  - "AAAI2026"
  - "LLM Safety"
  - "Universal Adversarial Perturbation"
  - "Multi-Image MLLM"
  - "Black-box Attack"
  - "Attention Manipulation"
  - "Transferable Attack"
date: 2026-05-08
content_hash: 15b06f577a9d60cf
---

# LAMP: Learning Universal Adversarial Perturbations for Multi-Image Tasks via Pre-trained Models

**Conference**: AAAI2026
**arXiv**: [2601.21220](https://arxiv.org/abs/2601.21220)  
**Code**: None  
**Area**: AI Security
**Keywords**: Universal Adversarial Perturbation, Multi-Image MLLM, Black-box Attack, Attention Manipulation, Transferable Attack

## TL;DR
This paper proposes LAMP, a black-box Universal Adversarial Perturbation (UAP) learning method targeting multi-image MLLMs. By incorporating attention constraints and a contagious loss, LAMP enables cross-model and cross-task transferable attacks by perturbing only a small subset of input images.

## Background & Motivation

### State of the Field

**Background**: Multimodal large language models (MLLMs) now support multi-image inputs (e.g., comparison, reasoning, temporal understanding), yet their adversarial robustness in such settings remains largely unexplored.

### Limitations of Prior Work

**Limitations of Prior Work**: Existing adversarial attacks are primarily designed for **single-image** scenarios and mostly operate under white-box settings, rendering them ill-suited for practical black-box deployment.

### Root Cause

**Key Challenge**: In real-world scenarios (e.g., images on social media processed by MLLMs), attackers cannot control the number or order of images received by the model. Existing single-image UAP methods therefore exhibit limited effectiveness in multi-image settings.

### Solution

**Goal**: How can one learn a small, fixed set of Universal Adversarial Perturbations under a black-box setting such that they effectively attack multi-image MLLMs, even when the attacker has no control over the number or order of images at inference time?

## Method

### Overall Architecture
A pre-trained surrogate model (Mantis-CLIP) is used to learn UAPs while keeping the MLLM parameters frozen; only the perturbations $\delta_k$ (subject to $\|\delta_k\|_\infty \leq \epsilon$) are optimized. The total loss comprises five terms:

$$\mathcal{L}_{adv} = \lambda_1 \mathcal{L}_{adv}^{lm} + \lambda_2 \mathcal{L}_{adv}^{dec} + \lambda_3 \mathcal{L}_{adv}^{h} + \lambda_4 \mathcal{L}_{adv}^{ctg} + \lambda_5 \mathcal{L}_{adv}^{ias}$$

### Key Designs

1. **Adversarial Language Modeling Loss** $\mathcal{L}_{adv}^{lm}$: Reduces the generation probability of correct tokens.
$$\mathcal{L}_{adv}^{lm} = -\frac{1}{N}\sum_{i=1}^{N}\log(1 - P_\theta(t_{i+1}|s_{1:i}))$$

2. **Hidden States Divergence Loss** $\mathcal{L}_{adv}^{dec}$: Maximizes the cosine distance between clean and adversarial hidden states.
$$\mathcal{L}_{adv}^{dec} = \frac{1}{L}\sum_{l=1}^{L}\cos(z_l^{adv}, z_l^{clean})$$

3. **Attention via Pompeiu-Hausdorff Distance** $\mathcal{L}_{adv}^{h}$: Employs the Hausdorff distance to measure the worst-case deviation between clean and adversarial attention weights, capturing local discrepancies more effectively than KL divergence.

4. **Contagious Loss** $\mathcal{L}_{adv}^{ctg}$ (core innovation): Encourages clean tokens to attend more strongly to perturbed image tokens in self-attention, thereby propagating adversarial effects from perturbed images to clean ones.
$$\mathcal{L}_{adv}^{ctg} = -\frac{1}{LH}\sum_{l}\sum_{h}\sum_{i \in \mathcal{C}}\sum_{j \in \mathcal{N}} A^{(l)}_{:,h,i,j}$$

5. **Index-Attention Suppression Loss** $\mathcal{L}_{adv}^{ias}$: Suppresses the attention of image tokens toward their positional index text tokens, enabling position-invariant attacks.

## Key Experimental Results

### Main Results

| Setting | Avg. Best Baseline | LAMP | Δ (pp) |
|---|---|---|---|
| Average across all models | 56.3% | 75.8% | **+19.5** |
| Mantis-CLIP | 51.5% | 71.9% | +20.4 |
| VILA-1.5 | 56.1% | 76.2% | +20.1 |
| LLaVA-v1.6 | 58.5% | 78.9% | +20.4 |
| Qwen-2.5 | 62.5% | 79.4% | +16.9 |

- Cross-model zero-shot transfer attacks substantially outperform all baselines.
- Under defense strategies, LAMP maintains ~70% ASR (vs. baseline 20–56%).
- The optimal number of perturbations is $|\delta|=2$; additional perturbations yield diminishing returns, attributable to the contagious loss.
- LPIPS is only 0.021 (best baseline: 0.068), indicating superior imperceptibility.

## Highlights & Insights
- **First adversarial attack on multi-image MLLMs**: Fills the gap in UAP attacks for multi-image scenarios.
- **Elegant Contagious Loss design**: A fixed number of UAPs can "infect" clean tokens, addressing the challenge of unknown image counts at inference time.
- **Position-invariant attack**: Index-attention suppression renders the attack independent of image position.
- **Strong transferability**: UAPs trained on a surrogate model effectively attack 7+ target models with diverse architectures.

## Limitations & Future Work
- Validation is limited to open-source models; closed-source models such as GPT-4V and Gemini are not evaluated.
- The perturbation budget $\epsilon=12/255$ is relatively large; performance under tighter budgets is not thoroughly investigated.
- Defense evaluation covers only query-based defenses; stronger adversarial training defenses are not assessed.
- Training requires an A100 GPU and 17K samples; computational costs are not analyzed in detail.

## Related Work & Insights
- vs. **CPGC-UAP / UAP-VLP / Doubly-UAP**: These methods target single-image encoder/decoder attacks; LAMP outperforms them by an average of 19.5 pp in multi-image ASR.
- vs. **Jailbreak-MLLM**: The latter improves transferability through model ensembles, whereas LAMP achieves higher ASR without ensembling.
- vs. **AnyDoor / MLAI**: These methods leverage multi-image capabilities but are not universal attacks; LAMP is the first multi-image UAP approach.

## Related Work & Insights
- The design philosophy of the contagious loss (encouraging clean tokens to attend to noisy tokens) is generalizable to other attention-based attack and defense scenarios.
- The index suppression strategy for position-invariant attacks offers a valuable reference for security evaluation of multi-image models.
- This work reveals a novel attack surface in multi-image MLLMs: corrupting a subset of images suffices to compromise overall reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (first multi-image UAP attack + contagious loss + position-invariant design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (7+ target models, 5 benchmarks, but no closed-source model evaluation)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, complete mathematical derivations)
- Value: ⭐⭐⭐⭐⭐ (significant implications for security research on multi-image MLLMs)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](../../ICLR2026/llm_safety/unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)
- [\[CVPR 2026\] PinPoint: Evaluation of Composed Image Retrieval with Explicit Negatives, Multi-Image Queries, and Paraphrase Testing](../../CVPR2026/llm_safety/pinpoint_evaluation_of_composed_image_retrieval_with_explicit_negatives_multi-im.md)

</div>

<!-- RELATED:END -->
