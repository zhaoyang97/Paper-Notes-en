---
title: >-
  [Paper Note] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models
description: >-
  [AAAI 2026][LLM Safety][Machine Unlearning] This paper proposes AUVIC, a framework that combines an adversarial perturbation generator with a dynamic anchor preservation mechanism to precisely unlearn target visual concepts (e.g., specific faces) in MLLMs, while avoiding collateral forgetting of semantically similar concepts. The paper also introduces VCUBench, the first evaluation benchmark for visual concept unlearning in group-scene scenarios.
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Adversarial Perturbation"
  - "Multimodal Large Language Models"
  - "Visual Concept Forgetting"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 291602e0f9fd04a3
---

# AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.11299](https://arxiv.org/abs/2511.11299)  
**Code**: Available (code link noted on arXiv page)  
**Area**: AI Safety / Multimodal VLM / Machine Unlearning
**Keywords**: Machine Unlearning, Adversarial Perturbation, Multimodal Large Language Models, Visual Concept Forgetting, Privacy Protection

## TL;DR

This paper proposes AUVIC, a framework that combines an adversarial perturbation generator with a dynamic anchor preservation mechanism to precisely unlearn target visual concepts (e.g., specific faces) in MLLMs, while avoiding collateral forgetting of semantically similar concepts. The paper also introduces VCUBench, the first evaluation benchmark for visual concept unlearning in group-scene scenarios.

## Background & Motivation

Multimodal large language models (MLLMs) are trained on massive datasets that may contain sensitive or copyright-protected content. Regulations such as the GDPR establish the "right to be forgotten," requiring models to remove the influence of personal data upon request. Machine unlearning addresses this need by selectively erasing target knowledge without retraining the entire model.

**Limitations of Prior Work**:
- Unlearning research has focused predominantly on text modalities, while **visual concept unlearning in MLLMs remains largely unexplored**.
- Naive methods such as gradient ascent (GA) cause severe **collateral forgetting**: when unlearning Trump, recognition rates for Biden and Boris Johnson also drop significantly (Biden recall falls to 11%), because these concepts share similar features in the embedding space.
- No benchmark exists for evaluating unlearning in multi-person scenarios—existing methods cannot selectively forget only the target individual in group photos.

A motivation experiment clearly illustrates the problem: after applying GA to unlearn eight concepts on LLaVA-1.5, a "collateral forgetting matrix" reveals that visually similar concepts are more susceptible to being collaterally forgotten. The target concept's BLEU score drops from 0.80 to 0.10, while the recall of visually similar individuals such as Biden also plummets.

## Core Problem

How to **precisely unlearn a specific visual concept** (e.g., a public figure's face) in an MLLM while satisfying three constraints:
1. **C1 – Utility Preservation**: The model's overall performance and multimodal alignment must not be degraded.
2. **C2 – Insufficient Forgetting Data**: Constructing multimodal unlearning training pairs is more challenging than text-only settings.
3. **C3 – Collateral Forgetting**: Gradient-based methods tend to over-generalize, erasing concepts that are visually or semantically adjacent to the target.

## Method

### Overall Architecture

AUVIC adopts a **Min-Max adversarial optimization** paradigm, alternating between an adversarial generator and a discriminator (the MLLM):
- **Generator**: Takes the input image, extracts visual features via a frozen CLIP encoder, and produces adversarial perturbations through a lightweight MLP. Its objective is to maximally activate the target concept's representation.
- **Discriminator (MLLM LoRA component)**: Receives adversarial images and perturbed text prompts, learning to suppress predictions of the target concept under adversarial conditions while preserving recognition of non-target concepts.
- Only the LoRA parameters of the visual encoder are updated (rank=32, α=32); the language component is fully frozen.

The overall optimization objective is:

$$\min_\theta \max_\phi \mathbb{E}_{x \sim \mathcal{D}} [\mathcal{L}_f + \lambda \mathcal{L}_p + \beta \mathcal{L}_c]$$

### Key Designs

1. **Multimodal Adversarial Perturbations**

    - **Text side**: Prompt perturbations enhance adversarial robustness by generating diverse semantically equivalent queries (greeting variants, paraphrases, contextual distractors). Prepending the target concept name to each query directs model attention and amplifies target activations to produce stronger gradients.
    - **Vision side**: A **feature-guided image perturbation module** is introduced. A frozen CLIP image encoder extracts feature vector $h = \text{CLIP}(x)$, which is mapped to an image-shaped perturbation $\delta_\theta \in \mathbb{R}^{3 \times H \times W}$ via a lightweight generator network $G_\theta$ consisting of three Linear-ReLU layers followed by tanh. Perturbation magnitude is constrained by $\ell_\infty$ ($\epsilon = 8/255$), yielding the adversarial image $x' = \text{clip}(x + \delta_\theta, 0, 1)$.

2. **Dynamic Anchor Preservation Mechanism**

    - GPT is used to generate a candidate list of $K$ public figure names.
    - The average token embedding $\bar{e}_i$ of each candidate is extracted from the frozen vocabulary, and cosine similarity with the target concept embedding $\bar{e}_T$ is computed.
    - **Gumbel-Softmax sampling** differentiably selects the top-$m$ most semantically similar concepts as the protection set $\mathcal{P}_{top}$.
    - The stochasticity introduced by Gumbel-Softmax causes the protection set to vary dynamically during training, improving robustness.

3. **Parameter-Efficient Fine-Tuning**

    - LoRA adapters are inserted only into the `q_proj`, `v_proj`, `fc1`, and `fc2` modules of the CLIP visual tower.
    - All original weights are frozen and the language head is entirely untouched, ensuring linguistic fluency is unaffected.

### Loss & Training

Three loss terms operate jointly:
- **$\mathcal{L}_f$ (Target Forgetting Loss)**: BCE loss that suppresses the maximum predicted logit for the target concept token, causing the model to "forget" the target.
- **$\mathcal{L}_p$ (Concept Preservation Loss)**: For each concept in the protection set, Gumbel-Softmax-weighted BCE encourages the model to maintain correct recognition, preventing collateral forgetting.
- **$\mathcal{L}_c$ (Consistency Regularization)**: KL divergence constrains the discrepancy between the model's output distributions on clean and adversarial inputs, preserving generation stability and downstream fluency.

Training uses the AdamW optimizer with a ReduceLROnPlateau scheduler; the generator and visual encoder are optimized independently on 3× RTX 4090 GPUs.

## Key Experimental Results

### Main Results: Trump as Target (Table 1)

| Method | TFA↑ | NTRA↑ | GRF-F1↑ | Efficacy↑ | Generality↑ | PPL↓ |
|--------|------|-------|---------|-----------|-------------|------|
| GA | 84.48% | 30.17% | 44.46% | 89.17% | 63.07 | 16.39 |
| PO | 49.14% | 54.48% | 51.67% | 80.42% | 62.91 | 7.58 |
| GA+KL | 85.86% | 26.55% | 40.56% | 90.62% | 62.98 | 8.92 |
| SIU | 92.35% | 63.49% | 75.25% | 100.0% | 61.2% | 11.26 |
| **AUVIC** | **93.64%** | **83.17%** | **88.10%** | 97.92% | **63.05%** | **8.14** |

### Average Results Across Six Targets (Table 2)

| Method | TFA↑ | NTRA↑ | GRF-F1↑ | Efficacy↑ | Generality↑ | PPL↓ |
|--------|------|------|--------|-----------|-------------|------|
| GA | 67.67 | 30.12 | 37.64 | 77.76 | 59.62 | 18.47 |
| PO | 49.87 | 55.91 | 50.19 | 70.71 | 61.62 | 9.86 |
| GA+KL | 77.36 | 32.99 | 43.80 | 82.95 | 60.83 | 11.06 |
| **AUVIC** | **96.99** | **75.34** | **84.94** | **96.82** | **62.69** | **8.34** |

**Key Comparison**: AUVIC's GRF-F1 (88.10%) substantially outperforms GA+KL (40.56%), with NTRA improving from 26–30% to 83.17%—indicating that the collateral forgetting problem is fundamentally addressed. Although GA achieves reasonable forgetting (TFA 84.48%), its NTRA of only 30.17% reveals extensive unintended erasure of non-target concepts.

### Ablation Study

| Variant | TFA | NTRA | GRF-F1 |
|---------|-----|------|--------|
| **AUVIC (Full)** | **93.64** | **83.17** | **88.10** |
| w/o Gumbel | 89.14 | 64.57 | 72.37 |
| w/o Adv Perturb | 83.20 | 60.43 | 70.98 |
| w/o Both | 27.43 | 75.43 | 38.55 |

- **Adversarial perturbation contributes most**: Removing it reduces GRF-F1 from 88.10 to 70.98 (−17.12), TFA by 10.44 points, and NTRA by 22.74 points.
- **Gumbel dynamic sampling is also critical**: Removing it drops NTRA from 83.17 to 64.57 (−18.6), demonstrating that dynamic sampling outperforms a fixed protection set.
- **Removing both**: TFA collapses to 27.43%, showing that the base framework alone is nearly incapable of completing the unlearning task—both components are essential.

## Highlights & Insights

- **Strong problem awareness**: The collateral forgetting matrix visualization and concrete examples clearly expose the systemic failure of GA-based unlearning, with well-motivated analysis.
- **Elegant Min-Max adversarial design**: The generator "maximally activates the target concept" while the model "learns to suppress it under the hardest conditions," naturally improving the robustness of unlearning through adversarial training.
- **Gumbel-Softmax dynamic protection set**: Ensures end-to-end differentiable training while introducing stochasticity to prevent overfitting to a fixed set of protected concepts—a clever design choice.
- **Vision-only LoRA modification**: Leaving the language head entirely untouched inherently preserves linguistic fluency (lowest PPL), representing a sound engineering decision.
- **VCUBench**: The first MLLM visual concept unlearning benchmark targeting group-scene scenarios, with 15k+ samples, establishing a meaningful baseline for the community.

## Limitations & Future Work

- **Limited concept types**: Experiments cover only facial unlearning of public figures, without addressing broader visual concepts such as object categories, scenes, or actions.
- **Narrow model coverage**: Validation is restricted to LLaVA-1.5 (7B); applicability to larger models (13B/70B+) or alternative MLLM architectures (e.g., InternVL, Qwen-VL) remains untested.
- **Dependency on GPT for candidate generation**: The dynamic anchor preservation mechanism requires GPT to generate a list of semantically similar public figures, introducing an external dependency of uncertain applicability to non-person concepts.
- **Adversarial robustness insufficiently validated**: The paper does not examine whether the post-unlearning model resists targeted knowledge extraction attacks (e.g., membership inference attacks); the thoroughness of unlearning requires further verification.
- **Efficacy is not highest**: SIU achieves 100% Efficacy on Trump, while AUVIC reaches 97.92%, indicating approximately 2% residual recognition in single-person scenarios.
- **Training cost not fully discussed**: The number of adversarial training iterations, convergence speed, and the overhead of simultaneously unlearning multiple concepts are not reported.
- **VCUBench scale is limited**: Only five target concepts are included, restricting the comprehensiveness of generalization evaluation.

## Related Work & Insights

- **vs. GA/GA+KL (Gradient Ascent Methods)**: GA-based methods achieve moderate forgetting but suffer from severe collateral forgetting (NTRA only 26–30%). AUVIC raises NTRA to 83% through adversarial training combined with anchor preservation. GA+KL improves stability with KL regularization but yields an even lower NTRA (26.55%), suggesting that simple regularization cannot resolve concept entanglement in the feature space.
- **vs. PO (Preference Optimization)**: TOFU-style preference optimization produces weak forgetting (TFA only 49.14%), as teaching the model to respond with "I don't know" does not genuinely erase internal representations.
- **vs. SIU**: SIU achieves competitive forgetting via visual subspace isolation (TFA 92.35%), but its NTRA of only 63.49% indicates insufficient precision in subspace separation. AUVIC substantially outperforms SIU in preservation capability (NTRA 83.17%) with lower PPL.
- **vs. Clear (Dontsov et al., 2024)**: Clear is a pioneer in multimodal unlearning but relies on synthetic benchmarks; AUVIC's VCUBench is grounded in real public figures, offering more convincing evaluation.
- AUVIC's adversarial perturbation approach to precise unlearning is **complementary** to topology-constrained methods for structure-preserving unlearning—the former operates from the input space perspective, the latter from the structural perspective of the feature space.
- AUVIC's Gumbel-Softmax dynamic protection set selection is transferable to unlearning in visual foundation models (e.g., CLIP), protecting zero-shot capabilities for similar concepts.
- The design philosophy of VCUBench (single-person + multi-person, cross-evaluation of forgetting and retention) offers a reusable template for constructing more general unlearning benchmarks.
- **Adversarial training for unlearning** is a promising paradigm: treating the generator as a "red team" that continuously challenges the unlearning effect. This idea is extensible to textual knowledge unlearning and multimodal hallucination suppression.
- AUVIC's vision-only LoRA strategy suggests that **freezing the language side while applying low-rank visual adaptation** may serve as a general paradigm for lightweight safety alignment in MLLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Applying adversarial training to visual concept unlearning in MLLMs is a novel combination, though individual components (adversarial training, LoRA, Gumbel-Softmax) are not themselves new.
- **Experimental Thoroughness**: ⭐⭐⭐ Evaluation is limited to LLaVA-1.5 with only five target concepts; large-scale and cross-model validation is absent; ablations are conducted but hyperparameter sensitivity analysis is missing.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation analysis with the collateral forgetting matrix and concrete examples is highly intuitive; method descriptions are clear, though the benchmark section is somewhat disorganized.
- **Value**: ⭐⭐⭐⭐ VCUBench establishes a meaningful benchmark as the first of its kind for MLLM visual concept unlearning, and the method demonstrates significant and genuine improvements in unlearning precision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](../../ACL2025/llm_safety/mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ICLR 2026\] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](../../ICLR2026/llm_safety/doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)
- [\[ACL 2025\] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](../../ACL2025/llm_safety/cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)

</div>

<!-- RELATED:END -->
