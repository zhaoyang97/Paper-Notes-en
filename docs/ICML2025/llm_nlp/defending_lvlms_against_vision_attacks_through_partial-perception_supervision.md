---
title: >-
  [Paper Note] Defending LVLMs Against Vision Attacks through Partial-Perception Supervision
description: >-
  [ICML2025][LLM (Other)][LVLM Security] Proposes DPS (Defense through Partial-Perception Supervision), which utilizes responses from cropped images as "weak supervision" to guide the full-image model for self-correction during inference. This achieves training-free, black-box visual attack defense for LVLMs, reducing the average attack success rate by 76.3%.
tags:
  - "ICML2025"
  - "LLM (Other)"
  - "LVLM Security"
  - "Visual Adversarial Attack"
  - "Partial-Perception Supervision"
  - "Weak-to-Strong Learning"
  - "Black-box Defense"
  - "Jailbreak Defense"
date: 2026-05-08
content_hash: 558dcdef4d465fd6
---

# Defending LVLMs Against Vision Attacks through Partial-Perception Supervision

**Conference**: ICML2025  
**arXiv**: [2412.12722](https://arxiv.org/abs/2412.12722)  
**Code**: [GitHub](https://github.com/tools-only/DPS)  
**Area**: LLM/NLP  
**Keywords**: LVLM Security, Visual Adversarial Attack, Partial-Perception Supervision, Weak-to-Strong Learning, Black-box Defense, Jailbreak Defense

## TL;DR

Proposes DPS (Defense through Partial-Perception Supervision), which utilizes responses from cropped images as "weak supervision" to guide the full-image model for self-correction during inference. This achieves training-free, black-box visual attack defense for LVLMs, reducing the average attack success rate by 76.3%.

## Background & Motivation

- **Core Problem**: LVLMs (e.g., GPT-4o, Gemini) are vulnerable to visual adversarial attacks (adversarial noise, typographic attacks), which mislead them into outputting incorrect or harmful content.
- **Limitations of Prior Work**: Methods like SmoothVLM defend against attacks using image cropping plus majority voting, but cropping disrupts semantic information, leading to a significant degradation in response quality on benign images.
- **Key Challenge**: Cropping can disrupt attack signals but also damages the semantics of normal images—how to achieve the best of both worlds?
- **Key Insight**:
    - **Sensitivity**: Visual attacks are highly sensitive to image modifications such as cropping, which disrupts the attack semantics.
    - **Confidence Discrepancy**: Models exhibit high confidence when processing clean images and are unaffected by distractor information; conversely, they display low confidence when handling attacked images and are easily influenced by distractors in the prompt.
- **Inspiration**: Weak-to-strong learning—a weak model (the model observing partial images) can effectively supervise and guide a strong model (the model observing the full image).

## Method

### DPS Framework (Two-Step Inference)

**Step 1: Partial-Perception Initial Response (Part-Perc Model)**

Generate an objective description of the cropped image:

> "Please provide an objective, detailed description of the image, avoiding subjective conjecture and associations. Then answer the question: (Original Question)."

Three cropping strategies are employed to generate three partial images:

- **Center Crop (CC)**: Extracts a region of 1/2 size from the center of the image.
- **Random Crop (RC)**: Extracts a region of 1/4 to 1/2 size from a random position.
- **Adaptive Crop (AC)**: Leverages the LVLM to extract the primary object region within the image.

**Step 2: Partial-Perception Supervision (Full-Perc Model)**

Utilize the responses from the Part-Perc model as supervisory information to guide the Full-Perc model in re-analyzing the full image:

> "Here is the information provided by the local observation agents: (Supervisory message). Re-analyze the given image, and provide your final answer to the question: (Original Question)."

**Mechanism**:

- Clean images $\rightarrow$ High model confidence $\rightarrow$ The (potentially inaccurate) descriptions from Part-Perc do not affect the final output.
- Attacked images $\rightarrow$ Low model confidence $\rightarrow$ The descriptions from Part-Perc (with attack signals removed) prompt the model to perform self-correction.

### Safety Enhancement: LS-DPS

For jailbreak attacks, a safety reminder is incorporated into the prompt of Step 2:

> "Consider whether you might be led into discussing harmful, malicious, or unethical topics."

An external LLM safety checker (LLM-Secured DPS) is connected to filter the final output:

$$\text{ASR}(\mathcal{D}_k) = \frac{1}{|\mathcal{D}_k|} \sum_{(x_i, q_i, t_i) \in \mathcal{D}_k} \mathbb{I}(\mathcal{F}(x_i, q_i), t_i)$$

Where $\mathcal{F}$ represents the LVLM, $\mathbb{I}$ is the indicator function of whether the attack succeeded, and $t_i$ denotes the attack goal or safety standard.

## Key Experimental Results

### Misleading Attack Defense (ASR ↓, Lower is Better)

| Model | Method | RTA-100 | Self-Gen | MultiTrust | Avg |
|------|------|---------|----------|------------|------|
| Qwen-VL-Plus | SmoothVLM | 0.92 | 0.83 | 1.00 | 0.91 |
| Qwen-VL-Plus | **DPS** | **0.24** | **0.30** | **0.40** | **0.31** |
| GPT-4o-Mini | SmoothVLM | 0.68 | 0.85 | - | 0.76 |
| GPT-4o-Mini | **DPS** | **0.35** | **0.43** | - | **0.39** |
| Gemini-1.5-Flash | SmoothVLM | 0.85 | 1.00 | 0.80 | 0.88 |
| Gemini-1.5-Flash | **DPS** | **0.58** | **0.49** | **0.11** | **0.39** |

DPS significantly reduces the average ASR under misleading attacks to 0.31~0.39, achieving **2x to 2.5x better defense efficacy** than the best baseline.

### Jailbreak Attack Defense (ASR ↓)

| Model | Method | MM-Safety | HADES | VisualAtt | Avg |
|------|------|-----------|-------|-----------|------|
| Qwen-VL-Plus | Protector | 0.07 | 0.22 | 0.18 | 0.16 |
| Qwen-VL-Plus | **LS-DPS** | **0.02** | **0.10** | **0.02** | **0.05** |
| GPT-4o-Mini | ECSO | 0.24 | 0.05 | 0.15 | 0.15 |
| GPT-4o-Mini | **LS-DPS** | **0.03** | **0.04** | **0.04** | **0.04** |
| Gemini-1.5-Flash | ECSO | 0.14 | 0.11 | 0.13 | 0.13 |
| Gemini-1.5-Flash | **LS-DPS** | **0.06** | **0.03** | **0.06** | **0.05** |

LS-DPS reduces the ASR under jailbreak attacks to 0.04~0.05, demonstrating superior performance over all baseline methods.

### Standard Performance (MM-Vet)

DPS has a minimal impact on standard performance, performing close to the vanilla model, whereas SmoothVLM suffers from significant performance degradation.

### Ablation Study (Comparison of Cropping Strategies, Qwen-VL-Plus)

| Strategy | Misleading Tasks (Avg) | Safety Tasks (Avg) |
|------|---------------|---------------|
| CC (Center Crop) | ~0.40 | ~0.08 |
| RC (Random Crop) | ~0.48 | ~0.09 |
| AC (Adaptive Crop) | ~0.37 | ~0.07 |
| DPS (Fusion of Three) | **~0.31** | **~0.05** |

The fusion of multiple cropping strategies significantly enhances defense capabilities.

## Highlights & Insights

1. **Novel Weak-to-Strong Defense Paradigm**: Analogizes "observing partial images vs. full images" to "weak model vs. strong model," utilizing weak supervision to guide the strong model toward self-correction rather than using simple voting.
2. **Fully Black-Box + Training-Free**: Requires no access to internal model weights or extra training, enabling defense purely through adjustments at the prompt level.
3. **Leveraging Confidence Discrepancies**: The insight that models exhibit high confidence on clean images but low confidence on attacked images serves as the theoretical foundation for maintaining both performance and defense efficacy.
4. **Addressing Both Misleading and Jailbreak Attacks**: Covers both mainstream attack scenarios under a single framework through prompt fine-tuning and the plug-and-play extension of LLM safety checkers.
5. **No Loss in Standard Performance**: In contrast to the significant performance drop observed with SmoothVLM, DPS barely impacts standard task performance.

## Limitations & Future Work

1. **Computational Overhead**: Requires multiple cropping operations and model inferences (at least 4 LVLM calls), resulting in a significant overhead increase compared to single inference.
2. **Failure Cases with Cropping**: If the attack information is globally distributed rather than localized, cropping might fail to eliminate the attack signal, rendering partial-perception supervision ineffective.
3. **Dependency on Confidence Discrepancy Assumption**: The method relies on the observation that "attacks cause model confidence to drop." If an attack technique keeps model confidence high despite being attacked, the defense may fail.
4. **Evaluation Limited to API Models**: Main experiments are conducted on proprietary API models (Qwen-VL-Plus, GPT-4o-Mini, Gemini-1.5-Flash), with open-source models verified only through additional experiments on Qwen2.5-VL-32B.
5. **Scope for Enhancing Interaction Strategies**: A simple two-step prompting approach is currently used; more sophisticated interaction mechanisms (e.g., multi-turn debates) could potentially yield further improvements.

## Rating

- Novelty: ⭐⭐⭐⭐ — Applying the weak-to-strong supervision paradigm to visual attack defense offers a brand-new perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across three models, six datasets, and seven baselines, covering both misleading and jailbreak attacks.
- Writing Quality: ⭐⭐⭐⭐ — The motivation analysis in Section 3 is progressive and clearly structured.
- Value: ⭐⭐⭐⭐ — High practicality (black-box and training-free), though the computational overhead may limit its deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More](../../ACL2025/llm_nlp/lm_graph_search_supervision.md)
- [\[CVPR 2025\] Building Vision Models upon Heat Conduction](../../CVPR2025/llm_nlp/building_vision_models_upon_heat_conduction.md)
- [\[ACL 2025\] Towards Robust ESG Analysis Against Greenwashing Risks: A3CG](../../ACL2025/llm_nlp/a3cg_esg_greenwashing.md)
- [\[ACL 2025\] Red-Teaming LLM Multi-Agent Systems via Communication Attacks](../../ACL2025/llm_nlp/red-teaming_llm_multi-agent_systems_via_communication_attacks.md)
- [\[ACL 2025\] SDD: Self-Degraded Defense against Malicious Fine-tuning](../../ACL2025/llm_nlp/sdd_self-degraded_defense_against_malicious_fine-tuning.md)

</div>

<!-- RELATED:END -->
