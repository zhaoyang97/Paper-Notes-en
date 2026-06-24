---
title: >-
  [Paper Note] Continual SFT Matches Multimodal RLHF with Negative Supervision
description: >-
  [CVPR 2025][LLM Alignment][RLHF] Through gradient analysis, it is discovered that the core advantage of multimodal RLHF over continual SFT lies in the negative supervision signal within the rejected responses. Based on this, the nSFT method is proposed, which uses an LLM to extract error information from rejected responses and construct corrective dialogue data. It matches or even outperforms RLHF methods like DPO/PPO using only SFT loss, requiring only one model and signific…
tags:
  - "CVPR 2025"
  - "LLM Alignment"
  - "RLHF"
  - "DPO"
  - "SFT"
  - "Preference Alignment"
  - "Multimodal Large Language Models"
  - "Negative Supervision"
  - "VLM"
date: 2026-05-08
content_hash: 056c0f765fed3ba2
---

# Continual SFT Matches Multimodal RLHF with Negative Supervision

**Conference**: CVPR 2025  
**arXiv**: [2411.14797](https://arxiv.org/abs/2411.14797)  
**Code**: [https://github.com/Kevinz-code/nSFT/](https://github.com/Kevinz-code/nSFT/)  
**Area**: Alignment RLHF  
**Keywords**: RLHF, DPO, SFT, Preference Alignment, Multimodal Large Language Models, Negative Supervision, VLM

## TL;DR
Through gradient analysis, it is discovered that the core advantage of multimodal RLHF over continual SFT lies in the negative supervision signal within the rejected responses. Based on this, the nSFT method is proposed, which uses an LLM to extract error information from rejected responses and construct corrective dialogue data. It matches or even outperforms RLHF methods like DPO/PPO using only SFT loss, requiring only one model and significantly improving GPU memory efficiency.

## Background & Motivation
**Background**: VLMs typically undergo pre-training followed by SFT, and then preference alignment via RLHF (e.g., DPO, PPO) to mitigate hallucinations and enhance multimodal understanding. It is widely believed in the community that RLHF strictly outperforms continual SFT in the preference alignment phase.

**Limitations of Prior Work**: Although multimodal RLHF is effective, it suffers from clear engineering bottlenecks: DPO requires concurrently loading two large models (policy + reference), while PPO requires four (policy + reference + reward + critic), leading to massive GPU memory overhead and training instability. Conversely, directly performing continual SFT (training with only the chosen responses of preference data) yields substantially worse results than RLHF.

**Key Challenge**: Why does SFT fail while RLHF succeeds given the same data? What exactly is the missing piece?

**Goal**: Identify the fundamental reason behind the superiority of RLHF over SFT, and implement this advantage in an SFT manner, achieving a balance of performance and efficiency.

**Key Insight**: Analyzing the DPO loss from a gradient perspective, the authors discover that its gradient is essentially a linear combination of the chosen SFT gradient and the rejected SFT gradient. What continual SFT lacks is precisely the SFT loss on the rejected response (negative supervision signal).

**Core Idea**: Use an LLM to extract error information from the rejected response and construct corrective training data, allowing standard SFT to obtain the advantage of the negative supervision signal in RLHF.

## Method

### Overall Architecture
The workflow of nSFT consists of two phases: **negative supervision data construction** and **continual SFT training**.

Input: A preference dataset containing images, questions, chosen responses $\mathbf{y}_c$, and rejected responses $\mathbf{y}_r$, which is exactly identical to the input of standard DPO.

Data construction phase: Use an LLM (GPT-4) to compare $\mathbf{y}_r$ and $\mathbf{y}_c$, utilizing a **Vision Error Codebook** to locate specific errors in the rejected response (e.g., object hallucination, attribute errors), and then construct new corrective question-answer pairs.

Training phase: Concurrently train the model on the original ground-truth (GT) dialogues and the newly constructed corrective dialogues using the standard SFT loss, requiring only a single model.

### Key Designs

1. **Relation between DPO and SFT Gradients (Theoretical Analysis)**:

    - Function: Reveal the intrinsic connection between DPO and SFT from a gradient perspective.
    - Mechanism: Expanding the logits in the DPO loss and omitting the reference model constraint simplifies the DPO logit to $p'_{\text{dpo}} = -(\mathcal{L}_{\text{sft}}(\mathbf{y}_c) - \mathcal{L}_{\text{sft}}(\mathbf{y}_r))$, which is the difference between two SFT losses. Further taking the derivative with respect to the parameters yields $\frac{\partial \mathcal{L}_d}{\partial \theta'} = \frac{1}{p_{\text{dpo}}} \left[ \frac{\partial \mathcal{L}_{\text{sft}}(\mathbf{y}_c)}{\partial \theta'} - \frac{\partial \mathcal{L}_{\text{sft}}(\mathbf{y}_r)}{\partial \theta'} \right]$, indicating that the DPO gradient is a linear combination of the SFT gradients for both chosen and rejected samples.
    - Design Motivation: This directly explains why continual SFT underperforms RLHF—it lacks the negative supervision gradient term $\mathcal{L}_{\text{sft}}(\mathbf{y}_r)$. Further analysis of the gradient update rate ratio $|t_2/t_1| < 1$ indicates that DPO optimization biassedly focuses on rejecting poor samples.

2. **Decoupling Negative Supervision and Data Construction (the $G(\cdot)$ Function)**:

    - Function: Explicitly transform the implicit negative supervision signal in RLHF into training data usable for SFT.
    - Mechanism: Since negative supervision is deeply entangled within DPO's pairwise logits, it cannot be optimized directly with SFT. Therefore, an LLM is introduced as a construction function $G(\cdot)$ that takes the rejected response $\mathbf{y}_r$, GT response $\mathbf{y}_c$, and the vision error codebook $Q$ as inputs to output corrective dialogues. The final nSFT loss is defined as: $\mathcal{L}_{\text{nSFT}} = \mathcal{L}_{\text{sft}}(\mathbf{y}_c) + \mathcal{L}_{\text{sft}}(G(\mathbf{y}_r; \mathbf{y}_c, Q))$.
    - Design Motivation: This breaks the pairwise coupling of chosen-rejected samples in RLHF, allowing the entire alignment process to use only SFT loss, thus avoiding loading the reference model.

3. **Vision Error Codebook (VEC)**:

    - Function: Provide the LLM with a comprehensive range of image-related error types, guiding it to accurately identify hallucinations in the rejected response.
    - Mechanism: The codebook covers errors at both the instance-level (object existence, attributes, and count) and the image-level (scene context and spatial relationships). The LLM first locates errors based on the GT and the codebook, then constructs corrective dialogues such as "Is this a travel book? Yes" / "Is this a cookbook? No" accordingly.
    - Design Motivation: Without the VEC, the dialogues constructed by the LLM are dominated by non-entity phrases (e.g., "might", "provide"). Introducing the VEC shifts the dialogic focus to concrete objects (e.g., "truck", "cup", "chair"). Key ablation studies confirm that the VEC provides a clear performance gain.

### Loss & Training
- nSFT loss: $\mathcal{L}_{\text{nSFT}} = \mathcal{L}_{\text{sft}}(\mathbf{y}_c) + \mathcal{L}_{\text{sft}}(G(\mathbf{y}_r; \mathbf{y}_c, Q))$
- Ablation studies show that incorporating a per-token KL constraint (similar to KL regularization in RLHF) can further enhance performance.
- Training setup: DeepSpeed + ZeRO-3, batch size 128, learning rate 2e-6, cosine scheduler.
- Construction methods vary across datasets: OCRVQA (short responses) manually constructs doubled Q-A pairs; TextCaps and LLaVA-150k (medium/long responses) utilize GPT-4 to construct 5-turn dialogues.

## Key Experimental Results

### Main Results
Continual alignment is conducted using LLaVA-1.5-7B on three different datasets (OCRVQA, TextCaps, LLaVA-150k), comparing five methods:

| Method | Dataset | SQA | GQA | VQAT | MMVet | MME | MMB | POPE | CHAIR↓ | MMHal |
|------|--------|-----|-----|------|-------|-----|-----|------|--------|-------|
| Baseline | — | 66.8 | 62.0 | 58.0 | 30.5 | 1510 | 64.3 | 85.9 | 32.0 | 2.80 |
| Cont. SFT | LLaVA-150k | 67.1 | 60.9 | 57.0 | 31.2 | 1480 | 64.0 | 86.3 | 29.1 | 2.91 |
| GT-DPO | LLaVA-150k | 68.1 | 61.6 | 57.6 | 33.9 | 1497 | 63.9 | 85.9 | 30.7 | 2.80 |
| SeVa | LLaVA-150k | 67.5 | 61.4 | 58.0 | 32.5 | 1490 | 64.7 | 85.6 | 28.2 | 2.94 |
| **nSFT** | LLaVA-150k | **68.4** | **62.3** | **58.4** | **34.2** | **1550** | **65.2** | **87.4** | **25.4** | **3.02** |

SOTA comparison (using mixed 15k data, compared with methods like SeVa and SIMA under their respective optimal settings):

| Method | VQA | SQA | GQA | MMB | MME | POPE | SEEDI | SHR↓ | MMVet |
|------|-----|-----|-----|-----|-----|------|-------|------|-------|
| LLaVA-1.5 | 58.2 | 66.8 | 62.0 | 64.3 | 1510 | 85.9 | 65.7 | 36.7 | 30.5 |
| SeVa-7B | 56.2 | 67.5 | 60.7 | 65.6 | 1450 | 86.7 | 65.8 | 34.9 | 36.8 |
| SIMA-7B | 58.3 | 68.1 | 62.2 | 64.9 | 1507 | 86.5 | 65.9 | 34.5 | 32.6 |
| **nSFT** | **58.7** | **68.5** | **62.9** | **67.1** | **1531** | **86.8** | **66.2** | **34.2** | 34.0 |

### Ablation Study

| Configuration | MMB | SQA | MME | POPE |
|------|-----|-----|-----|------|
| baseline | 64.3 | 66.8 | 1515 | 85.9 |
| + nSFT (Full) | 65.0 | 68.2 | 1533 | 86.5 |
| + nSFT w/o VEC | 64.4 | 67.6 | 1505 | 86.0 |
| + nSFT w/o chosen | 64.9 | 68.2 | 1523 | 86.4 |
| + nSFT + KL constraint | **65.2** | — | — | — |

### Key Findings
- **Negative supervision is the core**: Removing VEC significantly degrades all metrics, indicating that fine-grained error identification is crucial to the quality of negative supervision.
- **The chosen response is not essential**: Removing $\mathbf{y}_c$ barely impacts performance, as the negative supervision construction process already implicitly references the GT.
- **nSFT achieves the largest gain in hallucination metrics**: On the LLaVA-150k dataset, the total hallucination score improves by +11.8 (vs. +1.3 for GT-DPO), indicating that corrective data guided by the VEC is exceptionally effective at reducing object-level hallucinations.
- **DPO excels at suppressing the worst-case scenarios, while nSFT excels at improving the best-case scenarios**: In-domain evaluation shows that DPO has a higher ACC10w (accuracy of the 10 worst samples), whereas nSFT achieves a higher ACC10b (accuracy of the 10 best samples).
- **Transferable to larger models**: nSFT matches DPO and outperforms pure SFT on LLaVA-1.5-13B and LLaVA-NeXT-13B.
- **Pronounced training efficiency**: GPU memory consumption is about half of DPO's (no reference model required) with shorter training times.

## Highlights & Insights
- **Concise and elegant theoretical analysis**: Breaking down the DPO gradient into a linear combination of two SFT gradients through a few derivations directly identifies the absence of the rejected gradient term as the cause of continual SFT failure. This analysis is remarkably elegant and convincing.
- **The paradigm of "making implicit RLHF signals explicit" is highly inspiring**: Instead of optimizing a pairwise loss, this approach first extracts information using an external tool (an LLM coupled with an error codebook) and then undergoes SFT. This concept can be generalized to other RLHF scenarios, such as code generation and mathematical reasoning.
- **Clever design of the Vision Error Codebook**: Guiding LLM attention via predefined, fine-grained error types avoids generating overly general corrective data. The word cloud visualization demonstrates the distinct difference with and without VEC.

## Limitations & Future Work
- **Dependence on GPT-4 for data construction**: The data construction phase in nSFT relies on a strong LLM (GPT-4), increasing API costs and bounding the quality to the LLM's capability. Future efforts could explore utilizing open-source LLMs or self-model iterative construction.
- **Validated only in multimodal scenarios**: As noted by the authors, it remains uncertain whether nSFT is applicable to NLP-domain RLHF (e.g., detoxification, style transfer), where error types and construction methodologies might differ significantly.
- **Vision Error Codebook requires manual design**: The current codebook targets common errors in visual understanding. Relocating to alternative tasks necessitates manual redesign.
- **Online/iterative version unvalidated**: Current nSFT operates as an offline method with one-time data construction. Exploring the iterative construction of negative supervision data using the latest model outputs remains an potential future direction.

## Related Work & Insights
- **vs. DPO (SeVa/SIMA/GT-DPO)**: DPO-based methods implicitly utilize negative supervision via pairwise optimization, which requires two models. nSFT explicitly extracts negative supervision and employs SFT, requiring only one model while performing comparably or better. However, DPO exhibits more stability under worst-case scenarios.
- **vs. PPO**: PPO requires four models (policy + ref + reward + critic), causing the highest memory footprint. nSFT achieves comparable outcomes with significantly lower computational resources.
- **vs. Concurrent Work (NLP Domain)**: Similar findings exist in NLP (where negative feedback plays a more critical role in DPO) but have not been formally linked to SFT. This work is the first to systematically establish this connection in multimodal scenarios and offer a practical solution.

## Rating
- Novelty: ⭐⭐⭐⭐ Concise yet powerful theoretical analysis; the nSFT paradigm is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets $\times$ multiple baselines $\times$ multiple VLMs $\times$ 9 benchmarks, featuring very comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rich diagrams, and highly legible theoretical derivations.
- Value: ⭐⭐⭐⭐ Offers a highly efficient alternative to preference alignment with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mitigating Reward Hacking in RLHF via Bayesian Non-negative Reward Modeling](../../ICML2026/llm_alignment/mitigating_reward_hacking_in_rlhf_via_bayesian_non-negative_reward_modeling.md)
- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)
- [\[ACL 2025\] Aligning to What? Limits to RLHF Based Alignment](../../ACL2025/llm_alignment/aligning_to_what_limits_to_rlhf_based_alignment.md)
- [\[ACL 2026\] RbtAct: Rebuttal as Supervision for Actionable Review Feedback Generation](../../ACL2026/llm_alignment/rbtact_rebuttal_as_supervision_for_actionable_review_feedback_generation.md)
- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](../../ICML2025/llm_alignment/dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)

</div>

<!-- RELATED:END -->
