---
title: >-
  [Paper Note] Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs
description: >-
  [ACL 2026 (Findings)][LLM Safety][Low-rank decomposition] The first systematic study evaluating the impact of **low-rank decomposition (SVD/FWSVD/BASEL)** on LLM trustworthiness…
tags:
  - "ACL 2026 (Findings)"
  - "LLM Safety"
  - "Low-rank decomposition"
  - "PII leakage"
  - "Adversarial robustness"
  - "Ethical alignment"
  - "Fairness"
  - "Layer attribution"
date: 2026-05-08
content_hash: 4727883eda00132e
---

# Decomposed Trust: Privacy, Adversarial Robustness, Ethics, and Fairness in Low-Rank LLMs

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2511.22099](https://arxiv.org/abs/2511.22099)  
**Code**: To be confirmed  
**Area**: LLM Safety / Model Compression / Trustworthy AI  
**Keywords**: Low-rank decomposition, PII leakage, Adversarial robustness, Ethical alignment, Fairness, Layer attribution

## TL;DR
The first systematic study evaluating the impact of **low-rank decomposition (SVD/FWSVD/BASEL)** on LLM trustworthiness, discovering an asymmetric trade-off: "Training data privacy ↑, adversarial robustness ↑, PII protection ↓, ethical alignment ↓, fairness ↓." The study further localizes adversarial vulnerability to the `embed_tokens` and `down_proj` sub-layers through gradient attribution.

## Background & Motivation
**Background**: Beyond quantization (GPTQ) and pruning (Wanda), low-rank decomposition (SVD → FWSVD → BASEL → IMPACT) is becoming a mainstream LLM compression route, significantly reducing memory and increasing throughput while maintaining benign accuracy. While Hong et al. (2024, ICML) investigated the impact of quantization/pruning on trustworthiness, low-rank decomposition remains unexplored.

**Limitations of Prior Work**: The industry treats low-rank compression as a "side-effect-free slimming technique" for edge LLM deployment. However, no systematic study has questioned whether compressed models can still refuse PII, identify unethical prompts, or maintain fairness. This research gap poses compliance risks in sensitive scenarios like healthcare and finance.

**Key Challenge**: Low-rank decomposition **truncates the singular value subspace**. Arditi et al. (2024) demonstrated that the "safety subspace (refusal direction)" of LLMs resides within these truncated directions. As compression intensifies, the refusal vector becomes unconstructible, causing safety mechanisms to be silently removed despite stable benign accuracy.

**Goal**: (1) Systematically quantify the impact of low-rank compression across four trustworthy dimensions (privacy / adversarial robustness / ethics / fairness); (2) Decompose the interactions between model scale × fine-tuning × compression method; (3) Use gradient attribution to locate "which layers determine adversarial robustness" to guide future compression algorithms.

**Key Insight**: Selecting LLaMA-2 (7B/13B, Base/Chat) + Qwen-2.5 (7B/14B) as base models, an orthogonal evaluation was conducted using 3 low-rank methods (SVD / FWSVD / BASEL) × 3 compression rates (k%=70/50/30) across 4 trustworthy datasets (Enron / GLUE+AdvGLUE++ / ETHICS / Adult).

**Core Idea**: By evaluating "trustworthiness" across four independent quadrants, it was found that low-rank compression does **not result in uniform degradation** but directional changes. Rigorous explanations are provided using SVD safety subspace theory and condition number theory.

## Method

### Overall Architecture
This is an evaluation and interpretive study with the following pipeline: ① **Model Matrix Configuration**—LLaMA-2 Base/Chat (7B/13B) and Qwen-2.5 (7B/14B), each crossed with {fine-tune math, fine-tune code, no fine-tune} × {SVD, FWSVD, BASEL} × {k=70%, 50%, 30%}; ② **Trustworthy Evaluation**—Independent measurement of four dimensions: Enron Email (5 training-data leakage metrics) + Enron PII (leakage & rejection in zero-shot / few-shot protected / few-shot attack scenarios) + GLUE/AdvGLUE++ on SST-2/QQP/MNLI (accuracy drop $\Delta_{\text{robust}}$) + ETHICS commonsense (zero/few-shot accuracy + FPR under 5 jailbreak instructions) + UCI Adult (MDPD / MEOD on race/sex/age); ③ **Theoretical Explanation**—Explaining the causes of PII↓ ethics↓, adversarial robustness↑, and training-data privacy↑ using safety subspace, condition number, and capacity-memorization theories; ④ **Layer Attribution**—Quantifying per-layer contributions via first-order Taylor expansion $a_i = \|(\partial \ell / \partial \mathbf{h}_i) \mathbf{h}_i\|_2$, identifying critical layers using the difference $\Delta_i = |a_i^{\text{clean}} - a_i^{\text{adv}}|$.

### Key Designs

1.  **Four-dimensional Trust Evaluation Protocol**:
    -   Function: Systematically decomposes "trustworthiness" into four independent quadrants—privacy (training-data + PII), adversarial robustness, ethics (standard + jailbreaking), and fairness—to avoid the limitations of overall accuracy.
    -   Mechanism: Each dimension uses **industry-standard datasets and metrics**. Privacy uses Enron emails (context length $L \in \{50,100,200\}$); Adversarial uses the accuracy gap $\Delta_{\text{robust}}$ between GLUE and AdvGLUE++; Ethics uses ETHICS commonsense and FPR under 5 jailbreaks; Fairness uses MDPD/MEOD on Adult data.
    -   Design Motivation: A single aggregate metric hides opposing trade-offs, such as increased PII leakage alongside decreased training-data leakage. Dimensional decomposition allows for actionable deployment guides like the "✓✗ Decision Table" in Table 2.

2.  **Safety Subspace and Condition Number Theory**:
    -   Function: Mathematically explains why low-rank decomposition degrades PII/ethics but improves adversarial robustness.
    -   Mechanism: For weight matrix $\mathbf{W} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\top$, the refusal vector $\mathbf{v} = \sum_{k=1}^r \lambda_k \mathbf{u}_k$ spans the full subspace. Truncating to $r' < r$ introduces reconstruction error $\|\mathbf{v} - \hat{\mathbf{v}}\|_2^2 = \sum_{k=r'+1}^r \lambda_k^2$, breaking safety defenses. Conversely, robustness is determined by the condition number $\kappa(\mathbf{W}) = s_{\max}/s_{\min}$. Low-rank decomposition discards small singular values, increasing $s_{\min}$, decreasing $\kappa$, and thus improving robustness. Training-data privacy is governed by capacity; subspace compression reduces memorization.
    -   Design Motivation: These theories align with experimental observations, moving beyond "benchmark soup" and applying interpretability findings to compression guidelines.

3.  **Gradient Attribution Layer Sensitivity Analysis**:
    -   Function: Identifies which sub-layers (`embed_tokens`, `q/k/v/o_proj`, `gate/up/down_proj`) are most critical for adversarial robustness.
    -   Mechanism: Defines layer contribution as $a_i = \|(\partial \ell / \partial \mathbf{h}_i) \mathbf{h}_i\|_2$. By calculating the difference $\Delta_i = |a_i^{\text{clean}} - a_i^{\text{adv}}|$, a layer ranking is derived across 3 tasks and 8 model variants.
    -   Design Motivation: Existing low-rank methods (ASVD/AMC) allocate rank based on benign reconstruction error. Ours proves this harms `embed_tokens` and `down_proj`, which are trust-critical layers; future work should preserve more rank for these layers.

### Loss & Training
This is an evaluation paper and does not train new models. Fine-tuning uses standard GSM8K (math) and HumanEval-style (code) datasets. Low-rank compression follows original implementations with controlled $k\%$. Attribution experiments use first-order Taylor expansion without training.

## Key Experimental Results

### Main Results

Low-rank compression of LLaMA-2 Base 13B (k=70%) vs. original model, changes in four trust dimensions:

| Dimension | Metric | Base 13B | BASEL-70 | FWSVD-70 | SVD-70 | Trend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Training-data privacy | leakage @ L=200 (%↓) | 3.99 | **0.00** | 0.79 | 0.11 | ✓ Improved |
| PII (zero-shot) | leakage (%↓) | 2.42 | 0.00 | 0.00 | 0.00 | (Refusal) |
| PII (zero-shot) | **Actual Leakage** (%↓) | 5.67 | 42.00 | 26.25 | 47.25 | ✗ Degraded |
| PII (few-shot protected) | leakage (%↓) | 3.33 | 21.42 | 13.25 | 23.63 | ✗ Degraded |
| Adv. robustness SST-2 | acc drop (%↓) | 18.78 | **3.48** | 15.39 | 17.32 | ✓ BASEL Good |
| Adv. robustness QQP | acc drop (%↓) | 37.51 | **5.63** | 5.57 | 16.42 | ✓ Improved |
| Ethics zero-shot | accuracy (%↑) | 52.92 | 38.45 | 37.80 | 41.87 | ✗ Degraded |
| Ethics few-shot | accuracy (%↑) | 63.11 | 60.36 | **77.15** | 64.77 | ≈ Mitigated |
| Fairness | MDPD (%↓) | 0.01 | - | - | 2.00 | ✗ Degraded |

**Conclusion**: The four-dimensional trend is ✓✗✓✗✗—training data privacy and adversarial robustness benefit, while PII protection, ethical alignment, and fairness consistently deteriorate.

### Ablation Study

Relationship between compression rate $k\%$ and trustworthiness (LLaMA-2 Base 13B + BASEL):

| Metric | k=70% | k=50% | k=30% | Trend |
| :--- | :--- | :--- | :--- | :--- |
| Training-data leakage (%↓) | 0.0017 | 0.0300 | - | Constant low |
| PII zero-shot leakage (%↓) | 42.00 | 42.42 | - | High stability |
| Adv. SST-2 drop (%↓) | 3.48 | 13.46 | -0.61 | Non-monotonic |
| Ethics zero-shot acc (%↑) | 38.45 | 13.47 | 7.48 | Rapid collapse |
| Fairness MDPD (%↓) | - | 0.02 | 8.33 | Collapse at 30% |

Jailbreak FPR (Impact of fine-tuning):

| Model | FPR (%↓) | Model | FPR (%↓) |
| :--- | :--- | :--- | :--- |
| Base 7B | 10.20 | Chat 7B | 45.10 |
| Math Base 7B | **91.80** | Math Chat 7B | **99.40** |
| Prog Base 7B | 32.20 | Prog Chat 7B | 89.90 |

→ Math fine-tuning increases jailbreak FPR of 7B Base from 10.20% to 91.80%; **task-specific fine-tuning almost entirely destroys safety alignment**.

### Key Findings
- **PII vs. Training-data Privacy Divergence**: Compression reduces memory capacity (fewer training email leaks) but weakens the refusal subspace (more PII leaks); a single "privacy" label is insufficient to describe risks.
- **Non-linear Ethics/Fairness Collapse**: BASEL accuracy drops from 38.45% (k=70) to 7.48% (k=30), indicating ethical alignment mechanisms are completely destroyed at high compression ratios.
- **Math Fine-tuning as a High-Risk Action**: Math fine-tuning is more dangerous than programming fine-tuning, likely because math tasks rarely contain refusal samples, thus diluting the safety distribution.
- **`embed_tokens` and `down_proj` are Trust-Critical**: Across 8 LLaMA-2 variants, the top attribution ranking consistently highlights `embed_tokens` and layers near `down_proj`. Future compression should assign higher ranks to these layers.

## Highlights & Insights
- **"Four-Quadrant Decision Table" as a Key Takeaway**: Table 2 provides a simplified ✓✗ checklist for engineers to determine the suitability of low-rank compression.
- **Theory-Experiment Synergy**: The causal chain from single-direction refusal hypotheses to SVD truncation error formulas and observed PII/ethics degradation is rigorously constructed.
- **Reusable Attribution Method**: While $a_i$ is a standard first-order Taylor approach, applying the clean vs. adversarial difference to define "trust sensitivity" is a novel application.
- **Cross-Family Validation**: Testing both LLaMA-2 and Qwen-2.5 ensures findings are not artifacts of a single architecture or tokenizer.

## Limitations & Future Work
- **Untested Instruction-tuned / RLHF Models**: Only Base models were compressed; the impact of compression on active RLHF safety alignment remains unsolved.
- **Narrow Ethical Scope**: ETHICS only covers commonsense morality; conclusions may not generalize to deontology, justice, or virtue.
- **Qualitative Theoretical Depth**: The safety subspace argument assumes refusal is dominated by a single direction, whereas complex dialogues may involve multi-subspace encoding.
- **Diagnostic Focus**: Ours diagnoses rather than fixes; direct proposal of "trust-aware compression" algorithms (e.g., adaptive rank allocation) is deferred to future work.

## Related Work & Insights
- **vs. Hong et al. (2024, ICML) Decoding Compressed Trust**: Their study focuses on quantization/pruning, whereas ours covers low-rank factorization. Low-rank methods excel in adversarial robustness but lag in PII protection compared to quantization.
- **vs. DecodingTrust (Wang et al. 2023)**: Ours extends this systematic protocol from the GPT series to open-source compressed models.
- **vs. Arditi et al. (2024, NeurIPS) Single-Direction Refusal**: This study ports the "safety direction" theory to the compression context, establishing the link between SVD truncation and safety loss.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic trust evaluation + safety subspace explanation for low-rank LLMs, filling a significant gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across model families, methods, rates, and trustworthy dimensions.
- Writing Quality: ⭐⭐⭐⭐ Excellent distillation of findings into actionable tables; clear theoretical derivations.
- Value: ⭐⭐⭐⭐ Direct warning for industrial deployment; provides guidance for next-generation "trust-aware compression" algorithms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks](../../NeurIPS2025/llm_safety/on_the_robustness_of_verbal_confidence_of_llms_in_adversarial_attacks.md)
- [\[ICLR 2026\] Understanding Sensitivity of Differential Attention through the Lens of Adversarial Robustness](../../ICLR2026/llm_safety/understanding_sensitivity_of_differential_attention_through_the_lens_of_adversar.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[NeurIPS 2025\] Demystifying Language Model Forgetting with Low-Rank Example Associations](../../NeurIPS2025/llm_safety/demystifying_language_model_forgetting_with_low-rank_example_associations.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)

</div>

<!-- RELATED:END -->
