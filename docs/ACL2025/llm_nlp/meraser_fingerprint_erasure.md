---
title: >-
  [Paper Note] MEraser: An Effective Fingerprint Erasure Approach for Large Language Models
description: >-
  [ACL 2025][LLM (Other)][fingerprint erasure] Proposes MEraser (Mismatched Eraser), which completely removes backdoor-based fingerprint watermarks in LLMs using less than 1000 samples through a two-phase fine-tuning strategy (mismatched data erasure + clean data recovery) while preserving model performance, and pioneers transferable LoRA erasure adapters.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "fingerprint erasure"
  - "backdoor removal"
  - "model IP protection"
  - "LoRA transfer"
  - "LLM safety"
date: 2026-05-08
content_hash: 735708338fa6f672
---

# MEraser: An Effective Fingerprint Erasure Approach for Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.12551](https://arxiv.org/abs/2506.12551)  
**Code**: [GitHub](https://github.com/JingxuanZhang77/MEraser)  
**Area**: LLM/NLP  
**Keywords**: fingerprint erasure, backdoor removal, model IP protection, LoRA transfer, LLM safety

## TL;DR

Proposes MEraser (Mismatched Eraser), which completely removes backdoor-based fingerprint watermarks in LLMs using less than 1000 samples through a two-phase fine-tuning strategy (mismatched data erasure + clean data recovery) while preserving model performance, and pioneers transferable LoRA erasure adapters.

## Background & Motivation

**Urgent Need for Model Ownership Protection**: The widespread application of LLMs has introduced severe challenges in model provenance and intellectual property (IP) protection, with frequent issues of unauthorized copying and open-source license violations. Backdoor-based fingerprinting technologies are rapidly evolving as the mainstream solution for black-box model authentication.

**Severe Lack of Research on Fingerprint Erasure**: While substantial work has been done on fingerprint embedding, systematic fingerprint erasure methods are virtually non-existent. Existing erasure approaches exhibit critical flaws: incremental fine-tuning requires significant computational resources and fails against overfitted fingerprints, model pruning causes severe performance degradation, and model merging struggles to completely remove all types of fingerprints.

**Impracticality of Inference-Level Erasure Methods**: Token Forcing avoids triggers via exhaustive search but suffers from high computational costs and is ineffective against dynamic fingerprints; CleanGen requires a reference model sharing the same training distribution as the fingerprinted model, which is impractical in real-world scenarios.

**Triple-Unknown Constraints from the Attacker's Perspective**: Attackers face a triple challenge: the trigger composition strategy is unknown, the target fingerprint output is unknown, and the fingerprint-sensitive layers cannot be located. This demands a "blind erasure" capability from erasure methods.

**Diversity of Backdoor Fingerprints**: Existing backdoor fingerprints vary across three dimensions: trigger composition (rare tokens / under-trained tokens / common tokens), mapping architecture (one-to-one / many-to-one), and generalization strategies (overfitted / rule-based). Erasure methods must be universally applicable.

**Theoretical Inspiration**: SEAM utilizes catastrophic forgetting for blind backdoor unlearning, which is effective on small models; however, directly applying it to LLMs leads to irreversible performance collapse, necessitating the design of controllable erasure strategies specifically for LLMs.

## Method

### Overall Architecture

MEraser adopts a two-phase fine-tuning strategy: **Phase 1 (Erase)** fine-tunes the fingerprinted model using a carefully constructed mismatched dataset to break the association between triggers and predefined outputs; **Phase 2 (Recover)** uses a clean dataset to restore model performance while maintaining the erased state of the fingerprint. The entire process requires only 300 mismatched samples + 600 clean samples (less than 1000 samples in total).

### Key Designs

#### 1. Mismatched Dataset Generation

- **Function**: Constructing a dialogue dataset where inputs and outputs are semantically completely unrelated, aiming to disrupt the overfitted trigger-output association in the fingerprinted model.
- **Mechanism**: Starting with the Guanaco dataset, the original input-output pairs are first randomly shuffled to break semantic coherence, and then the shuffled pairs are reassembled into a dialogue format. Multilingual content and diverse task structures are introduced to increase dataset complexity.
- **Design Motivation**: Backdoor fingerprints inherently rely on overfitting during fine-tuning to establish a strong association between specific triggers and predetermined outputs. Mismatched data confuses the model by providing "incorrect" input-output pairs, thereby disrupting this overfitted association. Compared to completely random data, mismatched data retains the dialogue format, which aligns better with the training distribution of LLMs and facilitates subsequent recovery.

#### 2. Two-Phase Erasure Process

- **Function**: Phase 1 fine-tunes the fingerprinted model $M_\theta$ using mismatched data, causing the model to gradually lose its responsiveness to the original trigger $x_t$ for the predefined output $y_t$; Phase 2 fine-tunes the erased model using clean data to restore its language modeling capabilities.
- **Mechanism**: Based on the Neural Tangent Kernel (NTK) theoretical framework, training with mismatched data disrupts the specialized distribution of model parameters in the trigger-output space, while training with clean data pulls the parameters back to the normal language modeling manifold. The key is that the performance degradation of LLMs is controllable (rather than catastrophic forgetting), so the recovery phase can effectively restore performance.
- **Design Motivation**: Directly applying catastrophic forgetting (such as the SEAM method) on LLMs leads to irreversible performance collapse, necessitating the design of a more gentle "disrupt-recover" strategy. The fine-tuning intensity (learning rate, epochs, etc.) of the mismatched data can be finely tuned to adapt the optimal parameters for different fingerprinting methods.

#### 3. Transferable Erasure via LoRA

- **Function**: Training on a foundation model without embedded fingerprints using mismatched data, extracting the LoRA adapter as an "erasure adapter," and then directly merging it into different fingerprinted models to achieve fingerprint erasure, eliminating the need to train each fingerprinted model individually.
- **Mechanism**: The LoRA adapter $\Delta W$ captures the direction of parameter shift induced by mismatched training. This shift direction is generic—it is inherently a perturbation vector that "disrupts overfitted associations," making it transferable across models. Inspired by "LoRA-as-an-Attack" (using LoRA to propagate backdoors), this approach reversely utilizes LoRA to propagate erasure capabilities.
- **Design Motivation**: In practical deployments, training erasure from scratch for each fingerprinted model is costly. Achieving "train once, reuse multiple times" via a pluggable LoRA erasure module substantially reduces computational overhead, making the erasure scheme more viable in practice.

### Loss & Training

Both phases utilize the standard language modeling loss (next-token prediction cross-entropy loss). The key hyperparameter differences are:
- **Erase Phase**: Uses 300 mismatched samples with a relatively higher learning rate to quickly break the fingerprint association (specific parameters vary by fingerprinting method; IF-SFT requires stronger erasure intensity).
- **Recover Phase**: Uses 600 clean samples with a lower learning rate to gently restore language modeling capabilities while maintaining the erased state of the fingerprint.

## Key Experimental Results

### Main Results

MEraser's erasure performance across three model architectures $\times$ three fingerprinting methods (FSR: Fingerprint Success Rate, PPL: Perplexity):

| Model | Fingerprinting Method | Fingerprinted Model FSR | Fingerprinted Model PPL | Post-Erasure FSR | Post-Erasure PPL | Post-Recovery FSR | Post-Recovery PPL |
|------|----------|:----------:|:----------:|:---------:|:---------:|:---------:|:---------:|
| Llama2-7B | IF-SFT | 100% | 4.80 | 0% | 17.33 | 0% | 7.31 |
| Llama2-7B | UTF | 100% | 9.31 | 0% | 5.35 | 0% | 4.48 |
| Llama2-7B | HC | 100% | 6.71 | 0% | 5.53 | 0% | 4.65 |
| Mistral-7B | IF-SFT | 100% | 4.09 | 0% | 15.85 | 0% | 6.87 |
| Mistral-7B | UTF | 100% | 5.01 | 0% | 8.01 | 0% | 4.12 |
| Mistral-7B | HC | 100% | 5.11 | 0% | 5.87 | 0% | 4.00 |
| AmberChat-7B | IF-SFT | 100% | 4.26 | 0% | 25.2 | 0% | 9.10 |
| AmberChat-7B | UTF | 100% | 7.62 | 0% | 8.08 | 0% | 5.01 |
| AmberChat-7B | HC | 100% | 9.10 | 0% | 6.07 | 0% | 4.91 |

### Ablation Study

Comparison with baseline methods on Llama2-7B (evaluating the removal of IF-SFT / UTF / HC fingerprints):

| Method | IF-SFT FSR | UTF FSR | HC FSR | Retains PPL? | Universally Applicable? |
|------|:----------:|:-------:|:------:|:----------:|:-------:|
| Incremental FT (Guanaco) | 100% | 75% | 0% | ✓ | ✗ |
| Incremental FT (ShareGPT) | 100% | 3.125% | 0% | ✓ | ✗ |
| L1 Pruning (5%) | 87.5% | 3.125% | 30% | ✗ | ✗ |
| L2 Pruning (5%) | 100% | 81.25% | 40% | ✗ | ✗ |
| Random Pruning (20%) | 50% | 0% | 30% | ✗ | ✗ |
| Taylor Pruning (20%) | 100% | 3.125% | 70% | ✗ | ✗ |
| Model Merging (Task Arith.) | 0% | 0% | 50-90% | ✓ | ✗ |
| CleanGen | 0% | 0% | 0% | - | △ (Requires reference model) |
| Token Forcing | 0% | 0% | 90% | - | ✗ |
| **MEraser (Ours)** | **0%** | **0%** | **0%** | **✓** | **✓** |

### Key Findings

- **100% Erasure Rate**: MEraser reduces the FSR from 100% to 0% across all 9 combinations (3 models $\times$ 3 fingerprints), being the only method that is universally effective.
- **Recoverable Performance**: After the recovery phase, the PPL is close to or even better than the original level of the fingerprinted model (the post-recovery PPL of UTF/HC is even lower than that of the fingerprinted model, as mismatched training exerts a regularization effect).
- **Robustness of IF-SFT**: Due to its many-to-one overfitted mapping, IF-SFT is the hardest to erase, requiring higher erasure intensity. This leads to a larger intermediate spike in PPL (e.g., Llama2-7B rising from 4.80 to 17.33), but it can still be fully erased in the end.
- **Extremely Low Data Requirements**: Requiring only 300 + 600 = 900 samples in total, which is far fewer than the 6000+ samples used by methods like incremental fine-tuning.
- **Effectiveness of Transferable Erasure**: The LoRA erasure adapter achieves FSR = 0% in most cases, with only a 37.5% residue on UTF, verifying the feasibility of training once and reusing multiple times.
- **No Loss on Downstream Tasks**: Evaluations on SuperGLUE and SciQ ACC show minimal changes in downstream performance after erasure-recovery; some tasks even show improvements due to the regularization effect.

## Highlights & Insights

- **Dual Security Perspective**: Exposing the vulnerability of fingerprint protection technologies from an attacker's perspective provides an "offensive-defensive benchmark" for developing more robust model protection schemes. This dual research paradigm is highly valuable in the security domain.
- **Concise and Efficient Core Idea**: Breaking overfitted associations with mismatched data $\rightarrow$ restoring performance with clean data. The entire logic is intuitive, simple to implement, and highly effective, reflecting the engineering aesthetics of a "minimalist solution."
- **Reverse Utilization of LoRA**: Prior works use LoRA to propagate backdoor attacks; this paper reversely employs LoRA to propagate erasure capabilities, demonstrating the dual-use nature of the technology and innovative application of transfer learning in security scenarios.
- **Regularization Byproduct**: Mismatched data training unexpectedly exerts a regularization effect on overfitted fingerprinted models, with the post-recovery PPL dropping below that of the original fingerprinted model. This finding carries significant theoretical insight.

## Limitations & Future Work

1. **Limited to Backdoor-Based Fingerprints**: Ineffective against inference-stage watermarks (such as KGW sampling strategy modifications). The applicability is restricted to backdoor-based fingerprints/watermarks embedded during training.
2. **Incomplete Transferable Erasure on UTF**: The LoRA erasure adapter leaves a 37.5% FSR residue on UTF fingerprints, indicating room for improvement in generalizability.
3. **PPL Discrepancy After Recovery**: Specifically, the post-recovery PPL for IF-SFT fingerprints (7.31 vs. 4.80 originally) still exhibits a notable gap, suggesting that the trade-off between erasure intensity and recovery quality is not yet fully resolved.
4. **Validated Only at 7B Scale**: Main experiments are focused on 7B models. Effects on larger scales (13B/70B) or newer architectures (e.g., Mixtral MoE) remain to be verified.
5. **Insufficient Insights for Defense**: As primarily an offensive study, the paper only briefly discusses how to design "MEraser-resistant" robust fingerprinting schemes without providing concrete defensive solutions.

## Related Work & Insights

- **IF-SFT / UTF / HashChain**: Three representative backdoor fingerprinting methods that utilize rare tokens, under-trained tokens, and common tokens as triggers, respectively, forming the comprehensive evaluation targets of this paper.
- **SEAM (Zhu et al., 2023)**: Utilizes catastrophic forgetting for blind backdoor unlearning, providing inspiration via the NTK theoretical framework, but cannot be directly transferred to LLMs.
- **LoRA-as-an-Attack (Liu et al., 2024)**: Reveals that backdoors can propagate across models via LoRA adapters; this paper reversely utilizes this finding to achieve transferable erasure.
- **CleanGen (Li et al., 2024)**: An inference-level erasure method that requires a reference model for probability comparison; while theoretically general, its practical deployment requirements are stringent.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The two-phase mismatched erasure framework is concise and effective, and the transferable LoRA erasure shows innovation.
- **Effectiveness**: ⭐⭐⭐⭐⭐ 100% erasure rate across 9/9 scenarios, completely outperforming all baseline methods.
- **Practicality**: ⭐⭐⭐⭐ Minimal data requirements, simple workflow, and transferable, but limited to backdoor-based fingerprints.
- **Value**: ⭐⭐⭐⭐ A dual-security study exposing fingerprint vulnerabilities, pointing the way toward more robust IP protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Generative Psycho-Lexical Approach for Constructing Value Systems in Large Language Models](generative_psycholexical_approach_for_constructing_value.md)
- [\[ACL 2025\] A Semantic-Aware Layer-Freezing Approach to Computation-Efficient Fine-Tuning of Language Models](a_semantic-aware_layer-freezing_approach_to_computation-efficient_fine-tuning_of.md)
- [\[ICCV 2025\] VA-GPT: Aligning Effective Tokens with Video Anomaly in Large Language Models](../../ICCV2025/llm_nlp/va_gpt_aligning_effective_tokens_video_anomaly.md)
- [\[ACL 2025\] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](human_nlp_cooperation_survey.md)
- [\[ACL 2025\] Literature Meets Data: A Synergistic Approach to Hypothesis Generation](literature_meets_data_hypothesis.md)

</div>

<!-- RELATED:END -->
