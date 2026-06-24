---
title: >-
  [Paper Note] Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models
description: >-
  [ICML 2025][Model Compression][Quantization Safety] This paper systematically evaluates the impact of mainstream quantization methods (AWQ, AQLM, LLM-QAT, QLoRA) on LLM safety across different calibration datasets and bit-widths. It reveals that all quantization methods lead to a dramatic rise in Attack Success Rate (ASR) (from 0.3% to 85%). To address this, the authors propose the Q-resafe framework, which efficiently restores the safety capabilities of quantized models with…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Quantization Safety"
  - "Safety Patching"
  - "DPO Alignment"
  - "Safety-Critical Weights"
  - "LLM Quantization"
date: 2026-05-08
content_hash: 6f8f243f995dd7c8
---

# Q-resafe: Assessing Safety Risks and Quantization-aware Safety Patching for Quantized Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2506.20251](https://arxiv.org/abs/2506.20251)  
**Code**: [Available](https://github.com/Thecommonirin/Qresafe)  
**Area**: Model Compression  
**Keywords**: Quantization Safety, Safety Patching, DPO Alignment, Safety-Critical Weights, LLM Quantization

## TL;DR

This paper systematically evaluates the impact of mainstream quantization methods (AWQ, AQLM, LLM-QAT, QLoRA) on LLM safety across different calibration datasets and bit-widths. It reveals that all quantization methods lead to a dramatic rise in Attack Success Rate (ASR) (from 0.3% to 85%). To address this, the authors propose the Q-resafe framework, which efficiently restores the safety capabilities of quantized models with extremely low computational overhead through safety patch data construction, DPO alignment, and selective safety-critical weight updates.

## Background & Motivation

- **Background**: Quantization (16-bit to 4/8-bit) is a core compression technology for deploying LLMs on edge devices. Mainstream approaches include Post-Training Quantization (PTQ, e.g., AWQ/AQLM) and Quantization-Aware Training (QAT, e.g., LLM-QAT/QLoRA), both of which have matured in maintaining model utility.
- **Limitations of Prior Work**: Quantization alters weights to a much greater extent than lightweight fine-tuning, potentially severely disrupting the safety alignment achieved through RLHF or instruction tuning. Existing research indicates that even minor fine-tuning can cause safety degradation; hence, the impact of quantization is expected to be even more severe.
- **Limitations of Prior Work**: Existing work (e.g., Hong et al. 2024) focuses only on a few PTQ methods without calibration data (GPTQ/AWQ), lacking a systematic evaluation of the four major categories of mainstream methods (PTQ ± fine-tuning × QAT ± LoRA). Furthermore, an effective post-hoc safety remediation scheme is missing.
- **Overlooked Risks**: Quantization methods often utilize calibration data to assist the quantization process, but the inclusion of harmful samples in calibration data can further exacerbate safety degradation—a risk that has not been adequately investigated.
- **Ours**: (1) A comprehensive safety evaluation covering PTQ/QAT × calibration data types (benign/indirectly harmful/directly harmful) × bit-widths (INT4/INT8); (2) Q-resafe—an efficient safety patching framework guided by the pre-quantization model, optimized via DPO, and updating only safety-critical weights.
- **Core Idea**: Since the utility of the quantized model remains largely intact, safety restoration only requires a "minimally invasive" adjustment of a small set of safety-critical weights rather than a complete retraining.

## Method

### Overall Architecture

Q-resafe employs a three-stage pipeline to restore the safety capabilities of quantized LLMs:

1. **Safety Patch Dataset Construction**: Leveraging the pre-quantization full-precision LLM to generate preference data pairs (winner/loser), achieving knowledge distillation-based transfer of safety capabilities.
2. **DPO Safety Alignment**: Using the quantized model as the reference model, aligning the safety behavior of the quantized model to the full-precision model via DPO loss.
3. **Selective Safety-Critical Weight Update**: Identifying safety-critical weights based on SNIP scores and updating only the LoRA parameters of these weights to preserve utility.

### Key Designs

1. **Safety Patch Dataset Construction**: For each prompt $x$ in the calibration dataset, responses are generated using the full-precision model $\pi_{\mathbf{W}}$ and the quantized model $\pi_{\mathbf{Q}_0}$ respectively. The response from the full-precision model is labeled as the winner (preferred) $y_w$, and the response from the quantized model is labeled as the loser (dispreferred) $y_l$, forming a preference triplet $(x, y_w, y_l)$. The core advantages of this design are: (a) eliminating the need for manual preference labeling; (b) transferring the strong safety capability of the full-precision model to the quantized model from a knowledge distillation perspective; and (c) generating comparison pairs from actual models that present a greater challenge than manual reference responses, thereby realizing a more rigorous safety patch.

2. **DPO Safety Alignment Objective**: Based on the constructed preference dataset $\mathcal{D}_{patch}$, the DPO loss function is defined as:

$$\mathcal{L} = -\mathbb{E}_{\mathcal{D}_{patch}} \log \sigma\left(\beta \log \frac{\pi_{\mathbf{Q}}(y_w|x)}{\pi_{\mathbf{Q}_0}(y_w|x)} - \beta \log \frac{\pi_{\mathbf{Q}}(y_l|x)}{\pi_{\mathbf{Q}_0}(y_l|x)}\right)$$

where $\pi_{\mathbf{Q}_0}$ is the original quantized model serving as the reference, and $\pi_{\mathbf{Q}}$ is the patched model to be optimized. The constraints require the updates to satisfy both the low-rank structure of LoRA and the safety-critical weight mask: $\mathbf{Q} = \mathbf{Q}_0 + \text{Quant}(\mathbf{M}_Q \odot \mathbf{B}\mathbf{A})$. The DPO loss itself possesses a regularizing effect, preventing the patched model from deviating excessively from the quantized model to preserve utility.

3. **Safety-Critical Weight Identification (SNIP Score)**: The SNIP method is utilized to calculate the importance score of each weight:

$$I(W_{ij}, x) = |W_{ij} \cdot \nabla_{Q_{ij}} \mathcal{L}(x)|$$

Averaging over the calibration dataset yields $\text{SafeScore}(\mathbf{Q}) = \mathbb{E}_{x \in \mathcal{D}_{calib}} I(Q_{ij}, x)$. Weights with scores in the top-$\tau$ percentile are selected as safety-critical weights, based on which the mask matrix $\mathbf{M}_Q$ is constructed. Guided by the finding that "LLM capabilities are concentrated in a few weights," this design restores safety by modifying only a few key weights while keeping the majority unchanged to maintain utility.

4. **Periodic Re-identification and Mask Updating**: As training iterations progress, the distribution of safety-critical weights may shift. Therefore, the subset of safety-critical weights is re-identified every $K$ iterations to update the mask matrix. The weight mask $\mathbf{M}_Q$ is decomposed into a mask pair $(\mathbf{M}_A, \mathbf{M}_B)$ corresponding to the LoRA variables.

### Loss & Training

- **Optimization Objective**: DPO loss + LoRA low-rank constraints + safety-critical weight mask
- **Update Rules**: The SGD update for the LoRA matrix $\mathbf{A}$ is given by:

$$\mathbf{A}_{t+1} = \mathbf{M}_A \odot (\mathbf{A}_t - \eta \nabla_A \mathcal{L}(\mathbf{A}_t, \mathbf{B}_t)) + (\mathbf{1} - \mathbf{M}_A) \odot \mathbf{A}_t$$

This ensures that gradient updates are applied only to safety-critical positions masked with 1, while other positions remain unchanged. The update rule for $\mathbf{B}$ is formulated similarly.
- **Hyperparameters**: LoRA rank $r=128$, $\alpha=256$, DPO $\beta=0.01$, learning rate $5\times10^{-6}$, re-identification interval $K=1000$, safety-critical threshold $\tau=0.6$.
- **Computational Resources**: 4×NVIDIA A100 40GB, requiring only 1 epoch to complete safety patching.

## Key Experimental Results

### Main Results

**Safety Evaluation Panorama (INT4, ASR% ↓, lower is safer)**:

| Model | Method | Risk-I | Risk-II | Risk-III | MT-bench↑ | AlpacaEval↑ |
|------|------|:---:|:---:|:---:|:---:|:---:|
| Llama-2 (Baseline 0.3%) | AWQ | 42.4 | 42.4 | 42.4 | 6.51 | 68.37 |
| | AQLM | 18.5 | 75.5 | 77.4 | 6.40 | 66.42 |
| | LLM-QAT | 16.9 | 82.9 | 71.2 | 6.71 | 66.54 |
| | QLoRA | 42.3 | 83.4 | **85.3** | 6.40 | 63.92 |
| Gemma (Baseline 9.2%) | AWQ | 17.9 | 17.9 | 17.9 | 6.14 | 65.40 |
| | AQLM | 25.3 | 69.9 | 55.4 | 6.12 | 61.75 |
| | LLM-QAT | 20.7 | 68.4 | 52.9 | 6.28 | 62.85 |
| | QLoRA | 39.4 | 68.6 | 61.3 | 6.15 | 59.13 |

**Q-resafe Patching Performance**:

| Scenario | Model | Baseline Quantization ASR Increase | Q-resafe ASR Increase |
|------|------|:---:|:---:|
| Risk-I (Benign) | Llama INT4 | +16.6% | +1.5% |
| Risk-I (Benign) | Gemma INT4 | +11.5% | +0.9% |
| Risk-II (Indirectly Harmful) | Llama INT4 | +82.6% | +13.3% |
| Risk-III (Directly Harmful) | Llama INT4 | +92.3% | +13.6% |
| Risk-III (Directly Harmful) | Gemma INT4 | +66.7% | +1.8% |

**AWQ No Fine-tuning Scenario (Under Decoding Attacks)**:

| Method | Model | INT4 ASR | INT8 ASR | MT-Bench | AlpacaEval |
|------|------|:---:|:---:|:---:|:---:|
| AWQ | Llama | 42.4 | 39.1 | 6.51 | 68.37 |
| Q-resafe | Llama | 25.0 | 23.9 | 6.52 | 69.56 |
| AWQ | Gemma | 17.9 | 17.7 | 6.14 | 65.40 |
| Q-resafe | Gemma | 11.1 | 10.5 | 6.19 | 66.44 |

### Ablation Study

**Effect of Safety-Critical Weight Ratio $\tau$** (Llama INT4, Risk-I):

| $\tau$ | ASR (%) | GPU Time (h) | MT-Bench |
|:---:|:---:|:---:|:---:|
| 1.0 (All updated) | 1.6 | 2.1 | 7.3 |
| 0.8 | 1.6 | 1.8 | 7.2 |
| 0.6 | 1.8 | 1.2 | 7.1 |
| 0.4 | 5.5 | 0.8 | 6.8 |
| 0.2 | 13.9 | 0.5 | 6.6 |
| 0.0 (No identification) | 42.2 | - | 6.4 |

**Comparison of Different Safety Patching Methods** (INT4):

| Method | ASR (%) | GPU Time (h) |
|------|:---:|:---:|
| LLM-QAT + SFT | 12.4 | 8.4 |
| LLM-QAT + DPO | 1.5 | 9.6 |
| LLM-QAT + Q-resafe | 1.6 | **1.2** |
| QLoRA + SFT | 26.9 | 3.4 |
| QLoRA + DPO | 2.4 | 3.8 |
| QLoRA + Q-resafe | 2.4 | **1.2** |

**Multi-bitwidth Ablation** (Llama, UltraChat):

| Method | 8-bit | 4-bit | 3-bit | 2-bit |
|------|:---:|:---:|:---:|:---:|
| AQLM | 17.1 | 18.5 | 28.6 | 40.1 |
| QLoRA | 41.7 | 42.3 | 67.3 | 82.0 |
| AWQ | 10.5 | 17.4 | 29.5 | 38.6 |
| Q-resafe | **1.6** | **1.8** | **5.9** | **12.4** |

**Validation on More Quantization Methods** (bitsandbytes Series):

| Method | Pre-patch ASR | Post-patch ASR with Q-resafe |
|------|:---:|:---:|
| LLM.int8() | 19.2 | 5.2 |
| NF4 | 23.9 | 5.5 |
| FP4 | 35.2 | 6.0 |

### Key Findings

1. **All quantization methods degrade safety**: Even with benign calibration datasets, post-quantization ASR rises significantly, with QLoRA showing the most severe degradation (42.3%) and LLM-QAT showing the least (16.9%).
2. **The safety level of calibration datasets has a substantial impact**: Transitioning from benign to directly harmful data triggers an ASR surge from 18.5% to 77.4% (for AQLM), and indirectly harmful data (e.g., role-play/identity shift) exerts an even more pronounced effect.
3. **Lower bit-widths entail greater safety risks**: INT4 exhibits worse safety degradation than INT8, and 3-bit/2-bit quantization leads to even more drastic degradation, with the ASR climbing up to 82%.
4. **PTQ vs. QAT**: Under benign data, QAT achieves better safety than PTQ, as QAT adjusts parameters to compensate for information loss during the quantization process.
5. **Full-parameter fine-tuning vs. LoRA fine-tuning**: LLM-QAT (full-parameter) preserves more safety capabilities than QLoRA (LoRA).
6. **Q-resafe is highly efficient and generalizable**: Its GPU overhead is only 1/8 of standard DPO (1.2h vs. 9.6h), and it can be applied to any quantization method.
7. **Safety-critical weight identification is crucial**: Decreasing $\tau$ from 0.6 to 0 causes the ASR to skyrocket from 1.8% to 42.2%, validating the necessity of selective updates.

## Highlights & Insights

1. **A complete evaluation-remediation paradigm**: This work systematically evaluates four quantization methods × three datasets × multiple bit-widths, and further delivers a solutions framework (Q-resafe), demonstrating high completeness.
2. **"Minimally invasive" patching of safety-critical weights**: By utilizing SNIP scores to pinpoint safety-critical weights, modifying only the top 60% of these weights achieves safety comparable to full updates, significantly reducing computational overhead.
3. **Preference data construction from a knowledge-distillation perspective**: The automated scheme—generating winners from the full-precision model and losers from the quantized model—requires no human annotation and outperforms manual reference responses.
4. **Security auditing awareness for calibration data**: The counter-intuitive finding that indirectly harmful data (role-play/identity shift) has a more potent impact than directly harmful data reminds practitioners of the absolute necessity to audit calibration datasets.
5. **Method agnosticism**: Q-resafe is highly generalizable and can be seamlessly applied to any quantization method (including bitsandbytes methods such as LLM.int8(), NF4, and FP4).

## Limitations & Future Work

1. **Validation limited to 7B models**: Experiments are restricted to Llama-2-7B-Chat and Gemma-7B-Instruct; validation on larger-scale models (13B/70B) or newer releases (e.g., Llama-3) is yet to be explored.
2. **Safety-in-mind QAT**: The authors themselves point out that incorporating safety directly into the quantization pipeline (rather than as a post-hoc patch) is a more fundamental direction for future research.
3. **Limitations of SNIP scores**: Gradient-based importance scoring heavily relies on the representativeness of calibration data. If the calibration data distribution diverges drastically from actual deployment scenarios, the identified safety-critical weights could be inaccurate.
4. **Limited dimensions of safety evaluation**: The evaluation is primarily based on ASR and fails to cover fine-grained safety dimensions such as toxicity levels, bias, and privacy leakage.
5. **Requirement of full-precision model access**: Constructing safety patching data relies on the full-precision model to generate winner responses. If the full-precision model is inaccessible, one must rely on other aligned models as alternatives.

## Related Work & Insights

- **Safety fine-tuning degradation**: Qi et al. (2024b) showed that even benign fine-tuning can compromise safety alignment. This paper extends this finding to quantization.
- **Safety evaluation of quantization**: Hong et al. (2024) and Egashira et al. (2024) made preliminary explorations into the safety risks of quantized LLMs, yet with limited coverage of methods.
- **DPO alignment**: The DPO framework by Rafailov et al. (2024) is ingeniously adapted here for post-quantization safety remediation.
- **SNIP pruning**: The SNIP score originally proposed by Lee et al. (2019) for network pruning is creatively repurposed to identify safety-critical weights.
- **Insights**: Post-quantization/compression safety patching remains an important yet relatively under-explored area. The paradigm of "minimally invasive" patching can potentially be generalized to other model compression paradigms (such as pruning and distillation).

## Rating

- Novelty: ⭐⭐⭐⭐ The evaluation framework is highly systematic and comprehensive. The design combining safety-critical weight identification and masked DPO in Q-resafe is relatively novel, though the individual elements (DPO/SNIP/LoRA) are existing techniques.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Features a comprehensive evaluation matrix covering 4 quantization methods × 3 datasets × 2 bit-widths × 2 models. The ablation studies are highly detailed ($\tau$, method comparison, multi-bitwidth), with additional quantization techniques validated in the appendix.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-organized evaluation section (intra-method analysis $\rightarrow$ cross-method analysis $\rightarrow$ summary), with complete descriptions of formulas and algorithms.
- Value: ⭐⭐⭐⭐ It systematically evaluates the intersecting impacts of quantization × calibration data × safety for the first time. Q-resafe is highly practical (requiring only 1.2h of GPU time to patch), offering direct guidance for the secure deployment of quantized LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Granular Study of Safety Pretraining under Model Abliteration](../../NeurIPS2025/model_compression/a_granular_study_of_safety_pretraining_under_model_abliteration.md)
- [\[ACL 2025\] EfficientQAT: Efficient Quantization-Aware Training for Large Language Models](../../ACL2025/model_compression/efficientqat.md)
- [\[ICML 2025\] BoA: Attention-aware Post-training Quantization without Backpropagation](boa_attention-aware_post-training_quantization_without_backpropagation.md)
- [\[ICML 2025\] Weak-to-Strong Jailbreaking on Large Language Models](weak-to-strong_jailbreaking_on_large_language_models.md)
- [\[ICML 2025\] Persistent Topological Features in Large Language Models](persistent_topological_features_in_large_language_models.md)

</div>

<!-- RELATED:END -->
