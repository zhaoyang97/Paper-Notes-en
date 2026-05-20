---
title: >-
  [Paper Note] Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty
description: >-
  [AAAI 2026][Medical Imaging][LLM honesty] This paper reveals that the root cause of SFT-induced dishonesty in LLMs is **impaired self-expression** (rather than degraded self-knowledge)…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "LLM honesty"
  - "supervised fine-tuning"
  - "knowledge boundary"
  - "neuron restoration"
  - "parameter-efficient"
date: 2026-05-08
content_hash: d36f2227d1e05c4b
---

# Fine-Tuned LLMs Know They Don't Know: A Parameter-Efficient Approach to Recovering Honesty

**Conference**: AAAI 2026
**arXiv**: [2511.12991](https://arxiv.org/abs/2511.12991)  
**Code**: None  
**Area**: Medical Imaging / LLM Alignment
**Keywords**: LLM honesty, supervised fine-tuning, knowledge boundary, neuron restoration, parameter-efficient

## TL;DR

This paper reveals that the root cause of SFT-induced dishonesty in LLMs is **impaired self-expression** (rather than degraded self-knowledge), and proposes the HCNR framework accordingly. By identifying honesty-critical neurons via Fisher information and restoring them to their pre-trained states with Hessian-guided compensation, HCNR recovers 33.25% of honesty using only 256 data samples and 20% of parameters, achieving over 2.23× speedup.

## Background & Motivation

### Importance and Fragility of LLM Honesty

LLM honesty encompasses two dimensions:

**Self-knowledge**: the ability to recognize the boundaries of one's own knowledge

**Faithful self-expression**: the ability to respond truthfully based on that self-knowledge

Honesty is typically established during the alignment phase (e.g., RLHF), but **supervised fine-tuning (SFT) can severely undermine this property**. For example:
- After fine-tuning on legal QA, LLMs begin to confidently fabricate legal provisions
- After fine-tuning on medical diagnosis, LLMs produce plausible-sounding answers to questions beyond their knowledge
- Such "hallucinations" can have serious consequences in high-stakes domains

### Assumptions and Limitations of Prior Methods

Existing honesty recovery methods (e.g., RAIT, DPO, ORPO) share an implicit assumption: SFT deeply destroys the model's knowledge boundary capabilities, necessitating **large-scale data and full-parameter adjustment** for repair. This results in:
- Requirements for thousands of specially crafted IDK samples
- Long training times (30–40 minutes)
- Risk of catastrophic forgetting on downstream tasks

### Key Observation: Dishonesty as an "Illusory Phenomenon"

Two experiments reveal a counterintuitive finding:

**Observation 1**: During RAIT honesty-enhancement training, model honesty recovers rapidly within approximately 60 gradient steps—suggesting that the core knowledge boundary capability has not been destroyed.

**Observation 2**: Linear probes (logistic regression) trained on the hidden states of fine-tuned LLMs can distinguish answerable from unanswerable questions with high accuracy (high AUROC). Moreover, probes trained on the base model transfer directly to fine-tuned models while maintaining high AUROC—indicating that SFT does not alter the geometric structure of knowledge boundary representations.

**Conclusion**: SFT-induced dishonesty is a **failure of self-expression**, not a **loss of self-knowledge**.

## Method

### Overall Architecture

HCNR (Honesty-Critical Neurons Restoration) proceeds in two stages:

**Stage 1: Identifying and Restoring Honesty-Critical Neurons**
- Fisher information is used to assess each neuron's importance to honesty and downstream tasks
- Neurons with high honesty importance and low task importance are selected
- Among these, layers/neurons most perturbed by SFT are prioritized
- Selected neurons are restored to their pre-trained states

**Stage 2: Hessian-Guided Compensation**
- Restored neurons and unrestored task neurons introduce coordination misalignment
- The Hessian matrix is used to compute an optimal compensation vector that minimizes activation discrepancy

### Key Designs

#### 1. **Intra-layer Sensitivity Assessment**

Core Idea: The diagonal elements of the Fisher Information Matrix (FIM) serve as unbiased estimates of neuron importance.

For the $k$-th neuron in layer $j$, its importance on dataset $D$ is:

$$s_{j,k} = \mathbb{E}_{(x,y)\sim D}[(\partial_{W_{j,k}}\mathcal{L})^2]$$

$s_{j,k}^{hon}$ and $s_{j,k}^{task}$ are computed on honesty data $D^{hon}$ and task data $D^{task}$ respectively. A priority score is defined as:

$$r_{j,k} = s_{j,k}^{hon} \cdot \log\frac{s_{j,k}^{hon}}{s_{j,k}^{task}}$$

A high $r_{j,k}$ indicates that the neuron is critical for honesty but has minimal impact on downstream tasks—precisely the neurons to be preserved. The top $R_{IW}$ fraction of neurons per layer are selected as candidates.

Design Motivation: Restoring all neurons indiscriminately would degrade task performance; precise identification of neurons that are honesty-relevant but task-neutral is therefore necessary. The KL-divergence-based priority score better discriminates the two neuron types compared to a simple difference.

#### 2. **Cross-layer Perturbation Analysis**

SFT perturbs different layers to different degrees due to the hierarchical specialization of LLMs; layers with the greatest perturbation should be prioritized:

$$d_j = \frac{\|(W_j - W_j') \odot M_j\|_2}{\|W_j \odot M_j\|_2}$$

The top $R_{CW}$ fraction of highly perturbed layers are selected. The final honesty-critical neuron set $A^{hc}$ is obtained by intersecting the candidate layers and candidate neurons.

Design Motivation: Indiscriminately protecting all layers over-constrains downstream performance. In practice, certain layers (e.g., middle layers) exhibit greater perturbation in their honesty neurons and require priority restoration.

#### 3. **Hessian-Guided Compensation**

Simply restoring neurons to their pre-trained states introduces new misalignment—because all parameters are updated in coordination during SFT. Restoring a subset breaks this coordination, causing a rebound in honesty task loss.

The compensation vector is derived within the OBS framework:

$$c_{j,k} = \frac{W_{j,k}^{sft} - W_{j,k}^{orig}}{[H^{-1}]_{kk}} \cdot H_{:,k}^{-1}$$

The final weight update rule is:

$$W_{j,i}^{HCNR} = \begin{cases} W_{j,i}^{orig} + [\sum_{k \in A_j^{task}} c_{j,k}]_i & \text{if } i \in A_j^{hc} \\ W_{j,i}^{sft} & \text{if } i \in A_j^{task} \end{cases}$$

Design Motivation: Restoration without compensation causes honesty rebound (ablation shows F1 dropping from 72.84 to 65.96); Hessian compensation precisely bridges the coordination gap between restored and task neurons.

### Loss & Training

- HCNR is **training-free**—no additional training is required; only a small amount of data is needed to compute Fisher/Hessian statistics
- Only $|D^{hon}| = |D^{task}| = 128$ samples are required
- Hyperparameters: $R_{IW} = 0.5$ (select 50% of neurons per layer), $R_{CW} = 0.4$ (select 40% of layers)
- Only 20% of total parameters are modified
- Experiments are repeated 3 times and averaged
- Runs on Nvidia A800-80GB GPU

## Key Experimental Results

### Main Results

Results on Llama-3.1-8B-Instruct fine-tuned on HotpotQA and MedMCQA, followed by honesty recovery:

| Method | FalseQA F1 | NEC F1 | RefuNQ F1 | KUQ F1 | SelfAware F1 | Task Acc. |
|--------|-----------|--------|-----------|--------|-------------|----------|
| Fine-tuned | 56.51 | 35.46 | 32.43 | 68.50 | 67.01 | 30.65 |
| RAIT | 68.59 | 68.28 | 71.21 | 80.38 | 64.46 | 27.05 |
| DPO | 69.12 | 69.52 | 72.91 | 80.96 | 64.76 | 29.00 |
| ORPO | 65.83 | 70.03 | 71.26 | 79.21 | 65.21 | 29.60 |
| **HCNR** | **68.30** | **71.90** | **71.70** | **82.90** | **69.40** | **30.30** |

Efficiency comparison (recovery after HotpotQA fine-tuning):

| Method | Data Size | Param % | Time | Avg. F1 | Avg. RF Δ |
|--------|-----------|---------|------|---------|----------|
| RAIT | 5000 | 100% | 8.76 min | 70.58 | +33.40 |
| DPO | 5000 | 100% | 42.78 min | 71.45 | +37.41 |
| ORPO | 9000 | 100% | 30.97 min | 70.31 | +39.94 |
| **HCNR** | **256** | **20%** | **3.93 min** | **72.84** | **+42.64** |

### Ablation Study

| Stage 1 Config | Stage 2 Config | Avg. F1 | Avg. RF Δ | Task Acc. |
|----------------|---------------|---------|----------|----------|
| Random | Ours | 65.44 | +36.31 | 29.60 |
| w/o Task | Ours | 70.43 | +33.24 | 28.30 |
| Ours | w/o Compensation | 65.96 | +33.09 | 30.37 |
| Random | w/o Compensation | 54.21 | +23.04 | 29.70 |
| **Ours** | **Ours** | **72.84** | **+42.64** | **30.30** |

### Key Findings

1. **HCNR achieves top performance on 3–4 of 5 honesty benchmarks** while maintaining the highest task accuracy
2. **Efficiency advantages are pronounced**: only 256 data samples (20× reduction), 20% of parameters, and 3.93 minutes (2.23× speedup) suffice to outperform all baselines
3. **F1 saturates at 128 samples**: further data increases yield negligible gains, confirming the hypothesis that honesty degradation is a localized phenomenon
4. **Hessian compensation is indispensable**: removing compensation reduces F1 from 72.84 to 65.96 and RF Δ from 42.64 to 33.09
5. **ICL yields the worst recovery**: indicating that fine-tuning impairs in-context learning capabilities
6. **Cross-model generalization**: effective across 5 LLM families including Llama-3, Qwen2/3, and Mistral

## Highlights & Insights

1. **The core insight is highly valuable**: "SFT-induced dishonesty is a failure of expression, not a loss of cognition"—this finding reshapes the understanding of SFT side effects
2. **The linear probe transfer experiment is elegantly designed**: probes trained on the base model transfer directly to fine-tuned models with sustained effectiveness, providing strong evidence for the robustness of knowledge boundary representations
3. **Training-free design**: unlike RAIT/DPO/ORPO, HCNR requires no additional training—only statistical computation followed by direct weight modification
4. **Asymmetric behavior of $R_{IW}$ and $R_{CW}$**: $R_{IW}$ saturates rapidly (intra-layer neuron selection is relatively insensitive), while $R_{CW}$ has a clear optimum at 0.3–0.4 (indicating that cross-layer selection is more critical)
5. **Pareto frontier dominance**: on the task-honesty tradeoff plot, HCNR's Pareto frontier strictly dominates all baselines

## Limitations & Future Work

1. **Assumption of pre-trained state optimality**: the framework assumes that the pre-trained state represents optimal honesty, whereas the post-alignment state may in fact be superior
2. **Approximations in Fisher/Hessian computation**: diagonal Fisher approximation and finite-data Hessian estimation introduce accuracy limitations that scale with data availability
3. **Only LoRA and full fine-tuning evaluated**: other PEFT methods (e.g., Prefix Tuning, Adapter) remain untested
4. **Narrow definition of honesty**: only the dimension of "refusing to answer unknown questions" is considered; broader honesty aspects such as factual error correction and uncertainty calibration are not addressed
5. **Safety concerns**: whether honesty restoration may simultaneously revive certain behaviors intentionally suppressed by SFT warrants further analysis

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The insight that "dishonesty is a failure of expression" is highly innovative; the HCNR framework is elegantly designed
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 5 LLM families, 4 fine-tuning datasets, 5 honesty benchmarks, and detailed ablations
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative is fluent with a clear logical flow from observations to method to validation
- Value: ⭐⭐⭐⭐⭐ — Directly practical for safe LLM deployment; the method is efficient and readily applicable

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[AAAI 2026\] G2L: From Giga-Scale to Cancer-Specific Large-Scale Pathology Foundation Models via Efficient Fine-Tuning](g2lfrom_giga-scale_to_cancer-specific_large-scale_pathology_foundation_models_vi.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_imaging/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[AAAI 2026\] Unsupervised Multi-Parameter Inverse Solving for Reducing Ring Artifacts in 3D X-Ray CBCT](unsupervised_multi-parameter_inverse_solving_for_reducing_ring_artifacts_in_3d_x.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](../../CVPR2026/medical_imaging/parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)

</div>

<!-- RELATED:END -->
