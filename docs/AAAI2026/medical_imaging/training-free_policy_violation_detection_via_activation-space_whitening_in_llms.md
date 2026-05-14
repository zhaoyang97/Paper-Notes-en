---
title: >-
  [Paper Note] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs
description: >-
  [AAAI 2026][Medical Imaging][Policy violation detection] This work reformulates LLM policy violation detection as an out-of-distribution (OOD) detection problem in activation space. A training-free whitening approach is…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Policy violation detection"
  - "activation-space whitening"
  - "out-of-distribution detection"
  - "LLM internal representations"
  - "training-free"
date: 2026-05-08
content_hash: 2563bcba0625bea2
---

# Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs

**Conference**: AAAI 2026  
**arXiv**: [2512.03994](https://arxiv.org/abs/2512.03994)  
**Code**: [GitHub](https://github.com/orgs/FujitsuResearch/repositories)  
**Area**: Medical Imaging  
**Keywords**: Policy violation detection, activation-space whitening, out-of-distribution detection, LLM internal representations, training-free

## TL;DR

This work reformulates LLM policy violation detection as an out-of-distribution (OOD) detection problem in activation space. A training-free whitening approach is proposed: a whitening transform is fitted on compliant activations, and the Euclidean norm serves as the compliance score. Deployment requires only policy text and a small number of examples. The method achieves 86.0% F1 on DynaBench, outperforming fine-tuned baselines by 9.1 points and LLM-as-Judge by 16 points.

## Background & Motivation

- **Compliance challenges in enterprise LLM deployment**: Organizations deploying LLMs in sensitive domains such as legal, financial, and healthcare must ensure adherence to both internal organizational policies and external regulatory requirements.
    - Enterprise policies typically contain dozens of policies, each potentially comprising hundreds of rules.
    - Even high-performing LLMs may inadvertently violate organizational policies, creating legal and financial risk.
- **Limitations of prior work**:
    - **Guardrail systems** (e.g., LlamaGuard): Constrained to safety taxonomies and unable to generalize to complex organizational policies.
    - **LLM-as-a-Judge**: Flexible but introduces significant latency (1.47 s/sample).
    - **Fine-tuned detectors** (e.g., DynaGuard): Require large amounts of labeled data and training cost, with poor adaptability.
    - All of the above methods evaluate compliance at the **generated text level**.
- **Core insight**: The internal activation states of LLMs encode information about output correctness that is **not fully reflected in the generated tokens**.
- **Hypothesis**: Policy-violating states occupy distinct regions of the LLM embedding space and can be identified via OOD detection methods.

## Method

### Mechanism

Policy compliance detection is modeled as an OOD problem in activation space:
- **Compliant behavior** → in-distribution
- **Violating behavior** → out-of-distribution

### Offline Phase: Reference Statistics Preprocessing

#### Whitening Transform

Activation vectors $\{x_i^{(\ell)}\}_{i=1}^N$ are extracted from in-policy interactions at each layer. The empirical mean and covariance are computed:

$$\mu^{(\ell)} = \frac{1}{N} \sum_{i=1}^N x_i^{(\ell)}, \quad \Sigma^{(\ell)} = \frac{1}{N-1} \sum_{i=1}^N (x_i^{(\ell)} - \mu^{(\ell)})(x_i^{(\ell)} - \mu^{(\ell)})^\top$$

The whitening matrix $W^{(\ell)}$ satisfies ${W^{(\ell)}}^\top W^{(\ell)} = (\Sigma^{(\ell)})^{-1}$ and is computed via PCA whitening.

The whitened representation is:

$$y^{(\ell)} = W^{(\ell)}(x^{(\ell)} - \mu^{(\ell)})$$

#### Compliance Score

In the whitened space, deviation from compliant behavior is quantified by the **Euclidean norm**:

$$s^{(\ell)} = \|y^{(\ell)}\|_2$$

This score is equivalent to the Mahalanobis distance in the original space, but focuses on the principal directions of compliant variation.

#### Layer Selection

Whitening parameters are computed independently per layer. A small mixed set of compliant and violating samples is used to evaluate the separability of each layer, and the optimal operating layer $\ell^\star$ is selected.

#### Threshold Calibration

On the operating layer $\ell^\star$, the decision threshold $\tau$ is calibrated by maximizing the Youden statistic ($J = TPR - FPR$).

#### Policy-Conditioned Whitening

Policies are grouped into categories sharing common behavioral patterns. Independent whitening parameters are estimated for each category, enabling category-specific detection.

### Online Phase: Real-Time Detection

Each response is verified before being returned:

$$\hat{y} = \mathbb{I}[s^{(\ell^\star)} > \tau]$$

$\hat{y}=1$ indicates a violation; $\hat{y}=0$ indicates compliance. When policy grouping is used, the nearest policy category is selected via cosine similarity.

### Contrastive Data Construction

- For each policy rule in DynaBench, natural language prompts are generated using GPT-4.1.
- Contrastive sample pairs (compliant *good* + violating *bad*) are generated for each prompt.
- A GPT-4.1 validator is used to ensure data quality.

## Experiments

### Benchmarks and Setup

| Benchmark | Description |
|-----------|-------------|
| DynaBench | Policy compliance evaluation on multi-turn user–agent dialogues, covering 12 business impact categories |
| τ-bench | Tool-call correctness evaluation for AI agents (airline domain) |

Evaluated models: Mistral-7B, Gemma-2-9B, Llama-3.1-8B, Qwen3-8B, Qwen2.5-7B

### Main Results (DynaBench)

| Method Category | Model | F1 (%) |
|----------------|-------|--------|
| LLM-as-Judge | GPT-4o-mini | 70.1 |
| LLM-as-Judge | Qwen3-8B | 60.7 |
| Fine-tuned | LlamaGuard-3 | 20.9 |
| Fine-tuned | DynaGuard-8B | 73.1 |
| **Whitening (Ours)** | Mistral-7B | 66.8 |
| **Whitening (Ours)** | Gemma-2-9B | 75.2 |
| **Whitening (Ours)** | Llama-3.1-8B | 75.6 |
| **Whitening (Ours)** | Qwen3-8B | 78.4 |
| **Whitening (Ours)** | **Qwen2.5-7B** | **86.0** |

- Achieves state-of-the-art on 4 out of 5 backbones.
- Qwen2.5-7B reaches 86.0% F1, surpassing the strongest fine-tuned baseline DynaGuard-8B by 12.9 points.
- No fine-tuning required.

### Representation-Level vs. Generation-Level Analysis

| Model | Generation Classifier F1 | Whitening F1 |
|-------|--------------------------|--------------|
| DynaGuard-1.7B | 65.2 | **77.6** |
| DynaGuard-4B | 72.0 | **78.5** |
| DynaGuard-8B | 73.1 | **80.6** |

Key finding: for the same fine-tuned model, the whitening method outperforms its native generation classifier by 5–12 points, demonstrating that **internal representations encode richer policy-relevant information than output tokens**.

### Generalization on τ-bench

- **Synthetic data**: The whitening method substantially outperforms DynaGuard-8B and GPT-4o-mini.
- **Real trajectories**: The whitening method achieves AUC=0.87, confirming that activation-space separation generalizes across different interaction formats.

### Comparison with Other OOD Methods

| Method | Qwen2.5-7B F1 | Llama-3.1-8B F1 |
|--------|--------------|----------------|
| Mahalanobis | 67.2 | 65.8 |
| KNN | 78.5 | 66.2 |
| Energy Score | 66.4 | 72.1 |
| **Whitening (Ours)** | **82.2** | **74.3** |

The whitening method outperforms the strongest OOD baseline by 3.7%/2.2%. The Mahalanobis distance is constrained by the full covariance matrix in high-dimensional settings.

### Runtime Efficiency

| Category | Model | Time (s/sample) |
|----------|-------|----------------|
| LLM-as-Judge | GPT-4o-mini | 1.47 |
| Fine-tuned detector | DynaGuard-8B | 2.71 |
| **Same-model representation** | **Qwen2.5-7B** | **0.03** |
| Surrogate-model representation | Llama-3.1-8B | 0.98 |

Using internal representations adds only 0.03–0.05 s overhead, making the method suitable for real-time monitoring.

### Ablation Study

- **Top-K components**: F1 varies only 72.4%–76.7% across $K=10$–$50$, demonstrating robustness.
- **Samples per category**: 100 samples achieve 75.6%; increasing to 750 samples yields only 79.1% (diminishing returns).
- **Layer-wise analysis**: The optimal layer differs across policy categories (e.g., information leakage favors earlier layers; transaction-related policies favor mid-to-late layers).
- **Category-specific whitening** vs. unified whitening: Category-specific whitening consistently performs better.

## Highlights & Insights

1. **OOD framework for policy violation detection**: Shifting analysis from generated text to activation space represents a paradigm-level innovation.
2. **Core finding**: Policy compliance information is already encoded in the model's internal representations; the decoding process is a lossy bottleneck — whitening merely exposes pre-existing structure.
3. **0.03 s/sample** inference overhead enables real-time deployment, approximately 50× faster than LLM-as-Judge.
4. **Training-free with minimal calibration data** (approximately one sample per rule): Supports rapid policy updates.
5. Model safety benchmark scores (SORRY-Bench, HarmBench) are positively correlated with whitening separability, establishing a link between safety alignment and internal representation quality.

## Limitations & Future Work

- Performance depends on the quality of the model's internal representations — models with weaker safety awareness (e.g., Mistral-7B) exhibit lower separability.
- Threshold calibration is required and may need recalibration under distribution shift.
- **Detection only, not prevention**: The method does not directly intervene in the generation process.
- Access to model internal activations is required (or a surrogate model must be used), incurring additional inference cost for API-only models.
- Contrastive data is generated by GPT-4.1, which may introduce generative bias.
- The gap between the policy complexity of the DynaBench benchmark and real-world policies is not thoroughly discussed.

## Related Work & Insights

- **Guardrail systems**: LlamaGuard (Inan et al. 2023), NeMo Guardrails
- **Policy compliance**: DynaBench/DynaGuard (Hoover et al. 2025)
- **OOD detection**: Mahalanobis distance, Energy Score, KNN, whitening transform (Betser et al. 2025)
- **LLM internal representation analysis**: Zou et al. 2023 (truthfulness probes), Gekhman et al. 2025 (error detection)
- **Activation-space control**: SCANS (activation steering to mitigate over-safety)

## Rating ⭐⭐⭐⭐⭐

The approach is remarkably elegant — reducing complex policy violation detection to a norm computation in whitened space. Its training-free nature, extremely fast inference, and minimal calibration data requirements make it highly practical. The experiments are comprehensive and convincing (5 models, 2 benchmarks, multiple OOD comparisons, runtime analysis). The core finding — that internal representations outperform generated outputs — carries broad implications. This work sets a strong benchmark for LLM governance and enterprise safety deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] qa-FLoRA: Data-free Query-Adaptive Fusion of LoRAs for LLMs](qa-flora_data-free_query-adaptive_fusion_of_loras_for_llms.md)
- [\[AAAI 2026\] From Policy to Logic for Efficient and Interpretable Coverage Assessment](from_policy_to_logic_for_efficient_and_interpretable_coverage_assessment.md)
- [\[CVPR 2026\] InvAD: Inversion-based Reconstruction-Free Anomaly Detection with Diffusion Models](../../CVPR2026/medical_imaging/invad_inversion-based_reconstruction-free_anomaly_detection_with_diffusion_model.md)
- [\[ACL 2026\] DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training](../../ACL2026/medical_imaging/dart_mitigating_harm_drift_in_difference-aware_llms_via_distill-audit-repair_tra.md)
- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)

</div>

<!-- RELATED:END -->
