---
title: >-
  [Paper Note] Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection
description: >-
  [CVPR 2026][Multimodal VLM][VLM safety] The paper proposes CARE, a framework that first applies causal mediation analysis to precisely localize neurons and layers causally associated with unsafe behavior in VLMs (diagnos…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "VLM safety"
  - "causal mediation analysis"
  - "safety subspace projection"
  - "adversarial attack defense"
  - "dual-modal repair"
date: 2026-05-08
content_hash: a087c0c0497a7569
---

# Diagnosing and Repairing Unsafe Channels in Vision-Language Models via Causal Discovery and Dual-Modal Safety Subspace Projection

**Conference**: CVPR 2026
**arXiv**: [2603.27240](https://arxiv.org/abs/2603.27240)  
**Code**: N/A  
**Area**: Multimodal / VLM
**Keywords**: VLM safety, causal mediation analysis, safety subspace projection, adversarial attack defense, dual-modal repair

## TL;DR
The paper proposes CARE, a framework that first applies causal mediation analysis to precisely localize neurons and layers causally associated with unsafe behavior in VLMs (diagnosis), then constructs a dual-modal safety subspace via generalized eigendecomposition and projects activations onto it at inference time (repair), reducing attack success rates to below 10% with negligible loss of general capability.

## Background & Motivation
**Background**: Large vision-language models (LVLMs) demonstrate strong multimodal understanding, but remain vulnerable to jailbreak attacks—carefully crafted multimodal prompts that bypass safety alignment mechanisms.

**Limitations of Prior Work**: (1) Input preprocessing and adversarial training are computationally expensive and may degrade general performance; (2) existing activation-level defenses (ASTRA, SPO-VLM) lack precise localization of unsafe components, rely on a single modality, and employ heuristic linear steering that distorts general representations.

**Key Challenge**: How to precisely localize components within a VLM that are causally responsible for unsafe behavior, and repair them without compromising general capability?

**Goal**: To establish a causally driven, nonlinear, dual-modal framework for VLM safety diagnosis and repair.

**Key Insight**: First perform causal localization (which layers/neurons lead to unsafe outputs), then apply subspace projection (project activations onto safe directions).

**Core Idea**: Diagnosis—causal mediation analysis localizes middle FFN layers → Repair—generalized eigendecomposition identifies the malicious subspace → Project activations onto its orthogonal complement.

## Method

### Overall Architecture
Three steps: (1) layer-level causal tracing (ablation) → (2) dual-modal token attribution (RBF kernel) → (3) safety subspace projection (generalized eigendecomposition + orthogonal projection). The entire process requires no retraining and intervenes only at inference time.

### Key Designs
1. **Layer-Level Causal Tracing and Component Analysis**: Different layers are systematically blocked to observe changes in attack success rate (ASR), enabling coarse-grained localization of safety-critical layers. FFN and MHSA are then ablated separately, revealing that **FFN exerts substantially greater influence on safety than MHSA**. Design Motivation: FFN activations exhibit low inter-sample correlation (each sample projects independently), making safety signals more separable; MHSA integrates global context, causing safety signals to diffuse and become difficult to isolate. Three metrics—Silhouette coefficient, class separability, and Mahalanobis distance—are quantitatively verified to peak at layers 16–17.

2. **Dual-Modal Token Attribution**:

    - **Visual Attribution**: Cross-modal correlation between visual and text tokens is computed using an RBF kernel: $MI_i^v = \frac{s_i - s_{min}}{s_{max} - s_{min}}$, where $s_i = \|\tilde{K}_{i,:}\|_2^2$ is the L2 norm of rows of the centered cross-modal kernel matrix. The top-k visual tokens most correlated with the attack are selected.
    - **Text Attribution**: A self-modal RBF kernel matrix is used to compute semantic independence scores for text tokens, selecting the most influential ones.
   Design Motivation: Not all tokens are equally related to jailbreak behavior; focusing on high-attribution tokens enables more precise subsequent projection.

3. **Safety Subspace Projection**: Activations $A_b, A_m$ at the target layer are collected from benign and malicious samples respectively. After centering, covariance matrices $C_b, C_m$ are computed, and generalized eigendecomposition $C_m u = \lambda C_b u$ identifies the directions of maximal malicious deviation. The top-k eigenvectors $U_k$ form the malicious subspace, and the safety projection operator is constructed as:
   $P_{\text{safe}} = I - U_k U_k^T$
   At inference time, activations are projected as: $h' = P_{\text{safe}} h + \beta (1 - P_{\text{safe}}) h_{\text{benign}}$

   Separate safety subspaces are constructed for the visual and text modalities, and the two projected results are combined via adaptive fusion weights $w_{vis} = \frac{\|h'_{vis} - h_{txt}\|}{\|h'_{vis} - h\| + \|h'_{txt} - h\|}$. Design Motivation: Generalized eigendecomposition identifies the directions along which malicious activations deviate most from benign ones; projecting onto the orthogonal complement precisely suppresses unsafe components. Separate projection followed by fusion is used because the attack mechanisms differ between visual and text modalities.

### Loss & Training
**No training required**—the method intervenes entirely at inference time. Only a small number of benign/malicious samples need to be collected offline to extract activations for constructing the projection matrices.

## Key Experimental Results

### Main Results (Attack Success Rate ASR % ↓)

| Method | JailBreakV | MMSafety | PGD-Toxic κ=64 | PGD-Jailbreak κ=64 |
|--------|-----------|---------|---------------|-------------------|
| LLaVA (baseline) | 45.71 | 36.48 | 60.38 | 65.15 |
| SPO-VLM | 10.37 | 16.26 | 17.90 | 17.38 |
| ASTRA | 11.98 | 15.37 | 16.37 | 14.85 |
| **CARE (Ours)** | **7.03** | **9.13** | **12.78** | **8.46** |

**Similar trends on Qwen2.5-VL**: JailBreakV 6.55%, MMSafety 8.72%

### Ablation Study

| Configuration | JailBreakV ASR↓ | PGD-Toxic-64 ASR↓ | Note |
|--------------|----------------|-------------------|------|
| CARE (full) | 7.03 / 6.55 | 12.78 / 4.60 | Full dual-modal version |
| CARE w/o text | 15.26 / 14.3 | — | Text subspace is critical for language jailbreaks |
| CARE w/o visual | — | 45.71 / 46.13 | Visual subspace is critical for image attacks |

### Key Findings
- **FFN > MHSA**: Blocking FFN has substantially larger impact on ASR than blocking MHSA, confirming that FFN is the primary carrier of the safety mechanism.
- **Middle layers are most critical**: Safety-relevant representations reach peak cluster separability at layers 16–17 (LLaVA) or layers 12–14 (Qwen).
- **Both modalities are indispensable**: Removing the text subspace doubles ASR for language jailbreaks; removing the visual subspace increases ASR for PGD attacks by 10×.
- **General capability is preserved**: Performance drops of only 2–8% on MMBench, MM-Vet, and SQA.
- **Transferable defense**: Effective against unseen PGD attacks as well.

## Highlights & Insights
- **Causally driven precise localization**: Rather than blindly intervening across all layers, the method first identifies safety-critical layers and components (FFN), minimizing interference with irrelevant representations.
- **Theoretical elegance of generalized eigendecomposition**: Directly finding the direction of maximal deviation in the "benign vs. malicious" covariance space is more principled than heuristic linear steering.
- **Training-free**: Only a small number of offline activations are required; inference-time overhead consists solely of matrix multiplication.
- **FFN as a "discriminative projector"**: The low inter-sample correlation of FFN activations implies that safety signals are more "purely separable" within them—a finding of theoretical value for understanding the internal safety mechanisms of VLMs.

## Limitations & Future Work
- Construction of the safety subspace depends on offline collected malicious samples, which may provide insufficient coverage for entirely novel attack types.
- Although lightweight, the projection operation introduces additional computational overhead at each inference step.
- The benign regularization term $\beta$ requires tuning and may need different hyperparameter settings for different models.
- Validation is performed only on models at the 7–8B scale; whether the same safety mechanisms exist in larger models remains to be verified.
- The safety subspace dimensionality $k$ requires empirical tuning.
- Defense effectiveness against pure-text jailbreaks (without image input) is not evaluated separately.
- The phenomenon of safety mechanisms being diluted by "feature entanglement" in deeper layers warrants further investigation.

## Related Work & Insights
- **Distinction from ASTRA and SPO-VLM**: CARE employs causal analysis for precise localization rather than coarse-grained intervention, and uses a nonlinear RBF kernel with generalized eigendecomposition rather than linear steering.
- **Comparison with Refusal Pairs (fine-tuning methods)**: CARE requires no retraining and achieves superior performance.
- Causal mediation analysis has been applied in NLP interpretability; this work is the first to apply it for safety localization in VLMs.
- Generalized eigendecomposition is also used in classical discriminant analysis (e.g., LDA); its innovative application here serves to separate safe from malicious directions.
- The connection between the FFN "discriminative projector" role and the Neural Collapse phenomenon warrants deeper exploration.

## Technical Details
- **RBF kernel bandwidth**: $\sigma = \sqrt{0.5 \cdot \text{median}(D_{ij})}$, adaptive to the data distribution.
- **Kernel centering**: Visual one-sided $\tilde{K} = K_{cross}H_t$; text double-sided $\tilde{K} = HKH$.
- **Safety projection**: $h' = P_{safe}h + \beta(1-P_{safe})h_{benign}$
- **Fusion weight**: $w_{vis} = \frac{\|h'_{vis}-h_{txt}\|}{\|h'_{vis}-h\|+\|h'_{txt}-h\|}$
- **Localization validation**: Silhouette / Class Sep. / Mahalanobis metrics peak at layers 16–17.
- **Attack data**: JailbreakVBench + AdvBench + FigStep.
- **General performance**: MMBench −2–3%, MM-Vet −5–8%, SQA −2–4%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of causal diagnosis and dual-modal safety subspace projection is pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 2 VLMs × multiple benchmarks × PGD attacks × comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Framework is clearly presented and causal analysis is in-depth, though mathematical notation is dense.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for VLM safety defense; deployable without retraining.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Evolving Contextual Safety in Multi-Modal Large Language Models via Inference-Time Self-Reflective Memory](evolving_contextual_safety_in_multi-modal_large_language_models_via_inference-ti.md)
- [\[CVPR 2026\] HandVQA: Diagnosing and Improving Fine-Grained Spatial Reasoning about Hands in Vision-Language Models](handvqa_diagnosing_and_improving_fine-grained_spatial_reasoning_about_hands_in_v.md)
- [\[CVPR 2026\] DSCA: Dynamic Subspace Concept Alignment for Lifelong VLM Editing](dsca_dynamic_subspace_concept_alignment_for_lifelong_vlm_editing.md)
- [\[CVPR 2026\] Multi-Modal Representation Learning via Semi-Supervised Rate Reduction for Generalized Category Discovery](multi-modal_representation_learning_via_semi-supervised_rate_reduction_for_gener.md)
- [\[ACL 2026\] From Heads to Neurons: Causal Attribution and Steering in Multi-Task Vision-Language Models](../../ACL2026/multimodal_vlm/from_heads_to_neurons_causal_attribution_and_steering_in_multi-task_vision-langu.md)

</div>

<!-- RELATED:END -->
