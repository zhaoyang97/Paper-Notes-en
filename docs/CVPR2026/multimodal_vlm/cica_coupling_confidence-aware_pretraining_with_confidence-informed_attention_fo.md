---
title: >-
  [Paper Note] CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Sentiment Analysis] CICA enables each unimodal encoder to "self-evaluate" signal reliability (outputting confidence $s_m$ and uncertainty $u_m$) during pretraining. These signals are then used to modulate the output of a Confidence-Informed Attention mechanism. This allows the model to adaptively amplify reliable modalities and suppress noisy ones when text, visual, or audio signals conflict or are missing, setting a new SOTA across MOSI…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal Sentiment Analysis"
  - "Confidence-Aware"
  - "Reliability Modeling"
  - "Adaptive Fusion"
  - "Modality Conflict"
date: 2026-05-08
content_hash: 02a1636bdf2aee1b
---

# CICA: Coupling Confidence-Aware Pretraining with Confidence-Informed Attention for Robust Multimodal Sentiment Analysis

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Jiang_CICA_Coupling_Confidence_Aware_Pretraining_with_Confidence_Informed_Attention_for_Robust_Multimodal_CVPR_2026_paper.html)  
**Code**: None (Not released)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal Sentiment Analysis, Confidence-Aware, Reliability Modeling, Adaptive Fusion, Modality Conflict  

## TL;DR
CICA enables each unimodal encoder to "self-evaluate" signal reliability (outputting confidence $s_m$ and uncertainty $u_m$) during pretraining. These signals are then used to modulate the output of a Confidence-Informed Attention mechanism. This allows the model to adaptively amplify reliable modalities and suppress noisy ones when text, visual, or audio signals conflict or are missing, setting a new SOTA across MOSI, MOSEI, CH-SIMS, and CH-SIMSv2.

## Background & Motivation

**Background**: Multimodal Sentiment Analysis (MSA) jointly models language, visual, and acoustic signals to infer sentiment. Mainstream approaches have evolved from early tensor fusion (TFN, LMF) to Transformer-based cross-modal attention (MulT), and more recent "guided/asymmetric" fusion (ALMT assuming text dominance, KuDA/CLGSI using dynamic guidance).

**Limitations of Prior Work**: Real-world data is rarely clean or perfectly time-aligned. Modality conflicts are common—for instance, a person might be smiling (strong positive visual) but speaking with a frustrated tone (negative audio) and neutral text, where the ground truth follows the audio (-1.33). Most models are misled by the visual "smile" because they use **unified attention**, which defaults to treating all modalities as equally reliable, lacking an internal mechanism to determine which modality should dominate for a specific sample.

**Key Challenge**: Fusion quality depends on "how reliable each modality is at this moment," yet existing methods treat reliability as a global constant, lacking **sample-level and modality-level** reliability metrics. Existing uncertainty modeling (Evidential Deep Learning, Bayesian approximation) is mostly applied post-hoc at the decision layer. Three-way decision methods (like 3WD-DRT) only predict a single confidence score $s_m$ for discrete partition scaling, which results in overly coarse signals.

**Goal**: To construct a "perceive-and-decide" framework where the model must **first quantify the reliability of each modality for the current sample, and then decide how to combine them**.

**Core Idea**: Directly couple "self-evaluated confidence" with "fusion decision-making." During pretraining, encoders estimate their own reliability (confidence $s_m$ + uncertainty $u_m$). In the fusion stage, these signals act as continuous modulation factors on the attention output, where contributions are amplified only when they are both confident and consistent.

## Method

### Overall Architecture
CICA decouples "perceiving signal quality" and "deciding fusion strategy" into two collaborative phases. **Phase 1 (CAP, Confidence-Aware Pretraining)** independently trains each unimodal encoder $E_m$ ($m\in\{T,V,A\}$) to output not just representations $H_m$, but also a confidence score $s_m$ and uncertainty $u_m$ for each sample—this is "self-perception." **Phase 2 (CIF, Confidence-Informed Fusion)** freezes these encoders and trains a fusion module. It first uses Confidence-Informed Attention (CIA) to perceive the intrinsic structural quality of features, then applies the reliability signals $(s_m, u_m)$ from CAP as external modulation factors. Finally, a Mutual Information Contrastive Preservation (MCP) loss prevents any single modality from dominating the fusion ("modality collapse").

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    X["Text / Visual / Audio<br/>Triple Raw Inputs"] --> CAP
    subgraph CAP["Phase 1 · Confidence-Aware Pretraining CAP"]
        direction TB
        E["Unimodal Encoder + MixDomainAdapter<br/>Outputs H_m"] --> H["Three Auxiliary Heads<br/>Task / Confidence s_m / Uncertainty u_m"]
    end
    CAP -->|Freeze Encoders, pass H_m and (s_m,u_m)| CIA["Confidence-Informed Attention<br/>Q·K + S_mod perceives internal quality"]
    CIA --> MOD["Reliability Modulation Coupling<br/>r_m=ReLU(1+s_m-u_m), z×r_m"]
    MOD --> MCP["Mutual Information Contrastive Preservation (MCP)<br/>Prevents modality collapse"]
    MCP --> Y["Sentiment Prediction ŷ"]
```

### Key Designs

**1. CAP Confidence-Aware Pretraining: Teaching encoders to "self-evaluate" rather than just learn features**

To address the lack of internal mechanisms for judging modality reliability, CAP trains each unimodal encoder $E_m$ as a "perceiver": $(H_m, s_m, u_m)=E_m(X_m)$. The encoder uses a Transformer backbone where a MixDomainAdapter extracts task-specific features $h_{\text{spci}}$ from the last layer and domain-related features $h_{\text{domain}}$ from intermediate layers, concatenating them into $H_m=\text{Concat}(h_{\text{spci}},h_{\text{domain}})$. Three auxiliary heads are attached and pretrained with a joint objective:

$$\mathcal{L}_{\text{CAP}}=\mathcal{L}_{\text{task\_pre}}+\lambda_{\text{CA}}\mathcal{L}_{\text{CA}}+\lambda_{\text{uncert}}\mathcal{L}_{\text{uncert}}$$

The task head $C_{\text{pred}}$ predicts unimodal sentiment using MSE; the confidence head $C_{\text{CA}}$ predicts a scalar $s_m\in[0,1]$; and the uncertainty head $C_{\text{uncert}}$ predicts a bounded absolute error $u_m\in[0,1]$. A key step here is the **explicit calibration of the encoder's own uncertainty**: unlike 3WD-DRT which provides only a single confidence score, CICA models both "adaptive confidence" and "task uncertainty," allowing it to disentangle epistemic and aleatoric uncertainty.

**2. Adaptive CA Loss: Calibrating confidence into "true reliability" via learnable piecewise penalties**

Directly training confidence scores is problematic: classification losses (like BCE) treat it as binary, while static regression losses like $L=(s_m-1)^2$ are too rigid (blindly forcing all samples to be confident). The authors propose a **learnable calibration function** $\mathcal{L}_{\text{CA}}$, using parameters $\theta_{\text{CA}}=\{\hat\alpha,\hat\beta,\hat w_{\text{high}},\hat w_{\text{mid}},\hat w_{\text{low}}\}$ to adaptively define two reliability boundaries ($0<\beta<\alpha<1$, constrained by sigmoids $\beta=\sigma(\hat\beta),\ \alpha=\beta+(1-\beta)\cdot\sigma(\hat\alpha)$) and three levels of penalty weights (Softplus for positivity). The piecewise weight function is:

$$W(s_m,\theta_{\text{CA}})=\begin{cases}w_{\text{low}} & s_m\ge\alpha\\ w_{\text{mid}} & \beta\le s_m<\alpha\\ w_{\text{high}} & s_m<\beta\end{cases}$$

The loss is a weighted "encourage high confidence" term: $\mathcal{L}_{\text{CA}}=\frac{1}{B}\sum_i W(s_{m,i},\theta_{\text{CA}})\cdot(s_{m,i}-1)^2$. The uncertainty head target is given by the detached error $u_{\text{target}}=\tanh(|y_m-\hat y_m^{\text{detach}}|)$, fitted using MSE. ⚠️ The full parameterization and justification for $\theta_{\text{CA}}$ are in Appendix H.1 of the original paper. This ensures that confidence is not an arbitrary scalar but a reliability metric constrained by a **learnable calibration curve**.

**3. CIA Confidence-Informed Attention + Reliability Modulation: Decoupling and then coupling "internal quality perception" and "external reliability"**

A core observation is that standard attention $Q\cdot K^\top$ only models the **relevance** between query and key, ignoring the **intrinsic quality** of the key $K_m$ itself. CIA adds an internal structural modulation term $S_{\text{mod}}$ within the softmax, which captures local dependencies $N_m$ and token-level saliency $\rho_m$ to produce a structure-aware representation:

$$z_{\text{struct},m}=\text{softmax}\!\left(\frac{QK_m^\top}{\sqrt{d_k}}+S_{\text{mod}}\right)V_m$$

This is the "perception of internal quality." Then, the two reliability signals $(s_m, u_m)$ from CAP are synthesized via a projection $g(\cdot)$ into a unified reliability score $r_m=g(s_m,u_m)=\text{ReLU}(1+s_m-u_m)$—where $r_m$ is large only when confidence is high ($s_m$) and uncertainty is low ($u_m$). Finally, $r_m$ is broadcast across sequence and feature dimensions to continuously modulate the structural representation: $z_{\text{CIF},m}=z_{\text{struct},m}\times r_m$. Modalities judged unreliable ($r_m\approx0$) have their contributions naturally suppressed. This step is the "coupling point" of the framework: explicitly linking pretraining-phase "perception" and fusion-phase "deciding" via a multiplicative gate, which is stricter than the confidence scaling in 3WD-DRT (requiring both "confidence and consistency" for amplification).

**4. MCP Mutual Information Contrastive Preservation Loss: Preventing fusion dominance by a single modality**

Strong reliability gating risks "modality collapse," where one modality (typically text) dominates the fusion, discarding unique information from others. Inspired by Contrastive Predictive Coding (CPC)/NCE, MCP constrains the final fused representation $z_{\text{final}}$ to remain correspondent with each unimodal pooled representation $h_m=\text{MeanPool}(H_m)$: $\mathcal{L}_{\text{MCP}}=\sum_{m\in\{T,V,A\}}\mathcal{L}_{\text{NCE},m}(z_{\text{final}},h_m)$. This ensures that even when suppressing a "deceptive" modality, the fused representation retains sufficient information from all sources. The total objective during fine-tuning is $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{task}}+\lambda_{\text{mcp}}\mathcal{L}_{\text{MCP}}$, with L1 loss $\mathcal{L}_{\text{task}}=\frac{1}{B}\sum_i|y_i-\hat y_i|$ and $\lambda_{\text{mcp}}=0.1$.

### Loss & Training
Two-stage training: first pretrain unimodal encoders (including task/confidence/uncertainty heads) with $\mathcal{L}_{\text{CAP}}$, then freeze them and fine-tune the CIF fusion module using $\mathcal{L}_{\text{Total}}=\mathcal{L}_{\text{task}}+0.1\cdot\mathcal{L}_{\text{MCP}}$. The value $\lambda_{\text{mcp}}=0.1$ is given in the main text; other configurations like optimizers, learning rates, and epochs are in Appendices C and D.

## Key Experimental Results

### Main Results
SOTA performance is achieved across four benchmarks (English: MOSI/MOSEI, Chinese: CH-SIMS/CH-SIMSv2). The $\Delta$ row indicates gains relative to the strongest baseline:

| Dataset | MAE↓ | Corr↑ | Acc-7/3↑ | Acc-2↑ | vs. Strongest Baseline |
|--------|------|-------|----------|--------|--------------|
| MOSI | **0.630** | **0.855** | 49.56 | 88.19/90.24 | MAE ↓0.071, Corr ↑0.060 |
| MOSEI | **0.489** | **0.856** | 55.29 | 84.72/90.18 | MAE ↓0.029, Corr ↑0.077 |
| CH-SIMS | **0.378** | **0.754** | 76.37 | 86.00 | Corr ↑0.127, Acc-3 ↑9.73 |
| CH-SIMSv2 | **0.245** | **0.842** | 80.56 | 85.98 | Acc-3 ↑4.35, Corr ↑0.078 |

Gains are particularly significant on the more complex CH-SIMS (Corr +0.127, Acc-3 +9.73).

### Ablation Study
Components were removed selectively on MOSI/MOSEI (values in parentheses show the drop relative to "Full"):

| Configuration | MOSI Corr↑ | MOSI F1(non-0)↑ | MOSEI F1(non-0)↑ | Description |
|------|-----------|-----------------|------------------|------|
| Full | 0.855 | 90.23 | 90.16 | Full Model |
| (A) w/o CAP | 0.791 (↓0.064) | 85.05 (↓5.18) | 83.19 (↓6.97) | Remove Confidence-Aware Pretraining |
| (B) w/o CIF | 0.812 (↓0.043) | 86.13 (↓4.10) | 83.90 (↓6.26) | Remove Confidence Attention Fusion |
| (D) w/o Coupling | 0.831 (↓0.024) | 87.02 (↓3.21) | 84.13 (↓6.03) | CAP/CIF used independently |
| (C) w/o MCP | 0.847 (↓0.008) | 87.66 (↓2.57) | 84.71 (↓5.45) | Remove anti-collapse loss |
| (E) w/o S_mod | 0.840 (↓0.015) | 89.15 (↓1.08) | 88.05 (↓2.11) | Remove internal structure modulation |

### Key Findings
- **CAP and CIF are the two main pillars**: Removing either leads to the largest drops (MOSEI F1 non-0 dropped by 6.97/6.26), confirming that "perception" and "decision-making attention" are both indispensable.
- **Coupling has intrinsic value**: Even with both CAP and CIF present, lack of explicit coupling (D) results in a MOSI Corr drop of 0.024—indicating that modules being independent is insufficient; the key is linking perception signals into the fusion decision.
- **MCP and $S_{\text{mod}}$ act as stabilizers**: Their drops are smaller but consistent, primarily contributing to cross-modal balance and attention stability.
- **Robustness is the primary selling point**: Performance decays smoothly under Gaussian noise (variance 0.2–0.8). Under mild audio noise on MOSEI, Corr only drops from 0.856 to 0.843, as CAP detects contamination and CIF downweights it. For missing modalities (T+A on MOSEI), performance barely drops (MAE 0.483 vs 0.489) because CAP marks the missing modality as $s_V=0$ and CIF automatically adjusts.
- ⚠️ A notable phenomenon: On MOSI, text-only (T-only) Corr (0.857) is slightly higher than T+V+A (0.855). Since MOSI is highly text-dominant, 0.857 represents the text upper bound; the 0.002 gap stems from light MCP fusion regularization, a cost the authors deem worthwhile for cross-benchmark robustness.

## Highlights & Insights
- **Explicit "Perceive-and-Decide" Coupling**: Unlike most uncertainty methods that apply reliability post-hoc at the decision level, CICA shifts it to pretraining and closes the loop with a multiplicative gate at fusion. This "quantify then use" pipeline is clean and interpretable.
- **Learnable Piecewise Loss for Confidence Calibration**: Instead of a simple scalar, $\theta_{\text{CA}}$ learns reliability boundaries, avoiding the extremes of BCE or static regression. This "learnable calibration" trick is transferable to other tasks requiring self-evaluation.
- **Dual-Signal Gate $r_m=\text{ReLU}(1+s_m-u_m)$**: Requiring both "confidence" and "low uncertainty" is stricter than single-confidence scaling and naturally handles missing modalities (zeroing out when $s\approx0$).
- **MCP for Modality Balance**: Applying a strong reliability gate is balanced by a contrastive loss to ensure information retention, providing a clever patch between "suppressing noise" and "retaining information."

## Limitations & Future Work
- Code is not public; implementations for $S_{\text{mod}}$ (local dependency $N_m$, saliency $\rho_m$) and $\theta_{\text{CA}}$ are relegated to the appendix, making reproduction difficult.
- The two-stage process (freezing encoders then training fusion) is heavy. The addition of confidence/uncertainty heads and triple-loss pretraining incurs extra overhead not detailed in the training cost comparison.
- On MOSI, multimodal performance slightly lags behind pure text, suggesting limited multimodal gains on text-skewed data—the framework excels at "not being misled by noise" rather than necessarily improving points when text is sufficient.
- The quality of self-evaluated reliability depends on label quality; it remains to be seen if $s_m/u_m$ remains accurate in weak-label or domain-shift scenarios.

## Related Work & Insights
- **vs. 3WD-DRT (Three-Way Decision)**: It predicts only a single confidence $s_m$ for discrete partition scaling; CICA models both $s_m$ and $u_m$ with a continuous $r_m$ gate ("amplify only if confident and consistent"), providing better stability and interpretability.
- **vs. ALMT / KuDA (Guided Fusion)**: These rely on "text dominance" assumptions or external knowledge for dynamic guidance; CICA's reliability is self-perceived by encoders without prior assumptions.
- **vs. MMIM / MISA (Representation Enhancement)**: They improve representations or maximize mutual information; CICA uses mutual information (MCP) only as a secondary objective, with the main line being confidence-driven adaptive fusion.

## Rating
- Novelty: ⭐⭐⭐⭐ Explicit "perceive-and-decide" coupling + learnable calibration; highly relevant to conflict/missing modality issues.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ SOTA across four benchmarks + detailed ablation + dual stress tests (noise/missing modalities).
- Writing Quality: ⭐⭐⭐⭐ Clear main line and formulas, though much of $S_{\text{mod}}$ and CA loss parameterization is in the appendix.
- Value: ⭐⭐⭐⭐ Reusable paradigm for robust multimodal fusion; calibration and dual-signal gates are transferable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis](enhance-then-balance_modality_collaboration_for_robust_multimodal_sentiment_anal.md)
- [\[CVPR 2026\] Conflict-Aware Adaptive Cross-Reconstruction for Multimodal Sentiment Analysis](conflict-aware_adaptive_cross-reconstruction_for_multimodal_sentiment_analysis.md)
- [\[CVPR 2026\] Factorize, Reconstruct, Enhance: A Unified Framework for Multimodal Sentiment Analysis](factorize_reconstruct_enhance_a_unified_framework_for_multimodal_sentiment_analy.md)
- [\[CVPR 2026\] Prototype-as-Prompt: Multimodal Sentiment Prototypes Endowing Large Language Models the Capability to Perform Multimodal Sentiment Analysis](prototype-as-prompt_multimodal_sentiment_prototypes_endowing_large_language_mode.md)
- [\[CVPR 2026\] Linking Perception, Confidence and Accuracy in MLLMs](linking_perception_confidence_and_accuracy_in_mllms.md)

</div>

<!-- RELATED:END -->
