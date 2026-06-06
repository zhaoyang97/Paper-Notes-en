---
title: >-
  [Paper Note] TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning
description: >-
  [ICML 2026][LLM Safety][Backdoor defense] Addressing the issue of poisoned fine-tuning of MLLMs in Fine-Tuning-as-a-Service (FTaaS) scenarios…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Backdoor defense"
  - "MLLM fine-tuning"
  - "attention allocation"
  - "Gaussian Mixture Model"
  - "EM voting"
date: 2026-05-08
content_hash: ee3afa2ccc8625ef
---

# TCAP: Tri-Component Attention Profiling for Unsupervised Backdoor Detection in MLLM Fine-Tuning

**Conference**: ICML 2026  
**arXiv**: [2601.21692](https://arxiv.org/abs/2601.21692)  
**Code**: https://github.com/m1ng2u/TCAP (Available)  
**Area**: AI Safety / Multimodal Large Models / Backdoor Detection  
**Keywords**: Backdoor defense, MLLM fine-tuning, attention allocation, Gaussian Mixture Model, EM voting

## TL;DR
Addressing the issue of poisoned fine-tuning of MLLMs in Fine-Tuning-as-a-Service (FTaaS) scenarios, this paper identifies a universal fingerprint: "triggered samples abnormally polarize the attention of the first generated token among the three major components: system, vision, and text." Based on this, the unsupervised TCAP framework is proposed. It uses GMM on system attention to identify trigger-responsive attention heads, then aggregates them using EM-based Dawid–Skene voting. Across 5 trigger modes, 3 MLLMs, and 5 datasets, it reduces ASR from over 90% to approximately 0% with almost no loss in Clean Performance.

## Background & Motivation
**Background**: MLLMs (such as InternVL, LLaVA, and Qwen-VL) utilize FTaaS with LoRA/QLoRA for downstream adaptation, where users submit data and the service provider handles training. This "data as entry point" model gives poison-only backdoor attacks an extremely broad attack surface; polluting only 10% of samples can implant a backdoor that responds with attacker-specified text upon seeing a trigger.

**Limitations of Prior Work**: Existing defenses either require clean reference sets, supervised signals, or external modules (input preprocessing, trigger inversion, model pruning), or target only a single modality. The closest unsupervised solution, BYE, captures "visual attention collapse" through Shannon entropy, which is inherently effective only against **local patch triggers**. It fails against global triggers (such as Blend, SIG, WaNet, FTrojan) or text triggers; the paper's experiments show BYE's F1 is 0 on LLaVA-NeXT + Blend + ScienceQA.

**Key Challenge**: BYE assumes triggers will "concentrate" visual attention (low entropy), but the authors prove with an ideal model that the entropy upper bound for patch triggers is $\alpha_{\text{vis}}\log(|S_{\text{trig}}|/\alpha_{\text{vis}})$, whereas the upper bound for global triggers is $\alpha_{\text{vis}}\log(T/\alpha_{\text{vis}})$. Since $|S_{\text{trig}}|\ll T$, global triggers actually approach maximum entropy. Thus, entropy cannot measure global or text triggers.

**Goal**: To find an internal fingerprint **independent of trigger modality or form** as an unsupervised detection signal, covering various attacks like visual patch, blend, sinusoidal, warping, frequency, text prefix, and syntactic triggers.

**Key Insight**: Instead of examining "how attention is allocated within vision," the authors partition the MLLM input sequence into three functional blocks: system instructions (including role tags and special tokens), vision tokens, and user text. They observe the total attention mass of the **first generated token** toward these three blocks. The critical insight is that system instructions are an "anchor" that cannot be tampered with by an attacker and can serve as a noise-resistant baseline.

**Core Idea**: Triggers force a small number of attention heads in deeper layers to exhibit two types of complementary anomalies: Anomaly 1 ("system suppression + vision amplification") to extract trigger features and bypass safety constraints, and Anomaly 2 ("system amplification + vision suppression") to maintain output structural coherence. This Attention Allocation Divergence is a universal and measurable internal fingerprint of backdoors.

## Method
### Overall Architecture
TCAP is a **pure data cleaning** framework. After obtaining an MLLM fine-tuned on a poisoned dataset $\mathcal{D}$, TCAP does not modify the model or require a clean reference set. The pipeline consists of three steps: (1) Run inference on all training samples to extract tri-component attention vectors for system/vision/text; (2) Use GMM to model the system component of each (layer, head) and select the Top-K trigger-responsive heads based on a Separation Score; (3) Use EM-based Dawid–Skene voting to aggregate the binary decisions of these heads into a posterior probability $p_i$, and samples with $p_i>0.5$ are removed to obtain $\mathcal{D}_{\text{clean}}$, which is then used to retraining the MLLM to clear the backdoor.

### Key Designs

1.  **Tri-Component Attention Decomposition + Attention Allocation Divergence Fingerprint**:
    - **Function**: Decomposes the "attention of the first generated token toward all preceding tokens" into system, vision, and text components based on input sequence attribution as atomic features for backdoor detection.
    - **Mechanism**: For each (layer $l$, head $h$), let raw attention be $A^{l,h}=\{a_i^{l,h}\}_{i=1}^N$. Define the tri-component vector $\bm{\alpha}^{l,h}=(\alpha_{\text{sys}}^{l,h},\alpha_{\text{vis}}^{l,h},\alpha_{\text{txt}}^{l,h})$, where $\alpha_c^{l,h}=\sum_{i\in S_c}a_i^{l,h}$. Theoretically, entropy cannot distinguish global triggers (see Eq. 4/5 derivation), but tri-component "mass redistribution" can: triggered samples exhibit Anomaly 1 (system↓, vision↑) and Anomaly 2 (system↑, vision↓) polarization in deep layers.
    - **Design Motivation**: (a) elevates detection from "inner visual spatial distribution" to "cross-modality functional partitioning," naturally compatible with text triggers; (b) system instructions are "fixed points" that attackers cannot change, providing natural noise resistance; (c) focuses on deep layers because shallow layers handle local feature extraction while deep layers manage cross-modality decision fusion.

2.  **GMM + Separation Score Adaptive Selection of Trigger-Responsive Heads**:
    - **Function**: Unsupervisely identifies the few (layer, head) pairs that truly expose the backdoor to prevent the signal from being diluted by averaging over all heads.
    - **Mechanism**: Collect $\{\alpha_{\text{sys},i}^{l,h}\}_{i=1}^M$ for each head and apply min-max normalization to get $\tilde{\alpha}_{\text{sys},i}^{l,h}$. Use AIC to adaptively select the optimal number of components $K^*\in\{1,...,5\}$ to fit a GMM $\sum_{k=1}^{K^*}\pi_k\mathcal{N}(\mu_k,\sigma_k^2)$. Components are divided into a minority target group $\mathcal{G}_t$ (suspected backdoor mode) and a majority background group $\mathcal{G}_b$. The Separation Score is defined as the reciprocal of the overlap area of the two distribution groups: $\text{SS}^{l,h}=\bigl(\int\min(\sum_{k\in\mathcal{G}_t}\pi_k\phi_k,\sum_{k\in\mathcal{G}_b}\pi_k\phi_k)dx+\epsilon\bigr)^{-1}$. Select Top-$H_{\text{sens}}$ heads within the last $L_{\text{sens}}$ layers to form $\mathcal{H}_{\text{sens}}$.
    - **Design Motivation**: Given the low poisoning rate (10%), a bimodal assumption often fails as the secondary peak is submerged. Adaptive $K^*$ combined with the SS metric for "distribution separability" robustly identifies heads where clean and poison samples are truly separable without knowing the distribution shape.

3.  **EM-based Dawid–Skene Voting Aggregation**:
    - **Function**: Merges binary decisions from each head in $\mathcal{H}_{\text{sens}}$ into a single posterior probability to output a set of suspicious samples.
    - **Mechanism**: For each sensitive head $(l,h)$ and sample $i$, calculate the "cumulative probability of belonging to target components" using GMM posterior $\gamma_{i,k}^{l,h}$. A vote is cast if it exceeds a threshold $\tau_{\text{vote}}$: $v_i^{l,h}=\mathbf{1}[\sum_{k\in\mathcal{G}_t}\gamma_{i,k}^{l,h}>\tau_{\text{vote}}]$. Treating each sensitive head as a noisy annotator, Dawid–Skene EM is used to iteratively estimate the latent label of whether each sample is poisoned and the confusion matrix for each head, yielding the final posterior $p_i$. Samples with $p_i>0.5$ are excluded.
    - **Design Motivation**: Naive majority voting treats reliable and noisy heads equally. Dawid–Skene learns which heads are more trustworthy, effectively assigning learnable reliability weights, making it more robust to the reality of having few heterogeneous sensitive heads.

### Loss & Training
TCAP itself introduces no new losses; it serves as a pre-processor for dataset cleaning. After cleaning, standard $\mathcal{L}_c$ is used for LoRA fine-tuning on $\mathcal{D}_{\text{clean}}$. The paper uses InternVL2.5-8B, LLaVA-NeXT-8B, and Qwen3-VL-8B with LoRA, a target output "Backdoor Attack!", and a 10% poisoning rate as the evaluation protocol.

## Key Experimental Results

### Main Results
Clean Performance (CP) and Attack Success Rate (ASR) under Blend attack across 3 MLLMs × 5 datasets (selected columns for ScienceQA / DocVQA / SEED-Bench):

| Model | Method | ScienceQA CP/ASR | DocVQA CP/ASR | SEED-Bench CP/ASR |
|------|------|------------------|---------------|-------------------|
| InternVL2.5 | Vanilla FT | 96.88 / 93.60 | 57.17 / 91.16 | 77.83 / 94.10 |
| InternVL2.5 | BYE | 89.49 / 91.42 | 13.84 / 100.00 | 74.27 / 63.57 |
| InternVL2.5 | **TCAP** | **96.93 / 0.15** | **60.10 / 2.84** | **78.17 / 0.03** |
| LLaVA-NeXT | Vanilla FT | 89.19 / 96.03 | 31.66 / 100.00 | 72.13 / 96.27 |
| LLaVA-NeXT | BYE | 0.00 / 100.00 | 28.88 / 100.00 | 68.50 / 95.37 |
| LLaVA-NeXT | **TCAP** | **89.44 / 0.05** | **31.36 / 2.56** | **72.17 / 6.17** |
| Qwen3-VL | Vanilla FT | 96.58 / 86.17 | 89.07 / 98.93 | 80.37 / 97.37 |
| Qwen3-VL | **TCAP** | **96.68 / 15.62** | **90.57 / 0.33** | **81.27 / 0.37** |

Detection F1 (Precision / Recall / F1) across 5 attacks on ScienceQA (selected for InternVL2.5):

| Attack | BYE F1 | TCAP F1 |
|------|--------|---------|
| BadNet (patch) | 97.87 | **100.00** |
| Blend (global) | 0.00 | **98.34** |
| SIG (sinusoidal) | 9.94 | **92.11** |
| WaNet (warp) | 4.01 | **99.20** |
| FTrojan (frequency) | 9.58 | **85.24** |

### Ablation Study
F1 under Blend attack (Selected from Tab. 3):

| Configuration | InternVL2.5 | LLaVA-NeXT | Qwen3-VL |
|------|-------------|------------|----------|
| Full TCAP | 98.34 | 98.85 | 95.04 |
| w/o Head Selection | 43.02 | 22.31 | 15.33 |
| w/o Layer Filter | 67.21 | 31.94 | 11.02 |

### Key Findings
- **Head Selection is the lifeline**: Removing it drops F1 from 95–99 to 15–43, proving that "all-head averaging" is diluted by noise heads—only a few deep heads truly expose the backdoor.
- **Layer Filter is also critical**: Removing it drops F1 to 11–67, validating the hierarchy hypothesis that "shallow layers extract local features while deep layers make cross-modality decisions."
- **Visual evidence of entropy failure**: BYE achieves F1=0 and Recall=0 on InternVL2.5 + Blend + ScienceQA / PhD because global triggers increase entropy, causing BYE to misclassify clean as poison and let poison through. TCAP achieves F1=98 in the same setting.
- **Universal text trigger detection**: On text prefix "Hello!" triggers, TCAP achieves 100 F1 across three MLLMs; it also significantly reduces ASR for HiddenKiller (syntactic triggers) on the PhD dataset.
- **Qwen3-VL Blend ASR at 15.62** is the weakest result, suggesting stronger modern MLLMs may weaken the divergence signal, making sensitive heads harder to pick out.

## Highlights & Insights
- **Replacing Entropy with "Mass Redistribution"**: The authors provide a clean theoretical derivation (Eq. 3–5) explaining why entropy is inherently only effective for patch triggers and replace it with "system/vision/text tri-components." This approach of first falsifying existing metrics before changing coordinates is highly persuasive.
- **System Instruction as an Anchor**: Using immutable parts as a reference baseline is a highly transferable trick for any LLM/MLLM defense, such as jailbreak or prompt injection detection.
- **Dawid–Skene + Attention Heads = Noisy Label Fusion**: Treating candidate detectors as annotators with varying reliability is an elegant engineering pattern applicable to any "voting of multiple weak detectors" pipeline.
- **Truly Unsupervised**: The entire pipeline requires no clean reference set, no labels, and no external models, with retraining performed on the cleaned data itself, making it highly feasible for FTaaS.

## Limitations & Future Work
- The number of sensitive heads $H_{\text{sens}}$ and depth threshold $L_{\text{sens}}$ are hyperparameters; it remains unclear if they adapt across architectures (though GMM's $K^*$ is adaptive).
- The 15.62% ASR remaining on Qwen3-VL + Blend indicates that stronger MLLMs may reduce the number of anomalous heads, risking signal dilution.
- Evaluation only covers a 10% poisoning rate and simplified "Backdoor Attack!" settings; survival under extremely low rates (<1%), multi-target backdoors, or clean-label backdoors is not tested.
- It only cleans the training set and does not protect against inference-time attacks.
- The approach requires a forward pass for all training samples to gather attention, which may involve non-negligible costs for million-scale datasets.

## Related Work & Insights
- **vs BYE**: Both are unsupervised and focus on attention, but BYE uses Shannon entropy for spatial collapse (patch-only). TCAP uses tri-component mass + GMM + EM for cross-component divergence (modality-agnostic).
- **vs SampDetox (Diffusion Denoising)**: SampDetox uses input preprocessing to remove triggers but degrades CP by erasing semantics. TCAP removes samples without modifying them, keeping CP intact.
- **Transferable Insights**: (a) In any system prompt-based LLM security task, monitor the attention mass of the first token toward the system components. (b) "GMM + SS + EM voting" is a universal outlier detector applicable to scenarios with minority anomalies and multiple detectors, such as OOD or noise label cleaning.

## Rating
- Novelty: ⭐⭐⭐⭐ Moving backdoor fingerprints from "visual entropy" to "cross-component attention allocation" is a solid conceptual leap over BYE.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid across 3 MLLMs, 5 datasets, 5 visual attacks, and 2 text attacks, though lacking extreme low-rate or multi-target stress tests.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, and method phases are well-articulated.
- Value: ⭐⭐⭐⭐ Directly addresses FTaaS deployment pain points with an unsupervised, zero-dependency approach suitable for enterprise MLLM providers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[CVPR 2026\] Perturb and Recover: Fine-tuning for Effective Backdoor Removal from CLIP](../../CVPR2026/llm_safety/perturb_and_recover_fine-tuning_for_effective_backdoor_removal_from_clip.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)
- [\[ICML 2026\] PFT: Phonon Fine-tuning for Machine Learned Interatomic Potentials](pft_phonon_fine-tuning_for_machine_learned_interatomic_potentials.md)

</div>

<!-- RELATED:END -->
