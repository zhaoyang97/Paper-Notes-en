---
title: >-
  [Paper Note] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception
description: >-
  [ICML 2026][AI Safety][Collaborative Perception] UniTrans reformulates the traditional collaborative perception paradigm—from "training an adapter for every pair of vehicle modalities" to "inferring mapping in a modality…
tags:
  - "ICML 2026"
  - "AI Safety"
  - "Collaborative Perception"
  - "Heterogeneous Features"
  - "Modality Mapping"
  - "Zero-shot Translation"
  - "Parameter-level MoE"
date: 2026-05-08
content_hash: a1450f92ca029dec
---

# One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception

**Conference**: ICML 2026  
**arXiv**: [2605.17907](https://arxiv.org/abs/2605.17907)  
**Code**: https://github.com/CheeryLeeyy/UniTrans (Available)  
**Area**: Autonomous Driving / Collaborative Perception / Heterogeneous Feature Translation  
**Keywords**: Collaborative Perception, Heterogeneous Features, Modality Mapping, Zero-shot Translation, Parameter-level MoE  

## TL;DR
UniTrans reformulates the traditional collaborative perception paradigm—from "training an adapter for every pair of vehicle modalities" to "inferring mapping in a modality-intrinsic latent space → linearly combining expert parameters via a router → instantiating a mapping-specific translator on-the-fly." This achieves zero-shot BEV feature translation for unseen new vehicle models, improving average AP@0.7 by ~7 / 3 points on OPV2V-H / DAIR-V2X compared to the strongest baseline, while maintaining lower GFLOPs / CPU time than classic MoE.

## Background & Motivation

**Background**: Intermediate fusion-based collaborative perception is the mainstream paradigm for next-generation autonomous driving. Connected vehicles transmit their BEV features to neighbors to compensate for occlusions and long-range perception limitations in single-vehicle views. However, differing sensor configurations, backbones, and network depths across manufacturers mean that transmitted BEV features reside in heterogeneous "modality subspaces." Directly feeding these into an ego vehicle's fusion network leads to misinterpretation and degraded performance.

**Limitations of Prior Work**: There are two main approaches to heterogeneity. **One-to-one adaptation** (MPDA, PnPDA, PolyInter) trains dedicated adapters for every "source $\to$ target" pair, requiring new training whenever a new modality appears. **Two-step adaptation** (HEAL, STAMP, NegoCollab) introduces a unified protocol space as an intermediary. However, protocol spaces are pre-defined or negotiated; encountering a new modality often requires re-negotiating the protocol and re-training all mappings. Both categories rely on repeated joint training or fine-tuning, which is often infeasible across manufacturers due to model and data privacy constraints.

**Key Challenge**: In open-world deployments, modalities evolve continuously, yet translator training paradigms assume a closed set of modalities and repeatable training. If these assumptions are broken, the entire collaborative perception system loses scalability.

**Goal**: Pre-train a universal model capable of providing an appropriate feature translator for any newly emerging "source $\to$ target" modality pair during inference in a **zero-shot** manner, without additional training or fine-tuning.

**Key Insight**: Two observations are made. (i) Intermediate features entangle "scene content" with "modality statistics." However, mapping them to a **low-dimensional modality-intrinsic space** suppresses scene factors, allowing new modalities to be stably localized and compared. This moves the difficult "mapping estimation" task from a high-dimensional feature space to a compact space. (ii) Rather than training a massive monolithic translator to handle all mappings, mappings can be decomposed into a set of **reusable expert parameters**. A router translates "modality mapping" into "parameter combination coefficients," synthetically instantiating a mapping-specific translator on-demand. Consequently, inference runs only a single synthesized translator, avoiding the overhead of multi-expert parallelization in classic MoE.

**Core Idea**: Replace "training an adapter for each modality pair / maintaining a protocol space" with "inferring mapping in an intrinsic space + instantiating a translator via parameter-level composition conditioned on the mapping."

## Method

### Overall Architecture

In the system, each agent $i$ uses its own encoder $f^{\mathrm{enc}}_{i}$ to produce BEV intermediate features $F_i \in \mathcal{F}_{m_i}$. Neighbor $j$ transmits $\tilde{F}_j$ via communication. After geometric alignment, the ego vehicle obtains $\hat{F}_{j\to i}$, which is then passed through a translator $\mathcal{T}_{\phi_{j\to i}}$ to map it into the ego modality space before fusion and task head processing. All UniTrans innovations reside in the "instantiating $\phi_{j\to i}$ on-the-fly" step. Training is split into two stages: first, learning a Modality-Intrinsic Encoder (MIE), then learning a Translator Parameter Bank (TPB) and a Modality Mapping Router (MMR). During inference, all agents share the same TPB+MMR. Meeting a new modality, the MMR outputs combination coefficients $\boldsymbol{\alpha}_{j\to i}$ to linearly synthesize the translator from TPB expert parameters. Formally, the system learns a mapping $\delta_{j\to i} \in \mathcal{H} \to \phi_{j\to i} \in \Phi$ before performing feature translation. This "parameter-space composition" differs from classic "activation + weighted output" MoE, as it runs only one translator, ensuring inference cost does not scale linearly with the number of experts.

### Key Designs

1.  **Modality-Intrinsic Encoder (MIE)**:
    - **Function**: Compress high-dimensional BEV features $F \in \mathbb{R}^{C\times H\times W}$ into a low-dimensional intrinsic code $z \in \mathbb{R}^d$. It requires that $z$ from the same modality cluster together despite scene changes, while different modalities remain separated, providing a stable modality descriptor.
    - **Mechanism**: Extracts two types of scene-insensitive but modality-sensitive statistics: first-order channel statistics $\mu(F), \sigma(F)$, and a Gram descriptor $G(F)=\frac{1}{H'W'}\bar{F}\bar{F}^\top$ of pooled features, combined with a global response branch $r(F)$. These are fused via MLP: $z = \psi_I([\mu(F), \sigma(F), r(F), \psi_G(G(F))])$. The objective is an InfoNCE contrastive loss $\mathcal{L}_{\mathrm{IC}}$ plus a lightweight cross-entropy classification loss $\mathcal{L}_{\mathrm{IS}}$, resulting in $\mathcal{L}_{\mathrm{stage1}} = \mathcal{L}_{\mathrm{IC}} + \lambda_{\mathrm{IS}} \mathcal{L}_{\mathrm{IS}}$.
    - **Design Motivation**: Directly estimating $\Delta_{j\to i}:\mathcal{F}_{m_j}\to\mathcal{F}_{m_i}$ in the raw $\mathcal{F}$ space is ill-posed due to sparse samples, high dimensionality, and scene noise. Moving inference to the intrinsic space allows new modalities to land in consistent regions with few samples, providing the first source of zero-shot capability.

2.  **Translator Parameter Bank (TPB)**:
    - **Function**: Use a set of reusable "expert parameters" $\{\Theta^{(k)}\}_{k=1}^K$ and a shared expert $\Theta^{(0)}$ to represent the mapping space. Each $\Theta^{(k)}$ is a complete set of translator backbone parameters (Sparse Transformer-based MCT blocks).
    - **Mechanism**: Instead of training a monolithic translator for all $(m_j, m_i)$, the mapping space is decomposed, allowing experts to focus on "translation sub-patterns" which are mixed on-demand. 
    - **Design Motivation**: Mapping diversity is harder to fit than feature diversity. Monolithic translators often underfit large cross-modality gaps (e.g., LiDAR-to-Camera). TPB distributes the burden; the more complex the mapping, the more decomposition helps.

3.  **Modality Mapping Router (MMR) + Parameter-level Instantiation**:
    - **Function**: Given a code pair $(z_j, z_i)$, first compute a mapping descriptor $\delta_{j\to i} = g(z_j, z_i)$, then output coefficients $\boldsymbol{\alpha}_{j\to i} = \mathrm{softmax}(h(\delta_{j\to i})) \in \mathbb{R}^K$. Finally, synthesize the translator **at the parameter level**: $\phi_{j\to i} = \Theta^{(0)} + \sum_{k=1}^{K} \alpha^{(k)}_{j\to i}\, \Theta^{(k)}$.
    - **Mechanism**: Stage 2 jointly optimizes the task loss $\mathcal{L}_{\mathrm{task}}$, feature distillation loss $\mathcal{L}_{\mathrm{feat}} = \|F_{j\to i} - F^{\star}_{j\to i}\|_2^2$ (using $F^{\star}_{j\to i}=f^{\mathrm{enc}}_{m_i}(X_j)$ as a teacher), InfoNCE on routing vectors $\mathcal{L}_{\mathrm{ctr}}$, and a Switch-style balance regularizer $\mathcal{L}_r$.
    - **Design Motivation**: Modeling "mapping $\to$ translator" as coefficient prediction allows the MMR to learn extrapolatable rules across mappings—the second source of zero-shot capability. Parameter-level synthesis is more efficient than the "multi-expert forward + output mixing" of classic MoE.

### Loss & Training

Training is strictly two-stage. Stage 1 updates only MIE to build a stable intrinsic space. Stage 2 freezes MIE and task heads, backpropagating through MMR and TPB to minimize $\mathcal{L}_{\mathrm{stage2}} = \mathcal{L}_{\mathrm{task}} + \lambda_{\mathrm{feat}}\mathcal{L}_{\mathrm{feat}} + \lambda_{\mathrm{ctr}}\mathcal{L}_{\mathrm{ctr}} + \lambda_r \mathcal{L}_r$. Training only sees modalities in $\mathcal{M}_{\mathrm{tr}}$ and scenes in $\mathcal{D}_{\mathrm{tr}}$. Six emerging modalities $\{m_7, m_{13}, m_{17}, m_{25}, m_{27}, m_{30}\}$ are reserved for zero-shot testing.

## Key Experimental Results

### Main Results

Evaluations are conducted on OPV2V-H (simulated) and DAIR-V2X (real-world) with 30 LiDAR/Camera modality combinations. 6 emerging modalities serve as ego, with neighbors sampled from the emerging set for strict any-to-any zero-shot evaluation.

| Dataset | Metric | Ours (UniTrans) | Strongest Baseline (NegoCollab) | Gain |
| :--- | :--- | :--- | :--- | :--- |
| OPV2V-H Avg | AP@0.5 / AP@0.7 | **0.716 / 0.605** | 0.662 / 0.538 | +5.4 / +6.7 pt |
| OPV2V-H m27 (Cam ego) | AP@0.5 / AP@0.7 | **0.497 / 0.243** | 0.468 / 0.206 | +2.9 / +3.7 pt |
| OPV2V-H m30 (Cam ego) | AP@0.5 / AP@0.7 | **0.406 / 0.202** | 0.355 / 0.188 | +5.1 / +1.4 pt |
| DAIR-V2X Avg | AP@0.5 / AP@0.7 | **0.553 / 0.421** | 0.509 / 0.389 | +4.4 / +3.2 pt |

Inference cost is significantly lower (OPV2V-H): UniTrans requires 109.3 GFLOPs / 6.865 ms CPU, vs. Classic MoE (TopK=3) at 245.5 GFLOPs / 89.078 ms CPU, achieving ~2$\times$ compute savings.

### Ablation Study

| Configuration | AP@0.5 / AP@0.7 | Note |
| :--- | :--- | :--- |
| Full UniTrans | **0.716 / 0.605** | Full model |
| w/o $\mathcal{L}_{\mathrm{IC}}$ | 0.685 / 0.575 | Removed contrastive loss; intrinsic space unclustered |
| w/o $\mathcal{L}_{\mathrm{IS}}$ | 0.694 / 0.583 | Removed classification auxiliary |
| w/o $\mathcal{L}_{\mathrm{IC}} + \mathcal{L}_{\mathrm{IS}}$ | 0.662 / 0.540 | Stage 1 supervision removed; modality recognition failed |
| w/o $\mathcal{L}_{\mathrm{task}}$ | 0.691 / 0.579 | Lacks task supervision |
| w/o $\mathcal{L}_{\mathrm{feat}}$ | 0.653 / 0.531 | **Most critical** — Loss of ego-encoder distillation teacher |
| w/o $\mathcal{L}_{\mathrm{ctr}}$ | - | Routing vectors no longer cluster by mapping |

### Key Findings

- The distillation loss $\mathcal{L}_{\mathrm{feat}}$ is the most impactful component. This confirms that using the ego encoder to process neighbor observations as a teacher is the implicit key to zero-shot translation, providing a clear "how the ego domain should look" signal.
- Gains are most stable for LiDAR ego (m7–m25). While Camera ego (m27, m30) performance drops for all methods, UniTrans remains the strongest, proving the intrinsic space is robust even across large cross-modality gaps.
- Classic MoE only slightly outperforms some two-step methods (e.g., PolyInter), suggesting that routing directly on high-dimensional features struggles to learn mapping-aware expert selection.
- On DAIR-V2X, gains are smaller than simulation but the relative ranking holds, showing mapping-conditioned instantiation is superior under distribution shifts.

## Highlights & Insights

- **"Where to infer the mapping" matters more than "translator complexity"**: Moving mapping inference from the raw feature space to the intrinsic space independently accounts for ~5 AP@0.7 improvement. This concept is transferable to other heterogeneous alignment tasks (e.g., federated learning).
- **Parameter-level Expert Composition**: Unlike classic MoE "multi-expert forward + output mixing," UniTrans uses "weighted parameters + single forward," retaining MoE capacity while keeping inference latency comparable to a monolithic model.
- **Clever Teacher Design**: Using $F^\star_{j\to i}=f^\mathrm{enc}_{m_i}(X_j)$ as a teacher sidesteps the "unpaired feature" problem and helps the translator identify the target domain.
- **Plug-and-Play**: Once pre-trained, any new vehicle can be integrated zero-shot simply by sending one feature frame, allowing the ego to assemble a dedicated translator.

## Limitations & Future Work

- Modalities were derived from "backbone + depth" combinations. Whether the MIE space maintains clustering for truly new families (e.g., 4D radar, thermal) requires further verification.
- The number of experts $K$ and their full backbone parameters significantly affect VRAM ($K+1$ sets of parameters).
- The distillation teacher requires the ego to access raw neighbor observations $X_j$ during training, which may violate data privacy in cross-manufacturer training stages.
- Performance on Camera-ego scenarios remains lower than desired for deployment; incorporating geometric priors or temporal BEV history is a potential future direction.

## Related Work & Insights

- **vs MPDA / PnPDA / PolyInter**: These require new training for new modalities; UniTrans uses TPB to decompose all mappings into shared experts and MMR for online composition.
- **vs HEAL / STAMP / NegoCollab**: These rely on protocol spaces that may not suit new modalities; UniTrans estimates mappings directly in a data-driven intrinsic space.
- **vs Classic MoE**: Classic MoE routes on high-dimensional features and runs multiple experts. UniTrans routes in a low-dimensional space and uses parameter-level synthesis for faster single-forward inference.
- **Insight**: This "low-dimensional inference + parameter-pool instantiation" mirrors early hyper-network ideas but successfully scales them for collaborative perception. The decomposition into intrinsic space generalization and router generalization is a robust framework for heterogeneous device compatibility.

## Rating
- Novelty: ⭐⭐⭐⭐ Reformulating alignment as "intrinsic routing + parameter-level MoE" is sharp, though individual components have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive datasets and modality combinations. The ablation study and profiling are well-executed.
- Writing Quality: ⭐⭐⭐⭐ Clear notation and differentiation between modality and scene sets.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the pain point of cross-manufacturer deployment and evolving modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[ICML 2026\] Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference](partitioning_for_intrinsic_model_inversion_resistance_in_collaborative_inference.md)
- [\[CVPR 2026\] All Vehicles Can Lie: Efficient Adversarial Defense in Fully Untrusted-Vehicle Collaborative Perception via Pseudo-Random Bayesian Inference](../../CVPR2026/ai_safety/all_vehicles_can_lie_efficient_adversarial_defense_in_fully_untrusted-vehicle_co.md)
- [\[CVPR 2026\] FedRE: A Representation Entanglement Framework for Model-Heterogeneous Federated Learning](../../CVPR2026/ai_safety/fedre_a_representation_entanglement_framework_for_model-heterogeneous_federated_.md)
- [\[AAAI 2026\] Detect All-Type Deepfake Audio: Wavelet Prompt Tuning for Enhanced Auditory Perception](../../AAAI2026/ai_safety/detect_all-type_deepfake_audio_wavelet_prompt_tuning_for_enhanced_auditory_perce.md)

</div>

<!-- RELATED:END -->
