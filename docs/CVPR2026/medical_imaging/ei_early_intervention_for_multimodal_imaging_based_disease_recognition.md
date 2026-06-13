---
title: >-
  [Paper Note] EI: Early Intervention for Multimodal Imaging based Disease Recognition
description: >-
  [CVPR 2026][Medical Imaging][Multimodal medical imaging] EI proposes injecting cross-modal semantic guidance (the [INT] token) **before** unimodal embedding (UIE)…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Multimodal medical imaging"
  - "early intervention"
  - "LoRA"
  - "MoE"
  - "VFM adaptation"
  - "disease recognition"
date: 2026-05-08
content_hash: b71ff3d737a6fcc5
---

# EI: Early Intervention for Multimodal Imaging based Disease Recognition

**Conference**: CVPR 2026 Findings  
**arXiv**: [2603.17514](https://arxiv.org/abs/2603.17514)  
**Code**: [github.com/ruc-aimc-lab/EI](https://github.com/ruc-aimc-lab/EI)  
**Area**: Medical Imaging / Multimodal Fusion
**Keywords**: Multimodal medical imaging, early intervention, LoRA, MoE, VFM adaptation, disease recognition

## TL;DR

EI proposes injecting cross-modal semantic guidance (the [INT] token) **before** unimodal embedding (UIE), emulating the clinical workflow in which a clinician first examines one modality to form a preliminary judgment and then uses that judgment to guide interpretation of another modality. Simultaneously, EI introduces MoR (multi-rank LoRA with a relaxed bypass router) for parameter-efficient VFM adaptation to the medical domain. With fewer than 9M trainable parameters, EI surpasses all full fine-tuning and prompt-learning baselines on three datasets covering retinal, dermatological, and knee-joint imaging.

## Background & Motivation

**Background**: Disease recognition from multimodal medical images (e.g., CFP+OCT fundus, dermoscopy+clinical photography, multi-view MRI) is an important task in computer vision. Existing methods (MM-MIL, CosCatNet, RadDiag, MMRAD) all follow a **fusion-after-UIE** paradigm—independently extracting unimodal features with dedicated encoders and subsequently fusing them via concatenation, weighted summation, or attention.

**Limitations of Prior Work (1) — Late Fusion**: During the UIE stage, all existing methods remain entirely unaware of other modalities, preventing UIE from leveraging complementary information. This contradicts clinical practice, where clinicians never interpret a single modality in isolation but rather form an initial hypothesis from one modality and use it to guide interpretation of another.

**Limitations of Prior Work (2) — VFM Adaptation Difficulty**: Medical annotation data are scarce, and a large domain gap exists between natural and medical images. Directly applying VFMs such as CLIP or DINOv2 yields poor performance; full fine-tuning leads to overfitting; and prompt learning can only activate pre-existing knowledge without injecting new knowledge.

**Mechanism**: (a) The high-level semantic representation ([CLS] token) of the reference modality is used as an intervention token [INT] and injected at the **earliest stage** of target-modality UIE. (b) MoR—multi-rank LoRA combined with a relaxed bypass router—is designed to balance adaptation capacity and parameter efficiency.

## Method

### Overall Architecture

Input: a medical image sample with $M$ modalities → each modality serves in turn as the target while the remaining modalities serve as references → an auxiliary VFM extracts the [CLS] token of each reference modality → an Adapter transforms these into [INT] tokens → the [INT] tokens are prepended to the target modality's patch embedding sequence → the main VFM performs UIE (with [INT] interacting with target patch tokens across all Transformer layers) → each modality yields an [INT]-intervened CLS feature → adaptive gated weighted fusion → classification prediction.

### Key Designs

1. **[INT] Token Generation**:

    - **Function**: Extracts high-level semantic tokens from reference modalities as cross-modal intervention signals.
    - **Mechanism**: For each reference modality $r$, an auxiliary VFM $\phi_{a,r}$ extracts the final-layer [CLS] token $Z^L[0]$. The [CLS] tokens from all reference modalities are collected and transformed into the [INT] sequence via a two-layer MLP Adapter. The final layer is chosen because it encapsulates the richest high-level semantic information.
    - **Design Motivation**: This mimics the clinical workflow in which a physician first forms a preliminary diagnostic hypothesis from one examination (e.g., OCT cross-sectional imaging) and then uses that hypothesis to guide interpretation of another examination (e.g., CFP color fundus photography). The auxiliary VFM is responsible solely for generating [INT], keeping computational overhead manageable.

2. **Main Feature Extraction with Early Intervention**:

    - **Function**: Prepends the [INT] token to the target modality's patch embedding sequence so that it participates in self-attention from layer 0 onwards.
    - **Mechanism**: $\hat{Z}_t^0 = \text{Concat}(\text{Conv}(\mathbf{x}[t]),\ \text{INT})$, followed by standard forward propagation to obtain $\hat{\text{CLS}}_t = \phi_{p,t}(\hat{Z}_t^0, L)[0]$.
    - **Key Experimental Finding**: Ablation experiments clearly demonstrate that **earlier injection yields better performance**—injection at Layer 0 consistently achieves the best results (CLIP 0.824, DINOv2 0.841), while injection at Layer 11 degrades performance to 0.815, validating the early intervention principle.
    - **Visual Evidence**: Fig. 2 shows that after incorporating [INT], patch-level attention maps shift from diffuse to focused activation over lesion regions (e.g., drusen in OCT, hemorrhage spots in CFP), indicating that cross-modal guidance makes UIE more targeted.

3. **MoR (Mixture of Low-varied-Ranks Adaptation)**:

    - **Function**: Parameter-efficient fine-tuning of every linear layer in the VFM.
    - **Mechanism**: Three LoRA adapters of different ranks (2, 4, 8) are maintained simultaneously, together with a relaxed router (linear + softmax) that produces a 4-dimensional weight vector $[w_0, w_1, w_2, w_3]$, where $w_0$ is the **bypass weight**. The output is $h' = Wh + \sum_{k=1}^{3} w_k B_k A_k h$.
    - **vs. LoRA**: A single fixed rank cannot accommodate the varying complexity across different modalities and samples.
    - **vs. LoRAMoE**: In standard MoE routers the weights sum to 1, forcing the model to accept adaptation outputs. MoR's bypass mechanism allows the model to skip adaptation when the original weights are already sufficient (in the extreme case $w_0 = 1$, all LoRA branches are bypassed entirely).

4. **Adaptive Late Fusion**:

    - Each modality's $\hat{\text{CLS}}_t$ is projected by a linear layer to a $C$-dimensional prediction $\hat{y}_t$.
    - A gating layer (linear + softmax) produces modality importance weights $\{\alpha_1, \ldots, \alpha_M\}$.
    - The final prediction is $\hat{y} = \sum_{t=1}^{M} \alpha_t \hat{y}_t$.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_p + \lambda_1 \mathcal{L}_{aa} + \lambda_2 \mathcal{L}_{ag}$, comprising three terms:

- **Primary loss** $\mathcal{L}_p$: sum of cross-entropy losses over individual modality predictions and the fused prediction.
- **Auxiliary VFM supervision** $\mathcal{L}_{aa}$ ($\lambda_1=0.3$): ensures that the [INT] tokens generated by the auxiliary VFM are task-relevant.
- **Gating supervision** $\mathcal{L}_{ag}$ ($\lambda_2=0.1$): guides gating weight learning using a one-hot prior derived from the best-performing modality on the training set, addressing the tendency of VFM-based frameworks to overfit such that all modalities perform similarly during training, making it difficult for the gating module to distinguish true modality strengths.

## Key Experimental Results

### Main Results

| Dataset | Metric (mAP) | EI (DINOv2) | Best Baseline | Gain |
|---------|--------------|-------------|---------------|------|
| MMC-AMD (retinal 4-class) | mAP | **0.909** | MMRAD 0.821 | +10.7% |
| Derm7pt (skin 5-class) | mAP | **0.767** | MMRAD 0.566 | +35.5% |
| MRNet (knee 3-class) | mAP | **0.848** | MM-MIL 0.835 | +1.6% |
| Average across three datasets | MEAN | **0.841** | MMRAD 0.735 | +14.4% |

- EI uses only 8.9M trainable parameters, whereas full fine-tuning baselines typically require 200–400M.
- The largest gain is observed on Derm7pt, where domain shift is most severe, with mAP improving from 0.566 to 0.767.
- General-purpose VFMs (CLIP/DINOv2) consistently outperform domain-specific VFMs (RETFound/PanDerm/RadioDINO), suggesting that EI+MoR adaptation is more effective than training domain-specific models from scratch.

### Ablation Study

| Configuration | MEAN mAP | Note |
|---------------|----------|------|
| EI + MoR (full model) | **0.833** | Best |
| Fusion changed to after-UIE | 0.806 | Degrades to conventional late fusion |
| [INT] injected at Layer 11 instead of Layer 0 | 0.815 | Later injection is worse |
| Remove $\mathcal{L}_{aa}$ | 0.820 | Auxiliary VFM supervision is beneficial |
| Remove $\mathcal{L}_{ag}$ | 0.811 | Gating supervision contributes more |
| MoR → LoRA | ~2–3% drop | Fixed rank is insufficiently flexible |
| MoR without bypass | Slight drop | Bypass allows skipping unnecessary adaptation |

### Key Findings

- **Early intervention is the core contribution**: Degrading EI to conventional after-UIE fusion causes a significant performance drop, confirming that late fusion is indeed a bottleneck.
- **Layer 0 injection is optimal**: Earlier injection of [INT] consistently yields better performance, consistent with the design philosophy of introducing cross-modal information as early as possible.
- **MoR > LoRAMoE > LoRA > prompt learning > full fine-tuning**: In data-scarce medical settings, the quality of PEFT design is critical.
- **General VFMs outperform domain-specific VFMs**: Pretraining data scale and feature diversity matter more than domain alignment.

## Highlights & Insights

- **Clinical alignment of Early Intervention**: Translating the clinical workflow—"form a hypothesis from one examination, then use it to guide interpretation of another"—into [INT] token injection is conceptually concise and intuitively compelling. The visual evidence (attention maps shifting from diffuse to lesion-focused) is particularly persuasive.
- **Bypass design in MoR**: A minimalist improvement—expanding the router output dimension from 3 to 4—enables the model to adaptively decide whether LoRA adaptation is needed. This trick is transferable to any MoE-LoRA framework.
- **Adapter as a bridge**: The auxiliary VFM and main VFM differ in their roles (the former is frozen and lightweight; the latter requires fine-grained adaptation). A two-layer MLP transforms [INT] for compatibility, avoiding feature space misalignment between the two VFMs.

## Limitations & Future Work

- **Overhead of the auxiliary VFM**: Each reference modality requires one auxiliary VFM forward pass. With $M$ modalities, $2M$ VFM passes are needed ($M$ auxiliary + $M$ main), approximately doubling the compute relative to the unimodal case.
- **Validation limited to $M=2,3$**: When the number of modalities is larger (e.g., five or more medical imaging types), the [INT] sequence length grows linearly, and the complexity of self-attention may become a bottleneck.
- **Small dataset scale**: The largest dataset, Derm7pt, contains only 1,011 samples, and MMC-AMD only 768; whether the conclusions hold on large-scale datasets remains to be verified.
- **[INT] uses only the final-layer [CLS] token**: This restricts the intervention to high-level semantic information, discarding low-level texture and edge cues, which may be insufficient for tasks requiring low-level cross-modal complementarity.

## Related Work & Insights

- **vs. MMRAD**: Both leverage VFMs with PEFT; however, MMRAD uses prompt learning and performs fusion after UIE. EI improves along two independent dimensions: fusion before UIE and replacement of prompt learning with MoR.
- **vs. MM-MIL**: MM-MIL uses ResNet full fine-tuning with weighted-sum late fusion. Replacing its backbone with MoR (MM-MIL-MoR) raises performance from 0.733 to 0.823, yet still falls short of the full EI model (0.841), demonstrating that the contribution of early intervention is independent of the PEFT choice.
- **Transferable ideas**: The [INT] token injection mechanism can be transferred to any multimodal ViT framework (e.g., video+audio, RGB+depth), with the core principle being the use of high-level semantics from one modality to guide the embedding stage of another.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The early intervention concept is intuitively strong and clinically aligned; MoR is a well-motivated improvement over LoRA-MoE.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, two VFMs, and comprehensive ablation and hyperparameter analyses make for a very rigorous experimental evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ The structure is clear, motivation is developed fluently, and the clinical analogy is vivid.
- **Value**: ⭐⭐⭐⭐ Significant contribution to multimodal medical imaging; both MoR and the early intervention principle are transferable to other domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GLEAM: A Multimodal Imaging Dataset and HAMM for Glaucoma Classification](gleam_a_multimodal_imaging_dataset_and_hamm_for_gl.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](../../ICLR2026/medical_imaging/learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [\[CVPR 2026\] Robust Fair Disease Diagnosis in CT Images](robust_fair_disease_diagnosis_in_ct_images.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)

</div>

<!-- RELATED:END -->
