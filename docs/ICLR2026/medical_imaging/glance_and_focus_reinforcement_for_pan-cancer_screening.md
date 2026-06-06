---
title: >-
  [Paper Note] Glance and Focus Reinforcement for Pan-cancer Screening
description: >-
  [ICLR 2026][Medical Imaging][Pan-cancer screening] This paper proposes GF-Screen, a two-stage framework in which a lightweight Glance model employs reinforcement learning to rapidly localize CT sub-volumes containing les…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Pan-cancer screening"
  - "reinforcement learning"
  - "GRPO"
  - "CT segmentation"
  - "foreground-background imbalance"
date: 2026-05-08
content_hash: bb3d126ff1077957
---

# Glance and Focus Reinforcement for Pan-cancer Screening

**Conference**: ICLR 2026
**arXiv**: [2601.19103](https://arxiv.org/abs/2601.19103)  
**Code**: [GitHub](https://github.com/Luffy03/GF-Screen)  
**Area**: Medical Imaging / Cancer Screening
**Keywords**: Pan-cancer screening, reinforcement learning, GRPO, CT segmentation, foreground-background imbalance

## TL;DR

This paper proposes GF-Screen, a two-stage framework in which a lightweight Glance model employs reinforcement learning to rapidly localize CT sub-volumes containing lesions, while a Focus model performs fine-grained segmentation exclusively on the selected regions. By transferring GRPO's group-relative comparison paradigm from NLP to visual sub-volume groups, the method achieves RL optimization without a value network for the first time in a purely visual task. On the FLARE25 pan-cancer challenge, GF-Screen outperforms the champion solution by +25.6% DSC while achieving 5.7× faster inference.

## Background & Motivation

**Background**: Pan-cancer screening aims to detect and segment multiple lesion types from large-scale CT scans using a single unified model. Existing methods such as nnUNet, SwinUNETR, and CancerUniT adopt sliding-window strategies to traverse the entire CT volume for block-wise segmentation, achieving competitive performance on individual lesion types.

**Limitations of Prior Work**: Lesions occupy only approximately 0.085% of the CT volume, resulting in extreme foreground-background imbalance. Exhaustive inference introduces two critical problems: (1) substantial computational waste on healthy regions, with inference exceeding 100 seconds per scan and thus hindering large-scale deployment; and (2) redundant attention to healthy regions, which increases false positives and degrades screening precision.

**Key Challenge**: Accuracy and efficiency appear to be in conflict — higher detection sensitivity demands denser scanning, yet denser scanning generates more false positives and computational overhead. The root cause is that existing methods treat all regions uniformly and lack a selective attention mechanism.

**Goal**: (1) How can the model adopt a radiologist-like strategy of global coarse scanning followed by local fine inspection, skipping irrelevant regions? (2) How can selection behavior be trained with RL without introducing an additional value network? (3) How can the selection policy remain robust to severe foreground-background imbalance?

**Key Insight**: Radiologists reading CT scans employ a "glance-and-focus" strategy — rapidly surveying the global view to exclude normal regions and then carefully examining suspicious locations. The authors observe that a group of sub-volumes cropped from the same CT naturally constitutes the "candidate group" required by GRPO, enabling intra-group relative comparison directly without relying on an LLM to generate candidate answers.

**Core Idea**: A lightweight classification network performs "glancing" for selection, while a segmentation network performs "focusing" for fine inspection. Segmentation results serve as RL reward signals, and group-relative learning performs comparative optimization within sub-volume groups, simultaneously addressing efficiency and accuracy.

## Method

### Overall Architecture

GF-Screen consists of two collaborating models: a Glance model (lightweight 3D ResNet-18, ~32M parameters) that classifies cropped CT sub-volumes as "lesion-present" or "lesion-absent," and a Focus model (SwinUNETR) that performs pixel-level segmentation on selected sub-volumes. During training, $N=16$ randomly cropped sub-volumes are fed simultaneously into both models — the Focus model is trained with supervised segmentation using Dice-CE loss, and its segmentation output serves as a reward signal back-propagated to the Glance model via RL. During inference, the CT volume is first divided into sub-volumes via sliding window; the Glance model rapidly filters out approximately 83.3% of healthy sub-volumes, retaining only the 16.7% containing lesions for the Focus model to segment.

### Key Designs

1. **Binary Detection Reward Function**:

    - Function: Provides an RL reward signal for each sub-volume selection decision.
    - Mechanism: $r_i = \mathbb{1}(s_i \cap m_i \neq \emptyset)$, i.e., a reward of 1 is assigned if the Focus model's segmentation prediction $s_i$ has any overlap with the annotation $m_i$, and 0 otherwise. Segmentation DSC is deliberately not used as the reward.
    - Design Motivation: A DSC reward biases the model toward "easily segmented" views (well-defined boundaries, standard orientations), which are not necessarily the most clinically important. Partially lesion-containing sub-volumes with challenging angles may yield low DSC yet remain diagnostically critical. The binary reward focuses on "whether a lesion is detected," avoiding the omission of difficult but important cases.

2. **Group Relative Learning (GRL)**:

    - Function: Estimates the advantage value for each sub-volume selection action without a value network, replacing the critic network in conventional PPO.
    - Mechanism: $N$ sub-volumes cropped from the same CT form a comparison group, and relative advantages are computed via intra-group normalization: $A_i = (r_i - \text{mean}(r_{1..N})) / \text{std}(r_{1..N})$. Sub-volumes with positive advantage are encouraged to be selected, while those with negative advantage are encouraged to be discarded. The final objective $\mathcal{J}_{GRL}$ adopts a PPO-style clipped ratio multiplied by the advantage, with KL regularization ($\beta=0.01$) to constrain policy drift and a small-weight classification cross-entropy loss ($\alpha=0.1$) for basic supervision.
    - Design Motivation: GRPO in NLP requires an LLM to generate multiple candidate responses to form comparison groups, a mechanism absent in visual tasks. GF-Screen cleverly exploits the multi-candidate structure naturally arising from CT sub-volume cropping, representing the first successful application of GRPO to a purely visual perception task. Compared with PPO, this eliminates value network training overhead and yields more stable convergence.

3. **Glance-Focus Joint Training Strategy**:

    - Function: End-to-end joint training of both models.
    - Mechanism: The total loss is $L = L_{GRL} + L_{seg}$, where $L_{seg}$ is the standard Dice-CE segmentation loss. To maintain training stability, the Glance model maintains a frozen reference model $G_{ref}$ updated once per epoch. Training is conducted on a single A800 80G GPU with batch size 4, cropping 4 sub-volumes per volume (16 per group in total).
    - Design Motivation: The selection operation is non-differentiable, and segmentation gradients cannot be directly back-propagated to the Glance model, necessitating RL. Converting segmentation outputs into reward signals is the key to bridging the two models.

### Loss & Training

The Focus model is trained with the standard Dice-CE segmentation loss. The GRL loss for the Glance model comprises three terms: (1) a clipped policy gradient term — the primary optimization signal; (2) a KL divergence regularization term ($\beta=0.01$) — preventing excessive policy deviation; and (3) a cross-entropy classification term ($\alpha=0.1$) — providing weak supervision, with $\alpha$ set small to avoid model collapse to predicting all negatives due to foreground-background imbalance. The optimizer is AdamW with a learning rate of 3e-4 and cosine decay scheduling; sub-volume size is $96 \times 96 \times 64$.

## Key Experimental Results

### Main Results: Pan-cancer Segmentation and Detection (9 lesion types, 16 internal + 7 external datasets)

| Method | Seg. DSC (%) | Det. F1 (%) | FPR (%) | Inference Time (s/scan) |
|--------|-------------|------------|---------|------------------------|
| nnUNet | 53.3 | 90.2 | 30.4 | 136 |
| SwinUNETR | 48.6 | 92.5 | 47.5 | 114 |
| VoCo | 56.1 | 92.2 | 41.8 | 114 |
| SuPreM | 54.4 | 90.4 | 38.7 | 114 |
| PASTA | 52.8 | 88.6 | 42.6 | 197 |
| **GF-Screen** | **60.8** | **95.9** | **15.6** | **28** |

GF-Screen surpasses the second-best method VoCo by 4.7% in segmentation DSC, achieves a detection F1 of 95.9% (+3.4%), reduces the false positive rate from 30.4% (second-best) to 15.6% (−14.8%), and is 4–7× faster in inference. It also leads on external datasets (average DSC 54.1% vs. 49.0% for the runner-up). On the FLARE25 public validation leaderboard, GF-Screen achieves 58.6% DSC / 52.2% NSD, substantially outperforming the FLARE24 champion (33.0% / 24.0%) by +25.6% and +28.2%, respectively.

### Ablation Study: RL Training Strategy Comparison (FLARE23 dataset)

| Training Strategy | DSC (%) | Selection Rate (%) | Notes |
|-------------------|---------|--------------------|-------|
| SwinUNETR Baseline | 41.5 | 100 | No Glance; full-volume segmentation |
| Cross-Entropy (CE) | 37.6 | 3.1 | Collapse to negative class; too few selected |
| Balanced CE | 37.8 | 5.3 | Slightly better but still collapses |
| Focal Loss | 36.5 | 4.0 | Imbalance problem unresolved |
| OHEM | 39.5 | 7.2 | Hard example mining yields limited gains |
| PPO + Value Network | 24.5 | 51.6 | Unstable training; severe collapse |
| GRL + DSC Reward | 43.2 | 21.3 | Biased toward "easy" views |
| GRL + Binary Reward | 53.1 | 23.0 | Detection reward markedly outperforms DSC reward |
| **GRL + Binary Reward + αCE** | **56.7** | **16.7** | **Full method; best overall** |

### Key Findings

- **Direct classification training inevitably fails**: Extreme foreground-background imbalance causes CE/Focal/OHEM to collapse entirely to the negative class (selection rates of 3–7%), demonstrating that a pure classification paradigm cannot address this problem.
- **PPO is unstable**: Introducing an additional value network leads to severe training oscillation (DSC of only 24.5%), confirming the inadequacy of conventional RL methods for visual sub-volume selection.
- **Binary reward >> DSC reward**: 53.1% vs. 43.2%, a gap of 10 percentage points, indicating that the Glance model should focus on "whether a lesion is present" rather than "how well it is segmented."
- **Small-weight CE auxiliary term is effective**: Adding the CE term with $\alpha=0.1$ improves DSC from 53.1% to 56.7%, providing beneficial weak supervision without incurring collapse risk.
- **Group size sensitivity**: $N=16$ is optimal (56.5% DSC); performance drops substantially at $N=4$ (45.9%), demonstrating that a sufficiently large set of intra-group comparison candidates is critical for GRL.
- **High Glance model sensitivity**: Sensitivity of 97.7% (near-zero miss rate) and specificity of 75.9% (effective filtering of healthy regions).

## Highlights & Insights

- **Key insight for transferring GRPO from NLP to vision**: Sub-volume cropping naturally produces candidate groups, enabling intra-group relative comparison without requiring LLM generation capabilities. This insight is particularly elegant — it reinterprets the seemingly mundane data augmentation operation of "random cropping" in medical imaging as "candidate generation" in RL, allowing visual tasks to fully leverage the advantages of GRPO.
- **Positive coupling of efficiency and accuracy**: Conventional wisdom holds that "looking at more" yields greater accuracy, but this paper demonstrates that "selectively looking at less" is actually more accurate — discarding healthy regions not only saves computation but actively eliminates sources of false positives. This principle of "improving accuracy by subtraction" merits broader adoption in other foreground-sparse tasks.
- **Insight behind the binary reward design**: The paper deliberately rejects the seemingly more informative DSC reward in favor of a coarse binary detection reward. The underlying rationale is that the Glance model acts as a "gatekeeper," not a "diagnostician" — it only needs to determine "whether something suspicious is present," not to evaluate segmentation quality. Clear delineation of model roles is central to the system design.

## Limitations & Future Work

- **Miss detection risk of the Glance model**: Although sensitivity reaches 97.7%, extremely small lesions or those spanning sub-volume boundaries may still be missed, and missed detections carry severe consequences in clinical settings.
- **Fixed sub-volume size**: The fixed cropping size of $96 \times 96 \times 64$ may not be optimal for lesions of varying sizes, as large lesions may be split across multiple sub-volumes.
- **CT modality only**: The method has not been tested on MRI, ultrasound, PET, or other imaging modalities; whether the GRL paradigm transfers to foreground-sparse scenarios in other modalities remains to be verified.
- **Dependence on intra-group diversity**: If lesions are extremely dense (all positive) or the CT is entirely healthy (all negative), the gradient signal from intra-group comparison degrades.
- Adaptive sub-volume sizing or multi-scale cropping strategies could be explored to better handle lesions of varying sizes.

## Related Work & Insights

- **vs. CancerUniT**: CancerUniT employs a query-based Mask-Transformer for unified detection and segmentation of eight lesion types but still performs exhaustive inference without a foreground selection mechanism. GF-Screen outperforms it comprehensively in segmentation accuracy (+7.5% DSC) while being several times faster, demonstrating the superiority of "select then segment" over "full-volume inference."
- **vs. PPO-based Visual RL (Wang et al. 2020)**: Prior work applied PPO to discard redundant patches in image classification but required an additional value network and could not handle dense prediction tasks such as segmentation. GF-Screen replaces PPO with GRL, eliminating the value network while achieving markedly greater stability on segmentation tasks (PPO variant yields only 24.5% DSC).
- **vs. GRPO (Shao et al. 2024)**: GRPO in LLMs forms intra-group comparisons by having the model generate multiple responses; GF-Screen finds that CT sub-volume cropping naturally provides candidate groups, representing the first successful transfer of GRPO to a non-linguistic task.
- **vs. PASTA**: PASTA leverages tumor synthesis for pre-training but does not address inference efficiency (197 s/scan, the slowest among all methods). GF-Screen adopts an entirely different approach and simultaneously improves both accuracy and efficiency.

## Rating

- Novelty: ⭐⭐⭐⭐ The insight of transferring GRPO to visual tasks is elegant, though the overall "coarse-to-fine" framework is a well-established paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Nine lesion types, 23 datasets, internal and external validation, FLARE25 leaderboard, and highly systematic ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The logical chain is clear and the figures are well-designed, though certain passages contain redundant exposition.
- Value: ⭐⭐⭐⭐⭐ +25.6% DSC over the FLARE25 champion with 5.7× faster inference; extremely high practical deployment value.

- **vs. PASTA**: Ours proposes a distinct technical approach on this basis and achieves improvements on key metrics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First successful transfer of GRPO to vision + pan-cancer framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5,117 CT scans / 9 lesion types / 23 datasets / FLARE25 champion.
- Writing Quality: ⭐⭐⭐⭐ Method is clearly presented; clinical motivation is convincing.
- Value: ⭐⭐⭐⭐⭐ Direct clinical translation value for AI-assisted cancer screening.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PanFoMa: A Lightweight Foundation Model and Benchmark for Pan-Cancer Pathology Image Analysis](../../AAAI2026/medical_imaging/panfoma_a_lightweight_foundation_model_and_benchmark_for_pan-cancer.md)
- [\[CVPR 2026\] Association of Radiologic PPFE Change with Mortality in Lung Cancer Screening Cohorts](../../CVPR2026/medical_imaging/association_of_radiologic_ppfe_change_with_mortali.md)
- [\[AAAI 2026\] MedEyes: Learning Dynamic Visual Focus for Medical Progressive Diagnosis](../../AAAI2026/medical_imaging/medeyes_learning_dynamic_visual_focus_for_medical_progressive_diagnosis.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)

</div>

<!-- RELATED:END -->
