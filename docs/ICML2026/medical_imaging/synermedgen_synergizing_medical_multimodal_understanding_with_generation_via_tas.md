---
title: >-
  [Paper Note] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment
description: >-
  [ICML 2026][Medical Imaging][Unified Medical MLLM] SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (CTS / MI / TIA tasks). I…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Unified Medical MLLM"
  - "Generation-aligned Understanding"
  - "Cross-modal Synthesis"
  - "CTS/MI/TIA"
  - "SynerMed Dataset"
date: 2026-05-08
content_hash: 1413b5974c477fbf
---

# SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.08724](https://arxiv.org/abs/2605.08724)  
**Code**: https://github.com/Mhilab/SynerMedGen (available)  
**Area**: Medical Imaging / Multimodal VLM / Cross-modal Synthesis  
**Keywords**: Unified Medical MLLM, Generation-aligned Understanding, Cross-modal Synthesis, CTS/MI/TIA, SynerMed Dataset

## TL;DR
SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (CTS / MI / TIA tasks). It first uses two-stage training to enable the understanding branch to learn representations beneficial for synthesis, then transfers these to the latent flow matching generation branch. On 22 medical synthesis tasks, it outperforms both dedicated synthesis models and existing unified MLLMs.

## Background & Motivation

**Background**: Unified medical MLLMs (e.g., HealthGPT, UniMedVL) have begun to integrate "understanding" and "generation" into a single model—understanding handles VQA/report generation, while generation handles cross-modal synthesis such as CT↔MR, PET↔CT. Architectures often use dual-pathway or connector + diffusion hybrids.

**Limitations of Prior Work**: Existing unified frameworks treat understanding and generation as **two unrelated objectives**: the understanding side is trained with recognition-style tasks like lesion-level VQA, while the generation side is trained with pixel-level synthesis loss. As a result, models achieve high VQA scores but fail to preserve anatomical structures or adjust contrast correctly during cross-modal synthesis—supervision is completely misaligned.

**Key Challenge**: Cross-modal medical synthesis requires "understanding" at the **slice-level correspondence + modality identification + transformation direction**, whereas traditional understanding supervision only provides "global semantics," with little overlap in useful information.

**Goal**: To answer a "fundamental question avoided in unified medical MLLMs"—**what kind of 'understanding' is truly useful for generation?**—and to design specific tasks accordingly.

**Key Insight**: Since understanding should serve generation, **derive understanding tasks directly from the generation data**, ensuring naturally coupled training signals; also, use two-stage training to transfer multimodal priors learned in the first stage to the generation stage via shared parameters.

**Core Idea**: Define the "generation-aligned understanding" principle → design three understanding tasks directly corresponding to synthesis needs (CTS for pairing, MI for modality control, TIA for transformation direction) → two-stage training (understanding then generation) → release the SynerMed dataset with 1M paired samples + 2M understanding instances.

## Method

### Overall Architecture
Based on the Bagel unified architecture: an understanding encoder $E_{\text{ViT}}$ outputs semantic tokens $\mathbf{z}_{\text{ViT}}$, a generation encoder $E_{\text{VAE}}$ outputs latent tokens $\mathbf{z}_{\text{VAE}}$, both projected into a shared Mixture-of-Transformer-experts (MoT). MoT has two experts: the understanding expert for VLM-prompted learning, and the generation expert for conditional latent synthesis. **Stage I (GAU)**: train the understanding expert on three generation-aligned understanding tasks; **Stage II (UCG)**: perform flow matching in the VAE latent space on the same paired data, with the VAE decoder reconstructing pixels. All understanding tasks are formalized as "prompted short answer token generation" on the understanding expert, with loss computed only on answer tokens via masked NTP: $\mathcal{L}_{\text{NTP}}(\mathbf{y}^*)=-\sum_i\log p_\theta(y_i^*\mid \mathbf{y}^*_{<i},\mathbf{x}_{\text{text}},\mathbf{z}_{\text{ViT}})$.

### Key Designs

1. **Conditional Target Selection (CTS)—Capturing Slice-level Pairing**:

    - Function: Forces the model, under the explicit constraint of a target modality $m_{\text{tgt}}$, to select the true paired target slice $x^+=x_{\text{tgt}}$ from $N$ candidates for a given source slice $x_{\text{src}}$.
    - Mechanism: Formulated as a multiple-choice prompt, the model generates the correct option letter. The **key trick** is that hard negatives are not random slices, but **neighboring slices ($\pm 1\sim\pm K$) from the same target volume**—forcing the model to distinguish at fine-grained anatomical levels, not just coarse semantics like "oh, this is a brain." Loss: $\mathcal{L}_{\text{CTS}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{CTS}})$.
    - Design Motivation: Cross-modal synthesis requires "per-slice" preservation of patient-specific anatomy and lesions; coarse VQA cannot learn fine-grained slice correspondence. Hard negatives from neighbors are crucial for this capability.

2. **Modality Identification (MI)—Making Modality an Explicit Controllable Factor**:

    - Function: Enables the model to identify the modality of input images (or each panel)—CT / CBCT / PET / MRI (down to MRI sequence).
    - Mechanism: Also uses the prompted-generation framework, with the model outputting modality labels. The question bank **intentionally includes confusing pairs**—CT vs CBCT, similar MRI contrasts—to force the model to learn true features of the "modality" variable rather than superficial shortcuts. Loss: $\mathcal{L}_{\text{MI}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{MI}})$.
    - Design Motivation: Cross-modal synthesis requires "target modality" as a controllable input; if modality is not explicitly encoded during understanding, the generation stage must extract it from entangled appearance cues, leading to poor performance.

3. **Transformation Instruction Alignment (TIA)—Grounding Transformation Direction to Text**:

    - Function: Given a paired image $(x_1, x_2)$, the model selects the unique correct "route description" (e.g., "CT→MRI: change contrast, preserve anatomy") from a set of short descriptions.
    - Mechanism: Each synthesis route (ordered modality pair) maintains a description pool; positive examples $e^+$ are drawn from the ground-truth route pool, and $R-1$ distractors from other routes—distractors **specifically include "reversed direction" and "wrong modality pair" confusions**. Loss: $\mathcal{L}_{\text{TIA}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{TIA}})$.
    - Design Motivation: Slice correspondence and modality identification alone are insufficient—the model may still be unclear about "what to change, what to keep." TIA directly trains the ability to ground transformation semantics to text, making synthesis routes explicit model concepts.

### Loss & Training
**Stage I (GAU)**: The understanding expert is jointly trained on the three tasks, with total loss $\mathcal{L}_{\text{Stage I}}=\mathcal{L}_{\text{CTS}}+\mathcal{L}_{\text{MI}}+\mathcal{L}_{\text{TIA}}$. **Stage II (UCG)**: The generation expert performs flow matching-based conditional synthesis in VAE latent space; the understanding expert and shared MoT, already trained to be "generation-friendly" in Stage I, are further fine-tuned during generation training. The SynerMed dataset contains 1M paired synthesis samples and 2M generation-derived understanding instances.

## Key Experimental Results

### Main Results
On SynthRAD2023 (CBCT↔CT, MRI↔CT, PET↔CT across multiple sites) and BraTS (T1/T2/T1c/FLAIR four-modality conversions), a total of 22 synthesis tasks were compared using SSIM (× 100):

| Task Direction | Pix2Pix | CycleGAN | BBDM | ResViT | SynDiff | RCD | HealthGPT | UniMedVL | **SynerMedGen** |
|----------|---------|----------|------|--------|---------|-----|-----------|----------|-----------------|
| Brain CBCT→CT | 66.17 | 53.32 | 71.09 | 85.00 | 85.47 | 85.97 | 57.37 | 51.48 | **87.15** |
| Pelvis CBCT→CT | 63.55 | 55.87 | 60.49 | 84.00 | 83.21 | 86.22 | 46.89 | 43.94 | **87.14** |
| Brain MRI→CT | 74.33 | 52.65 | 68.99 | 86.39 | 87.19 | 86.12 | 84.29 | 54.11 | **88.87** |
| Whole-Body CT→PET | 72.21 | 65.98 | 67.68 | 87.07 | 88.12 | 88.90 | 66.54 | 74.12 | **91.10** |
| BraTS T2→T1 | 59.34 | 57.19 | 56.77 | 86.94 | 88.31 | 88.01 | 60.13 | 77.26 | **90.58** |
| BraTS T1→T2 | 62.10 | 53.31 | 56.41 | 85.78 | 84.25 | 86.19 | 70.32 | 78.88 | **87.14** |

SynerMedGen ranks first in all 22 tasks, with especially large gains (+15~30 SSIM points) over unified model baselines (HealthGPT, UniMedVL).

### Ablation Study

| Configuration | Average SSIM Trend | Notes |
|------|----------------|------|
| Full SynerMedGen (CTS+MI+TIA → UCG) | Optimal | All three tasks enabled |
| Only traditional VQA-style understanding → UCG | Significant drop | Validates that "task misalignment" is the root cause |
| Stage I (GAU) only, zero-shot generation | Strong zero-shot SSIM on 22 tasks | Synthesis possible without generation training, proving understanding phase learns generative knowledge |
| Remove CTS | Increased slice-level misalignment, anatomical drift | Pairing constraint is key |
| Remove MI | Target modality control fails | Lacks controllable modality factor |
| Remove TIA | Direction reversal / incorrect changes | Lacks route-level grounding |

### Key Findings
- **Training only understanding, without generation**, achieves strong SSIM on 22 synthesis tasks—this is the most striking result, directly proving the effectiveness of the "generation-aligned understanding" principle: the understanding phase already learns representations needed for generation, and the generation phase simply "translates" them back to pixels.
- The hard negative design in CTS (neighboring slices from the same volume) is irreplaceable; using random slices degrades the task to coarse semantic recognition, with a sharp drop in synthesis performance.
- In cross-dataset zero-shot tests, SynerMedGen maintains its advantage, indicating that the representations derived from the three tasks are truly modality/task-agnostic general priors.
- Unified MLLM baselines (HealthGPT, UniMedVL) perform especially poorly on subtle contrast synthesis tasks like CBCT, further confirming that simply attaching a generation head to an understanding model without supervision alignment is ineffective.

## Highlights & Insights
- The principle that "understanding tasks must be derived from generation data" is a **transferable design principle**—any "unified understanding + generation" work (video generation, 3D asset generation, code generation) can apply it: first clarify "what priors generation needs," then design corresponding understanding tasks, rather than defaulting to VQA/caption tasks.
- The three tasks correspond to "content, modality, direction" priors, geometrically covering all necessary dimensions for cross-modal synthesis, and are cleanly separated—almost serving as a **checklist for understanding task design**.
- The phenomenon that "Stage I alone achieves strong zero-shot generation" suggests that, in unified models, what is shared between understanding and generation is not token representations, but **multimodal priors implicitly encoded in MoT shared parameters**—providing concrete, measurable experimental evidence for "understanding-to-generation transfer."

## Limitations & Future Work
- Still limited to cross-modal slice synthesis; higher-order tasks such as 3D volume and temporal (4D-CT) synthesis are not yet covered.
- All three tasks are multiple-choice/classification; whether more open-ended generation-description understanding (e.g., detailed imaging reports) can further improve synthesis quality remains unexplored.
- The serial Stage I + Stage II training incurs high cost on very large datasets; whether end-to-end joint training is feasible is worth exploring.
- Robustness to domain shift across institutions/scanners is not deeply tested; further evaluation is needed for clinical deployment.

## Related Work & Insights
- **vs HealthGPT**: HealthGPT uses task-specific adapters to separately connect understanding and generation, but supervision is not aligned; SynerMedGen does not add adapters but redesigns supervision, achieving a 15~30 SSIM point improvement.
- **vs UniMedVL**: UniMedVL adopts a "progressive learning curriculum," but still uses traditional understanding tasks; SynerMedGen demonstrates that curriculum alone is insufficient—supervision alignment is key.
- **vs General-domain Bagel / Show-o / Janus**: These works focus on architectural unification, while this paper aligns at the "task design" level; the approaches are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The principle of "deriving understanding tasks from generation data" is explicitly proposed and cleanly instantiated for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ 22 synthesis tasks + zero-shot + unseen datasets + large-scale dataset release, with sufficient scale.
- Writing Quality: ⭐⭐⭐⭐ The design motivations for the three tasks are clearly explained using the "content/modality/direction" triad.
- Value: ⭐⭐⭐⭐ Establishes a new SOTA for unified medical MLLMs; the SynerMed dataset (1M paired + 2M understanding) is itself a community contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[CVPR 2026\] MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](../../CVPR2026/medical_imaging/medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [\[ICML 2026\] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)

</div>

<!-- RELATED:END -->
