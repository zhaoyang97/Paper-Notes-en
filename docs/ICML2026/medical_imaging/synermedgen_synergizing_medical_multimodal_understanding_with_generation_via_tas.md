---
title: >-
  [Paper Note] SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment
description: >-
  [ICML 2026][Medical Imaging][Unified Medical MLLM] SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (via CTS, MI…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "Unified Medical MLLM"
  - "generation-aligned understanding"
  - "cross-modal synthesis"
  - "CTS/MI/TIA"
  - "SynerMed dataset"
date: 2026-05-08
content_hash: 8f66bb8ed3844f5e
---

# SynerMedGen: Synergizing Medical Multimodal Understanding with Generation via Task Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.08724](https://arxiv.org/abs/2605.08724)  
**Code**: https://github.com/Mhilab/SynerMedGen (Yes)  
**Area**: Medical Imaging / Multimodal VLM / Cross-modal Synthesis  
**Keywords**: Unified Medical MLLM, generation-aligned understanding, cross-modal synthesis, CTS/MI/TIA, SynerMed dataset

## TL;DR
SynerMedGen proposes the "generation-aligned understanding" principle—deriving understanding tasks directly from the same paired synthetic data (via CTS, MI, and TIA tasks). By using a two-stage training process where the understanding branch first learns representations beneficial for synthesis, the model subsequently transfers these to a latent flow matching generation branch, outperforming both specialized synthesis models and existing unified MLLMs across 22 medical synthesis tasks.

## Background & Motivation

**Background**: Unified medical MLLMs (e.g., HealthGPT, UniMedVL) have begun integrating "understanding" and "generation" into a single model—where understanding handles tasks like VQA and report generation, while generation manages cross-modal synthesis such as CT↔MR or PET↔CT. Architecturally, these models typically adopt dual-pathway or connector+diffusion hybrid designs.

**Limitations of Prior Work**: Existing unified frameworks treat understanding and generation as **unrelated objectives**. The understanding side is trained with "recognition-style" tasks like lesion-level VQA, while the generation side is trained with pixel-level synthesis loss. Consequently, models may achieve high VQA scores but fail to preserve anatomical structures or correctly adjust contrast during cross-modal synthesis due to misaligned supervision.

**Key Challenge**: The "understanding" required for medical cross-modal synthesis involves **slice-level correspondence, modality identification, and transformation direction**, whereas traditional understanding supervision provides only global semantics, resulting in non-overlapping information across tasks.

**Goal**: To address a fundamental question often avoided in unified medical MLLMs: **"What kind of understanding is truly useful for generation?"** and design specific tasks based on this insight.

**Key Insight**: Since understanding should serve generation, **understanding tasks should be derived from the generation data itself**, naturally coupling the training signals. Simultaneously, a two-stage training strategy ensures that the multimodal priors learned in the first stage are naturally transferred to the generation stage via shared parameters.

**Core Idea**: Define the "generation-aligned understanding" principle → Design three understanding tasks directly corresponding to synthesis requirements (CTS for pairing, MI for modality control, TIA for transformation direction) → Execute two-stage training (understanding then generation) → Release the SynerMed dataset containing 1M paired samples and 2M understanding instances.

## Method

### Overall Architecture
Based on the Bagel unified architecture: an understanding encoder $E_{\text{ViT}}$ produces semantic tokens $\mathbf{z}_{\text{ViT}}$, and a generation encoder $E_{\text{VAE}}$ produces latent tokens $\mathbf{z}_{\text{VAE}}$. Both sets of tokens pass through a projection layer into a shared Mixture-of-Transformer-experts (MoT). The MoT contains two experts: an understanding expert for VLM prompted learning and a generation expert for conditional latent synthesis. **Stage I (GAU)**: The understanding expert is trained on three categories of generation-aligned understanding tasks. **Stage II (UCG)**: Flow matching is performed in the VAE latent space using the same paired data, with the VAE decoder reconstructing pixels. All understanding tasks are formalized as "generating short answer tokens prompted by the understanding expert," using a masked NTP loss calculated only on answer tokens: $\mathcal{L}_{\text{NTP}}(\mathbf{y}^*)=-\sum_i\log p_\theta(y_i^*\mid \mathbf{y}^*_{<i},\mathbf{x}_{\text{text}},\mathbf{z}_{\text{ViT}})$.

### Key Designs

1.  **Conditional Target Selection (CTS) — capturing slice-level pairing**:
    *   **Function**: Forces the model to select the target slice $x^+ = x_{\text{tgt}}$ that is truly paired with the source slice $x_{\text{src}}$ from $N$ candidates, under the constraint of an explicitly given target modality $m_{\text{tgt}}$.
    *   **Mechanism**: Formulated as a multiple-choice prompt where the model generates the correct option letter. A **key trick** is that hard negatives are not random slices, but **adjacent slices $\pm 1 \sim \pm K$ within the same target volume**—this forces the model to differentiate at a fine-grained anatomical level rather than relying on coarse semantics. Loss: $\mathcal{L}_{\text{CTS}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{CTS}})$.
    *   **Design Motivation**: Cross-modal synthesis requires "slice-by-slice" preservation of patient-specific anatomy and lesions; coarse VQA fails to learn fine-grained slice correspondence. Hard negatives from neighbors are the critical design for this capability.

2.  **Modality Identification (MI) — making modality an explicitly controllable factor**:
    *   **Function**: Requires the model to identify the modality of the input image (or each panel)—CT, CBCT, PET, or MRI (down to the MRI sequence).
    *   **Mechanism**: Uses the same prompted-generation framework where the model outputs modality labels. The task pool **intentionally includes easily confused pairs** (e.g., CT vs. CBCT, neighboring MRI contrasts) to force the model to learn genuine features of the "modality" variable. Loss: $\mathcal{L}_{\text{MI}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{MI}})$.
    *   **Design Motivation**: Cross-modal synthesis requires the "target modality" as a controllable input condition; if the modality is not explicitly encoded during the understanding stage, the generation stage must struggle to disentangle it from visual cues.

3.  **Transformation Instruction Alignment (TIA) — grounding transformation directions to text**:
    *   **Function**: Given a pair of aligned images $(x_1, x_2)$, the model must pick the correct "route description" from a set of short descriptions (e.g., "CT→MRI: change contrast, preserve anatomy").
    *   **Mechanism**: Each synthesis route maintains a description pool. A positive example $e^+$ is drawn from the ground-truth route, while $R-1$ distractors are drawn from other routes—**distractors specifically include "reversed direction" and "wrong modality pair" cases**. Loss: $\mathcal{L}_{\text{TIA}}=\mathcal{L}_{\text{NTP}}(\mathbf{y}^*_{\text{TIA}})$.
    *   **Design Motivation**: Slice correspondence and modality identification are insufficient; models may still confuse "what to change" versus "what to keep." TIA trains the ability to ground transformation semantics to text.

### Loss & Training
**Stage I (GAU)**: The understanding expert is jointly trained on three tasks with a total loss $\mathcal{L}_{\text{Stage I}}=\mathcal{L}_{\text{CTS}}+\mathcal{L}_{\text{MI}}+\mathcal{L}_{\text{TIA}}$. **Stage II (UCG)**: The generation expert performs conditional flow matching on VAE latents. Since the understanding expert and shared MoT are already "generation-friendly" from Stage I, parameters continue to be fine-tuned during generation training. The SynerMed dataset contains 1M paired synthesis samples and 2M derived understanding instances.

## Key Experimental Results

### Main Results
Comparison across 22 synthesis tasks including SynthRAD2023 (CBCT↔CT, MRI↔CT, PET↔CT) and BraTS (T1/T2/T1c/FLAIR conversions). Metric: SSIM (× 100):

| Task Direction | Pix2Pix | CycleGAN | BBDM | ResViT | SynDiff | RCD | HealthGPT | UniMedVL | **Ours** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Brain CBCT→CT | 66.17 | 53.32 | 71.09 | 85.00 | 85.47 | 85.97 | 57.37 | 51.48 | **87.15** |
| Pelvis CBCT→CT | 63.55 | 55.87 | 60.49 | 84.00 | 83.21 | 86.22 | 46.89 | 43.94 | **87.14** |
| Brain MRI→CT | 74.33 | 52.65 | 68.99 | 86.39 | 87.19 | 86.12 | 84.29 | 54.11 | **88.87** |
| Whole-Body CT→PET | 72.21 | 65.98 | 67.68 | 87.07 | 88.12 | 88.90 | 66.54 | 74.12 | **91.10** |
| BraTS T2→T1 | 59.34 | 57.19 | 56.77 | 86.94 | 88.31 | 88.01 | 60.13 | 77.26 | **90.58** |
| BraTS T1→T2 | 62.10 | 53.31 | 56.41 | 85.78 | 84.25 | 86.19 | 70.32 | 78.88 | **87.14** |

SynerMedGen ranks first in all 22 tasks, with a significant lead over unified model baselines (HealthGPT, UniMedVL), typically gaining +15~30 SSIM points.

### Ablation Study

| Configuration | Avg. SSIM Trend | Explanation |
| :--- | :--- | :--- |
| Full SynerMedGen (CTS+MI+TIA → UCG) | Optimal | All three tasks enabled |
| Traditional VQA-style understanding → UCG | Significant drop | Confirms "task misalignment" is the root cause |
| Stage I Only (GAU), zero-shot generation | Strong zero-shot | Synthesis achieved without generation training |
| W/O CTS | Drop in SSIM | Increased slice misalignment; unstable anatomy |
| W/O MI | Drop in SSIM | Failures in target modality control |
| W/O TIA | Drop in SSIM | Direction reversals and incorrectly modified regions |

### Key Findings
- **Strong zero-shot generation** is achieved on 22 tasks by training only on understanding without generation training. This striking result proves the "generation-aligned understanding" principle: the understanding stage acquires the representations needed for synthesis, which the generation stage merely "translates" back to pixels.
- The hard negative design in CTS (adjacent slices from the same volume) is irreplaceable; replacing it with random slices degrades the task to coarse semantic recognition, causing synthesis performance to plummet.
- SynerMedGen maintains its advantage in cross-dataset zero-shot tests, indicating that the representations derived from the three tasks are truly modality/task-agnostic universal priors.
- Unified MLLM baselines fail most significantly on subtle contrast synthesis like CBCT, corroborating that simply attaching a generation head to an understanding model without aligned supervision is ineffective.

## Highlights & Insights
- The principle that "understanding tasks must be derived from generation data" is a **transferable design principle**—it can be applied to any work involving unified understanding and generation (e.g., video, 3D assets, code).
- The three tasks correspond to "content, modality, and direction" priors, geometrically covering all necessary dimensions for cross-modal synthesis and serving as a **checklist for understanding task design**.
- The phenomenon of "Stage I alone yielding strong zero-shot generation" suggests that in unified models, what is shared is not just token representations but **multimodal priors implicit in the MoT shared parameters**, providing measurable evidence for the understanding-to-generation transfer.

## Limitations & Future Work
- Currently limited to cross-modal slice synthesis; higher-order tasks like 3D volumes or temporal (4D-CT) sequences are not yet covered.
- All three tasks use multiple-choice or classification formats; whether open-ended generative understanding (e.g., detailed reports) can further improve synthesis quality remains unexplored.
- Sequential training of Stage I and Stage II is computationally expensive for ultra-large datasets; end-to-end joint training is worth investigating.
- Domain shift robustness across different institutions or scanners was not extensively tested.

## Related Work & Insights
- **vs. HealthGPT**: HealthGPT uses task-specific adapters to connect understanding and generation separately, but supervision is misaligned. SynerMedGen improves supervision design without adding adapters, outperforming it by 15–30 SSIM.
- **vs. UniMedVL**: UniMedVL uses a progressive learning curriculum but sticks to traditional understanding tasks. SynerMedGen proves that curriculum is insufficient; supervision alignment is the key.
- **vs. General Domain Bagel/Show-o/Janus**: These models focus on architectural unification. This work aligns tasks at the "design level," making the approach orthogonal and potentially stackable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Deriving understanding from generation data" as a clean, instantiated principle is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ 22 synthesis tasks, zero-shot tests, and large-scale dataset release.
- Writing Quality: ⭐⭐⭐⭐ The three-part motivation (content/modality/direction) is very clear.
- Value: ⭐⭐⭐⭐ Establishes a new SOTA for unified medical MLLMs; the SynerMed dataset is a significant community contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)
- [\[ICML 2026\] Seizure-Semiology-Suite (S³): A Clinically Multimodal Dataset, Benchmark, and Models for Seizure Semiology Understanding](seizure-semiology-suite_s3_a_clinically_multimodal_dataset_benchmark_and_models_.md)
- [\[ICML 2026\] CAME-Grad: The Double Dilemma in Multi-Task Radiology Report Generation — A Gradient Dynamics Analysis and Solution](the_double_dilemma_in_multi-task_radiology_report_generation_a_gradient_dynamics.md)
- [\[CVPR 2026\] MedGEN-Bench: Contextually Entangled Benchmark for Open-Ended Multimodal Medical Generation](../../CVPR2026/medical_imaging/medgen-bench_contextually_entangled_benchmark_for_open-ended_multimodal_medical_.md)
- [\[CVPR 2026\] CURE: Curriculum-guided Multi-task Training for Reliable Anatomy Grounded Report Generation](../../CVPR2026/medical_imaging/cure_curriculum-guided_multi-task_training_for_reliable_anatomy_grounded_report_.md)

</div>

<!-- RELATED:END -->
