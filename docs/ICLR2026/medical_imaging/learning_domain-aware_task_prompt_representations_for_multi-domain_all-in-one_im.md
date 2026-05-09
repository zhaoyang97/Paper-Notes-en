---
title: >-
  [Paper Note] Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration
description: >-
  [ICLR 2026][Medical Imaging][All-in-one image restoration] This paper proposes DATPRL-IR, the first multi-domain all-in-one image restoration method, which learns domain-aware task prompt representations via a dual prompt pool (task prompt pool + domain prompt pool). Domain priors are distilled from an MLLM and injected into the backbone through adaptive gated fusion, achieving significant improvements over SOTA across 9 tasks spanning natural, medical, and remote sensing domains.
tags:
  - ICLR 2026
  - Medical Imaging
  - All-in-one image restoration
  - multi-domain restoration
  - prompt learning
  - dual prompt pool
  - cross-modal alignment
date: 2026-05-08
content_hash: 33eac1c515b9534a
---

# Learning Domain-Aware Task Prompt Representations for Multi-Domain All-in-One Image Restoration

**Conference**: ICLR 2026
**arXiv**: [2603.01725](https://arxiv.org/abs/2603.01725)
**Code**: [GitHub](https://github.com/GuangluDong0728/DATPRL-IR)
**Area**: Medical Imaging
**Keywords**: All-in-one image restoration, multi-domain restoration, prompt learning, dual prompt pool, cross-modal alignment

## TL;DR
This paper proposes DATPRL-IR, the first multi-domain all-in-one image restoration method, which learns domain-aware task prompt representations via a dual prompt pool (task prompt pool + domain prompt pool). Domain priors are distilled from an MLLM and injected into the backbone through adaptive gated fusion, achieving significant improvements over SOTA across 9 tasks spanning natural, medical, and remote sensing domains.

## Background & Motivation

**Background**: Existing all-in-one image restoration (AiOIR) methods (e.g., PromptIR, MoCE-IR) can handle multiple degradation tasks with a single model, but are confined to a single image domain (e.g., natural or medical images). No prior method simultaneously addresses multi-task restoration across multiple domains.

**Limitations of Prior Work**: (1) Images from different domains (natural, medical, remote sensing) exhibit distinct visual characteristics, making single-domain methods non-transferable; (2) existing methods focus on distinguishing task-specific differences while neglecting shared knowledge across tasks; (3) learning difficulty increases sharply as the number of tasks and domains grows.

**Key Challenge**: In a multi-domain, multi-task setting, models must simultaneously capture task-specific knowledge, domain-specific knowledge, and shared knowledge across both dimensions. Existing single-prompt or single-encoding mechanisms cannot effectively represent this hierarchical knowledge structure.

**Goal**: How can a single model handle diverse restoration tasks across three domains (natural, medical, remote sensing)? How can shared knowledge across tasks and domains be exploited to reduce learning difficulty?

**Key Insight**: Although images from different domains have unique characteristics, they also exhibit overlapping visual properties (e.g., "grayscale + anatomical structures" for medical; "aerial view + built structures" for remote sensing). A dual prompt pool is employed to encode task and domain knowledge separately, with instance-level adaptive combination and fusion.

**Core Idea**: A dual prompt pool is used to learn task- and domain-specific/shared knowledge independently. A prompt combination mechanism and cross-attention fusion generate domain-aware task prompt representations to guide multi-domain all-in-one restoration.

## Method

### Overall Architecture
Degraded input image → encoder-decoder backbone extracts features → intermediate features query the Task Prompt Pool (TP Pool) to obtain task representation $\mathbf{PR}_t$ → shallow features query the Domain Prompt Pool (DP Pool) to obtain domain representation $\mathbf{PR}_d$ → cross-attention fusion produces domain-aware task prompt representation $\mathbf{PR}_{dt}$ → Adaptive Gated Fusion (AGF) injects $\mathbf{PR}_{dt}$ into each backbone layer → restored image output.

### Key Designs

1. **Task Prompt Pool and Prompt Combination Mechanism (PCM)**:

    - **Function**: Implicitly encodes task-specific and shared knowledge across restoration tasks, and adaptively generates instance-level task representations for each input image.
    - **Mechanism**: A pool of $N_t=15$ key-value prompt pairs $(\mathbf{K}_j^{\text{task}}, \mathbf{V}_j^{\text{task}})$ is constructed. A learnable projector maps intermediate encoder features to a query $\mathbf{Q}^{\text{task}}$. The top-$k$ ($k=3$) prompts are selected by cosine similarity and combined via temperature-scaled softmax: $\mathbf{PR}_t = \sum_{j \in k} \alpha_j^{\text{task}} \mathbf{V}_j^{\text{task}}$. Prompts are jointly optimized with the restoration objective.
    - **Design Motivation**: Different tasks may share certain prompts (e.g., both super-resolution and deblurring require sharpening). PCM achieves a balance between knowledge sharing and task specificity through weighted combination rather than hard assignment.

2. **Domain Prompt Pool and MLLM Knowledge Distillation**:

    - **Function**: Learns domain-related visual priors and distills domain knowledge from a multimodal large language model.
    - **Mechanism**: A pool of $N_d=15$ domain prompts is constructed and queried using shallow features. During training, LLaVA-1.5-7B generates multi-perspective textual descriptions of high-quality images (content, color, objects, brightness, viewpoint). These descriptions are encoded by a CLIP text encoder to obtain $\mathbf{F}_{\text{text}}$, and domain priors are distilled into the domain prompt pool via a cross-modal alignment loss $\mathcal{L}_{\text{align}} = 1 - \cos(\mathbf{PR}_d, \mathbf{F}_{\text{text}})$. At inference time, neither LLaVA nor CLIP is required, introducing no additional overhead.
    - **Design Motivation**: Domain awareness requires understanding semantic-level image features (content type, capture modality, etc.). MLLMs possess strong image understanding capabilities and can provide rich domain descriptions. Distillation ensures that knowledge is acquired during training without incurring any inference-time cost.

3. **Adaptive Gated Fusion (AGF)**:

    - **Function**: Dynamically controls the fusion ratio between prompt representations and feature maps at each backbone layer.
    - **Mechanism**: Task and domain representations are fused into $\mathbf{PR}_{dt}$ via cross-attention. At each layer, a learnable gate $\alpha_l \in [0,1]$ controls the mixing ratio: $\mathbf{F}_l^e = \text{CrossAttn}(\alpha_l \mathbf{F}_l, (1-\alpha_l) \mathbf{PR}_{dt})$, allowing each layer to independently learn its optimal fusion strategy.
    - **Design Motivation**: Shallow layers may require more domain information (to identify input type), while deeper layers may require more task information (to execute specific restoration operations). A fixed fusion ratio is overly rigid for such varying demands.

### Loss & Training
The total loss is $\mathcal{L} = \lambda_{\text{pix}}\mathcal{L}_{\text{pix}} + \lambda_{\text{fft}}\mathcal{L}_{\text{fft}} + \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{bal}}\mathcal{L}_{\text{bal}} + \lambda_{\text{con}}\mathcal{L}_{\text{con}}$. $\mathcal{L}_{\text{pix}}$ and $\mathcal{L}_{\text{fft}}$ are $\ell_1$ losses in the RGB and Fourier domains, respectively; $\mathcal{L}_{\text{div}}$ encourages prompt diversity (cosine similarity threshold $\tau=0.1$); $\mathcal{L}_{\text{bal}}$ is a prompt usage balance regularizer (maximizing selection entropy). Adam optimizer with learning rate $4 \times 10^{-4}$ + cosine annealing, batch size 12, trained for 1000K iterations.

## Key Experimental Results

### Main Results

| Task / Dataset | Metric | DATPRL-IR (6T) | Prev. SOTA (MoCE-IR) | Gain |
|--------|------|------|----------|------|
| Natural SR / DIV2K-Val | PSNR | 28.98 | 28.16 | +0.82 |
| Deraining / Rain100L | PSNR | 39.56 | 38.64 | +0.92 |
| MRI SR / IXI MRI | PSNR | 27.88 | 27.75 | +0.13 |
| CT Denoising / AAPM-Mayo | PSNR | 33.80 | 33.74 | +0.06 |
| Remote Sensing SR / UCMerced | PSNR | 28.29 | 28.06 | +0.23 |
| Cloud Removal / CUHK CR1 | PSNR | 26.12 | 26.06 | +0.06 |
| **6-Task Average** | **PSNR** | **30.77** | **30.40** | **+0.37** |

### Ablation Study

| Configuration | Deraining PSNR | CT Denoising PSNR | Remote Sensing SR PSNR |
|------|---------|------|------|
| w/o TP + w/o DP (baseline) | 38.34 | 33.70 | 28.02 |
| TP Pool only | 39.32 | 33.76 | 28.16 |
| DP Pool only | 38.88 | 33.74 | 28.12 |
| TP + DP (full model) | 39.56 | 33.80 | 28.29 |

### Key Findings
- Scaling from 6 tasks to 9 tasks does not degrade performance on existing tasks; in fact, performance improves (e.g., natural SR: 28.98 → 29.05), confirming the existence of transferable shared knowledge across tasks.
- Replacing the MLLM with models of varying scales (LLaVA-7B/13B, Qwen3-VL-2B) has negligible impact on performance, indicating that the method relies only on coarse-grained domain semantics.
- Replacing the domain prompt pool with fixed text prompts (e.g., "This is an MRI image") degrades performance, validating the necessity of adaptive selection and shared knowledge modeling.
- A prompt pool size of 15 and top-$k$ of 3/5 constitute the optimal configuration; performance degrades with values that are too large or too small.

## Highlights & Insights
- This work is the first to extend all-in-one image restoration to a multi-domain setting. The proposed dual prompt pool architecture elegantly decouples the learning of task and domain knowledge, achieving an adaptive balance between shared and task-specific knowledge through PCM. The property that adding more tasks does not degrade existing task performance demonstrates strong scalability.
- The design of distilling domain priors from an MLLM is particularly elegant: LLaVA's strong comprehension capability is leveraged during training, while inference requires no MLLM at all, achieving "free" domain awareness.

## Limitations & Future Work
- Domain coverage is currently limited to three domains (natural, medical, remote sensing); scalability to additional domains (e.g., underwater, night vision, satellite) remains to be validated.
- The prompt pool size and top-$k$ require manual tuning; an adaptive mechanism for these hyperparameters is absent.
- Evaluation relies solely on PSNR/SSIM; perceptual quality metrics (e.g., LPIPS) and downstream task evaluations are not included.

## Related Work & Insights
- **vs. PromptIR**: PromptIR employs a single learnable prompt to encode degradation information. The proposed method achieves more flexible instance-level representations via a prompt pool with PCM, while additionally introducing a domain-awareness dimension.
- **vs. MoCE-IR**: MoCE-IR uses a mixture-of-experts architecture to allocate task-specific resources. The proposed dual prompt pool "query–retrieve–combine" paradigm achieves similar functionality but is more lightweight, outperforming MoCE-IR by 0.37 dB on average across 6 tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First multi-domain all-in-one restoration method; dual prompt pool + MLLM distillation design is innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive experiments across 3 domains and 9 tasks, with thorough ablation and scalability validation.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, rich figures and tables, well-articulated motivation.
- **Value**: ⭐⭐⭐⭐ Multi-domain unified restoration has significant practical implications; the dual prompt pool is transferable to other multi-domain, multi-task scenarios.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Adaptive Domain Shift in Diffusion Models for Cross-Modality Image Translation](adaptive_domain_shift_in_diffusion_models_for_cross-modality_image_translation.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](../../CVPR2026/medical_imaging/interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[AAAI 2026\] Learning Cell-Aware Hierarchical Multi-Modal Representations for Robust Molecular Modeling](../../AAAI2026/medical_imaging/learning_cell-aware_hierarchical_multi-modal_representations.md)
- [\[CVPR 2026\] Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization](../../CVPR2026/medical_imaging/human_knowledge_integrated_multi-modal_learning_for_single_source_domain_general.md)
- [\[CVPR 2026\] Mind the Discriminability Trap in Source-Free Cross-domain Few-shot Learning](../../CVPR2026/medical_imaging/mind_the_discriminability_trap_in_source-free_cross-domain_few-shot_learning.md)

<!-- RELATED:END -->
