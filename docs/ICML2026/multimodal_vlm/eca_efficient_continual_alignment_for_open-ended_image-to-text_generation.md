---
title: >-
  [Paper Note] ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation
description: >-
  [ICML2026][Multimodal VLM][Continual Alignment] ECA proposes exemplar-free incremental learning on the "alignment module" (the Q-Former in BLIP-2) of pretrained VLMs. By utilizing Mixture-of-Query to compositionally aggregate task-specific queries per image, expanding parallel adapters on-demand based on a Fisher Information Matrix criterion, and preserving prior knowledge via sparse dictionary replay, the method successfully learns new subjects without catastrophic forgettin…
tags:
  - "ICML2026"
  - "Multimodal VLM"
  - "Continual Alignment"
  - "Exemplar-free Incremental Learning"
  - "Open-Ended Image-to-Text Generation"
  - "Q-Former"
  - "Fisher Information Matrix"
date: 2026-05-08
content_hash: 962a19e5cb16e887
---

# ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation

**Conference**: ICML2026  
**arXiv**: [2606.12633](https://arxiv.org/abs/2606.12633)  
**Code**: https://github.com/Snowball0823/ECA  
**Area**: Multimodal VLM / Incremental Learning  
**Keywords**: Continual Alignment, Exemplar-free Incremental Learning, Open-Ended Image-to-Text Generation, Q-Former, Fisher Information Matrix

## TL;DR
ECA proposes exemplar-free incremental learning on the "alignment module" (the Q-Former in BLIP-2) of pretrained VLMs. By utilizing Mixture-of-Query to compositionally aggregate task-specific queries per image, expanding parallel adapters on-demand based on a Fisher Information Matrix criterion, and preserving prior knowledge via sparse dictionary replay, the method successfully learns new subjects without catastrophic forgetting in open-ended image-to-text generation tasks where visual topics drift over time.

## Background & Motivation
**Background**: Open-ended image-to-text generation (OpenITG, e.g., image captioning, open-ended VQA) relies on VLMs to transform images into context-aware text. Modern VLMs typically freeze the visual encoder and LLM, using only an **alignment module** (e.g., Q-Former in BLIP-2, projector in LLaVA) to bridge visual features to the LLM token space. In real-world scenarios, visual content drifts with environment and time, necessitating incremental learning (IL) for OpenITG.

**Limitations of Prior Work**: Existing OpenITG incremental methods suffer from three major flaws. First, they assume **disjoint classes/backgrounds** between tasks and discard images containing multiple topics—yet real-world images often feature co-existing objects where dominant semantics fluctuate (e.g., an indoor scene focusing first on "appliances" and later on "vehicles"). Second, most methods rely on storing raw samples and full fine-tuning of fusion/language components to resist forgetting, which is inefficient, erodes pretraining gains, and poses privacy and memory risks. Third, the assumption of disjoint distributions prevents them from handling **semantic overlap between tasks**.

**Key Challenge**: Under the realistic setting of "tasks partitioned by main topic, semantic overlap, and no sample buffer," the model must simultaneously maintain cross-modal alignment, resist catastrophic forgetting, and avoid parameter conflicts caused by overlapping semantics. The authors decompose this into three specific challenges: **C1**: Recurring semantics appear without task identifiers, requiring compositional reuse of early cues; **C2**: Cross-modal alignment must be preserved under distribution drift without storing raw samples; **C3**: Semantic overlap between tasks triggers parameter conflicts that must be suppressed during adaptation.

**Goal**: To propose the new concept of **continual alignment**—incrementally adapting only the alignment module in the VLM while keeping large frozen backbones intact, thereby efficiently preserving high-quality cross-modal representations.

**Key Insight**: The authors use the Q-Former of BLIP-2 as an isolatable carrier for the alignment module (positioned between the frozen visual encoder and frozen LLM), allowing "continual alignment" to be observed and optimized independently.

**Core Idea**: Implement exemplar-free IL solely on the alignment module using a three-part suite: MoQ for C1, DR for C2, and FeDEx for C3.

## Method

### Overall Architecture
ECA is implemented on BLIP-2: the frozen visual encoder outputs patch embeddings, which **Mixture of Query (MoQ)** uses to learn task-specific query tokens and aggregate them via image-based attention. These are fed to the Q-Former equipped with **Fisher Dynamic Expansion (FeDEx)**. FeDEx determines whether to initialize a new parallel adapter for the current task based on a Fisher Information Matrix (FIM) criterion, absorbing new features while preserving existing alignment. Meanwhile, **Dictionary Replay (DR)** maintains an embedding dictionary and replays it during training to distill knowledge from previous tasks. This pipeline only updates the alignment module (12.29M trainable parameters), while the visual encoder and LLM remain frozen throughout, requiring no raw sample buffer or task IDs.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Image<br/>Frozen Visual Encoder → Patch Embeddings"] --> B["Mixture of Query<br/>Task-specific query tokens aggregated via image attention"]
    B --> C["Q-Former Alignment Module<br/>Loaded with Fisher Dynamic Expansion"]
    C -->|S(ω)>0.5 Interference Detected| D["Create New Parallel Adapter<br/>Freeze old PAs, average all PA outputs"]
    C -->|S(ω)≤0.5 Reusable| E["Reuse Current Parallel Adapter"]
    D --> F["Soft Visual Prompts → Frozen LLM generates text"]
    E --> F
    G["Dictionary Replay<br/>Sparse Dictionary Learning + KD Replay"] -.Preserve alignment during training.-> C
```

### Key Designs

**1. Mixture of Query (MoQ): Compositional Task-Specific Queries via Image-Attention**

Addressing C1. The Q-Former relies on learnable query tokens $Q_t$ to expose visual evidence to the frozen LLM. In IL, updating $Q_t$ based only on the current task overwrites historical cues. MoQ learns a set of queries $v_t$ and a task key $k_t$ for each task. For every image, it takes the mean patch embedding $\overline{e}_{t,i}$ and uses attention to **dynamically aggregate weighted queries** from all tasks, adding them to the fixed pretrained query $Q_\star$:

$$Q_{t,i}=Q_\star+\mathrm{Attention}(\overline{e}_{t,i},K_t,V_t),$$

where $K_t=[k_1,\dots,k_t]$ and $V_t=[v_1,\dots,v_t]$. To prevent entanglement, $V_{<t}, K_{<t}, Q_\star$ are fixed, and an orthogonality constraint $\mathcal{L}_{\text{orth}}=\|v_tV_{<t}^\top\|_F^2+\|k_tK_{<t}^\top\|_F^2$ is applied. A key alignment loss $\mathcal{L}_{\text{key}}$ ensures $k_t$ aligns with the current task's visual embedding direction.

**2. Fisher Dynamic Expansion (FeDEx): FIM-based Criterion for On-demand Expansion**

Addressing C3. Full fine-tuning is costly and destructive, so Parallel Adapters (PA) are used for parameter-efficient fine-tuning. However, a single PA has limited capacity, while creating a new PA per task destroys positive transfer from overlapping features. FeDEx introduces a principled criterion based on a second-order Taylor expansion and Fisher approximation:

$$S(\omega_t)=\frac{I_+(\omega_t)}{I_+(\omega_t)+|I_-(\omega_t)|}\in[0,1],$$

where $I_+$ and $I_-$ represent the gain and harm of an update to old task data $\mathcal{D}_t$. A theorem states: **when $S(\omega_t)\le 0.5$, training on the new task does not harm old tasks ($\Delta\mathcal{L}_{\mathcal{D}_t}\le 0$)**. Thus, a new PA is only created when $S(\omega_t)>0.5$ (interference detected). This ensures capacity is spent only during genuine conflicts, while overlapping tasks continue sharing PAs to maintain positive transfer.

**3. Dictionary Replay (DR): Sparse Dictionary as Compact Memory for Exemplar-Free Replay**

Addressing C2. Exemplar-free IL lacks old data, and traditional "one prototype per class" fails in OpenITG due to dispersed visual embeddings. DR learns an overcomplete dictionary $D\in\mathbb{R}^{m\times d_v}$ ($m\gg d_v$), allowing any patch embedding $e_k$ to be represented as a sparse linear combination of dictionary atoms via non-negative Lasso:

$$\alpha_k=\arg\min_{\alpha}\tfrac12\|e_k-D_{t-1}^\top\alpha\|_F^2+\gamma\|\alpha\|_1,\ \text{s.t.}\ \alpha\ge 0.$$

When training task $t+1$, the dictionary $D_t$ is fed into both the "frozen old parameters" and "current parameters," and a knowledge distillation loss aligns their outputs:

$$\mathcal{L}_{\text{DR}}(\Omega_{t+1})=\tfrac1m\|A(D_t;sg(\Omega_t))-A(D_t;\Omega_{t+1})\|_F^2.$$

The dictionary compresses old visual components into compact, replayable memories, preserving alignment without storing raw images.

### Loss & Training
Within task $t$, the model first utilizes $S(\omega_t)$ to decide PA expansion, then jointly optimizes Q-Former parameters $\omega_t$, keys $k_t$, and query values $v_t$:

$$\mathcal{L}=\mathcal{L}_{\text{ce}}+\mathcal{L}_{\text{MoQ}}+\lambda\mathcal{L}_{\text{DR}},$$

where $\mathcal{L}_{\text{ce}}$ is the cross-entropy loss for generation, and $\lambda$ balances the dictionary replay term. Following task training, the embedding dictionary is updated via dictionary learning.

## Key Experimental Results

### Main Results
The authors re-partitioned four OpenITG benchmarks based on "Main Topic"—ToS-COCO Caption, ToS-VQAv2, ToS-TextCaps, and ToS-TextVQA—retaining realistic semantic overlaps. The backbone is BLIP-2 with a frozen visual encoder and LLM. Metrics include Avg (Final Average), BWT (Backward Transfer/Forgetting), and FWT (Forward Transfer).

| Method | Trainable Params | COCO CIDEr (Avg↑) | BWT↑ | VQAv2 Acc (Avg↑) | BWT↑ | FWT↑ |
|------|------|------|------|------|------|------|
| ZeroShot | 0 M | 104.65 | – | 48.33 | – | – |
| Vanilla (PA) | 12.29 M | 123.00 | -4.50 | 64.39 | -2.00 | 12.02 |
| LwF | 12.29 M | 123.88 | -3.78 | 64.92 | -0.99 | 14.65 |
| Dual-Prompt | 14.30 M | 123.59 | -1.60 | 65.03 | 1.27 | 12.74 |
| CODA-Prompt | 15.41 M | 124.20 | -1.19 | 65.64 | 1.38 | 13.71 |
| MoE-LoRA | 98.84 M | 122.77 | -3.53 | 61.02 | -3.90 | 10.27 |
| **ECA (Ours)** | **12.29 M** | **125.56** | **-1.86** | **68.05** | **1.81** | **16.38** |
| Upper-bound (PA) | 12.29 M | 126.91 | – | 68.18 | – | – |

ECA uses the same 12.29M trainable parameters as the smallest baseline but achieves an Avg of 68.05 on VQAv2, significantly outperforming CODA-Prompt (65.64) and approaching the Joint-Training upper bound (68.18).

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Full ECA (MoQ+FeDEx+DR) | VQAv2 Avg 68.05 / BWT 1.81 | Synergy of components; resists forgetting with high transfer. |
| w/o MoQ | Loss of compositional queries | C1 failure; new and old cues overwrite each other. |
| w/o FeDEx (Fixed single/multi PA) | Improper capacity | Single PA hits capacity; blind expansion ruins positive transfer. |
| w/o DR | No old knowledge replay | C2 failure; alignment lost under distribution drift. |

### Key Findings
- **Alignment Module Tuning is Sufficient**: ECA surpasses MoE-LoRA (98.84M) and full Q-Former tuning (107M) with only 12.29M parameters, validating the "continual alignment" hypothesis.
- **Positive BWT as Evidence of Forgetting Resistance**: While Vanilla(PA) shows a BWT of -2.00 (forgetting), ECA raises this to +1.81, indicating that learning new tasks slightly benefits old ones.
- **Fisher Threshold 0.5 is Theoretically and Empirically Optimal**: The theorem guarantees that $S\le 0.5$ prevents harm to old tasks, and empirical scans confirm its peak performance at this value.

## Highlights & Insights
- **Focusing IL on the "Alignment Module"**: By keeping the visual encoder and LLM frozen and learning only on the Q-Former/projector, ECA saves computation while preserving pretraining gains. This paradigm is transferable to any projector-based MLLM (e.g., LLaVA).
- **FeDEx's Fisher Decision for Expansion**: While dynamic expansion is common, deciding "when to expand" based on $S(\omega)>0.5$ elevates capacity allocation from a heuristic to a theoretically grounded decision.
- **Sparse Dictionary as Memory**: Using an overcomplete dictionary + sparse coding to capture essential visual components for distillation is a clever design for exemplar-free replay in dispersed embedding spaces.

## Limitations & Future Work
- **Primary Validation on BLIP-2 / Q-Former**: Although theoretically generalizable to all projector-based MLLMs, the main experiments focus on Q-Former; results on LLaVA-style architectures are needed.
- **Linear Growth of PAs and Queries**: PAs and query sets grow with the number of detected interference events and tasks, potentially accumulating storage and inference costs over very long task sequences.
- **FIM Threshold Dependencies**: The non-degradation theorem relies on small-update and Fisher approximations; the robustness of the 0.5 threshold under large-step training relies primarily on empirical evidence.
- **Custom Benchmarks**: The four ToS-* benchmarks are re-partitioned by the authors, making them realistic but potentially limiting external comparability until the community adopts them.

## Related Work & Insights
- **vs VQACL (Zhang et al. 2023)**: VQACL uses prototype learning and sample buffers, posing privacy issues and assuming disjoint tasks. ECA is exemplar-free and handles semantic overlap under visual drift.
- **vs Prompting (Dual-Prompt / CODA-Prompt)**: While these are strong for single-modal classification, ECA outperforms them in OpenITG by targeting the alignment module specifically rather than adding generic prompts.
- **vs Multimodal Instruction Tuning (Continual LLaVA / MoE-LoRA)**: These focus on **textual instruction** drift. ECA targets **visual topic** drift and achieves better results with far fewer parameters (12.29M vs 98.84M).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Continual alignment" concept + main-topic setting + Fisher expansion criterion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four benchmarks, multiple baselines, and ablation studies, though focused on BLIP-2.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mapping between the C1/C2/C3 challenges and the MoQ/DR/FeDEx solutions.
- Value: ⭐⭐⭐⭐ The paradigm of efficient continual learning on the alignment module is highly practical for VLM deployment and incremental updates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] OpenING: A Comprehensive Benchmark for Judging Open-ended Interleaved Image-Text Generation](../../CVPR2025/multimodal_vlm/opening_a_comprehensive_benchmark_for_judging_open-ended_interleaved_image-text_.md)
- [\[ICLR 2026\] Decoding Open-Ended Information Seeking Goals from Eye Movements in Reading](../../ICLR2026/multimodal_vlm/decoding_open-ended_information_seeking_goals_from_eye_movements_in_reading.md)
- [\[CVPR 2026\] GUIDE: A Benchmark for Understanding and Assisting Users in Open-Ended GUI Tasks](../../CVPR2026/multimodal_vlm/guide_a_benchmark_for_understanding_and_assisting_users_in_open-ended_gui_tasks.md)
- [\[ECCV 2024\] Towards Open-ended Visual Quality Comparison](../../ECCV2024/multimodal_vlm/towards_open-ended_visual_quality_comparison.md)
- [\[CVPR 2026\] Text-Printed Image: Bridging the Image-Text Modality Gap by "Printing" Text into Images](../../CVPR2026/multimodal_vlm/text-printed_image_bridging_the_image-text_modality_gap_for_text-centric_trainin.md)

</div>

<!-- RELATED:END -->
