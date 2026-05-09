---
title: >-
  [Paper Note] cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning
description: >-
  [ICLR 2026][CAD reconstruction] cadrille is the first multi-modal CAD reconstruction model capable of handling point cloud, multi-view image, and text inputs simultaneously. Through a three-stage training paradigm of VLM backbone + SFT + RL fine-tuning, it achieves state-of-the-art performance across 10 CAD reconstruction benchmarks, with RL fine-tuning reducing the invalid rate to near 0%.
tags:
  - ICLR 2026
  - CAD reconstruction
  - multi-modal
  - reinforcement learning
  - VLM
  - code generation
date: 2026-05-08
content_hash: 4fe7acf9a72db3af
---

# cadrille: Multi-modal CAD Reconstruction with Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2505.22914](https://arxiv.org/abs/2505.22914)
**Code**: [https://github.com/col14m/cadrille](https://github.com/col14m/cadrille)
**Area**: Others
**Keywords**: CAD reconstruction, multi-modal, reinforcement learning, VLM, code generation

## TL;DR
cadrille is the first multi-modal CAD reconstruction model capable of handling point cloud, multi-view image, and text inputs simultaneously. Through a three-stage training paradigm of VLM backbone + SFT + RL fine-tuning, it achieves state-of-the-art performance across 10 CAD reconstruction benchmarks, with RL fine-tuning reducing the invalid rate to near 0%.

## Background & Motivation

**Background**: CAD models are critical in engineering and manufacturing. Existing CAD reconstruction methods primarily recover CAD models from a single modality—point clouds, images, or text. Recent work CAD-Recode represents CAD models as executable Python code, replacing the conventional special-token representation.

**Limitations of Prior Work**:
   - Single-modal methods are inherently limited—point clouds require specialized equipment, while image- and text-based methods each have their own shortcomings.
   - Existing multi-modal methods (CAD-GPT, CAD-MLLM) lag far behind single-modal SOTAs in quality.
   - SFT-trained models generalize poorly to cross-domain scenarios (e.g., real-world scan data), achieving only ~60% IoU on CC3D with an invalid rate approaching 10%.

**Key Challenge**: Manually annotated CAD datasets are small and limited in diversity; programmatically generated datasets are large but exhibit a domain gap with real data; naively mixing both data sources for SFT degrades performance due to inconsistent CAD operations.

**Goal**:
   - How to handle point cloud, image, and text modalities simultaneously within a unified framework?
   - How to improve cross-domain generalization without requiring large-scale manual annotations?
   - How to minimize the invalid rate in generated Python code?

**Key Insight**: Drawing inspiration from the LLM training paradigm (pre-training → SFT → RLHF), the paper introduces RL fine-tuning into CAD reconstruction. The RL stage requires only 3D meshes—not CAD sequence annotations—to compute IoU-based rewards, enabling the use of more readily available unannotated datasets.

**Core Idea**: Large-scale programmatically generated data is used in SFT to learn general CAD reconstruction capabilities, while scarce manually annotated or real-world data is used in RL fine-tuning to bridge the domain gap. A single unified model thereby surpasses all single-modal SOTAs across all three input modalities.

## Method

### Overall Architecture
Any one of point cloud / multi-view images / text is accepted as input → encoded by a VLM (Qwen2-VL-2B) → executable Python code is generated → executed to produce a parametric B-Rep 3D shape. Training proceeds in three stages: (1) pre-trained VLM (used as-is) → (2) SFT on programmatically generated data → (3) RL fine-tuning on manually annotated data.

### Key Designs

1. **Unified Multi-modal Architecture**:

    - **Function**: Unified processing of point cloud, image, and text modalities.
    - **Mechanism**: Built on Qwen2-VL-2B. Text and images are handled by the model's native embedding and visual encoder; point clouds are sampled via furthest point sampling and projected into the VLM's representation space through a single linear projection layer. All modalities produce Python code as output.
    - **Design Motivation**: VLMs natively support text and image processing and already possess Python code generation capability. Extending to three modalities requires only an additional point cloud projection layer, which is far more efficient than designing separate architectures for each modality.

2. **Staged Data Usage Strategy**:

    - **Function**: SFT leverages large-scale programmatically generated data (~1M samples from CAD-Recode); RL uses small-scale manually annotated data (DeepCAD + Fusion360).
    - **Mechanism**: Programmatically generated data is large and diverse but exhibits CAD operation inconsistencies with manually annotated data (e.g., DeepCAD includes symmetric extrusion and extruded cuts absent in CAD-Recode). Naively mixing both datasets for SFT causes mutual interference. The solution is to let SFT focus on learning general CAD reconstruction capabilities, while RL fine-tuning focuses on domain adaptation.
    - **Design Motivation**: RL fine-tuning requires only 3D meshes (not CAD sequence annotations) for IoU reward computation, enabling the use of datasets previously unusable for SFT (e.g., the Fusion360 training set).

3. **RL Fine-tuning Methods (DPO + Dr. CPPO)**:

    - **Function**: Two RL strategies—offline DPO and online Dr. CPPO—with the latter yielding substantially better results.
    - **Mechanism**:
        - **DPO**: For each input $q$, $K=5$ Python code samples are drawn from the SFT model. Two are randomly selected and ranked by reward to form preference pairs $(\tau_w, \tau_l)$, which are used to train with the standard DPO loss. Every 10 epochs, the reference model is replaced with the current model, allowing the policy to gradually diverge from the original SFT policy.
        - **Dr. CPPO**: Combines Dr. GRPO (reference-model-free) with CPPO (using only the highest-signal samples). For each input, $G$ sequences are sampled and advantages $A_g = r_g - \text{mean}(\{r_i\})$ are computed. The $N$ samples with the largest $|A_g|$ are selected to form the batch, and the policy is updated using PPO's clipped objective.
    - **Design Motivation**: DPO performance is bounded by the quality of the best sampled candidate. As an online method, Dr. CPPO continuously generates new samples, breaking this ceiling.

### Loss & Training
- **SFT stage**: Standard cross-entropy $\mathbb{E}_{(q,\tau)\sim\mathcal{D}}[\log \pi_\theta(\tau|q)]$
- **RL reward function**: $R(\tau) = r_{\text{IoU}}(\tau) + r_{\text{invalid}}(\tau)$, where the IoU reward is scaled by a factor of 10 and invalid code incurs a penalty of $-10$
- **Hard sample mining**: Only samples with a mean reward $< R_{th}=7.5$ across three SFT model samples are used for RL training, accelerating convergence
- **Single-modality RL improves multi-modality**: Performing RL fine-tuning on the image modality alone also improves point cloud and text modalities (via shared parameters)

## Key Experimental Results

### Main Results (DeepCAD test set, SFT stage)

| Method | Training Data | PC CD↓ | PC IoU↑ | PC IR↓ | Img CD↓ | Img IoU↑ | Img IR↓ |
|--------|--------------|--------|---------|--------|---------|---------|---------|
| CAD-SIGNet | Dp | 0.29 | 77.3 | 5.0 | - | - | - |
| CADCrafter | Di | - | - | - | 0.26 | - | 3.6 |
| CAD-Recode | Rp | 0.18 | 87.1 | 3.1 | - | - | - |
| **cadrille** | Dpit | 0.25 | 79.4 | **0.4** | 0.25 | 78.2 | **0.5** |
| **cadrille** | Rpi+Dt | **0.18** | **87.1** | 2.1 | **0.18** | **86.1** | 1.5 |

### RL Fine-tuning Results (Cross-dataset — Image Modality)

| Configuration | DeepCAD IoU↑ | Fusion360 IoU↑ | CC3D IoU↑ | CC3D IR↓ |
|--------------|-------------|---------------|----------|---------|
| SFT only (Rpi) | 86.1 | 77.6 | 56.1 | 7.7% |
| SFT + mixed data (Rpi+Dpi) | 85.6 | 75.2 | 53.1 | 6.0% |
| + DPO | 86.9 | 78.5 | 56.0 | 3.9% |
| + **Dr. CPPO** | **92.2** | **84.6** | **65.0** | **0.1%** |

### Key Findings
- **Remarkable RL fine-tuning gains**: Dr. CPPO improves CC3D IoU from 56.1% to 65.0% (+9%) and reduces IR from 7.7% to 0.1%.
- **Single-modality RL improves multi-modality**: RL fine-tuning on the image modality alone simultaneously improves point cloud reconstruction (CC3D point cloud IoU from 61.8% to 67.9%).
- **Mixed-data SFT hurts performance**: Naively mixing CAD-Recode and DeepCAD for SFT yields worse results on Fusion360 and CC3D than using CAD-Recode alone.
- **Online RL >> Offline RL**: DPO primarily reduces IR without substantially improving accuracy; Dr. CPPO achieves comprehensive gains across all metrics.
- **Hard sample mining is effective**: Restricting RL training to samples where the SFT model performs poorly accelerates convergence.

## Highlights & Insights
- **The SFT→RL divide-and-conquer strategy is particularly elegant**: Large-scale synthetic data is used in SFT to acquire general capabilities, while a small amount of real-world data drives RL-based domain adaptation. This avoids the data inconsistency issues inherent in conventional mixed training, and the RL stage requires no CAD sequence annotations (only meshes for IoU computation), substantially lowering data requirements. This strategy is transferable to any setting where synthetic data is abundant but real-world data is scarce.
- **Single-modality RL improving multi-modal performance** is a noteworthy finding—it suggests that RL enhances the model's internal capacity to generate high-quality CAD code, rather than merely improving its ability to interpret inputs from a specific modality. This implies that RL primarily acts on the decoding/generation side of the model.
- **The natural advantage of programmatic rewards**: Rewards for CAD reconstruction (IoU, code validity) can be computed precisely and automatically, making the task inherently well-suited for RL training—analogous to the use of verifiable rewards in LLM reasoning tasks such as DeepSeek-R1.

## Limitations & Future Work
- The model currently accepts only a single modality per input; combining multiple modalities within a single prompt for complementary information (e.g., image + partial text description) has not been explored.
- RL fine-tuning has not been directly applied to the point cloud modality; the observed improvements in point cloud reconstruction are indirect.
- The complexity of programmatically generated data is limited and unlikely to cover the full diversity of real-world CAD operations.
- The backbone is only 2B parameters; larger VLMs may yield further improvements.
- Evaluation is restricted to closed-set CAD reconstruction benchmarks; open-world zero-shot generalization has not been investigated.

## Related Work & Insights
- **vs. CAD-Recode**: Both represent CAD as Python code, but CAD-Recode is single-modal (point cloud) + SFT only; cadrille extends to three modalities and adds RL fine-tuning, improving IoU by 3–7%.
- **vs. CAD-MLLM**: Both are multi-modal CAD methods, but CAD-MLLM uses special tokens for CAD sequences and performs far below single-modal SOTAs; cadrille comprehensively surpasses it via Python code representation and RL.
- **vs. CADCrafter**: Both target image-to-CAD reconstruction, and both employ DPO; however, CADCrafter applies SFT and RL on the same dataset, while cadrille separates the SFT and RL data sources, yielding superior results.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of unified multi-modal processing and RL fine-tuning is pioneering in the CAD reconstruction domain, though the core techniques (VLM + RL) are not novel in themselves.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 10 benchmarks including real-world data (CC3D), with extensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, persuasive motivation, and information-dense tables.
- **Value**: ⭐⭐⭐⭐ Highly practical, open-sourced, and the SFT→RL divide-and-conquer strategy offers transferable insights for other tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/others/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[NeurIPS 2025\] MiCADangelo: Fine-Grained Reconstruction of Constrained CAD Models from 3D Scans](../../NeurIPS2025/others/micadangelo_fine-grained_reconstruction_of_constrained_cad_models_from_3d_scans.md)
- [\[ICLR 2026\] Distributionally Robust Classification for Multi-Source Unsupervised Domain Adaptation](distributionally_robust_classification_for_multi-source_unsupervised_domain_adap.md)
- [\[ICLR 2026\] Completing Missing Annotation: Multi-Agent Debate for Accurate and Scalable Relevance Assessment](completing_missing_annotation_multi-agent_debate_for_accurate_and_scalable_relev.md)
- [\[CVPR 2026\] Rooftop Wind Field Reconstruction Using Sparse Sensors: From Deterministic to Generative Learning Methods](../../CVPR2026/others/rooftop_wind_field_reconstruction_using_sparse_sen.md)

</div>

<!-- RELATED:END -->
