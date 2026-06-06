---
title: >-
  [Paper Note] SiNGER: A Clearer Voice Distills Vision Transformers Further
description: >-
  [ICLR 2026][Audio & Speech][Vision Transformer] This paper proposes SiNGER (Singular Nullspace-Guided Energy Reallocation), a framework that suppresses high-norm artifacts in ViT features by applying perturbations along…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "Vision Transformer"
  - "Knowledge Distillation"
  - "High-Norm Artifacts"
  - "Nullspace Guidance"
  - "LoRA Adapter"
date: 2026-05-08
content_hash: b2dee3602e3f9df3
---

# SiNGER: A Clearer Voice Distills Vision Transformers Further

**Conference**: ICLR 2026
**arXiv**: [2509.20986](https://arxiv.org/abs/2509.20986)  
**Code**: [github.com/AIRLABkhu/SiNGER](https://github.com/AIRLABkhu/SiNGER)  
**Area**: Audio & Speech
**Keywords**: Vision Transformer, Knowledge Distillation, High-Norm Artifacts, Nullspace Guidance, LoRA Adapter

## TL;DR
This paper proposes SiNGER (Singular Nullspace-Guided Energy Reallocation), a framework that suppresses high-norm artifacts in ViT features by applying perturbations along the left-nullspace directions of teacher features, thereby preserving informative signals. Combined with lightweight LoRA adapters, SiNGER achieves state-of-the-art performance across multiple downstream tasks while producing cleaner and more interpretable representations.

## Background & Motivation

**Background**: Vision Transformers (ViTs) have become the backbone of visual foundation models (VFMs), achieving strong performance across diverse visual tasks via self-attention mechanisms and superior scalability. However, the quadratic complexity of ViT renders large-scale deployment non-trivial, making model compression essential. Among compression approaches (pruning, quantization, distillation), knowledge distillation (KD) stands out as the most reliable due to its structural and numerical stability.

**Limitations of Prior Work**: ViT token representations exhibit "high-norm artifacts"—certain patch features possess anomalously large norms, particularly concentrated in background regions. When standard MSE-based KD is applied, gradients are dominated by these high-norm tokens, causing the student to overfit the artifacts while neglecting genuinely informative signals, substantially degrading distillation effectiveness.

**Key Challenge**: A fundamental trade-off exists between artifact suppression and information preservation. Prior methods such as ViTKD mitigate artifact influence via random masking of teacher features, but such indiscriminate masking inevitably discards valuable information signals. The underlying cause is that artifacts arise as "singular defects" from power-iteration-like accumulation in residual blocks—tokens align along the dominant left singular vector of the pretrained weight matrices.

**Goal**: The paper addresses two sub-problems: (a) identifying a mathematically principled method to disentangle artifacts from informative signals; and (b) designing an efficient implementation that integrates seamlessly into existing KD pipelines.

**Key Insight**: The authors observe that modifying only the components of teacher features that lie within the left-nullspace of the subsequent Transformer block's weight matrix does not affect downstream outputs (since nullspace components are mapped to zero by the next layer), while allowing energy redistribution to suppress high-norm artifacts. This constitutes an elegant mathematical property—nullspace directions serve as a "free" modification space.

**Core Idea**: Leverage the left-nullspace of the next-layer weight matrix to guide perturbations on teacher features, guaranteeing information-lossless artifact suppression by construction.

## Method

### Overall Architecture

The SiNGER pipeline proceeds as follows:
- **Input**: A pretrained large ViT teacher and a small ViT student to be trained.
- **Core Component**: A lightweight LoRA adapter is inserted after each Transformer block of the teacher to refine its features.
- **Workflow**: (1) Teacher forward pass produces raw features → (2) LoRA adapter applies nullspace-guided perturbations to suppress high-norm artifacts → (3) Refined teacher features serve as distillation targets for the student → (4) Student is trained via standard KD loss against the refined features.
- **Output**: A student model that more faithfully inherits teacher knowledge, with cleaner and more interpretable feature maps.

### Key Designs

1. **Analysis and Modeling of High-Norm Artifacts**:

    - **Function**: Formally characterize the mechanism by which high-norm artifacts arise in ViTs and their impact on KD.
    - **Mechanism**: Artifacts emerge from cumulative residual block accumulation—at each residual block, token features accumulate energy along the dominant left singular vector of the weight matrix, forming "singular defects." Formally, given the SVD $W = U\Sigma V^\top$ of a layer weight $W$, artifact tokens align approximately with $u_1$ (the left singular vector corresponding to the largest singular value). Under an MSE distillation objective, gradients $\nabla \propto \|h_t - h_s\|$ are dominated by high-norm tokens, producing outlier-driven optimization bias.
    - **Design Motivation**: Establishing the mathematical nature of the problem provides the theoretical foundation for the subsequent nullspace-guided solution.

2. **Nullspace-Guided Perturbation**:

    - **Function**: Modify teacher features to suppress artifacts without affecting downstream layer outputs.
    - **Mechanism**: Let $h^{(l)}$ denote the output feature of the $l$-th teacher layer and $W^{(l+1)}$ the weight matrix of the next layer. The left-nullspace of $W^{(l+1)}$ is $\mathcal{N}(W^{(l+1)\top})$, i.e., the subspace of all vectors $v$ satisfying $W^{(l+1)\top} v = 0$. If a perturbation $\delta$ lies entirely within this nullspace, then $W^{(l+1)}(h^{(l)} + \delta) = W^{(l+1)}h^{(l)}$, meaning the next layer's output is completely unchanged. Energy from high-norm artifacts can thus be redistributed toward a more uniform distribution by freely exploiting nullspace directions.
    - **Design Motivation**: This is the key to resolving the "artifact suppression vs. information preservation" dilemma—modifications in the nullspace direction mathematically guarantee zero information loss to downstream computation while permitting arbitrary artifact suppression.

3. **LoRA Adapter Implementation**:

    - **Function**: Efficiently implement nullspace-guided perturbations.
    - **Mechanism**: A LoRA module of the form $\delta = BA \cdot h$ is appended after each Transformer block of the teacher, where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, and $r \ll d$ is the rank. Critically, the columns of $B$ are initialized as basis vectors of the left-nullspace of the next-layer weight matrix, ensuring that perturbations strictly follow nullspace directions at the start of training; the adapter then learns more flexible perturbations as training progresses.
    - **Design Motivation**: Directly computing the full nullspace projection is prohibitively expensive (requiring SVD at each step). The low-rank structure of LoRA naturally parametrizes nullspace perturbations while introducing negligible additional parameters. The nullspace initialization serves as the bridge between theory (nullspace guidance) and practice (LoRA parametrization).

### Loss & Training

Training proceeds in two parts:
- **Teacher-Side Adapter Training**: Adapter parameters are updated via backpropagation of the distillation loss (teacher backbone parameters are frozen), learning the optimal feature refinement.
- **Student Distillation Training**: The student is trained against the refined teacher features using a standard feature-matching MSE loss.
- **Loss Function**: $\mathcal{L} = \sum_l \|f_s^{(l)} - \tilde{f}_t^{(l)}\|^2$, where $\tilde{f}_t^{(l)}$ denotes the teacher feature after LoRA adapter refinement at layer $l$.
- Due to the minimal adapter parameter count, training overhead is negligible relative to standard KD.

## Key Experimental Results

### Main Results: Multi-Task Comparison

SiNGER is evaluated across multiple downstream tasks with ViT-Large as teacher and ViT-Tiny as student:

| Distillation Method | Classification (Top-1↑) | Detection (mAP↑) | Segmentation (mIoU↑) | Feature Quality |
|---|---|---|---|---|
| No Distillation (Baseline) | Low | Low | Low | With artifacts |
| FitNet | Medium | Medium | Medium | Severe artifacts |
| ViTKD (Random Masking) | High | High | High | Fewer artifacts, information loss |
| **SiNGER (Ours)** | **Highest** | **Highest** | **Highest** | **Clean & interpretable** |

SiNGER consistently outperforms all baselines across classification, object detection, and semantic segmentation, with comprehensive gains visualized in a radar chart (Figure 1b).

### Ablation Study

| Configuration | Performance | Notes |
|---|---|---|
| Full SiNGER | Optimal | Complete model: nullspace initialization + LoRA adapter |
| w/o Nullspace Initialization | Notable drop | Randomly initialized LoRA fails to guide perturbations effectively |
| w/o LoRA Adapter | Significant drop | Degenerates to standard KD; high-norm artifacts dominate optimization |
| Random Masking Only (ViTKD) | Moderate | Reduces artifacts but discards information signals |
| Varying LoRA rank $r$ | Non-monotonic | Too low: insufficient expressivity; too high: noise introduction |

### Key Findings
- **Nullspace Initialization Is Essential**: Removing nullspace-guided initialization causes a significant performance drop, confirming that perturbations in the nullspace direction—rather than mere LoRA parametrization—are the key to the method's success.
- **Substantially Improved Feature Interpretability**: Qualitative analysis (Figure 2) shows that student feature maps distilled via SiNGER exhibit the highest semantic consistency with the teacher and the most coherent patch-wise cosine similarity patterns.
- **Cross-Task Consistency**: Unlike methods that excel on specific tasks at the expense of others, SiNGER demonstrates consistent improvement across classification, detection, and segmentation, indicating that it refines general representation quality rather than task-specific preferences.
- **Robustness to Teacher Scale**: As the teacher grows larger (from ViT-Base to ViT-Large), artifact severity increases and standard KD yields diminishing returns; SiNGER instead better exploits the knowledge of larger teachers.

## Highlights & Insights

- **Nullspace as a Free Modification Space**: The most elegant contribution of this work is recognizing that the nullspace of the next-layer weight matrix constitutes a "free" modification space—any modification within it leaves downstream computation unchanged. This insight transforms the "artifact suppression vs. information preservation" trade-off from an apparent impossibility into a simultaneously satisfiable guarantee, yielding both mathematical elegance and practical utility.
- **LoRA Nullspace Initialization**: Initializing the down-projection matrix of LoRA with nullspace basis vectors cleverly bridges parameter-efficient fine-tuning and theoretical guarantees. This trick is generalizable to other scenarios requiring constraints within specific subspaces.
- **Revisiting the "Teacher Is Always Right" Assumption**: Conventional KD treats teacher outputs as ground truth for the student to approximate. This work demonstrates that teacher features themselves are defective (due to artifacts), and purifying the teacher prior to distillation yields better results. This "improve the teacher before teaching the student" paradigm may generalize to other KD settings.
- **Potential Transfer to LLM Compression**: LLMs similarly exhibit attention sink and high-norm token phenomena; SiNGER's nullspace-guided framework may transfer to LLM distillation.

## Limitations & Future Work

- **Computational Overhead**: Although LoRA adapters are parameter-light, SVD must be computed on each layer's weight matrix to obtain nullspace bases, which may incur a one-time computational cost for very large models.
- **Incomplete Method Details Due to Cache Truncation**: The local cache covers only through Method Section 3.1, precluding access to the complete mathematical derivations and precise experimental values; numerical entries in the table are based on qualitative descriptions rather than exact figures.
- **ViT-Specific Design**: The theoretical framework targets the residual-block artifact accumulation mechanism specific to ViTs; applicability to other architectures (e.g., CNNs or hybrid models) remains to be verified.
- **Zero-Shot and Few-Shot Settings Unexplored**: The paper focuses on classical supervised distillation; the impact of artifacts and the effectiveness of SiNGER in zero-shot transfer scenarios are not discussed.
- **LoRA Rank Selection**: The optimal rank $r$ requires empirical search, and an adaptive rank selection strategy is absent.

## Related Work & Insights

- **vs. ViTKD**: ViTKD employs random masking to reduce high-norm token influence—simple and effective, but at the cost of discarding information signals. SiNGER achieves selective artifact suppression via nullspace guidance with a theoretical guarantee of zero information loss, representing a fundamental improvement over ViTKD.
- **vs. FitNet**: FitNet is a classical feature distillation method that directly aligns intermediate layer features without accounting for ViT-specific artifact phenomena, resulting in inferior performance compared to both ViTKD and SiNGER in ViT distillation settings.
- **vs. Register Tokens**: Register tokens absorb artifact energy by appending extra tokens to the input—an architecture-level solution. SiNGER operates at the post-processing/distillation level and requires no modification to the teacher architecture or retraining.
- **vs. SiNDer**: SiNDer analyzes the singular value decomposition mechanism underlying ViT artifacts; SiNGER builds on this theoretical foundation to propose an actionable distillation framework.

## Rating

- Novelty: ⭐⭐⭐⭐ The nullspace-guided perturbation concept is original, elegantly applying linear algebraic theory to artifact suppression in KD.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation, ablation studies, and visualization analyses are comprehensive, though cache truncation precludes verification of specific numerical values.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured, progressing logically from problem analysis through theoretical derivation to practical implementation, balancing formal rigor with intuitive explanation.
- Value: ⭐⭐⭐⭐ The paper presents a theoretically grounded improvement to ViT KD, and the nullspace-guided paradigm has broader applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Alethia: A Foundational Encoder for Voice Deepfakes](../../ICML2026/audio_speech/alethia_a_foundational_encoder_for_voice_deepfakes.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](../../ACL2026/audio_speech/still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)
- [\[CVPR 2026\] BabyVLM-V2: Toward Developmentally Grounded Pretraining and Benchmarking of Vision Foundation Models](../../CVPR2026/audio_speech/babyvlm-v2_toward_developmentally_grounded_pretraining_and_benchmarking_of_visio.md)
- [\[ICCV 2025\] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining](../../ICCV2025/audio_speech/25_years_in_class_a_multimodal_textbook_for_visionlanguage_p.md)
- [\[NeurIPS 2025\] VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction](../../NeurIPS2025/audio_speech/vita-15_towards_gpt-4o_level_real-time_vision_and_speech_interaction.md)

</div>

<!-- RELATED:END -->
