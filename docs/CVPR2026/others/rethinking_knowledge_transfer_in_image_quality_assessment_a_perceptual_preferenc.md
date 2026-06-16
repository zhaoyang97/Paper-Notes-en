---
title: >-
  [Paper Note] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective
description: >-
  [CVPR 2026][Others][Paper Note] The authors attribute the failure of cross-dataset transfer in IQA to the mismatch of "perceptual preference structure" (i.e., differences in conditional distributions $P(Y|X)$ across datasets). They propose Perceptual Preference Representation (PPR) as a feature-score correlation vector to quantify these preferences,
tags:
  - CVPR 2026
  - Others
date: 2026-05-08
content_hash: d8ea1e54208d724e
---
# Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Knowledge_Transfer_in_Image_Quality_Assessment_A_Perceptual_Preference_CVPR_2026_paper.html)  
**Code**: https://github.com/Li-aobo/PreSTA  
**Area**: Image Quality Assessment / Knowledge Transfer / Low-level Vision  
**Keywords**: Image Quality Assessment, Knowledge Transfer, Perceptual Preference Structure, Conditional Distribution Alignment, Data Efficiency  

## TL;DR
The authors attribute the failure of cross-dataset transfer in IQA to the mismatch of "perceptual preference structure" (i.e., differences in conditional distributions $P(Y|X)$ across datasets). They propose Perceptual Preference Representation (PPR) as a feature-score correlation vector to quantify these preferences, Perceptual Preference Compatibility (PPC) using cosine similarity to measure dataset compatibility, and a greedy pruning strategy (PreSTA) to select source samples aligned with the target domain. Using only 20% of source data, this method outperforms full-data baselines.

## Background & Motivation

**Background**: Image Quality Assessment (IQA) aims to align algorithmic scoring with human subjective perception. Deep Blind IQA (BIQA) methods (meta-learning, hypernetworks, Transformers, large pre-trained vision models) have shown strength on single datasets. however, the cost of collecting subjective annotations for every new imaging scenario (new devices, distortions, or content) is extremely high. Thus, transferring perceptual knowledge from existing annotated datasets to new scenarios has become a critical direction.

**Limitations of Prior Work**: Transfer is exceptionally difficult in practice. Directly applying a model trained on a source dataset to a target dataset (cross-domain like synthetic $\rightarrow$ authentic, or even within-domain like synthetic $\rightarrow$ synthetic) results in a significant drop in SRCC. While joint training on multiple datasets improves overall robustness, it rarely provides stable gains for a specific target; sometimes, increasing data diversity even **hurts** target performance.

**Key Challenge**: Existing transfer methods mostly focus on aligning **marginal distributions**—addressing $P(X)$ through feature alignment/domain selection, and $P(Y)$ through learn-to-rank/rescaling. They implicitly assume that the conditional distribution $P(Y|X)$ remains stable across domains. In IQA, this assumption fails. Human attention fluctuates with context (focusing on high-frequencies for blur, smooth areas for noise, faces for portraits, or legibility for documents). These "perceptual cues and their relative importance" constitute the **perceptual preference structure**, which shifts systematically across scenes. Grad-CAM evidence shows that models trained on different datasets exhibit entirely different attention patterns for the **same image**, indicating learned $P(Y|X)$ is fundamentally different.

**Goal**: (1) Identify a **training-free, interpretable** way to quantify the perceptual preference structure of each dataset and measure compatibility between datasets; (2) Use this to select source samples truly aligned with the target domain's preferences to achieve robust and data-efficient transfer.

**Key Insight**: Since the issue lies in $P(Y|X)$ rather than $P(X)/P(Y)$, one should not simply align marginal distributions or blindly stack data. Instead, the focus should be on characterizing the "feature-to-quality score" mapping itself and performing alignment at the **sample level**.

**Core Idea**: Use a "correlation vector between each dimension of visual features and quality scores" as the perceptual preference fingerprint (PPR) of a dataset. Use their cosine similarity (PPC) as a training-free metric for transfer compatibility. Finally, apply greedy pruning to select a source subset that makes the source PPR most similar to the target PPR—**aligning preference structures is more important than increasing data scale**.

## Method

### Overall Architecture
The input to PreSTA is a source dataset $D_s=\{(x_i,y_i)\}$ (images + subjective scores) and a target dataset $D_t$. It outputs a subset $D_s' \subseteq D_s$ such that an IQA model trained only on $D_s'$ generalizes well to $D_t$. The pipeline: First, use an ImageNet pre-trained backbone to extract hierarchical perceptual features and compress each dataset into a PPR vector. Next, calculate the source-target preference consistency (PPC) using cosine similarity. Then, perform a greedy pruning algorithm to remove source samples that hinder alignment, aiming to maximize the PPC between the subset PPR and the target PPR. Finally, train a standard IQA regression model on the preference-aligned subset. The entire PPR/PPC calculation and sample selection process **requires no training**; models are trained only in the final step.

```mermaid
flowchart TD
    A["Source/Target Datasets<br/>(Images + Scores)"] --> B["Perceptual Preference Representation (PPR)<br/>Correlation vector between features and scores"]
    B --> C["Perceptual Preference Compatibility (PPC)<br/>Cosine similarity between source/target PPR"]
    C --> D["Preference-Aligned Sample Selection<br/>Greedy Pruning to maximize PPC"]
    D -->|Incremental Stats O(Nd)→O(d)| D
    D --> E["Preference-Aligned Subset D_s'"]
    E -->|PreSTA-S Cross/Within-domain · PreSTA-J Joint| F["Train IQA Regression Model"]
```

### Key Designs

**1. Perceptual Preference Representation (PPR): Compressing "How Features Determine Quality" into a Correlation Vector**

Design Motivation: To align $P(Y|X)$, one must first quantify this "conditional mapping." PPR achieves this by extracting perceptual features $F\in\mathbb{R}^{N\times d}$ and quality scores $y\in\mathbb{R}^{N}$ for all samples in a dataset. It calculates the Pearson correlation coefficient between each feature dimension and the quality scores to form a $d$-dimensional vector:

$$r=[\rho_1,\rho_2,\dots,\rho_d],\qquad \rho_k=\frac{\mathrm{Cov}(f_k,y)}{\sigma_{f_k}\cdot\sigma_y}$$

where $f_k$ is the $k$-th column of $F$. $\rho_k$ measures the importance of a specific perceptual cue for quality judgment in that dataset. The entire vector characterizes how observers weight different perceptual signals in that scenario. For features, ResNet-50 (layer1–layer4) or Swin-B (stage1–stage4) pre-trained on ImageNet is used. Feature maps from each layer are global average pooled and concatenated—covering cues from low-level textures to high-level semantics. Using a frozen, general backbone ensures a **domain-independent** feature space where PPRs from different datasets can be compared. This is effective because it bypasses the need to train a model to "know" the preference, relying instead on interpretable statistical fingerprints.

**2. Perceptual Preference Compatibility (PPC): A Training-Free Gauge for Source Selection**

With PPR, judging compatibility becomes a comparison of vector directions. PPC is the cosine similarity between the source and target PPRs:

$$\mathrm{PPC}(r_s,r_t)=\frac{r_s\cdot r_t}{\lVert r_s\rVert_2\,\lVert r_t\rVert_2}$$

A high PPC indicates that two datasets are highly consistent in how features map to quality, which empirically corresponds to better cross-dataset transfer. Notably, PPC is **not used to predict absolute SRCC**, as absolute scores are affected by target difficulty and distortion types. Instead, it measures **relative compatibility**—given a target domain, which candidate source is likely to transfer better. In 11 out of 12 experimental groups (6 targets × 2 backbones), higher PPC correctly predicted higher SRCC, validating it as a reliable training-free selection criterion.

**3. Preference Structure Aligned Greedy Sample Selection: Aligning Source PPR to Target PPR at the Sample Level**

While dataset-level PPC selects the source, different samples within a source contribute differently to the preference. This step is formulated as finding a subset $D_s' \subseteq D_s$ that maximizes $\mathrm{PPC}(r_{s'},r_t)$. Since searching for subsets is combinatorially explosive, a **greedy pruning** approach is used: starting from the full source set, in each round, the algorithm calculates the potential PPC for the remaining set if sample $i$ were removed. The sample $i^*$ that maximizes the resulting PPC is deleted. This continues until three stopping criteria balance alignment quality and data diversity: minimum retention ratio $\alpha_{\min}$ (fixed at 20%), sufficient similarity threshold $\tau_{\mathrm{sim}}$ (0.9 for cross/within, 0.95 for joint), and minimum improvement threshold $\epsilon$ ($10^{-6}$). By removing samples that cause the source preference to deviate from the target, the training set's perceptual structure is calibrated to the target domain.

**4. Incremental Statistical Updates: Strategy to Reduce Complexity from $O(Nd)$ to $O(d)$**

Greedy pruning requires calculating the potential PPR for every remaining sample in each round. A naive implementation would recompute all statistics, which is computationally prohibitive. The authors maintain the means $\mu_f, \mu_y$, variances $\sigma_f^2, \sigma_y^2$, and covariance $\mathrm{Cov}(f,y)$. When sample $i$ (with features $f^{(i)}$ and score $y^{(i)}$) is removed, statistics are updated using closed-form incremental formulas:

$$\mu_f'=\frac{N\mu_f-f^{(i)}}{N-1},\qquad (\sigma_f')^2=\frac{N\sigma_f^2-(f^{(i)}-\mu_f)(f^{(i)}-\mu_f')}{N-1}$$

$$\mathrm{Cov}'(f,y)=\frac{N\,\mathrm{Cov}(f,y)-(f^{(i)}-\mu_f)(y^{(i)}-\mu_y')}{N-1}$$

This reduces the cost of evaluating a single "hypothetical removal" from $O(Nd)$ to $O(d)$, making greedy selection feasible for large datasets like KADID-10k or KonIQ-10k.

### Loss & Training
PPR/PPC and sample selection require no training. The final IQA model consists of a backbone (ResNet-50 / Swin-B) and a regression head, optimized with L1 loss. For joint training, each dataset is assigned an independent regression head. Training uses Adam, a learning rate of $2\times10^{-5}$, weight decay of $5\times10^{-4}$, batch size 32, and 32 epochs. Random cropping ($224\times224$) and horizontal flipping are used. Results are reported as the median SRCC/PLCC over 10 runs.

## Key Experimental Results

The datasets cover synthetic distortions (LIVE, CSIQ, TID2013, KADID-10k) and authentic distortions (LIVEC, KonIQ-10k, BID, SPAQ).

### Main Results: PreSTA-S Cross-domain / Within-domain Transfer (Swin-B)

| Setting | Source → Target | Baseline SRCC | PreSTA-S SRCC | Data Used |
| :--- | :--- | :--- | :--- | :--- |
| Cross (Syn → Auth) | KADID-10k → LIVEC | 0.589 | **0.744** | 20% |
| Cross (Syn → Auth) | KADID-10k → BID | 0.739 | **0.833** | 20% |
| Cross (Syn → Auth) | KADID-10k → KonIQ-10k | 0.682 | **0.774** | 20% |
| Cross (Auth → Syn) | KonIQ-10k → LIVE | 0.812 | **0.849** | 20% |
| Cross (Auth → Syn) | KonIQ-10k → TID2013 | 0.452 | **0.516** | 20% |
| Within (Syn → Syn) | KADID-10k → CSIQ | 0.773 | **0.840** | 27.9% |
| Within (Auth → Auth) | KonIQ-10k → SPAQ | 0.865 | **0.880** | 20% |

The improvements are most significant in cross-domain settings where perceptual gaps are largest. PreSTA-S outperforms the 100% data baseline using only 20% of source data.

### PreSTA-J Joint Transfer (Table 3)

| Method | LIVEC SRCC/PLCC | Extra Samples | BID SRCC/PLCC | Extra Samples |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (Target only) | 0.883 / 0.909 | 0 | 0.868 / 0.897 | 0 |
| + Cross-domain KADID-10k | 0.876 / 0.902 | 10,125 | 0.858 / 0.886 | 10,125 |
| + Full Within-domain Joint | 0.899 / 0.915 | 21,198 | 0.883 / 0.908 | 21,198 |
| **PreSTA-J** | **0.905 / 0.919** | **2,793** | **0.885 / 0.898** | **5,343** |

PreSTA-J outperforms full within-domain joint training on LIVEC while using only 2.8k extra samples compared to 21k. Blindly adding the incompatible KADID-10k reduced LIVEC performance (negative transfer), confirming that structural compatibility is more critical than data scale.

### Key Findings
- **PPC is an Effective Source Criterion**: In 11 of 12 comparisons, higher PPC led to higher SRCC.
- **Alignment > Scaling**: Adding incompatible sources triggers negative transfer; selective addition of aligned samples ensures stable gains even with small data volumes.
- **Extreme Data Efficiency**: Using 20%~38% of source data typically reaches or exceeds the full-data baseline, revealing that many samples in existing datasets "dilute" performance.
- **Visual Evidence**: After aligning KADID-10k to LIVEC, the per-channel PPR distribution shifts toward the target, and Grad-CAM attention patterns migrate accordingly, proving that alignment reshapes the model's decision-making focus.

## Highlights & Insights
- **Novel Problem Diagnosis**: Identifying $P(Y|X)$ (perceptual preference) as the bottleneck of IQA transfer rather than $P(X)/P(Y)$ is a significant contribution.
- **Training-Free Preference Fingerprints**: PPR/PPC provide an interpretable, zero-cost way to quantify compatibility. This approach could be transferred to any "image-to-scalar" scoring task (e.g., aesthetics, video quality).
- **Incremental Stats as an Engineering Key**: The closed-form updates for greedy pruning transform a theoretically sound idea into a practical tool for large-scale datasets.
- **Value of Data Efficiency**: Outperforming full datasets with 20% data is highly attractive for IQA, where annotations are expensive.

## Limitations & Future Work
- **Dependence on Target PPR**: PPR requires images and scores from the target domain. Estimating target PPR in completely unlabeled scenarios remains a challenge.
- **Linearity Assumption**: PPR relies on Pearson correlation, which assumes a monotonic linear relationship and might miss non-linear perceptual interactions.
- **Greedy Bias & Thresholds**: Greedy search is a heuristic and does not guarantee global optimality. Thresholds ($\alpha_{\min}, \tau_{\mathrm{sim}}$) currently require manual tuning.
- **Backbone Dependency**: PPR is built on ImageNet-frozen features. Any bias in the backbone propagates to the preference vector.

## Related Work & Insights
- **vs. Domain Adaptation (FreqAlign / DGQA)**: Prior works align marginal features $P(X)$. Ours argues that $P(X)$ alignment cannot solve $P(Y|X)$ mismatch and instead targets the mapping directly.
- **vs. Joint Training (UNIQUE / LIQE / Q-Align)**: These use ranking or unified scales to harmonize $P(Y)$, but blind joint training often suffers from negative transfer. PreSTA-J filters for compatible samples first.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] A Unified Framework for Knowledge Transfer in Bidirectional Model Scaling](a_unified_framework_for_knowledge_transfer_in_bidirectional_model_scaling.md)
- [\[CVPR 2026\] DPGF-Net: Dual-Prior Guided Fusion Network for Joint Assessment of Perceptual Quality and Semantic Consistency in AI-Generated Images](dpgf-net_dual-prior_guided_fusion_network_for_joint_assessment_of_perceptual_qua.md)
- [\[CVPR 2026\] Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling](life-iqa_boosting_blind_image_quality_assessment_through_gcn-enhanced_layer_inte.md)
- [\[CVPR 2026\] Rethinking BCE Loss for Multi-Label Image Recognition with Fine-Tuning](rethinking_bce_loss_for_multi-label_image_recognition_with_fine-tuning.md)
- [\[ICML 2026\] Complexity as Advantage: A Regret-Based Perspective on Emergent Structure](../../ICML2026/others/complexity_as_advantage_a_regret-based_perspective_on_emergent_structure.md)

</div>

<!-- RELATED:END -->
