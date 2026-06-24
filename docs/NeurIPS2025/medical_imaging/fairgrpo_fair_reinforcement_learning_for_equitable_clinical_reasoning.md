---
title: >-
  [Paper Note] FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning
description: >-
  [NeurIPS 2025][Medical Imaging][Fairness] This work proposes FairGRPO, a hierarchical fair reinforcement learning algorithm that addresses demographic performance disparities in clinical AI through adaptive importance weighting (based on group representation size and task difficulty). Evaluated across 7 clinical datasets (280K samples, 5 modalities), it reduces predictive parity (PP) by 27.2%, improves F1 by 12.49%, and releases the first fairness-optimized clinical VLLM—Fair…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "Fairness"
  - "Reinforcement Learning"
  - "GRPO"
  - "Clinical Reasoning"
  - "Vision-Language Models"
  - "Demographic Bias"
date: 2026-05-08
content_hash: f4b0ac33ab86ad61
---

# FairGRPO: Fair Reinforcement Learning for Equitable Clinical Reasoning

**Conference**: NeurIPS 2025  
**arXiv**: [2510.19893](https://arxiv.org/abs/2510.19893)  
**Code**: Yes (anonymous link, including FairMedGemma-4B model weights)  
**Area**: Medical Fairness / Clinical Reasoning  
**Keywords**: Fairness, Reinforcement Learning, GRPO, Clinical Reasoning, Vision-Language Models, Demographic Bias

## TL;DR

This work proposes FairGRPO, a hierarchical fair reinforcement learning algorithm that addresses demographic performance disparities in clinical AI through adaptive importance weighting (based on group representation size and task difficulty). Evaluated across 7 clinical datasets (280K samples, 5 modalities), it reduces predictive parity (PP) by 27.2%, improves F1 by 12.49%, and releases the first fairness-optimized clinical VLLM—FairMedGemma-4B.

## Background & Motivation

**Medical AI Bias**: Clinical datasets are heavily biased toward majority groups (categorized by race, gender, age, or socioeconomic status), leading to significant performance degradation of AI systems on minority groups.

**Feedback Loops**: Conventional optimization naturally favors well-represented groups (which contribute more gradient updates and dominate the loss landscape), forming a vicious cycle where the model increasingly focuses on the majority, while minority performance stagnates or even declines.

**Limitations of Prior Work**: Methods like Group DRO are designed for fixed output spaces of discriminative models and cannot be directly applied to generative multi-step reasoning processes. Data augmentation, reweighting, and post-hoc calibration yield limited effectiveness in VLLMs.

**RL Training Amplifies Bias**: Reasoning training via RL inherits and amplifies biases present in the training data, while fairness in RL remains unexplored within the medical reasoning domain.

**Lack of Demographic Labels**: In real-world clinical datasets, demographic labels are often incomplete or unavailable, further increasing the difficulty of fairness optimization.

## Method

### Overall Architecture

FairGRPO introduces a hierarchical fair scaling mechanism on top of standard GRPO, consisting of three stages:

**Stage 1: Standard GRPO Normalization**

For a prompt $q$ and its generated response group $G_{(q,t)}$ at iteration $t$, each response $o_{(q,i,t)}$ receives a reward $r_{(q,i,t)}$, and the normalized score is computed as:

$$s_{(q,i,t)} = \frac{r_{(q,i,t)} - \hat{\mu}_{G_{(q,t)}}}{\hat{\sigma}_{G_{(q,t)}} + \varepsilon}$$

**Stage 2: Group Discovery**
- **Explicit Groups**: Use labeled demographic attributes (e.g., age, gender).
- **Implicit Groups**: When demographic labels are missing, a feature vector $\mathbf{v}_q \in \mathbb{R}^{|G_{(q,t)}|}$ is constructed for each unlabeled prompt (where each dimension is the raw reward of a rollout). K-means clustering is then applied to discover latent groups (with the optimal number of clusters determined automatically via the elbow method).
- This reward-based representation directly captures task-level difficulty patterns, offering much higher computational efficiency than CNN/ViT embeddings.

**Stage 3: Demographic-Based Reward Scaling**

Two-level temperature factors are computed:

$$T_{(g,t)} = \sqrt{N_{(g,t)}} \cdot \bar{r}_{(g,t)}, \quad T_{(\gamma,g,t)} = \sqrt{N_{(\gamma,g,t)}} \cdot \bar{r}_{(\gamma,g,t)}$$

where $N$ represents the sample size and $\bar{r}$ represents the average raw reward. The inverse temperature scaling is formulated as:

$$s_{(q,i,t)}^{\text{scaled}} = \frac{s_{(q,i,t)}}{\max(T_{(g_{(q,t)},t)} \cdot T_{(\gamma_{(q,t)},g_{(q,t)},t)}, \varepsilon)}$$

**Key Insight**: For minority groups (small $N$) or difficult groups (low $\bar{r}$), the temperature factor is small, which amplifies the learning signal after scaling, while majority group signals are attenuated.

### Loss & Training

The training objective maintains the policy gradient formulation of GRPO with clipped importance sampling:

$$J_{\text{FairGRPO}}(\theta) = \mathbb{E}_{q,o}\left[\sum_{k=1}^{n_o} \min\left(\varphi_k(\theta)\hat{A}^{\text{FairGRPO}}, \text{clip}(\varphi_k(\theta), 1\pm\varepsilon)\hat{A}^{\text{FairGRPO}}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

- **Reward Design**: Simple binary accuracy reward (1 for correct, 0 for incorrect).
- **Base Models**: Qwen-2.5-VL-7B and MedGemma-4B.
- **Unified Fine-Tuning**: Simultaneous training on 7 clinical datasets without dataset-specific adapters.
- **Hardware**: 4 × NVIDIA H200 GPUs.

## Key Experimental Results

### Dataset Configuration

7 public clinical datasets, 5 modalities, totaling 280.2K samples:

| Dataset | Sample Size | Modality | Demographics |
|--------|--------|------|----------|
| CheXpert | 212K | Chest X-ray | Age, Sex |
| Hemorrhage | 2.5K | CT | Age, Sex |
| VinDr-Mammo | 20K | Mammography | Age |
| ISIC-2020 | 33K | Dermoscopy | Age, Sex |
| HAM10000 | 10K | Dermoscopy | Age, Sex |
| PAD-UFES-20 | 2.3K | Dermoscopy | Age, Sex |
| COVID-BLUES | 362 | Ultrasound | Age |

### Main Results on MedGemma-4B

| Method | PP↓ | EOD↓ | σ_F1↓ | ΔF1↓ | F1↑ | Acc↑ | F1_ES↑ |
|------|-----|------|-------|------|-----|------|--------|
| REINFORCE++ | 20.99 | 8.749 | .0518 | .1033 | .2978 | 78.60 | .2831 |
| RLOO | 23.68 | 10.37 | .0600 | .1170 | .3047 | 80.62 | .2875 |
| GRPO | 22.42 | 6.476 | .0418 | .0795 | .3123 | 80.02 | .2998 |
| GRPO+RS | 23.76 | 6.664 | .0433 | .0835 | .2843 | 80.76 | .2725 |
| GRPO+DRO | 16.04 | 7.367 | .0447 | .0871 | .3271 | 81.19 | .3009 |
| FairGRPO_ND | 25.15 | 11.56 | .0547 | .1067 | **.3513** | 79.23 | .3331 |
| **FairGRPO** | **11.67** | **6.663** | **.0383** | **.0721** | .3218 | **81.83** | **.3100** |

### Key Findings

- **PP reduced by 27.2%**: FairGRPO (11.67) vs. the best performing baseline Group DRO (16.04).
- **F1 improved by 12.49%**: FairGRPO_ND (.3513) vs. GRPO (.3123).
- **Improvement on 25/33 groups**: Out of 33 demographic subgroups, FairGRPO outperforms GRPO on 25 groups.
- **Significant gains for the 75+ age group**: On PAD-UFES-20, the accuracy for the 75+ age group improved by 73.08%.
- **Runtime overhead**: Reward computation accounts for < 0.1% of total training time, introducing virtually no extra overhead.

### Training Dynamics Analysis

- FairGRPO **consistently improves fairness** during training (the F1 disparity drops monotonically).
- The fairness of GRPO **deteriorates** as training progresses (the F1 disparity expands).
- FairGRPO extends the Pareto frontier, comprehensively outperforming GRPO on the performance-fairness trade-off.

### Qualitative Analysis

- Dermoscopy image of an 84-year-old female: FairGRPO correctly identifies irregular borders, central necrosis, and pigment patterns -> correctly diagnoses basal cell carcinoma. GRPO hallucinates non-existent features -> misdiagnoses as AKIEC.
- Mammography of an elderly female: FairGRPO correctly identifies high-density shadows and rates it BI-RADS 2. GRPO underestimates the severity -> misclassifies as BI-RADS 1.

## Highlights & Insights

- ⭐⭐⭐⭐ **Novelty**: The first clinical VLLM method to integrate fairness optimization into critic-free RL training.
- ⭐⭐⭐⭐ **Implicit Group Discovery**: Discovers latent vulnerable groups via clustering without requiring demographic labels, addressing the practical issue of missing labels in clinical data.
- ⭐⭐⭐⭐ **Large-Scale Validation**: Extensive evaluation on 7 datasets × 5 modalities × 280K samples × 2 base models.
- ⭐⭐⭐ **Zero Extra Overhead**: Reward computation accounts for < 0.1% of training time, functioning as a plug-and-play solution.
- ⭐⭐⭐ **Open-Source Contribution**: Releases FairMedGemma-4B, the first publicly available fairness-optimized clinical VLLM.

## Limitations & Future Work

1. **Limited Demographic Dimensions**: Only considers age and gender, neglecting other critical dimensions such as race and socioeconomic status.
2. **Coarse Age Grouping**: 25-year interval grouping may obscure within-group variances.
3. **Simple Reward Design**: Binary accuracy rewards (0/1) may fail to capture subtle differences in the quality of clinical reasoning.
4. **Intersectionality Unexplored**: Fails to analyze fairness performance for intersecting groups (e.g., "elderly + female").
5. **Restricted Model Scale**: Only validated on 4B/7B models; the behaviors of larger models might differ.

## Rating ⭐⭐⭐⭐

This work addresses an important and overlooked problem: fairness in RL training. The design of hierarchical temperature scaling is intuitive and straightforward to implement. Implicit group discovery is an especially valuable contribution, as demographic labels are frequently unavailable in clinical data. The large-scale multi-dataset validation makes the findings highly convincing. The primary shortcomings are the single demographic dimensions and the overly simplistic reward design. The release of FairMedGemma is a practical contribution to the community.
- 75+ age group accuracy on PAD-UFES-20 improved by 73.08%.

## Highlights & Insights

1. **First Fair RL Algorithm for Clinical VLLMs**: Setting fairness as a foundational optimization objective rather than a post-processing step establishes a new paradigm.
2. **Unlabeled Fairness Optimization**: Discovers implicit groups via clustering based on reward patterns, solving the common issue of missing demographic labels in medical data.
3. **Pareto Frontier Improvement**: FairGRPO improves both performance and fairness simultaneously, avoiding a trade-off between the two.
4. **In-Depth Qualitative Analysis**: Demonstrates progress in diagnostic reasoning quality (reducing hallucinations, more accurately identifying key features) in models trained with FairGRPO.

## Limitations & Future Work

- Currently only evaluates vision-language tasks, without covering broader medical modalities (such as time-series and EHR).
- Demographic dimensions are limited to age and gender, lacking factors like race and socioeconomic status.
- Lacks a theoretical analysis on the convergence properties of FairGRPO.
- As a research prototype, it should not be directly applied to clinical decision-making.

## Related Work & Insights

- **vs. Group DRO**: DRO is designed for discriminative models, while FairGRPO is tailored for RL scenarios with generative multi-step reasoning.
- **vs. Resampling**: Resampling is a static method, whereas FairGRPO is dynamically adaptive.
- **Insight**: The idea of reward-pattern-based clustering can be generalized to other RL settings requiring fairness (such as long-tail scenarios in autonomous driving).

## Rating

- Novelty: ⭐⭐⭐⭐ (First work to integrate fairness into VLLM RL training)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 datasets × 5 modalities × 2 base models, multidimensional evaluation)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, well-motivated)
- Value: ⭐⭐⭐⭐ (Solves the critical medical AI fairness issue) GRPO treats all prompts equally, ignoring their source domain and demographic representation.

**Implicit Group Discovery** (for samples without demographic labels):

- Constructs a feature vector $\mathbf{v}_q \in \mathbb{R}^{|G_{(q,t)}|}$ for each unlabeled prompt, where each dimension represents the raw reward of a rollout.
- For example: a chest X-ray prompt generates five rollouts with rewards $[0.2, 0.8, 0.7, 0.9, 0.3]$.
- Uses **K-means clustering** to group prompts with similar reward distributions.
- Automatically determines the optimal number of clusters using the **elbow method**.
- Key advantage: Highly computationally efficient (dimension equals the number of rollouts rather than high-dimensional CNN/ViT features) and directly captures task-specific difficulty patterns.

**Hierarchical Temperature Scaling**:

$$T_{(g,t)} = \sqrt{N_{(g,t)}} \cdot \bar{r}_{(g,t)}, \quad T_{(\gamma,g,t)} = \sqrt{N_{(\gamma,g,t)}} \cdot \bar{r}_{(\gamma,g,t)}$$

where $N$ represents the sample size and $\bar{r}$ represents the average raw reward. **Inverse temperature scaling** enables minority/difficult groups to receive amplified learning signals:

$$s^{\text{scaled}}_{(q,i,t)} = \frac{s_{(q,i,t)}}{\max(T_{(g_{(q,t)},t)} \cdot T_{(\gamma_{(q,t)},g_{(q,t)},t)}, \varepsilon)}$$

Finally, the scores are re-normalized to zero-mean and unit variance.

### Loss & Training

The training objective maintains the policy gradient formulation of GRPO with clipped importance sampling:

$$J_{\text{FairGRPO}}(\theta) = \mathbb{E}_{q,o}\left[\sum_{k=1}^{n_o} \min\left(\varphi_k(\theta)\hat{A}^{\text{FairGRPO}}, \text{clip}(\varphi_k(\theta), 1\pm\varepsilon)\hat{A}^{\text{FairGRPO}}\right) - \beta D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right]$$

**Reward Design**: Simple correctness reward—1 point for correct answers, 0 for incorrect ones.

**Training Configuration**: 4 × NVIDIA H200 GPUs, conducting simultaneous multi-task unified fine-tuning across 7 datasets.

## Key Experimental Results

### Main Results

**Dataset Scale**: 7 public datasets, 5 clinical modalities (X-ray/CT/dermoscopy/mammography/ultrasound), totaling 280.2K samples.

| Method | PP↓ | EOD↓ | F1↑ | Acc↑ | F1_ES↑ |
|------|-----|------|-----|------|--------|
| GRPO (MedGemma) | 22.42 | 6.476 | .3123 | 80.02 | .2998 |
| GRPO+DRO | 16.04 | 7.367 | .3271 | 81.19 | .3009 |
| **FairGRPO (FairMedGemma)** | **11.67** | **6.663** | .3218 | **81.83** | **.3100** |
| FairGRPO_ND (unlabeled version) | 25.15 | 11.56 | **.3513** | 79.23 | .3331 |

On MedGemma, FairGRPO reduces PP by 27.2% (vs. the best fairness baseline DRO) and improves EOD by 23.8%.

| Method | PP↓ | EOD↓ | F1↑ | F1_ES↑ |
|------|-----|------|-----|--------|
| GRPO (Qwen-2.5-VL) | 11.39 | 9.091 | .2550 | .2437 |
| **FairGRPO** | 16.80 | **5.546** | **.2647** | **.2588** |

On Qwen-2.5-VL, EOD is reduced by 15.7%, and the maximum F1 gap is reduced by 28.9%.

### Ablation Study

**Performance of FairGRPO_ND** (completely without demographic labels):

- Maximum accuracy gap improved by 10.81%, and accuracy standard deviation improved by 13.38%.
- F1 improved by 12.49% (potentially due to implicit clustering providing a better alignment with downstream tasks).
- Demonstrates that even without demographic information, latent group discovery alone can improve fairness.

**Training Dynamics Analysis** (Fig 2):

- The F1 disparity of FairGRPO remains consistently lower than that of GRPO, with the gap widening as training progresses.
- FairGRPO extends the performance-fairness Pareto frontier.
- Negligible runtime overhead: advantage computation consumes less than 0.1% of the total training time.

### Key Findings

- FairGRPO outperforms GRPO on 25 out of 33 demographic subgroups (Fig 3).
- On CheXpert, F1 improves by 24.4% for females and 34.4% for males.
- On PAD-UFES-20, performance improves by 6.33% for patients aged 75+ and 3.68% for those aged 51–75.
- Qualitative analysis: FairGRPO reduces hallucinations on minority groups and improves the diagnostic reasoning chain.

## Highlights & Insights

1. **Novelty**: First work to address fairness in VLLMs within critic-free RL training.
2. **Effective Without Labels**: Implicit group discovery through reward vector clustering improves fairness even without demographic labels.
3. **Computationally Efficient**: The reward-based feature representation only requires vectors of the rollout dimension, which is significantly lower than traditional visual features.
4. **Balancing Performance and Fairness**: Avoids the trade-off of "sacrificing majority performance for fairness"; the performance of majority groups is also enhanced.
5. **Pareto Frontier Expansion**: FairGRPO provides a superior performance-fairness trade-off point throughout the entire training process.

## Limitations & Future Work

- Currently only evaluated on vision-language tasks, without covering more medical modalities (e.g., time series, EHR).
- Fairness dimensions are limited to age and gender, lacking factors like race and socioeconomic status.
- Lack of theoretical analysis on the convergence properties of FairGRPO.
- As a research prototype, it should not be directly applied to clinical decision-making.

## Related Work & Insights

- **vs. Group DRO**: DRO is designed for discriminative models, while FairGRPO introduces fairness to critic-free RL for the first time.
- **vs. Resampling**: Resampling is a static method, whereas FairGRPO is dynamically adaptive.
- **Insight**: The concept of clustering via reward vectors can be generalized to other RL scenarios requiring the discovery of latent subgroups.
- **Key Insight**: During RL training, the fairness of standard methods (GRPO/RLOO) deteriorates as training progresses, whereas FairGRPO ensures continuous improvement.

## Rating

- Novelty: ⭐⭐⭐⭐ (First fair RL method for VLLMs, with an ingenious implicit group discovery design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 datasets across 5 modalities, 2 VLLM architectures, multi-dimensional fairness metrics)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, logical flow from motivation to method and experiments)
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Boundaries of Fair AI in Medical Image Prognosis: A Causal Perspective](the_boundaries_of_fair_ai_in_medical_image_prognosis_a_causal_perspective.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](../../CVPR2026/medical_imaging/orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)
- [\[CVPR 2026\] OralGPT-Plus: Learning to Use Visual Tools via Reinforcement Learning for Panoramic X-ray Analysis](../../CVPR2026/medical_imaging/oralgpt-plus_learning_to_use_visual_tools_via_reinforcement_learning_for_panoram.md)
- [\[ICLR 2026\] Glance and Focus Reinforcement for Pan-cancer Screening](../../ICLR2026/medical_imaging/glance_and_focus_reinforcement_for_pan-cancer_screening.md)
- [\[CVPR 2026\] MedGRPO: Multi-Task Reinforcement Learning for Heterogeneous Medical Video Understanding](../../CVPR2026/medical_imaging/medgrpo_multi-task_reinforcement_learning_for_heterogeneous_medical_video_unders.md)

</div>

<!-- RELATED:END -->
