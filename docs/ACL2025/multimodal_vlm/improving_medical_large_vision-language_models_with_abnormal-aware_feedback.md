---
title: >-
  [Paper Note] Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback
description: >-
  [ACL 2025][Multimodal VLM][Medical LVLM] This paper proposes UMed-LVLM, which enhances the abnormal region localization capability of medical LVLMs through Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding (including Relevance Reward, Abnormal Localization Reward, and Vision Relevance Reward) training strategies. It achieves a 58% improvement over the baseline on the MAU dataset and demonstrates excellent cross-modal and OOD generalization capabilities.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Medical LVLM"
  - "Abnormality Detection"
  - "Visual Localization"
  - "Reinforcement Learning"
  - "instruction tuning"
  - "Abnormal-Aware Reward"
date: 2026-05-08
content_hash: a4f12ee667109d24
---

# Improving Medical Large Vision-Language Models with Abnormal-Aware Feedback

**Conference**: ACL 2025  
**arXiv**: [2501.01377](https://arxiv.org/abs/2501.01377)  
**Code**: Not released  
**Area**: Multimodal VLM / Medical Image Analysis  
**Keywords**: Medical LVLM, Abnormality Detection, Visual Localization, Reinforcement Learning, instruction tuning, Abnormal-Aware Reward

## TL;DR

This paper proposes UMed-LVLM, which enhances the abnormal region localization capability of medical LVLMs through Abnormal-Aware Instruction Tuning and Abnormal-Aware Rewarding (including Relevance Reward, Abnormal Localization Reward, and Vision Relevance Reward) training strategies. It achieves a 58% improvement over the baseline on the MAU dataset and demonstrates excellent cross-modal and OOD generalization capabilities.

## Background & Motivation

**Background**: Med-LVLMs (e.g., LLaVA-Med, MedVInt, Med-Flamingo) are already capable of understanding medical images and answering questions, but they suffer from significant deficiencies in **visual localization** (especially abnormal region localization). Even though GPT-4V performs well in modality recognition and anatomical structure identification within medical images, it still struggles with disease diagnosis and precise localization.

**Significance of Visual Localization**:
   - Localization bias leads to unreliable diagnoses, undermining the credibility and interpretability of Med-LVLMs.
   - Enhancing visual localization capability can conversely improve visual understanding (which has been validated in general-domain LVLMs).

**Why General Detectors Cannot Be Used**: General-domain scenarios can leverage general detectors (such as YOLO) to assist in visual localization. However, medical abnormality detection lacks sufficient data to train specialized detectors, particularly for rare diseases.

**Ours**: Enhance the **intrinsic** visual localization capability of Med-LVLMs (without relying on external detectors). Through abnormal-aware training strategies, the model is guided to focus on abnormal regions while generating diagnoses.

## Method

### Overall Architecture

UMed-LVLM employs a two-stage training scheme: Stage 1 - Abnormal-Aware Instruction Tuning; Stage 2 - Abnormal-Aware Rewarding (AAR), building upon MedVInt for continuous training.

### Stage 1: Abnormal-Aware Instruction Tuning

Given a medical image x and a user query q, the model generates a response a containing the **diagnosis** and **abnormal region descriptions**:
$$p(a|x,q;\theta) = \prod_{t=1}^{T} p(a_t|a_{<t},x,q;\theta)$$

The training loss is the standard cross-entropy:
$$\mathcal{L}_{it} = -\sum_{i=1}^{T} \log p_i$$

Although this stage enables the model to learn to output textual descriptions of abnormal regions, it **cannot directly guide the model to focus on abnormal regions visually**.

### Stage 2: Abnormal-Aware Rewarding (AAR)

AAR incorporates three reward mechanisms, optimized based on a reinforcement learning framework (improved PPO):

#### 1. Relevance Reward
- **Policy Network π**: Generates responses based on the state $s_t$ (image + query).
- **Value Network V**: Estimates the expected return of states.
- **LLM Relevance Reward r_t^LLM**: An external LLM evaluates the relevance of the response.
- Total reward: $r_t^{\pi,V,LLM} = A(s_t, a_t; \theta, \phi) + r_t^{LLM}$, where A is the advantage function.
- The Q-function is updated using the Bellman equation.

#### 2. Abnormal Localization Reward (ALR)
$$r_t^{loc} = \frac{\text{Overlap}(\text{Pred-BBox}, \text{GT-BBox})}{\text{Union}(\text{Pred-BBox}, \text{GT-BBox})}$$
Directly uses the IoU between the predicted bounding box and the ground-truth bounding box as the reward to encourage precise localization of abnormal regions.

#### 3. Vision Relevance Reward (VRR)
$$r_t^{att} = \sum_{i \in N} \sum_{j \in \bar{N}} \frac{\exp(Q_i \cdot K_j^\top / \sqrt{d_k})}{\sum_{k \in \bar{N}} \exp(Q_i \cdot K_k^\top / \sqrt{d_k})}$$
where N is the set of abnormal class tokens, and $\bar{N}$ is the set of image patches corresponding to the abnormal region. By aggregating Transformer attention weights, it measures whether the model concentrates its attention on the abnormal image regions when processing abnormal class tokens.

#### Reward Normalization and Aggregation
$$r_t = r_t^{\pi,V,LLM} + \frac{r_t^{loc}}{\max(r_t^{loc})} + \frac{r_t^{att}}{\max(r_t^{att})}$$
ALR and VRR are normalized independently across all response candidates for the same query to ensure that each reward contributes equally.

#### Optimization Objective
The modified objective function based on PPO is:
$$\mathcal{L}^{\text{CLIP+ENT}}(\theta) = \hat{\mathbb{E}}[\mathcal{L}^{\text{CLIP}}(\theta) + c_1 r_t - c_2 \mathcal{L}^{VF}(\phi) + c_3 S[\pi(\cdot|s_t)]]$$
It consists of the PPO clip term, combined rewards, value function loss, and policy entropy regularization.

### MAU Dataset Construction

- **Data Sources**: Five medical datasets — DeepLesion (CT, 32,120 images), KidneyStone (Kidney CT, 1,300 images), NIH (Chest X-ray, 112,120 images), TBX11K (Chest X-ray/Tuberculosis, 11,200 images), and KVASIR (Endoscopy, 8,000 images).
- **Data Construction**: A Prompt Method was designed to generate diagnostic annotations using GPT-4V: first providing the image + abnormal category + abnormal region location $\rightarrow$ GPT-4V generates the diagnosis $\rightarrow$ a reflection prompt reorganizes it into a step-by-step diagnosis flow of "detection $\rightarrow$ localization $\rightarrow$ recognition".
- **Scale**: 5,817 medical images containing user queries and diagnostic responses with annotated abnormal regions.
- **Expert Review**: Reviewed by 3 medical PhD students; only 13 erroneous samples were found and manually corrected.

## Experiments

### Main Results (Table 2 - MAU Test Set)

| Method | DeepLesion | KidneyStone | KVASIR | NIH | TBX11K | Avg |
|------|-----------|-------------|--------|-----|--------|-----|
| MedVInt | 0.29 | 0.11 | 0.27 | 0.08 | 0.09 | 0.17 |
| GPT-4V | 0.27 | 0.36 | 0.53 | 0.18 | 0.19 | 0.31 |
| MedVInt (SFT) | 0.42 | 0.93 | 0.93 | 0.28 | 0.78 | 0.67 |
| MedVInt (SFT+PPO) | 0.44 | 0.94 | 0.95 | 0.30 | 0.80 | 0.69 |
| **UMed-LVLM** | **0.53** | **0.99** | **0.98** | **0.37** | **0.86** | **0.75** |
| GPT-4V w/ bbox | 0.50 | 0.95 | 0.95 | 0.32 | 0.81 | 0.72 |

UMed-LVLM outperforms the baseline MedVInt (0.17) significantly with an average accuracy of 0.75, representing a 142% improvement over GPT-4V (0.31). It even exceeds the performance of GPT-4V when provided with abnormal region location information (0.72).

### External Benchmark Performance

| Benchmark | Metric | MedVInt | UMed-LVLM |
|------|------|---------|-----------|
| VQA-RAD Open | ACC | 69.3 | **74.9** |
| VQA-RAD Close | ACC | 84.2 | **87.6** |
| SLAKE Open | ACC | 88.2 | **90.4** |
| PMC-VQA Choice | ACC | 39.2 | **42.6** |
| MedMNIST Pneumonia AUC | AUC | 98.5 | **99.1** |

Outperforms SOTA on external benchmarks such as VQA-RAD, SLAKE, PMC-VQA, and MedMNIST.

### Ablation Study (Table 6)

| Method | DeepLesion | KidneyStone | KVASIR | NIH | TBX11K | Avg |
|------|-----------|-------------|--------|-----|--------|-----|
| UMed-LVLM | 0.53 | 0.99 | 0.98 | 0.37 | 0.86 | **0.75** |
| w/o VRR | 0.49 | 0.97 | 0.95 | 0.30 | 0.82 | 0.71 |
| w/o ALR | 0.48 | 0.96 | 0.96 | 0.35 | 0.83 | 0.72 |
| w/o AAR | 0.42 | 0.93 | 0.93 | 0.28 | 0.78 | 0.67 |

All three AAR components contribute to the performance: the full AAR brings an 8 percentage point (+8pp) gain compared to SFT-only, while ALR and VRR contribute approximately 3-4pp improvements each.

### In-depth Analysis

1. **Localization Accuracy and Diagnostic Performance**: Diagnostic performance tends to saturate once the IoU reaches 0.6, indicating that Med-LVLMs can benefit without requiring extremely high localization accuracy.
2. **Training Epochs**: Performance peaks at epoch 4 (~0.75).
3. **Data Scale**: Performance continuously improves as the training data increases from 20% to 100%, demonstrating a positive correlation.
4. **Unseen Category Generalization (Table 7)**: When training without Abdomen/Lung/Pelvis categories, UMed-LVLM still far outperforms MedVInt on these categories (0.35 vs. 0.05), showcasing the generalization capability of abnormal-aware learning.
5. **Cross-Dataset Generalization (Table 8)**: It still reaches 0.57 on TBX11K without TBX11K training data, and 0.42 on DeepLesion when trained without DeepLesion data.
6. **Cross-Modal Generalization (Table 9)**: Even when trained solely on CT, it performs well on X-ray and Gross Pathology, indicating that abnormal-aware training captures universal medical abnormality recognition capabilities.

## Highlights & Insights

1. **Innovative Application of RL in Medical Imaging**: Adapts PPO from 'aligning with human preferences' to 'aligning with medical abnormalities', designing two domain-specific rewards: ALR (localization accuracy) and VRR (attention focusing).
2. **Validation of 'Localization Enhances Understanding'**: Experiments confirm that reinforcing abnormal localization ability can significantly improve medical image understanding, showing a consistent causal relationship across different experimental settings.
3. **No Detector/Segmenter Required**: Achieves abnormal localization without relying on external detectors, lowering deployment barriers.
4. **Outstanding Generalization Ability**: Comprehensive and convincing generalization experiments across categories, datasets, and modalities prove that the model learns a universal abnormality recognition capability rather than merely memorizing specific patterns.
5. **Insights from GPT-4V**: GPT-4V with bounding boxes (0.72) is close to the trained UMed-LVLM (0.75), indicating that abnormal region information is critical for diagnosis.

## Limitations & Future Work

1. Constrained by computational resources, the method has not been validated on larger open-source LVLMs (e.g., LLaVA-1.5-13B, InternVL).
2. The MAU dataset contains only 5,817 images, which is limited in scale.
3. Evaluated only on specific medical imaging datasets, lacking validation in broader clinical scenarios (e.g., pathological slides, MRI).
4. The stability of reinforcement learning training and sensitivity to hyperparameters are not fully discussed.

## Related Work & Insights

- **Med-LVLMs**: Medical language-vision models such as LLaVA-Med, MedVInt, XrayGPT, and Med-Flamingo.
- **Region-Aware LVLMs**: RegionGPT (relying on external detectors), Shikra, etc.
- **Visual Localization**: LVLM localization in general scenes (e.g., BBox-GPT).
- **RLHF/Reinforcement Learning**: Application of PPO, DPO in LLM alignment.

## Rating ⭐⭐⭐⭐

- **Novelty**: ⭐⭐⭐⭐ The triple reward mechanism of AAR is cleverly designed, and VRR is a novel design that directly constrains the model to focus on the abnormal regions based on attention weights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablation and generalization analyses are extremely comprehensive (cross-category/cross-dataset/cross-modality/data scale/IoU impact).
- **Method Interpretability**: ⭐⭐⭐⭐ The reward design has clear medical motivation, and the IoU threshold analysis is intuitive.
- **Practicality**: ⭐⭐⭐ The dataset scale is small, and the model is not built on the latest LVLM backbones.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Harnessing PDF Data for Improving Japanese Large Multimodal Models](harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)
- [\[ICML 2025\] MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization](../../ICML2025/multimodal_vlm/mmedpo_aligning_medical_vision-language_models_with_clinical-aware_multimodal_pr.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ICCV 2025\] CompCap: Improving Multimodal Large Language Models with Composite Captions](../../ICCV2025/multimodal_vlm/compcap_improving_multimodal_large_language_models_with_composite_captions.md)
- [\[ACL 2025\] HSCR: Hierarchical Self-Contrastive Rewarding for Aligning Medical Vision Language Models](hscr_hierarchical_self-contrastive_rewarding_for_aligning_medical_vision_languag.md)

</div>

<!-- RELATED:END -->
