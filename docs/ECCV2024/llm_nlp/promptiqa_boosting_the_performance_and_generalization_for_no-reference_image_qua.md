---
title: >-
  [Paper Note] PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts
description: >-
  [ECCV 2024][LLM (Other)][no-reference IQA] The authors propose PromptIQA, which uses a small number of "image-score pairs" (ISPs) as prompts. This allows the trained NR-IQA model to adapt to new quality assessment requirements without fine-tuning, achieving SOTA performance and generalization capabilities across 12 datasets and 5 categories of IQA tasks.
tags:
  - "ECCV 2024"
  - "LLM (Other)"
  - "no-reference IQA"
  - "image-score pairs prompt"
  - "mixed training"
  - "data augmentation"
  - "assessment requirement adaptation"
date: 2026-05-08
content_hash: 5c3d3e0232869e83
---

# PromptIQA: Boosting the Performance and Generalization for No-Reference Image Quality Assessment via Prompts

**Conference**: ECCV 2024  
**arXiv**: [2403.04993](https://arxiv.org/abs/2403.04993)  
**Code**: None (The paper mentions it will be open-sourced but provides no link)  
**Area**: LLM/NLP  
**Keywords**: no-reference IQA, image-score pairs prompt, mixed training, data augmentation, assessment requirement adaptation

## TL;DR
The authors propose PromptIQA, which uses a small number of "image-score pairs" (ISPs) as prompts. This allows the trained NR-IQA model to adapt to new quality assessment requirements without fine-tuning, achieving SOTA performance and generalization capabilities across 12 datasets and 5 categories of IQA tasks.

## Background & Motivation
**Background**: Once a traditional NR-IQA model is trained, its evaluation criteria are fixed. However, the evaluation requirements vary significantly across different application scenarios (natural image assessment, face quality, AI-generated image quality, underwater image quality, etc.)—the same quality score can represent completely different subjective perceptions in different datasets.

**Limitations of Prior Work**:
   - Traditional IQA models $S = \mathcal{R}(\mathcal{V}(I))$ maintain static evaluation metrics after training, requiring retraining or fine-tuning to adapt to new requirements.
   - Constructing IQA datasets is extremely time-consuming and labor-intensive (requiring a massive amount of manual annotation).
   - Although mixed training methods (such as UNIQUE and StairIQA) allow cross-dataset training, they still require designing independent regression heads or complex data transformations for each dataset.

**Key Challenge**: How to enable the model to understand new assessment requirements at an extremely low data cost (a few samples) without retraining?

**Ours**: The authors design a prompt mechanism that allows the IQA model to work like a human annotator—viewing a few standard examples to understand the assessment criteria before grading image quality.

**Key Insight**: Inspired by the human annotation process—annotators need to review "standard exemplars" to understand assessment requirements before starting work, which is in-context few-shot prompting in nature.

**Core Idea**: Using Image-Score Pairs (ISPs) as prompts to encode assessment requirement information, which is then fused with the target image features to output tailored quality scores.

## Method

### Overall Architecture
ISP Prompt ($n$ image-score pairs) → Visual Encoder + Score Expansion → Prompt Encoder (3 ViT blocks) → ISPP Fusion Module (3 ViT blocks) → Image-Prompt Fusion Module (8 ViT blocks) → Quality Regression (2 FC layers) → Quality Score $S$

Core formula: $S = \mathcal{R}(\mathcal{FM}_{IP}(\mathcal{V}(I), \mathcal{F}_{AP}))$

### Key Designs

1. **Image-Score Pairs Prompt (ISPP)**:

    - **Function**: Combines $n$ (default 10) image-score pairs as prompts to intuitively represent assessment requirements.
    - **Mechanism**: $\mathbf{P} = [\mathcal{ISP}_1, \mathcal{ISP}_2, \ldots, \mathcal{ISP}_n]$, where each $\mathcal{ISP}_i = (I_i, S_i)$. The ISP sampling strategies include:
        - Interval Sampling: Sampling uniformly according to the score distribution, which better reflects the dataset distribution.
        - Random Sampling: Sampling randomly, which is closer to the practical few-shot scenarios.
    - **Design Motivation**: Compared to text prompts which might introduce cognitive bias, ISPs directly demonstrate "what kind of image should get what score," being more intuitive and unambiguous.

2. **Prompt Encoder + Fusion Module**:

    - **Function**: First understands the relationship between the image and score in each ISP, then integrates all ISPs to form the assessment requirement features, and finally fuses them with the target image.
    - **Mechanism**:
        - Each ISP feature: $\mathcal{F}_{\mathcal{ISP}_i} = \text{CAT}(\mathcal{V}(I_i), \mathcal{E}(S_i)) \in \mathbb{R}^{2 \times N}$
        - Prompt Encoder (3 ViT blocks): Explores the deep attention relationship between image and score features to obtain $\mathcal{F}_{P_i} \in \mathbb{R}^{1 \times M}$
        - ISPP Fusion Module (3 ViT blocks): Enables interaction among $n$ prompt features to form the assessment requirement representation $\mathcal{F}_{AC} \in \mathbb{R}^{n \times M}$
        - Image-Prompt Fusion Module (8 ViT blocks): Fuses the target image features with the assessment requirement representation to obtain $\mathcal{F}_{IPF} \in \mathbb{R}^{(n+1) \times M}$
    - **Design Motivation**: Hierarchical fusion ensures the model first understands the "standards" before performing assessment, rather than simply concatenating them.

3. **Data Augmentation Strategy (Random Scaling + Random Flipping)**:

    - **Function**: Breaks the "shortcut learning" dependency of ground-truth (GT) labels on the prompts, forcing the model to learn the assessment requirements from the ISPP.
    - **Mechanism**:
        - Random Scaling (probability 0.5): $f_{RS}(\mathbf{S}) = \frac{1}{\max(\mathbf{S})} \cdot \mathbf{S}$, which scales both ISPP and GT scores simultaneously.
        - Random Flipping (probability 0.1): $f_{RF}(\mathbf{S}) = \alpha - \mathbf{S}$, transforming the MOS semantics into DMOS semantics ($\alpha=1$).
    - **Design Motivation**: If the GT does not change with the ISPP, the model might ignore the prompt and simply memorize the mapping from the input image to the score ("shortcut"). Data augmentation forces the model to generate different scores for the same image under different prompts, demanding it to "read the prompt before answering."

### Loss & Training
- Loss function: $\mathcal{L}_1$ loss (absolute difference between predicted scores and GT)
- Visual Encoder: MoNet (pre-trained IQA encoder)
- Optimizer: Adam, lr=$1 \times 10^{-5}$, weight decay=$1 \times 10^{-5}$, Cosine Annealing every 50 epochs
- Training for 100 epochs, batch size of 66, on 6 Nvidia RTX 3090 GPUs
- Mixed training on 12 datasets (LIVE, CSIQ, TID2013, Kadid-10k, BID, SPAQ, LIVEC, KonIQ-10K, GFIQA20k, AGIQA3k, AIGCIQA2023, UWIQA), covering 5 categories of IQA tasks

## Key Experimental Results

### Main Results (Typical Results on ADN-IQA Task)

| Dataset | Metric | PromptIQA | Prev. SOTA (MoNet) | Gain |
|--------|------|-----------|-------------------|------|
| BID | SROCC↑ | **0.9152** | 0.9012 | +1.53% |
| BID | PLCC↑ | **0.9341** | 0.9152 | +2.07% |
| LIVEC | SROCC↑ | **0.9125** | 0.8998 | +1.41% |
| LIVEC | PLCC↑ | **0.9280** | 0.9169 | +1.21% |
| GFIQA20k | SROCC↑ | **0.9698** | TOPIQ-Face 0.9664 | +0.35% |
| UWIQA | SROCC↑ | **0.8766** | UIQI 0.7423 | +18.1% |

### Generalization Experiment (Simulating New Assessment Requirements with FR-IQA, 10-shot)

| Model | Training Scheme | TID2013-SSIM | TID2013-FSIM | TID2013-LPIPS | Kadid-SSIM |
|------|---------|-------------|-------------|---------------|------------|
| MANIQA-SDT | Zero-shot | 0.5391 | 0.8245 | -0.7486 | 0.5553 |
| MANIQA-MDT&FT | Few-shot | 0.4507 | 0.6925 | -0.6202 | 0.5652 |
| MoNet-SDT&FT | Few-shot | 0.5473 | 0.8380 | -0.7478 | 0.5708 |
| **PromptIQA-MDT** | **Few-shot** | **0.5992** | **0.8802** | **0.8064** | **0.5717** |

**Key Findings**: Traditional models show a negative correlation (up to -0.7486) on LPIPS (DMOS type) because they were trained with MOS. PromptIQA correctly identifies DMOS semantics via the prompts, achieving a positive correlation of 0.8064.

### Ablation Study

| Configuration | TID2013 | SPAQ | GFIQA20k | UWIQA | Generalization-FSIM | Generalization-LPIPS |
|------|---------|------|----------|-------|-----------|------------|
| w/o mixed training | 0.8849 | 0.9228 | 0.9696 | 0.8781 | 0.8579 | 0.6511 |
| w/o prompt | 0.8929 | 0.9220 | 0.9665 | 0.8602 | 0.7851 | -0.7712 |
| w/o random scaling | 0.9218 | 0.9252 | 0.9683 | 0.8766 | 0.8614 | 0.7797 |
| w/o random flipping | 0.9080 | 0.9245 | 0.9691 | 0.8754 | 0.8646 | 0.6538 |
| **Full PromptIQA** | **0.9223** | **0.9261** | **0.9702** | **0.8839** | **0.8802** | **0.8064** |

### Key Findings
- **Prompts are Critical for Generalization**: Removing prompts results in a minor performance drop under mixed training, but when generalizing to new requirements, LPIPS drastically plunges from +0.8064 to -0.7712 (negative correlation!).
- **Random Flipping is Critical for DMOS Generalization**: Removing it drops the generalization LPIPS from 0.8064 to 0.6538.
- **Interval Sampling > Random Sampling**: Interval sampling better reflects the dataset distribution, though random sampling exhibits a small standard deviation and acceptable robustness.
- **ISP Number Effect**: Performance increases monotonically as the number of ISPs scales from 3 to 10; more ISPs provide richer assessment criteria information.
- **Prompt Validity Verification**: Randomizing either images or scores in the ISPs leads to a substantial performance drop; inverting the scores causes SROCC/PLCC to become nearly identical negative values (e.g., 0.9698 → -0.9698), proving that the model indeed learns the assessment requirements through prompts.

## Highlights & Insights
- **Transferring In-context Learning to IQA**: Utilizing ISPs as prompts is fundamentally an application of in-context learning to low-level vision tasks, conveying complex assessment semantics without text prompts.
- **Clever Data Augmentation to Solve the "Shortcut" Issue**: Random Scaling and Flipping seem simple but precisely resolve the core issue where the prompt fails because GT does not scale or change accordingly with the prompt.
- **Cross-Task Unification**: A single model covers five major tasks (SDN-IQA, ADN-IQA, F-IQA, AIG-IQA, U-IQA) without any architectural modifications.
- **New Requirement Adaptation with Only 10-shot**: In contrast to traditional methods that require fine-tuning on entire datasets, PromptIQA requires only 10 samples.
- **Automatic MOS/DMOS Identification**: The model automatically distinguishes between MOS and DMOS semantics through the pattern of score arrangements in the prompt, eliminating the need for manual specification.

## Limitations & Future Work
- Performance on LIVE and CSIQ (DMOS datasets) is slightly lower than that of models trained on a single dataset, indicating that the distribution gap between DMOS and MOS is not fully resolved.
- The ISP prompt requires a few pre-annotated samples (though only 10 are needed), rendering it inapplicable in completely unlabeled scenarios.
- The Visual Encoder (MoNet) freezes its pre-trained weights, so the potential of the prompt mechanism might be bounded by the quality of bottom-level features.
- The explainability of the prompt has not been explored—specifically, what concrete assessment criteria has the model learned from the ISPs?

## Related Work & Insights
- **vs UNIQUE**: UNIQUE requires annotation variance information for each image, and its experiments are incomplete (lacking multiple datasets); PromptIQA does not require any additional annotations.
- **vs StairIQA**: StairIQA sets up an independent regression head for each dataset, which is structurally redundant and cannot generalize to new requirements; PromptIQA utilizes a unified regression head along with prompt adaptation.
- **vs MANIQA/MoNet**: On unseen assessment criteria, even with 10-shot fine-tuning, they cannot adapt effectively (especially for DMOS-to-MOS conversion), whereas PromptIQA achieves this with zero-shot/zero fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The design of using ISPs as prompts is highly original, bringing in-context learning concepts into the IQA task. The data augmentation strategy brilliantly solves the prompt invalidation issue.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive experiments covering 12 datasets, 5 categories of tasks, comparisons with 21 SOTA methods, generalization tests, analysis of prompt impacts, and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, well-defined problems, though the abundance of tables and experiments demands a heavy reading load.
- Value: ⭐⭐⭐⭐⭐ Solves the fundamental pain point in the IQA field where "changing requirements necessitates retraining." The 10-shot adaptation to new requirements possesses immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective](../../ICML2025/llm_nlp/product_of_experts_with_llms_boosting_performance_on_arc_is_a_matter_of_perspect.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](../../AAAI2026/llm_nlp/soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[ACL 2025\] QualiSpeech: A Speech Quality Assessment Dataset with Natural Language Reasoning](../../ACL2025/llm_nlp/qualispeech_a_speech_quality_assessment_dataset_with_natural_language_reasoning_.md)
- [\[ACL 2025\] Conversational Quality Assessment: A Large-Scale Corpus and Comprehensive Study](../../ACL2025/llm_nlp/conversational_quality_assessment_a_large-scale_corpus_and_comprehensive_study.md)
- [\[AAAI 2026\] Rectification Reimagined: A Unified Mamba Model for Image Correction and Rectangling with Prompts](../../AAAI2026/llm_nlp/rectification_reimagined_a_unified_mamba_model_for_image_cor.md)

</div>

<!-- RELATED:END -->
