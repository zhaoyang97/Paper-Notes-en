---
title: >-
  [Paper Note] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking
description: >-
  [ACL 2025][Audio & Speech][dementia detection] To address the gender confounding bias in speech transcript-based dementia detection, this paper proposes Extended Confounding Filter (ECF) and Dual Filter (DF), two weight masking methods that require no additional training modules. By tracking weight updates during fine-tuning, the methods locate and zero out gender-associated parameters, significantly reducing gender gaps in false positive rates and statistical parity while ma…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "dementia detection"
  - "confounding bias"
  - "weight masking"
  - "Transformer debiasing"
  - "gender fairness"
date: 2026-05-08
content_hash: e5e75d2bd598c83c
---

# Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking

**Conference**: ACL 2025  
**arXiv**: [2506.05610](https://arxiv.org/abs/2506.05610)  
**Code**: [GitHub](https://github.com/LinguisticAnomalies/DualFilter.git)  
**Area**: Audio & Speech  
**Keywords**: dementia detection, confounding bias, weight masking, Transformer debiasing, gender fairness

## TL;DR

To address the gender confounding bias in speech transcript-based dementia detection, this paper proposes Extended Confounding Filter (ECF) and Dual Filter (DF), two weight masking methods that require no additional training modules. By tracking weight updates during fine-tuning, the methods locate and zero out gender-associated parameters, significantly reducing gender gaps in false positive rates and statistical parity while maintaining robust dementia detection performance across various distribution shifts.

## Background & Motivation

Transformer models perform well in dementia detection using patient speech transcripts but suffer from severe **confounding bias**. In dementia speech datasets, the gender variable simultaneously affects two aspects: (1) linguistic patterns—males and females exhibit different linguistic styles (word choice, syntax, narrative structure, etc.) when completing the same picture description task; (2) dementia prevalence—females present a significantly higher risk of dementia than males. This causes models to potentially learn gender-specific linguistic cues (such as vocabulary patterns more commonly used by females) as a "shortcut" for dementia prediction, rather than relying on genuine signs of cognitive decline.

The authors first experimentally verify the existence of this bias: after fine-tuning BERT-base on the DementiaBank and CCC datasets, a significant difference in predictive performance is observed between male and female subgroups (Mann-Whitney-Wilcoxon test $p < 0.001$). This disparity persists even after balancing the gender distribution, proving that the bias stems from linguistic patterns rather than simple distributional imbalances.

Prior Confounding Filter methods operate solely on the classification head, failing to explore the distribution of confounding information across the entire Transformer network. This paper extends weight masking to the full network architecture and proposes a more efficient dual-model comparison scheme.

## Method

### Overall Architecture

A two-stage pipeline: **Stage 1** fine-tunes the model normally for dementia detection (or simultaneously trains a gender classification model); **Stage 2** tracks weight updates to locate gender-associated parameters, generates a binary mask matrix to zero out these parameters, and produces the debiased model.

### Key Designs

1. **Extended Confounding Filter (ECF)**:

    - **Function**: Extends the localization scope of the original Confounding Filter's weight masking from the classification head to the full Transformer network.
    - **Mechanism**: Stage 1 fine-tunes the dementia detection model $f(x;\hat{\theta})$ normally and saves a snapshot. Stage 2 starts from the classification head and **unfreezes layer-by-layer** (cls $\rightarrow$ layer12 $\rightarrow$ layer11 $\rightarrow$ ... $\rightarrow$ layer1 $\rightarrow$ emb) to train the model on gender label prediction. Under each unfreezing configuration, the change magnitude of each element in all trainable weight matrices ($W_Q, W_K, W_V, W_O, W_1, W_2$) is tracked, normalized, and accumulated. The top-15% parameters with the largest changes in each weight matrix are selected to generate the mask (zeroed out).
    - **Design Motivation**: Semantic information is dynamically distributed across layers in Transformers; identifying gender-related weights solely on the classification head is insufficient. The layer-by-layer unfreezing probing scheme allows flexible localization of confounding information at different network depths.
    - **Key Findings**: The model's dementia detection performance remains stable after eliminating top-layer gender weights, only degrading sharply once the bottom layers (especially the token embedding layer) are affected.

2. **Dual Filter (DF)**:

    - **Function**: Locates confounding weights by comparing global weight changes between two models, which is more efficient and flexible.
    - **Mechanism**: Two models are initialized from the **same pre-trained checkpoint**: $f$ is fine-tuned for dementia detection, and $g$ is fine-tuned for gender classification. The change magnitudes $\Delta_p$ and $\Delta_c$ of all parameters across the entire network are tracked. Top-$k\%$ weight positions with the largest changes in each model are selected, and three types of masks are generated via set operations:
        - **Intersection Mask $M_I = \Delta_{p,k} \cap \Delta_{c,k}$**: Weights that change significantly in both tasks—likely encoding entangled gender-dementia information.
        - **Difference Mask $M_D = \Delta_{c,k} \setminus \Delta_{p,k}$**: Weights that change significantly only in the gender model—pure gender information.
        - **Union Mask $M_I \cup M_D$**: Equivalent to the top-$k\%$ changing weights in the gender model.
    - The selected mask is applied to zero out the corresponding parameters in the dementia detection model $f$.
    - **Design Motivation**: The layer-by-layer probing in ECF requires multiple Stage 2 training runs, incurring high computational cost. DF only requires two fine-tuning processes, offering a complexity linear to the dataset size. Furthermore, DF uses global weight ranking instead of layer-wise local ranking, capturing cross-layer confounding patterns.
    - **Key Details**: Classification heads are excluded from weight tracking (since they are inherently different and incomparable between tasks). Stage 2 training for gender classification uses only non-dementia samples (healthy controls) to avoid mixing in dementia signals.

3. **Confounding Shift Evaluation Framework**:

    - **Function**: Systematically evaluates model robustness under different distribution shifts.
    - **Mechanism**: Introduces a parameter $\alpha = P(\text{dementia}|\text{female}) / P(\text{dementia}|\text{male})$ to control the conditional distribution of gender and dementia in train/test sets. $P(\text{gender}=1) = P(\text{dementia}=1) = 0.5$ is fixed to ensure balance. Training is conducted on $\alpha_{\text{train}}$ and testing on $1/\alpha_{\text{train}}$ to simulate extreme shifts.
    - **Design Motivation**: In real-world clinical deployments, the population distributions of training and deployment sites often differ; hence, models must remain fair under various distribution shifts.

### Loss & Training

- Stage 1: Fine-tune BERT-base using standard cross-entropy loss for dementia detection.
- Stage 2 (ECF): Train gender classification using cross-entropy loss, but tracking weight updates rather than using the final model.
- Stage 2 (DF): Fine-tune two models independently, requiring no joint training or adversarial loss.
- After applying the mask, **no further fine-tuning** is performed—direct zero-out yields the final model.

## Key Experimental Results

### Main Results: Verification of Gender Confounding Bias

| Dataset | Setting | Mean Gender Difference in AUPRC | p-value |
|--------|------|-------------------|------|
| DementiaBank (DB) | Original Distribution | 0.055 | < 0.001 |
| DementiaBank (DB) | Balanced Distribution | 0.068 | < 0.001 |
| CCC | Original Distribution | 0.152 | 0.002 |
| CCC | Balanced Distribution | 0.102 | 0.007 |

The difference remains significant or even larger after balancing the gender distribution, proving that the bias originates from differences in linguistic patterns.

### Debiasing Effect Example (DB Dataset)

| Method | $\alpha_{\text{train}}$ | Masking Ratio | AUPRC | ΔFPR |
|------|---------|---------|-------|------|
| Original Model | 0.2 | 0% | 0.83 | 0.23 |
| DF ($M_I$) | 0.2 | 10% | 0.80 (-0.03) | **0.03** (-0.20) |

Performance drops by only 0.03, while the gender gap in False Positive Rate (FPR) decreases from 0.23 to 0.03—approaching near-perfect fairness.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Original CF (classification head mask only) | No improvement in AUPRC | Classification head is insufficient for capturing confounding info |
| ECF layer-by-layer expansion | Performance is stable with top-layer masking but degrades with bottom-layer masking | Gender information is mainly located in the middle-to-top layers; token embedding layer is crucial for performance |
| DF $M_I$ vs $M_D$ vs $M_I \cup M_D$ | $M_I$ and $M_D$ show higher resilience; $M_I \cup M_D$ occasionally degrades | Indiscriminately removing all gender-related weights harms task performance |
| Different $\alpha_{\text{train}}$ Configurations | Performance degrades as $\alpha$ deviates further from 1 | "Step effect"—the degree of distribution shift is positively correlated with performance degradation |

### Method Comparison ($\alpha_{\text{train}}=3, \alpha_{\text{test}}=1/3$)

On the AUPRC-ΔFPR trade-off curve:
- **CCC Dataset**: ECF achieves the best trade-off.
- **DB Dataset**: DF ($M_D$) outperforms all other methods.
- Both methods **consistently outperform** adapter-based baselines (ConGater, ModDiffy) and the original Confounding Filter.
- Weight masking methods provide a more **fine-grained trade-off trajectory**: continuously adjusting the masking ratio (0-60 with step 1) allows precise control over the fairness-performance balance.

### Key Findings

1. **Simple classification head masking is completely insufficient**: The original Confounding Filter only operates on the classification layer and barely mitigates confounding bias, showing that gender information is distributed throughout the entire Transformer network.
2. **Top-layer masking is safe, bottom-layer masking is hazardous**: The model remains resilient when gender weights are removed starting from top layers, and sharp degradation only occurs once bottom layers (especially the token embedding layer) are affected. In some configurations, removing gender weights from certain layers even improves dementia detection performance.
3. **Weight entanglement**: Gender information and dementia information are partially entangled within Transformer weights—the intersection mask $M_I$ is non-empty, and masking some intersection weights slightly hurts task performance.
4. **Demographic parity also significantly improves**: On balanced test sets ($\alpha=1$), the difference in demographic parity (ΔSP) is significantly reduced.
5. **Even with balanced label and gender distributions, failing to address confounding shift leads to performance degradation**: The distribution shift problem cannot be solved solely through data balancing.

## Highlights & Insights

- **Model-agnostic methodology**: ECF and DF can be applied to any Transformer architecture without introducing extra training modules or objective functions, operating solely via weight-change tracking and zeroing out.
- **Concise and elegant "dual-model comparison" paradigm of DF**: It requires no joint training, adversarial losses, or adapter modules—only two standard fine-tunings and simple set operations. Simple in concept but highly effective.
- **Scalable to non-binary confounding variables**: By adapting Stage 2 into multi-class classification, other confounders such as age or education level can be handled.
- **Weight masking vs. loss optimization methods**: Masking methods provide a continuously adjustable fairness-performance trade-off curve, offering greater flexibility than adapter-based approaches with discrete hyperparameters.
- The confounding shift framework (controlled by the $\alpha$ parameter) in the experimental design is a valuable reference for other fairness research.

## Limitations & Future Work

- **Small dataset scale**: DB has only 290 participants with 548 samples, and CCC has only 70 participants with 394 transcripts. Simulating different $\alpha$ configurations requires resampling, leading to substantial data duplication.
- Only one encoder model (BERT-base) is evaluated; other architectures like RoBERTa or DeBERTa are not tested.
- The layer-by-layer probing of ECF incurs high computational overhead (each unfreezing configuration requires a separate Stage 2 training run).
- Only binary gender is considered; multivariate confounding (e.g., the joint effect of gender $\times$ age $\times$ education level) is not explored.
- The study assumes that gender should not affect dementia prediction, although clinically, gender is indeed a risk factor. The proposed method might inadvertently remove some clinically meaningful gender-related signals.
- Weight zeroing is a hard operation—exploring softer weight adjustment methods (such as scaling instead of zeroing) remains future work.

## Related Work & Insights

- **vs. Confounding Filter (Wang et al. 2019)**: The original method operates only on the classification head and targets CNNs/non-Transformer architectures; ECF extends it to the full Transformer network, and DF further proposes a more efficient global scheme.
- **vs. ConGater (Masoudian et al. 2024)**: An adapter-based approach that debiases via extra modules and joint loss functions; the proposed method is more lightweight and introduces no additional parameters.
- **vs. ModDiffy (Hauzenberger et al. 2023)**: Another modular debiasing method; the proposed approach shows better trade-offs in AUPRC-ΔFPR.
- **vs. INLP (Ravfogel et al. 2022)**: Removes protected attribute information via linear projection; the proposed method operates directly in the parameter space rather than the representation space, aligning more closely with the fine-tuning process.

## Rating

- Novelty: ⭐⭐⭐⭐ The dual-model weight comparison mechanism in Dual Filter is novel and elegant. Introducing confounding bias mitigation into speech transcript-based dementia detection is highly pioneering.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes systematic distribution shift experiments under various $\alpha$ configurations, comparisons of multiple masking strategies, evaluations on two datasets and multiple baselines, and assessment across both fairness and performance dimensions.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, rigorous methodological descriptions, and detailed ethical considerations.
- Value: ⭐⭐⭐⭐ The methods are generalizable and transferable to other clinical NLP tasks affected by confounding bias; the weight masking perspective provides a novel tool for Transformer interpretability research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Analyzing and Mitigating Inconsistency in Discrete Audio Tokens for Neural Codec Language Models](audio_token_consistency.md)
- [\[ACL 2025\] ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors](atri_mitigating_multilingual_audio_text_retrieval_inconsistencies_by_reducing_da.md)
- [\[ICML 2025\] Teaching Physical Awareness to LLMs through Sounds](../../ICML2025/audio_speech/teaching_physical_awareness_to_llms_through_sounds.md)
- [\[ACL 2025\] Double Entendre: Robust Audio-Based AI-Generated Lyrics Detection via Multi-View Fusion](double_entendre_robust_audio-based_ai-generated_lyrics_detection_via_multi-view_.md)
- [\[ACL 2026\] RTCFake: Speech Deepfake Detection in Real-Time Communication](../../ACL2026/audio_speech/rtcfake_speech_deepfake_detection_in_real-time_communication.md)

</div>

<!-- RELATED:END -->
