---
title: >-
  [Paper Note] I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts
description: >-
  [ICML 2025][Medical Imaging][multimodal fusion] I2MoE proposes an interpretable multimodal interaction-aware mixture-of-experts framework. By incorporating four interaction experts (uniqueness $\times 2$ + synergy + redundancy) combined with a weakly supervised interaction loss, it explicitly models heterogeneous interactions between modalities. Furthermore, it provides sample-level and dataset-level interpretability through a reweighting model…
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "multimodal fusion"
  - "mixture-of-experts"
  - "interpretability"
  - "partial information decomposition"
  - "modality interaction"
date: 2026-05-08
content_hash: 4e15913c3f94014e
---

# I2MoE: Interpretable Multimodal Interaction-aware Mixture-of-Experts

**Conference**: ICML 2025  
**arXiv**: [2505.19190](https://arxiv.org/abs/2505.19190)  
**Code**: [https://github.com/Raina-Xin/I2MoE](https://github.com/Raina-Xin/I2MoE)  
**Area**: Medical Imaging / Multimodal VLM  
**Keywords**: multimodal fusion, mixture-of-experts, interpretability, partial information decomposition, modality interaction

## TL;DR

I2MoE proposes an interpretable multimodal interaction-aware mixture-of-experts framework. By incorporating four interaction experts (uniqueness $\times 2$ + synergy + redundancy) combined with a weakly supervised interaction loss, it explicitly models heterogeneous interactions between modalities. Furthermore, it provides sample-level and dataset-level interpretability through a reweighting model, improving accuracy on the ADNI dataset by 5.5%.

## Background & Motivation

**Background**: Multimodal fusion is a core task in multimodal learning. Existing fusion methods (early fusion, late fusion, Transformers, etc.) usually use a single set of parameters to handle all types of modality interactions, without distinguishing the nature of relationships between different modalities.

**Limitations of Prior Work**: Partial Information Decomposition (PID) theory decomposes multimodal information into four types: unique information of modality 1, unique information of modality 2, synergistic information (information that emerges only when combining both modalities), and redundant information (information shared by both modalities). However, PID has remained at the level of theoretical analysis and has not been integrated into end-to-end deep learning frameworks. Existing attempts either handle only pairwise interactions (Wörtwein et al., 2022), require estimating each interaction separately (MMoE, Yu et al., 2024), or lack interpretability (Dufumier et al., 2024).

**Key Challenge**: In multimodal data, different samples may rely on distinctly different interaction modes—some samples mainly rely on unique image information, while others rely on synergistic information—yet existing methods process all samples uniformly, which both compromises performance and lacks interpretability.

**Goal**: (1) Explicitly model four modality interactions in an end-to-end framework; (2) Provide sample-level and dataset-level interpretability; (3) Remain compatible with arbitrary fusion backbone networks.

**Key Insight**: Utilizing a Mixture-of-Experts (MoE) architecture naturally matches the four interaction types—each expert is responsible for one interaction, and experts are specialized through weakly supervised losses.

**Core Idea**: Use four interaction experts + weakly supervised perturbation loss to explicitly model the four PID elements, combined with a reweighting model to provide interpretable, multi-granularity decision analysis.

## Method

### Overall Architecture

I2MoE inserts a mixture-of-experts layer into the standard multimodal fusion pipeline. After the input data passes through modality-specific encoders, the four interaction experts perform fusion with independent parameters and output predictions. A reweighting model assigns weights to each expert based on the input, and their weighted combination forms the final prediction. During training, an additional modality-perturbation forward pass is conducted to calculate the interaction loss. During inference, only one complete forward pass is required.

### Key Designs

1. **The Design and Weakly Supervised Training of Four Interaction Experts**:

    - **Function**: Enable four fusion models with identical structures but independent parameters to specialize in capturing one specific interaction type.
    - **Mechanism**: During training, execute an additional perturbed forward pass for each sample—replacing a modality's embedding with a random vector to simulate single-modality scenarios. The perturbed outputs are then used as weakly supervised signals to train the experts:
        - **Unique Expert 1**: The complete prediction should be close to the prediction keeping only modality 1 (positive sample) and far from the prediction keeping only modality 2 (negative sample).
        - **Unique Expert 2**: Symmetric to the above.
        - **Synergy Expert**: The complete prediction should be far from any single-modality prediction—since synergistic information strictly requires both modalities.
        - **Redundant Expert**: The complete prediction should be close to any single-modality prediction—since redundant information is preserved even under a single modality.
    - For classification tasks, Triplet Margin Loss is used to model uniqueness interactions, and Cosine Similarity models synergy and redundancy. Regression tasks uniformly use MSE.
    - **Design Motivation**: Simulate PID info operations through modality perturbation, using prediction consistency as a proxy instead of directly calculating mutual information.

2. **Learnable Reweighting Model**:

    - **Function**: Adaptively allocate weights for each interaction expert per sample, providing sample-level interpretability.
    - **Mechanism**: Use an MLP to map the concatenated modality embeddings to a 4D weight vector $[w_{\mathrm{uni1}}, w_{\mathrm{uni2}}, w_{\mathrm{syn}}, w_{\mathrm{red}}]$, where the final prediction is $\hat{y} = \sum_i w_i \cdot \hat{y}_i$. **Local Explanation**: Analyze individual sample weight distributions to understand which interaction the prediction primarily relies on. **Global Explanation**: Aggregate weight distributions across all samples on the test set to reveal dataset-level interaction factors.
    - **Design Motivation**: Compared to Simple-Weight (globally shared weights), MLP reweighting allows assigning different weights to different samples. Ablation studies show this design yields a 4.93% accuracy gain on ADNI.

3. **Extension to More Modalities**:

    - **Function**: Generalize the bi-modal framework to $n$ modalities.
    - **Mechanism**: The number of unique experts grows linearly to $n$ (one per modality), while keeping one synergy expert and one redundancy expert, resulting in $n+2$ total experts. The interaction loss is adjusted accordingly—for unique expert $i$, the negative sample is the output when modality $i$ is masked, and the positive sample is the output when other modalities are masked.
    - **Design Motivation**: Avoid combinatorial explosion—if experts were set up for every pair of modalities, the complexity would be $O(n^2)$ or higher.

### Loss & Training

The total loss is $L_{\text{total}} = L_{\text{task}} + \frac{\lambda_{\text{int}}}{E} \sum_{i=1}^{E} L_{\text{int}}^i$, where $L_{\text{task}}$ is the standard task loss (reweighted prediction vs. ground truth), and $L_{\text{int}}^i$ is the interaction specialization loss for the $i$-th expert. $\lambda_{\text{int}}$ balances the two losses.

## Key Experimental Results

### Main Results

Comparison of I2MoE-MulT with 7 fusion methods across 5 datasets:

| Method | ADNI Acc | ADNI AUROC | MIMIC Acc | IMDB Micro-F1 | MOSI Acc |
|---|---|---|---|---|---|
| Early Fusion | 52.01 | 65.69 | 67.63 | 56.10 | 72.16 |
| MulT | 59.57 | 77.21 | 72.42 | 59.68 | 68.80 |
| MoE++ | 58.08 | 75.18 | 72.51 | 58.15 | 70.85 |
| SwitchGate | 62.28 | 79.70 | 70.98 | 55.92 | 72.35 |
| **I2MoE-MulT** | **65.08** | **81.09** | 69.78 | **61.00** | 71.91 |

### Ablation Study

Ablation of key components (ADNI dataset):

| Ablation Variant | Description | Accuracy Change |
|---|---|---|
| No-Interaction | Remove interaction loss | Significant decrease—proves interaction loss is crucial for expert specialization |
| Simple-Weight | Globally shared weights instead of MLP | Decreased by ~4.93%—proves the necessity of adaptive reweighting |
| Synergy-Redundancy | Keep only synergy and redundancy experts | Decrease—proves unique experts are indispensable |
| Latent-Contrastive | Apply interaction loss at embedding layer | Decrease—interaction loss in output space is more effective |

### Key Findings

- I2MoE improves accuracy by 5.5% on ADNI compared to vanilla MulT, and by 3% on MOSI.
- Predictions among the four experts are highly differentiated: in ADNI and MIMIC data, 81%/85% of samples show disagreement among experts. When disagreement exists, I2MoE can still correctly predict about 49%/64% of samples.
- Global explanation analysis shows: the weight distribution for ADNI is relatively even (all four interactions are important), MIMIC shows high weight variability (significant differences among patients), and ENRICO is dominated by the uniqueness of the screenshot modality.
- I2MoE combined with three different fusion backbones (SwitchGate, InterpretCC, MoE++) consistently yields improvements, demonstrating the framework's universality.

## Highlights & Insights

- For the first time, PID information decomposition theory is integrated into an end-to-end deep learning framework, establishing a solid theoretical foundation.
- The design of the weakly supervised interaction loss is clever—simulating PID operations through modality perturbation without explicitly computing mutual information.
- Provides genuinely meaningful interpretability—not only indicating "which modality is important" but also "which kind of interaction is important."

## Limitations & Future Work

- Replacing modality embeddings with random vectors in the interaction loss is only an approximation of PID, and the choice of replacement strategy may affect the quality of expert specialization.
- On the MIMIC dataset, the accuracy of I2MoE slightly drops instead of rising (especially when paired with InterpretCC, dropping by 2.49%), despite an increase in AUROC.
- Although generalizing to more modalities avoids combinatorial explosion, whether unified synergy/redundancy experts can adequately capture interactions between subsets of modalities is questionable.
- Training requires $n+1$ forward passes ($n$ perturbations + 1 complete pass), which introduces a non-trivial computational overhead.

## Related Work & Insights

- MMoE (Yu et al., 2024) is the most direct preceding work, which also uses interaction experts but as a pre-processing step rather than for end-to-end learning.
- The PID framework (Liang et al., 2023) provides the theoretical foundation, and I2MoE is the first deep learning implementation to end-to-endize it.
- InterpretCC (Swamy et al., 2024a) focuses on interpretable fusion but does not explicitly model interaction types.
- Insights: In medical multimodal scenarios, interpretability is a must-have rather than just the icing on the cake. The design logic of I2MoE can be transferred to more domains requiring explainable decision-making.

## Rating

⭐⭐⭐⭐ (7/10)

The combination of the theoretical foundation (PID) and the engineering implementation (MoE) is natural and elegant. The design of the weakly supervised interaction loss is clever, and the interpretability analysis (local + global) holds practical value. The experiments are thorough, covering 5 datasets across medical and general domains. The limitations lie in its limited efficacy on certain datasets (such as the drop in MIMIC accuracy), insufficient attention paid to computational overhead, and the lack of a rigorous analysis of the approximation level to the PID theory. Overall, it is a solid paper on multimodal fusion methodology.

From an application perspective, the value of I2MoE is particularly prominent in healthcare scenarios: the 5.5% accuracy improvement on the ADNI dataset is clinically significant for Alzheimer's disease diagnosis, while the interpretability of weights can help physicians understand 'which information source' drives the prediction—which is crucial for medical AI approval and building clinical trust. The backbone-agnostic nature of the framework also means it can be rapidly adapted to other clinical multimodal fusion systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MoRE-Brain: Routed Mixture of Experts for Interpretable and Generalizable Cross-Subject fMRI Visual Decoding](../../NeurIPS2025/medical_imaging/more-brain_routed_mixture_of_experts_for_interpretable_and_generalizable_cross-s.md)
- [\[ICML 2026\] EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature Experts](../../ICML2026/medical_imaging/eeg-based_multimodal_learning_via_hyperbolic_mixture-of-curvature_experts.md)
- [\[NeurIPS 2025\] Dual Mixture-of-Experts Framework for Discrete-Time Survival Analysis](../../NeurIPS2025/medical_imaging/dual_mixture-of-experts_framework_for_discrete-time_survival_analysis.md)
- [\[NeurIPS 2025\] Mamba Goes HoME: Hierarchical Soft Mixture-of-Experts for 3D Medical Image Segmentation](../../NeurIPS2025/medical_imaging/mamba_goes_home_hierarchical_soft_mixture-of-experts_for_3d_medical_image_segmen.md)
- [\[CVPR 2025\] DFLMoE: Decentralized Federated Learning via Mixture of Experts for Medical Data](../../CVPR2025/medical_imaging/dflmoe_decentralized_federated_learning_via_mixture_of_experts_for_medical_data_.md)

</div>

<!-- RELATED:END -->
