---
title: >-
  [Paper Note] Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering
description: >-
  [ICLR 2026][Medical Imaging][Medical VQA] This paper proposes the Q-FSRU framework, which transforms medical image and text features into the frequency domain via FFT for fusion…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Medical VQA"
  - "Frequency-Domain Fusion"
  - "Quantum Retrieval Augmentation"
  - "Multimodal Fusion"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: b9d52c71155efaf0
---

# Q-FSRU: Quantum-Augmented Frequency-Spectral Fusion for Medical Visual Question Answering

**Conference**: ICLR 2026
**arXiv**: [2509.23899](https://arxiv.org/abs/2509.23899)  
**Code**: None  
**Area**: Medical Imaging
**Keywords**: Medical VQA, Frequency-Domain Fusion, Quantum Retrieval Augmentation, Multimodal Fusion, Contrastive Learning

## TL;DR
This paper proposes the Q-FSRU framework, which transforms medical image and text features into the frequency domain via FFT for fusion, and introduces a quantum-inspired retrieval augmentation mechanism (Quantum RAG) to retrieve medical facts from an external knowledge base, achieving 90.0% accuracy on the VQA-RAD dataset.

## Background & Motivation
- Medical Visual Question Answering (Med-VQA) requires simultaneous understanding of medical images and clinical questions; existing methods face challenges including data scarcity, complex clinical terminology, and diverse imaging modalities.
- Most approaches (e.g., LLaVA-Med, STLLaVA-Med) operate solely in the spatial domain, potentially overlooking pathological pattern information encoded in the frequency domain.
- Existing retrieval augmentation methods rely on classical cosine similarity, which may fail to fully capture the complex semantic relationships required for clinical reasoning.
- **Core Motivation**: Frequency-domain transformations can capture global contextual patterns missed by spatial processing; quantum-inspired similarity metrics may outperform classical retrieval methods.

## Method

### Overall Architecture
Q-FSRU consists of four core modules: (1) multimodal feature extraction, (2) FFT frequency-domain processing, (3) quantum-inspired knowledge retrieval, and (4) multimodal fusion with contrastive learning. The pipeline is: image and text features → FFT spectral transformation → cross-modal co-selection → Quantum RAG retrieval → MLP classification.

### Key Designs

1. **Frequency-Spectral Representation and Fusion (FSRU)**:

    - 1D FFT is applied separately to text features $t$ and projected image features $v_{\text{proj}}$, with magnitude spectra extracted: $t_{\text{freq}} = |\mathcal{F}(t)|$, $v_{\text{freq}} = |\mathcal{F}(v_{\text{proj}})|$
    - A learnable filter bank ($K=4$) compresses the frequency representations.
    - Gated attention implements cross-modal co-selection: $g_{\text{text}} = \sigma(W_{\text{gate1}} \cdot \text{AvgPool}(v_{\text{compressed}}))$, yielding enhanced text features $t_{\text{enhanced}} = t_{\text{compressed}} \odot g_{\text{text}}$
    - **Design Motivation**: Frequency-domain transformation captures global pathological patterns in medical images; the gating mechanism enables mutual enhancement between modalities.

2. **Quantum-Inspired Retrieval Augmentation (Quantum RAG)**:

    - Embedding vectors are represented as quantum states: $|\psi(x)\rangle = x / \|x\|_2$
    - Density matrices $\rho(x) = |\psi(x)\rangle\langle\psi(x)|$ provide statistical robustness.
    - Uhlmann fidelity measures similarity between the query and knowledge base entries: $\text{Fid}(\rho_q, \rho_{k_i})$
    - Top-3 retrieved entries are aggregated via softmax-weighted summation: $k_{\text{agg}} = \sum_{j=1}^{3} \text{softmax}(\text{Sim}_j / \tau) \cdot k_j$, with temperature $\tau = 0.1$
    - **Design Motivation**: Quantum fidelity may more effectively capture semantic relationships in high-dimensional spaces.

3. **Dual Contrastive Learning Framework**:

    - Intra-modal contrastive loss: $\mathcal{L}_{\text{intra}}$, temperature $\tau = 0.07$
    - Cross-modal contrastive loss: $\mathcal{L}_{\text{cross}}$, temperature $\tau = 0.05$
    - **Design Motivation**: Pulls representations of same-class samples closer while pushing apart those of different classes.

### Loss & Training
- Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + (0.3 \cdot \frac{\mathcal{L}_{\text{intra-text}} + \mathcal{L}_{\text{intra-image}}}{2} + 0.7 \cdot \mathcal{L}_{\text{cross}})$
- Optimizer: Adam, learning rate $5 \times 10^{-5}$, L2 regularization weight $10^{-5}$
- 5-fold cross-validation, batch size 32, maximum 50 epochs, step-based decay (0.98 per 5 epochs), early stopping patience of 10
- Image encoder: ViT-B/16 (ImageNet pretrained); text encoder: 300-dimensional word embeddings with mean pooling

## Key Experimental Results

### Main Results

| Dataset | Metric | Q-FSRU | Prev. SOTA (FSRU) | Gain |
|--------|------|--------|----------------|------|
| VQA-RAD | Accuracy | 90.0% | 87.1% | +2.9% |
| VQA-RAD | F1-Score | 85.2% | 82.3% | +2.9% |
| VQA-RAD | AUC | 0.954 | 0.921 | +0.033 |
| VQA-RAD→PathVQA | Accuracy | 81.7% | 78.4% | +3.3% |
| PathVQA→VQA-RAD | Accuracy | 80.3% | 76.9% | +3.4% |

### Ablation Study

| Configuration | Accuracy | Δ Acc. | Note |
|------|---------|--------|------|
| Q-FSRU (Full) | 90.0% | — | Complete model |
| w/o Frequency Processing | 85.1% | -4.9% | Largest contributor |
| w/o Quantum Retrieval | 86.8% | -3.2% | Significant contribution |
| w/o Contrastive Learning | 87.3% | -2.7% | Notable contribution |
| Spatial-only Fusion | 84.2% | -5.8% | Worst configuration |
| Cosine Similarity (replacing quantum) | 88.1% | -1.9% | Quantum similarity outperforms cosine |

### Key Findings
- Frequency-domain processing contributes the largest performance gain (−4.9%), indicating that spectral representations capture clinically relevant patterns missed in the spatial domain.
- Quantum-inspired retrieval outperforms classical cosine similarity by 1.9%, though the margin is modest.
- The model contains only 92.4M parameters, far fewer than LLaVA-Med/STLLaVA-Med (7B), yet achieves superior performance on VQA-RAD.
- Strong cross-dataset generalization (+3.3%/+3.4%) suggests that the learned representations transfer well across domains.

## Highlights & Insights
- Introducing frequency-domain analysis into Med-VQA is a novel research direction; the global information captured by FFT may be particularly beneficial for medical image analysis.
- Quantum-inspired retrieval is an interesting yet relatively preliminary exploration, applying quantum state representations to knowledge retrieval.
- The compact model size (92.4M parameters) offers practical deployment value in resource-constrained environments.

## Limitations & Future Work
- Validation is limited to two datasets (VQA-RAD and PathVQA), both of modest scale (VQA-RAD contains only 3,515 QA pairs).
- The theoretical advantages of quantum retrieval are extensively discussed, but the empirical gains remain relatively limited (only 1.9% over cosine similarity).
- Comparisons with recent large language models (e.g., GPT-4V) are absent.
- The construction and maintenance of the knowledge base are not described in detail.
- Performance on more complex multi-choice or open-ended question answering remains unknown.

## Related Work & Insights
- Frequency-domain approaches have demonstrated success in image analysis (FDTrans) and rumor detection (Lao et al. 2024); this work extends the paradigm to Med-VQA.
- Quantum-inspired information retrieval (Uprety et al. 2021) offers a novel perspective on similarity computation.
- Cross-modal contrastive learning has become a standard practice in multimodal fusion.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of frequency-domain processing and quantum retrieval is novel in the Med-VQA context.
- **Experimental Thoroughness**: ⭐⭐⭐ Datasets are small; comparisons with recent large vision-language models are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Structure is clear and mathematical derivations are complete.
- **Value**: ⭐⭐⭐ The lightweight design is valuable, but the practical advantages of quantum retrieval require more rigorous validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CARE: Towards Clinical Accountability in Multi-Modal Medical Reasoning with an Evidence-Grounded Agentic Framework](care_towards_clinical_accountability_in_multi-modal_medical_reasoning_with_an_ev.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [\[ICLR 2026\] Towards Interpretable Visual Decoding with Attention to Brain Representations](towards_interpretable_visual_decoding_with_attention_to_brain_representations.md)
- [\[ICLR 2026\] LaVCa: LLM-assisted Visual Cortex Captioning](lavca_llm-assisted_visual_cortex_captioning.md)
- [\[ICLR 2026\] Improving 2D Diffusion Models for 3D Medical Imaging with Inter-Slice Consistent Stochasticity](improving_2d_diffusion_models_for_3d_medical_imaging_with_inter-slice_consistent.md)

</div>

<!-- RELATED:END -->
