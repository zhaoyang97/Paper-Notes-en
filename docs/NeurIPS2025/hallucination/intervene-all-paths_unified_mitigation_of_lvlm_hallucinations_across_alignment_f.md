---
title: >-
  [Paper Note] Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats
description: >-
  [NeurIPS 2025][Hallucination Detection][hallucination] This paper proposes AllPath, a multi-path hallucination intervention framework grounded in the Transformer causal architecture. It is the first to demonstrate that h…
tags:
  - "NeurIPS 2025"
  - "Hallucination Detection"
  - "hallucination"
  - "attention head intervention"
  - "causal path"
  - "multi-path framework"
  - "training-free"
date: 2026-05-08
content_hash: f9d5ecd1d3b2404a
---

# Intervene-All-Paths: Unified Mitigation of LVLM Hallucinations across Alignment Formats

**Conference**: NeurIPS 2025
**arXiv**: [2511.17254](https://arxiv.org/abs/2511.17254)  
**Code**: [https://github.com/SooLab/AllPath](https://github.com/SooLab/AllPath)  
**Area**: Hallucination Detection
**Keywords**: hallucination, attention head intervention, causal path, multi-path framework, training-free

## TL;DR
This paper proposes AllPath, a multi-path hallucination intervention framework grounded in the Transformer causal architecture. It is the first to demonstrate that hallucinations in LVLMs do not stem from a single causal path but from the interaction of three paths — image-to-input-text, image-to-output-text, and text-to-text — and that models adaptively rely on different paths depending on the question-answer alignment format. By designing lightweight key-head identification methods for each path and performing adaptive intervention, AllPath consistently reduces hallucinations across four benchmarks covering different alignment formats: POPE, MCQ-POPE, CHAIR, and MME.

## Background & Motivation

**Background**: LVLM hallucination mitigation methods fall broadly into two categories — contrastive decoding methods (VCD, ICD) that reduce language bias by calibrating output distributions, and intervention methods (PAI, AD-HH) that directly manipulate attention weights/heads to enhance visual grounding or suppress text-dominant behavior.

**Limitations of Prior Work**: Each existing method typically intervenes on only a single causal path — PAI targets only the image→output-text path, while AD-HH targets only the input-text→output-text path — causing each to excel on only a subset of benchmarks (PAI performs well on CHAIR but moderately on POPE; VCD exhibits the opposite pattern).

**Key Challenge**: Hallucinations arise not from a single path but from the interaction of multiple paths. More critically, **LVLMs rely on different causal paths for different question-answer formats (binary / multiple-choice / open-ended description)**, making single-path intervention insufficient to cover all scenarios.

**Key Insight**: Starting from the Transformer causal architecture, the paper systematically analyzes all possible information propagation paths and designs targeted key-head identification and intervention methods for each.

**Core Idea**: A multi-path framework with adaptive path-selection intervention, requiring only a single forward pass to score all heads.

## Method

### Overall Architecture
AllPath proceeds in three steps: (1) identify text-to-text (T2T) and image-to-text (I2T) key attention heads using two lightweight methods; (2) analyze the roles of these heads across different paths and alignment formats; (3) adaptively select intervention paths based on question type, amplifying positive heads and suppressing negative ones.

### Key Designs

1. **Text-to-Text Head Identification (LPI Score)**:

    - Function: Quantifies the degree to which each attention head promotes hallucinated vs. non-hallucinated tokens.
    - Mechanism: Defines the Log Probability Increase (LPI) score $\text{logProb}_{\uparrow}^{(l,n)}(\mathcal{B}_t) = \log\sum_{b\in\mathcal{B}_t}\mathbb{P}(b|h_t^{(l-1)}+H_t^{(l,n)}) - \log\sum_{b\in\mathcal{B}_t}\mathbb{P}(b|h_t^{(l-1)})$, computed separately over the hallucination set $\mathcal{B}_t^-$ and the non-hallucination set $\mathcal{B}_t^+$; the T2T Score is $S_{\text{T2T}}^{(l,n),+} - S_{\text{T2T}}^{(l,n),-}$, where a lower value indicates a stronger tendency to promote hallucinations.
    - Design Motivation: Compared to zero-out strategies (which require a full forward pass per head) and training-based methods (which require annotated data), LPI scores all heads in a single forward pass.

2. **Image-to-Text Head Identification (I2T Score)**:

    - Function: Identifies which attention heads exhibit semantically aligned attention patterns over visual tokens.
    - Mechanism: Target tokens $\mathcal{T}_{\text{I2T}}$ (first occurrences of object words) are partitioned into those present in the image $\mathcal{T}_{\text{I2T}}^+$ and those absent $\mathcal{T}_{\text{I2T}}^-$. For $\mathcal{T}_{\text{I2T}}^+$, the total attention of the head over the corresponding region $M_r$ is computed; for $\mathcal{T}_{\text{I2T}}^-$, the total attention over the entire image is used. $S_{\text{I2T}}^{(l,n)} = S_{\text{I2T}}^{(l,n),+} - S_{\text{I2T}}^{(l,n),-}$.
    - Design Motivation: A good I2T head should concentrate attention on the relevant region when the object is present, and distribute attention diffusely when the object is absent.

3. **Key Finding — Strong Correlation between T2T Heads and Alignment Format**:

    - T2T heads are highly correlated across different benchmarks sharing the same alignment format (e.g., Yes/No), with $\rho=0.82$, but correlation drops sharply across formats (Yes/No vs. MCQ), with $\rho=0.12$.
    - This indicates that T2T heads primarily govern instruction following and format alignment rather than visual understanding.

4. **Key Finding — Independence of Image-to-Input-Text and Image-to-Output-Text Paths**:

    - Although both types of I2T heads attend to visual content more than average heads, they exhibit almost no correlation with each other.
    - This implies that intervening solely on the output path is insufficient; heads on the input path also require intervention.

5. **Adaptive Intervention Strategy**:

    - Function: Selects the combination of paths to intervene based on question type, applying different scaling factors to selected heads.
    - Mechanism: The top-$\xi$ and bottom-$\xi$ T2T heads ($Z_{\text{T2T}}^+$, $Z_{\text{T2T}}^-$) and top-$\zeta$ I2T heads ($Z_{\text{I2T}}^+$) are selected. The MHA output is modified as $\tilde{H}_{\leq t}^{(l)} = \sum_n \lambda^{(l,n)} H_{\leq t}^{(l,n)}$, where $\lambda^{(l,n)} = \gamma^+$ if $(l,n)\in Z^+$, $\gamma^-$ if $(l,n)\in Z^-$, and $1$ otherwise.
    - Default settings: $\gamma^+=2.0$, $\gamma^-=0.0$ (negative heads are directly zeroed out).

### Loss & Training
- **Training-free**: The entire method requires no training; head identification is performed via a single forward pass at inference time, followed by scaling-based intervention.
- Short-answer tasks: $\xi=20, \zeta=10$; open-ended tasks: $\xi=40, \zeta=50$.

## Key Experimental Results

### Main Results — POPE & MCQ-POPE & CHAIR

| Method | POPE-Random Acc | POPE-Advers. Acc | MCQ-POPE-Random Acc | CHAIR $C_S$↓ | CHAIR $C_I$↓ |
|------|----------------|-----------------|---------------------|-------------|-------------|
| Vanilla | 85.1 | 80.9 | 72.8 | 52.2 | 14.6 |
| VCD | 86.3 | 81.4 | 78.2 | 58.2 | 16.1 |
| PAI | 86.4 | 82.5 | 78.0 | 28.8 | 7.9 |
| AD-HH | 85.0 | 80.9 | 78.5 | 33.2 | 7.5 |
| **AllPath** | **87.2** | **82.8** | **80.5** | **26.6** | **7.2** |

### MME Hallucination Subset

| Method | Existence↑ | Count↑ | Position↑ | Color↑ | Total↑ |
|------|-----------|--------|-----------|--------|--------|
| Vanilla | 180.0 | 113.9 | 116.7 | 129.4 | 540.0 |
| VCD | 177.8 | 122.8 | 122.2 | 141.7 | 564.4 |
| PAI | 185.0 | 122.8 | 114.4 | 144.4 | 566.7 |
| **AllPath** | **188.3** | **126.1** | **132.2** | **153.3** | **600.0** |

### Ablation Study — Number of Heads and Scaling Factors

| $\xi$ (T2T heads) | $\zeta$ (I2T heads) | POPE-Rand Acc | POPE-Advers. Acc |
|---|---|---|---|
| 0 | 10 | 86.2 | 81.6 |
| 20 | 0 | 86.4 | 82.5 |
| **20** | **10** | **87.2** | **82.8** |
| 30 | 10 | 88.3 | 82.9 |
| 20 | 15 | 88.2 | 83.0 |

### Key Findings
- VCD performs well on POPE but degrades on CHAIR ($C_S$ worsens from 52.2 to 58.2) due to intervening only on the text path; PAI performs well on CHAIR but yields limited improvement on POPE/MCQ-POPE — validating the limitations of single-path intervention.
- AllPath is the only method that consistently improves performance across all benchmarks spanning three distinct alignment formats.
- Completely removing attention from output-text to image results in only ~2% degradation on POPE (indicating the text-to-text path is dominant for Yes/No formats) but a 10% drop on CHAIR (indicating the image-to-text path is critical for open-ended description).
- Directly reusing the POPE head configuration for MME still yields consistent improvement, validating the generalizability of the method.

## Highlights & Insights
- The design of **scoring all heads in a single forward pass** is a core technical contribution, offering orders-of-magnitude efficiency gains over zero-out strategies that require one forward pass per head.
- The finding that **T2T heads are strongly correlated with alignment format** has theoretical value — it explains why different methods perform inconsistently across benchmarks and provides structured guidance for future intervention method design.
- The discovery that **image-to-input-text and image-to-output-text paths are independent** challenges the intuitive assumption that only the output path needs attention.
- The plug-and-play nature of the method allows direct application to any Transformer-based LVLM.

## Limitations & Future Work
- Validation is primarily conducted on LLaVA-v1.5-7B; results on Qwen-VL and Qwen2.5-VL are included in the appendix but remain limited in scope.
- The scaling factors $\gamma^+, \gamma^-$ and head counts $\xi, \zeta$ require manual tuning for different task formats.
- The selection of T2T and I2T heads relies on annotated hallucinated/non-hallucinated tokens, necessitating a small labeled set.
- Whether head re-identification is needed for entirely new question-answer formats (e.g., chain-of-thought) remains unexplored.
- Whether higher-order path interactions (e.g., second-order effects) exist in multimodal large models warrants further investigation.

## Related Work & Insights
- **vs. VCD**: VCD calibrates output distributions by contrasting distorted visual inputs, effectively intervening only on the image-to-output-text path, and even degrades on CHAIR; AllPath's multi-path intervention is more comprehensive.
- **vs. PAI**: PAI enhances image attention (image-to-text path) and excels on CHAIR but yields limited improvement on POPE/MCQ-POPE; AllPath additionally covers the T2T path.
- **vs. AD-HH**: AD-HH suppresses "lazy" text-dominant heads but addresses only the text-to-text path; AllPath demonstrates the necessity of simultaneously intervening across multiple paths.

## Rating
- Novelty: ⭐⭐⭐⭐ The multi-path framework is conceptually clear; the T2T/I2T head identification methods are concise and elegant; the finding that "format determines path" is valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks spanning different alignment formats, correlation analysis, ablations, and multi-model validation — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Logically clear; the narrative structure of analysis → finding → method → validation is well-organized.
- Value: ⭐⭐⭐⭐ Training-free and applicable to diverse hallucination scenarios; provides meaningful insights into the internal mechanisms of LVLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SHIELD: Suppressing Hallucinations In LVLM Encoders via Bias and Vulnerability Defense](../../ICLR2026/hallucination/shield_suppressing_hallucinations_in_lvlm_encoders_via_bias_and_vulnerability_de.md)
- [\[CVPR 2026\] Fighting Hallucinations with Counterfactuals: Diffusion-Guided Perturbations for LVLM Hallucination Suppression](../../CVPR2026/hallucination/fighting_hallucinations_with_counterfactuals_diffusion-guided_perturbations_for_.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/hallucination/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)
- [\[CVPR 2026\] Beyond the Global Scores: Fine-Grained Token Grounding as a Robust Detector of LVLM Hallucinations](../../CVPR2026/hallucination/beyond_global_scores_fine_grained_token_grounding_as_robust_detector_of_lvlm_hallucinations.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)

</div>

<!-- RELATED:END -->
