---
title: >-
  [Paper Note] Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning
description: >-
  [CVPR2026][Multimodal VLM][Active Learning] This paper proposes the Similarity-as-Evidence (SaE) framework, which reinterprets VLM text-image similarities as Dirichlet evidence. A Similarity Evidence Head (SEH) is introd…
tags:
  - "CVPR2026"
  - "Multimodal VLM"
  - "Active Learning"
  - "Vision-Language Models"
  - "Uncertainty Quantification"
  - "Dirichlet Distribution"
  - "Evidential Deep Learning"
  - "Medical Image Classification"
  - "Calibration"
date: 2026-05-08
content_hash: b3aedfd817f45c41
---

# Similarity-as-Evidence: Calibrating Overconfident VLMs for Interpretable and Label-Efficient Medical Active Learning

**Conference**: CVPR2026
**arXiv**: [2602.18867](https://arxiv.org/abs/2602.18867)  
**Code**: To be confirmed  
**Area**: Multimodal VLM
**Keywords**: Active Learning, Vision-Language Models, Uncertainty Quantification, Dirichlet Distribution, Evidential Deep Learning, Medical Image Classification, Calibration

## TL;DR

This paper proposes the Similarity-as-Evidence (SaE) framework, which reinterprets VLM text-image similarities as Dirichlet evidence. A Similarity Evidence Head (SEH) is introduced to calibrate overconfident softmax outputs, and a dual-factor acquisition strategy based on vacuity and dissonance enables interpretable, label-efficient medical active learning, achieving a SOTA macro-average accuracy of 82.57% across 10 datasets under a 20% annotation budget.

## Background & Motivation

**High annotation cost**: Expert annotation in medical image analysis is constrained by time, cost, and privacy regulations. Active learning (AL) maximizes model performance under limited annotation budgets by selecting the most informative samples.

**Cold-start problem**: Traditional AL methods yield unreliable predictions when initial labels are extremely scarce (e.g., 1–3 per class), leading to inefficient sample selection in early rounds and wasted annotation resources.

**VLM overconfidence**: VLMs convert cosine similarities to probabilities via temperature-scaled softmax, effectively treating geometric proximity as certainty, resulting in severe calibration bias—high confidence is assigned even to incorrect predictions.

**Misleading acquisition functions**: Overconfidence causes models to favor samples they already "understand" rather than those most beneficial for performance improvement, wasting the annotation budget.

**Lack of interpretability**: Existing AL strategies rely on scalar uncertainty scores (entropy/margin), which quantify uncertainty magnitude but cannot reveal its source—whether it stems from knowledge deficiency or conflicting hypotheses.

**Clinical need**: In clinical workflows, experts need to understand why a case is selected for annotation—whether it represents an unseen phenotype or an ambiguous decision boundary—yet existing methods cannot provide such interpretable selection rationale.

## Method

### Overall Architecture

SaE comprises three synergistic components: (1) PubMed-augmented prompts that construct semantically rich textual prototypes; (2) a Similarity Evidence Head (SEH) that maps similarity vectors to Dirichlet evidence parameters; and (3) a dual-factor AL acquisition strategy based on vacuity and dissonance. The VLM image encoder is frozen; only the SEH and CoOp-style learnable prompts are trained.

### Key Designs

**PubMed-augmented prompts**: For each class $k$, $\delta_k$ descriptive sentences are retrieved from PubMed, encoded by a frozen text encoder, L2-normalized, and averaged to produce semantically rich class prototype embeddings $\bar{\hat{\mathbf{e}}}^k_{\text{txt}}$. The cosine similarity vector $\mathbf{s} = [s_1, \dots, s_K]$ is computed as input to the evidence model.

**Similarity Evidence Head (SEH)**: A dual-branch MLP architecture—a feature branch encodes the frozen VLM image embedding $\mathbf{x}$ into $z_f$, while a similarity branch maps $\mathbf{s}$ into $z_s$. The concatenated representation passes through a shallow MLP with softplus activation to output a strictly positive evidence intensity scalar $\lambda$. The core idea is to treat the similarity vector as the allocation proportions of a total evidence budget, whose magnitude is controlled by $\lambda$.

**Similarity-to-evidence mapping**: Dirichlet concentration parameters are defined as $\alpha_k(x) = \lambda(x) \cdot p_k(x) + 1$, where $p_k$ denotes class probabilities from VLM softmax. This decomposition yields:

- **Vacuity**: $\text{Vac}(x) = K / \sum_k \alpha_k(x)$, measuring insufficient total evidence to flag rare or unseen phenotypes
- **Dissonance**: A balance-based measure over belief masses capturing inter-class evidence conflict to flag ambiguous decision boundaries

**Dual-factor acquisition strategy**: A linear schedule $w_v(t) = 1 - (t-1)/(T-1)$, $w_d(t) = (t-1)/(T-1)$ prioritizes high-vacuity samples in early rounds (coverage of unseen phenotypes) and high-dissonance samples in later rounds (refinement of decision boundaries).

### Loss & Training

The dual-objective loss $\mathcal{L}_{\text{SEH}}$ for the SEH:

$$\mathcal{L}_{\text{SEH}} = \text{MSE}\left(\frac{1}{\lambda_i + \epsilon},\; l_{\text{cls},i}\right) + \beta \cdot \text{MSE}\left(\lambda_i,\; \frac{1}{H[\mathbf{p}_i] + \epsilon}\right)$$

- First term: aligns inverse evidence with observed classification difficulty (hard samples → high $l_{\text{cls}}$ → low $\lambda$)
- Second term: enforces consistency with the intrinsic certainty of the frozen VLM (low entropy → high $\lambda$); $H[\mathbf{p}_i]$ is a detached target with no gradient back-propagation
- $\beta = 0.5$ balances the two terms

## Key Experimental Results

### Main Results

On 10 public medical datasets spanning 9 organ types, SaE achieves a macro-average accuracy of 82.57% under a 20% annotation budget, surpassing the strongest baseline MedCoOp+BADGE at 77.75% (+4.82%).

| Dataset | Random | PCB | MedCoOp+Coreset | MedCoOp+Entropy | MedCoOp+BADGE | **SaE** |
|---|---|---|---|---|---|---|
| DermaMNIST | 69.42 | 71.07 | 74.11 | 74.56 | 75.46 | **80.21** |
| Kvasir | 71.10 | 72.92 | 80.83 | 81.92 | 81.42 | **88.58** |
| RETINA | 51.48 | 53.55 | 62.78 | 65.22 | 66.88 | **75.22** |
| LC25000 | 93.92 | 95.71 | 96.93 | 97.47 | 97.25 | **99.23** |
| BTMRI | 83.40 | 85.50 | 86.26 | 89.92 | 89.57 | **93.46** |
| BUSI | 57.10 | 58.47 | 66.53 | 72.03 | 72.88 | **79.15** |
| **Macro Avg.** | 68.01 | 71.41 | 73.84 | 77.39 | 77.75 | **82.57** |

### Ablation Study

| Variant | Macro Avg. (%) |
|---|---|
| Random | 68.01 |
| + Dual-factor score (classifier logits) | 73.35 (+5.34) |
| + VLM similarity replacing logits | 78.62 (+10.61) |
| **SaE: + SEH calibration** | **82.57 (+14.56)** |

The SEH contributes the largest incremental gain (+3.95%), confirming that calibration is the critical factor.

### Key Findings

1. **Cold-start mitigation**: By round 3 (60% budget), SaE reaches 96.7% of its final accuracy on average; on BTMRI, round 3 achieves 92.92% (final: 93.46%, ratio: 99.42%).
2. **Calibration superiority**: On BTMRI, SaE achieves ECE=0.021 and NLL=0.425, substantially outperforming PCB (ECE=0.116, NLL=0.757) and BADGE (ECE=0.036, NLL=0.548).
3. **Training stability**: SaE maintains the lowest and most stable training loss from the first epoch, whereas BADGE exhibits high initial loss and significant instability.
4. **Largest improvement scenarios**: The most pronounced gains appear on RETINA (+8.34%), Kvasir (+6.66%), and BUSI (+6.27%), indicating greater advantages under class imbalance and data scarcity.

## Highlights & Insights

- **Theoretical novelty**: This is the first work to reinterpret VLM similarities as evidence for parameterizing a Dirichlet distribution, providing a principled solution to VLM overconfidence.
- **Interpretability**: The vacuity/dissonance decomposition endows annotation selection with clinically intelligible justifications (unseen phenotype vs. ambiguous diagnosis) rather than black-box scores.
- **Elegant dual-factor scheduling**: The adaptive early-coverage → late-refinement strategy aligns naturally with clinical reasoning logic.
- **Comprehensive experiments**: Evaluated across 10 datasets, 9 organs, and 5 seeds, with consistent results and small standard deviations.
- **Lightweight and efficient**: Only the SEH and learnable prompts are trained while the VLM encoder is frozen; the framework runs on a single RTX 4090.

## Limitations & Future Work

- Evaluation is limited to classification tasks; generalization to other medical imaging tasks such as segmentation and detection remains unverified.
- The quality of PubMed-augmented prompts depends on retrieval results; high-quality descriptions may be unavailable for rare diseases.
- The linear schedule $w_v(t)/w_d(t)$ is a heuristic design that has not been compared against adaptive scheduling strategies.
- Only BiomedCLIP (ViT-B/16) is used as the backbone; applicability to other VLMs (e.g., CONCH, UNI) has not been validated.
- The dual-factor considers only vacuity and dissonance, without incorporating complementary signals such as sample diversity or representativeness.

## Related Work & Insights

- **Medical AL**: Uncertainty sampling methods such as Least-Confidence/Margin/Entropy are sensitive to artifacts and class imbalance; diversity-based methods such as Coreset/BADGE incur high computational overhead.
- **VLM calibration**: CLIP-family models suffer from severe overconfidence; post-hoc temperature scaling provides only global adjustment and does not explain the source of uncertainty.
- **Evidential Deep Learning (EDL)**: Modeling predictions as Dirichlet distributions enables uncertainty decomposition, but standard EDL transforms evidence directly from classification logits, making it fragile under early-stage AL and distribution shift.
- **VLM + AL**: Methods such as PCB compress VLM uncertainty into softmax scalars, inheriting the overconfidence problem while lacking decomposition of uncertainty sources.

## Rating

- Novelty: ⭐⭐⭐⭐ — The similarity-to-evidence reinterpretation is highly elegant; the vacuity/dissonance decomposition introduces a new paradigm for medical AL.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 datasets, multiple baselines, extensive ablations, calibration analysis, and cold-start analysis constitute a highly complete evaluation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and high-quality figures and tables.
- Value: ⭐⭐⭐⭐ — Delivers practical clinical value in improving medical image annotation efficiency; the framework demonstrates strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Relational Visual Similarity](relational_visual_similarity.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](../../ICCV2025/multimodal_vlm/babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[ACL 2026\] Tree-of-Evidence: Efficient "System 2" Search for Faithful Multimodal Grounding](../../ACL2026/multimodal_vlm/tree-of-evidence_efficient_34system_234_search_for_faithful_multimodal_grounding.md)
- [\[CVPR 2026\] ApET: Approximation-Error Guided Token Compression for Efficient VLMs](apet_approximation-error_guided_token_compression_for_efficient_vlms.md)

</div>

<!-- RELATED:END -->
