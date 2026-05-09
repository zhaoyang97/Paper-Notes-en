---
title: >-
  [Paper Note] Asymmetric Duos: Sidekicks Improve Uncertainty
description: >-
  [NeurIPS 2025][LLM Evaluation][Uncertainty Quantification] Asymmetric Duos (AD) pairs a large model with a small "sidekick"—combining their predictions via temperature-weighted logit averaging—achieving near-5× deep ensemble uncertainty estimation quality at only 10–20% additional FLOPs. RN50 AD (5% FLOPs overhead) approaches an $m=5$ deep ensemble (400% FLOPs overhead) on AUROC/AURC/SAC@98.
tags:
  - NeurIPS 2025
  - LLM Evaluation
  - Uncertainty Quantification
  - Deep Ensembles
  - Asymmetric Pairing
  - Temperature Scaling
  - FLOPs Efficiency
date: 2026-05-08
content_hash: fe149c3f902d6ec0
---

# Asymmetric Duos: Sidekicks Improve Uncertainty

**Conference**: NeurIPS 2025
**arXiv**: [2505.18636](https://arxiv.org/abs/2505.18636)
**Code**: [https://github.com/timgzhou/asymmetric-duos](https://github.com/timgzhou/asymmetric-duos)
**Area**: Uncertainty Estimation
**Keywords**: Uncertainty Quantification, Deep Ensembles, Asymmetric Pairing, Temperature Scaling, FLOPs Efficiency

## TL;DR
Asymmetric Duos (AD) pairs a large model with a small "sidekick"—combining their predictions via temperature-weighted logit averaging—achieving near-5× deep ensemble uncertainty estimation quality at only 10–20% additional FLOPs. RN50 AD (5% FLOPs overhead) approaches an $m=5$ deep ensemble (400% FLOPs overhead) on AUROC/AURC/SAC@98.

## Background & Motivation

**Background**: Deep ensembles (DE) are the gold standard for uncertainty estimation—training 2–5 independent models and averaging their predictions. However, computational cost scales linearly with ensemble size, making it impractical for large models such as ViT-H.

**Limitations of Prior Work**: (a) Deep ensembles require 200–500% additional FLOPs, which is unacceptable for practical deployment; (b) lightweight alternatives such as MC-Dropout fall far short of ensemble performance; (c) fine-tuning and storing multiple large models incurs prohibitive computational and memory costs.

**Key Challenge**: High-quality uncertainty estimation requires model diversity (different models making different errors on different samples), yet diversity comes at the cost of multi-model inference.

**Goal**: Achieve near-ensemble uncertainty estimation quality at minimal additional cost (10–20% FLOPs).

**Key Insight**: Diversity need not come from models of equal size—an asymmetric pairing of a large model and a small model can provide sufficient diversity.

**Core Idea**: Large model + small sidekick → temperature-weighted logit averaging → temperature optimization on a validation set via L-BFGS (seconds) → 10–20% additional FLOPs approaching the quality of a 5× ensemble.

## Method

### Overall Architecture
Train a large model $f_{large}$ (e.g., ViT-L) and a small model $f_{small}$ (e.g., RN18) independently → **Temperature Calibration**: minimize NLL on a validation set via L-BFGS to obtain $T_{large}, T_{small}$ → **Inference**: $f_{Duo}(X) = f_{large}(X) \cdot T_{large} + f_{small}(X) \cdot T_{small}$ → Uncertainty $= 1 - [\sigma(f_{Duo})]_{\hat{Y}}$

### Key Designs

1. **Temperature-Weighted Fusion**:

    - Function: Automatically adjusts the relative contribution of the large and small models.
    - Mechanism: $f_{Duo}(X) = T_{large} \cdot f_{large}(X) + T_{small} \cdot f_{small}(X)$; temperatures $T$ are solved via L-BFGS by minimizing validation NLL (requiring only seconds of computation).
    - Design Motivation: Temperature scaling automatically down-weights a poor sidekick—if the sidekick is entirely uninformative, $T_{small} \to 0$ and the method degenerates to the single large model.

2. **Asymmetric Pairing Strategy**:

    - Function: Selects model pairs with large capacity gaps.
    - Mechanism: The sidekick requires only 5–20% of the large model's FLOPs. Sidekick accuracy is secondary—diversity matters more than accuracy.
    - Design Motivation: Experiments show that even a lower-accuracy sidekick provides valuable uncertainty signals as long as its error patterns differ from those of the large model.

3. **Compatibility with Model Soups**:

    - Function: AD can be stacked on top of Model Soups.
    - Mechanism: First apply soups (weight averaging), then attach a sidekick → Soup+Duo > Soup alone.
    - Design Motivation: The two techniques are orthogonal—soups improve the base model while Duo improves uncertainty estimation.

### Loss & Training
- The large and small models are trained independently with standard cross-entropy loss.
- Temperature calibration: NLL minimization on a validation set, solved via L-BFGS within seconds.
- Evaluation metrics: AUROC (correctness prediction), AURC (selective classification), SAC@98 (coverage at 98% accuracy).

## Key Experimental Results

### Main Results (ImageNet)

| Method | Extra FLOPs | AUROC | AURC | SAC@98 |
|--------|-------------|-------|------|--------|
| RN50 Single Model | 0% | baseline | baseline | baseline |
| **RN50 AD (RN18 sidekick)** | **5%** | **≈ DE-5** | **≈ DE-5** | **+10%** |
| RN50 DE ($m=5$) | 400% | best | best | best |

### Ablation Study

| Configuration | Finding |
|---------------|---------|
| Unweighted vs. weighted Duo | Unweighted fusion degrades significantly at very low FLOPs—temperature calibration is critical. |
| UQ only (predictions fixed) | Uncertainty improvement is not solely attributable to accuracy gain—there is an independent UQ contribution. |
| Cross-domain evaluation (ImageNet-V2, iWildCam) | AD remains effective under distribution shift. |
| AD + Model Soups | Combination outperforms either technique used alone. |

### Key Findings
- 5% additional FLOPs approaches the quality of a deep ensemble costing 400% FLOPs—an 80× efficiency improvement.
- Uncertainty improvement is not solely due to accuracy gains—ablations confirm that using only the Duo's uncertainty while fixing predictions still yields better calibration.
- Cross-domain generalization is robust—consistent improvements on ImageNet-V2, Caltech, and iWildCam.
- Temperature calibration is critical—seconds of computation yield substantial gains.

## Highlights & Insights
- **Extreme efficiency–quality trade-off**: 5% FLOPs overhead delivers near-5× ensemble UQ quality—highly valuable for practical deployment.
- **"Diversity does not require parity"**: Small models naturally exhibit different error patterns from large models, requiring no deliberate diversity engineering.
- **Automatic safeguard via temperature calibration**: If the sidekick is entirely uninformative, its temperature automatically decays to zero—performance cannot fall below the single-model baseline.

## Limitations & Future Work
- Validated on image classification only—segmentation, regression, and NLP tasks remain untested.
- Evaluated in fine-tuning workflows only—training from scratch has not been verified.
- The large model still requires full training—training cost is not reduced.
- Temperature calibration depends on the availability of a validation set.

## Related Work & Insights
- **vs. Deep Ensembles**: DE requires 200–500% additional FLOPs; AD achieves comparable performance with only 5–20%.
- **vs. MC-Dropout**: MC-Dropout falls far short of DE; AD more closely approaches DE quality.
- **vs. Model Soups**: Soups improve accuracy; AD improves UQ—the two are orthogonal and composable.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet effective concept of asymmetric pairing.
- Experimental Thoroughness: ⭐⭐⭐⭐ ImageNet + cross-domain + multiple backbones + ablations.
- Writing Quality: ⭐⭐⭐⭐ Concise and clear.
- Value: ⭐⭐⭐⭐⭐ Extremely high practical utility—near-zero-cost UQ improvement.

### Supplementary Notes on the Method
- **Mathematical basis of temperature calibration**: Minimize NLL $= -\sum_i \log \sigma(f_{Duo}(X_i))_{y_i}$ with respect to $T_{large}, T_{small}$—a low-dimensional convex optimization problem (only 2 parameters) solved by L-BFGS within seconds.
- **Why small-model "errors" are valuable**: Large and small models make mistakes on different samples—when the large model is highly confident yet wrong, the small model may be uncertain (and vice versa), so their fusion yields uncertainty estimates that more accurately reflect true risk.
- **Distinction from knowledge distillation**: Distillation uses the large model to guide the small model (unidirectional); AD lets large and small models complement each other (bidirectional)—no model parameters are modified, and fusion occurs only at inference time.
- **Practical significance of SAC@98**: The fraction of samples covered at a 98% accuracy requirement directly corresponds to safety-critical applications such as "the proportion of trustworthy scenarios in autonomous driving."
- **Applicability to large-scale models**: Pairing ViT-H and similar very large models with an RN18 sidekick adds less than 1% FLOPs while improving uncertainty—extremely practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Efficient Semantic Uncertainty Quantification in Language Models via Diversity-Steered Sampling](efficient_semantic_uncertainty_quantification_in_language_models_via_diversity-s.md)
- [\[ICLR 2026\] Measuring Uncertainty Calibration](../../ICLR2026/llm_evaluation/measuring_uncertainty_calibration.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](../../ACL2026/llm_evaluation/made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[NeurIPS 2025\] Bayesian Evaluation of Large Language Model Behavior](bayesian_evaluation_of_large_language_model_behavior.md)
- [\[NeurIPS 2025\] Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator](your_pre-trained_llm_is_secretly_an_unsupervised_confidence_calibrator.md)

</div>

<!-- RELATED:END -->
