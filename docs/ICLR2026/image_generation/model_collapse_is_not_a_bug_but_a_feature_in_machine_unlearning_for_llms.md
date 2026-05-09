---
title: >-
  [Paper Note] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs
description: >-
  [ICLR 2026][Image Generation][machine unlearning] This paper repositions "model collapse"—commonly regarded as a negative phenomenon—as a tool for machine unlearning, proposing the PMC method. By iteratively fine-tuning on retained data and the model's own generated outputs, PMC achieves targeted information removal without directly optimizing over the forget targets, and validates its effectiveness through both theoretical analysis and empirical experiments.
tags:
  - ICLR 2026
  - Image Generation
  - machine unlearning
  - model collapse
  - partial model collapse
  - LLM privacy
  - iterative relearning
date: 2026-05-08
content_hash: 121810d0937758fc
---

# Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs

**Conference**: ICLR 2026
**arXiv**: [2507.04219](https://arxiv.org/abs/2507.04219)
**Code**: [TUM DAML - Partial Model Collapse](https://www.cs.cit.tum.de/daml/partial-model-collapse/)
**Area**: Image Generation
**Keywords**: machine unlearning, model collapse, partial model collapse, LLM privacy, iterative relearning

## TL;DR

This paper repositions "model collapse"—commonly regarded as a negative phenomenon—as a tool for machine unlearning, proposing the PMC method. By iteratively fine-tuning on retained data and the model's own generated outputs, PMC achieves targeted information removal without directly optimizing over the forget targets, and validates its effectiveness through both theoretical analysis and empirical experiments.

## Background & Motivation

Privacy regulations such as the GDPR require the ability to selectively remove the influence of specific data from machine learning models. For LLMs, full retraining is computationally infeasible, necessitating efficient **machine unlearning** techniques.

Existing LLM unlearning methods share a fundamental problem: they **counterintuitively rely on the very data to be forgotten** to drive the unlearning optimization. For example, gradient ascent (GA) trains against the forget targets, while NPO (Negative Preference Optimization) treats them as negative samples. This paradigm entails two serious issues:

**Violation of the data minimization principle**: The unlearning process still utilizes sensitive data, increasing the risk of data exposure.

**Uncharacterized side effects**: Adversaries may exploit probabilistic probing to infer forgotten information (information leakage).

The paper's core insight draws from the **model collapse** phenomenon—when a generative model is iteratively trained on its own outputs, the output distribution gradually collapses, effectively losing information. If this collapse can be triggered **partially and controllably**, unlearning can be achieved without accessing sensitive data.

## Method

### Overall Architecture

The core mechanism of **Partial Model Collapse (PMC)**:
1. For retain questions: standard fine-tuning using ground-truth answers.
2. For forget questions: multiple responses are sampled from the model itself, and the response with the highest "forget quality" is selected via a Bradley-Terry preference model for fine-tuning.
3. The process iterates without ever accessing the ground-truth answers of the forget targets.

### Key Designs

1. **From Full Collapse to Partial Collapse (Lemma 1)**:

    - Pure iterative self-training leads to complete distributional collapse (loss of all class information).
    - Through the "anchoring" effect of retained data, **partial collapse** is achievable: probability mass on retained classes remains unchanged, while probability mass on non-retained classes converges to zero.
    - This mechanism naturally suits the requirements of machine unlearning.

2. **Theoretical Derivation for Continuous Distributions (Theorem 2)**:
   The iterative unlearning process is defined as:
    $p_{t+1} = \arg\min_{p \in \mathcal{P}} \frac{\alpha}{1+\alpha} \mathbb{E}_{x \sim p_r}[-\log p(x)] + \frac{1}{1+\alpha} \mathbb{E}_{x \sim p_t}[-\log p(x)]$
   Under the assumption of no statistical error, $p_t$ converges exponentially to the retain distribution $p_r$ at rate $\frac{1}{1+\alpha}$. The closed-form solution is:
    $p_t(x) = [1 - (\frac{1}{1+\alpha})^t] p_r(x) + (\frac{1}{1+\alpha})^t p_0(x)$

3. **Preference-Model-Based Q&A Unlearning (Corollary 3)**:
   A Bradley-Terry model is introduced for sample selection, choosing the response with the highest forget quality from $n$ candidates:
    $\theta_{t+1} = \arg\min_\theta \lambda \mathbb{E}_{(q,x) \in D_r}[-\log f_\theta(x|q)] + \mathbb{E}_{q \in D_f, \hat{x} \sim \mathcal{BT}}[-\log f_\theta(\hat{x}|q)]$
   Theoretical guarantee: the expected reward converges to its maximum $e^{r^*}$ and the variance converges to zero.

4. **Reward Function Design**: $r(x) = 1 - \text{ROUGE-L}(\hat{x}, y)$, where $y$ is the ground-truth answer for the forget question. In practice, argmax approximation is used in place of full BT sampling, i.e., the highest-scoring response is selected.

### Loss & Training

- Experiments are conducted on the "forget10" split of the TOFU dataset (400 forget samples).
- Models: Phi-1.5 and Llama-3.2-3B-Instruct.
- Each method is evaluated over 100 hyperparameter configurations × 5 random seeds = 500 runs.
- Evaluation metrics: ROUGE-L recall score; Unlearning Quality (UQ) = max score − actual score (higher is better); Utility = aggregate ROUGE-L on retain set + world knowledge + real-author Q&A.

## Key Experimental Results

### Main Results: Pareto Frontier

| Method | Unlearning Quality (UQ) | Utility | Pareto Performance |
|--------|------------------------|---------|-------------------|
| GA/GD/NPO/SimNPO | Low–Medium | Medium–High | NPO is the best baseline |
| IDK | Medium | Medium | Simple yet effective |
| **PMC (Ours)** | **Highest** | **Highest** | **Substantially expands the Pareto frontier** |

PMC achieves a utility–unlearning quality trade-off that simultaneously dominates all baselines on Phi-1.5. On Llama-3.2-3B-Instruct, the unlearned model accurately refuses to answer only the forget questions while preserving all other capabilities.

### Ablation Study

| Configuration | Key Metric | Observation |
|---------------|-----------|-------------|
| Iterations (2→20) | UQ improves steadily; utility nearly unchanged | PMC's advantage lies in gaining more UQ without sacrificing utility over more iterations |
| Sample count (1→20) | More samples = higher UQ; utility unaffected for the first 6 iterations | Large sample sizes increase variance |
| λ (0.5→1.5) | Larger λ improves utility but reduces UQ | Careful balancing required |
| Temperature (1.0→1.5) | Higher temperature improves UQ | Utility degrades beyond 1.5 |
| BT temperature τ | UQ is optimal as τ→0 | Validates the argmax approximation |

### Side-Effect Analysis of Existing Methods

| Side Effect | NPO | PMC |
|-------------|-----|-----|
| Token probability shift on unrelated datasets | Severely left-shifted (mean −0.12) | Zero-mean Gaussian (unbiased) |
| Accuracy of lowest-probability option in MCQ | High accuracy in low-quantile range (leakage) | No pattern (no leakage) |
| Minimum probability distribution | Concentrated near zero | Normal distribution |

### Key Findings

- **PMC's core advantage lies in not directly optimizing over forget targets**: the model learns to generate "I don't know" or related but harmless responses to forget questions, rather than simply suppressing the probability of correct answers.
- **Information leakage**: Target-dependent methods such as NPO create exploitable signals for adversaries—forgotten knowledge can be recovered by selecting the option with the lowest probability.
- **Generation coherence**: NPO significantly reduces the generation probability of forgotten tokens on unrelated datasets (e.g., wikitext), degrading normal text generation; PMC exhibits no such issue.
- **Better performance on Llama-3.2**: Instruction-tuned models can learn to precisely refuse forget questions while leaving other responses unchanged.

## Highlights & Insights

- **Paradigm innovation**: Redefining model collapse from a "bug" to a "feature" opens a new collapse-based unlearning paradigm.
- The ability to **forget without accessing sensitive data** offers unique advantages in stricter privacy-compliance settings (e.g., GDPR).
- **Theoretical completeness**: The progressive derivation from discrete class distributions (Lemma 1) → continuous distributions (Theorem 2) → Q&A settings (Corollary 3) is logically clear.
- **Revealing hidden risks in existing methods**: Probabilistic probing attacks and cross-context token probability shifts are important but previously overlooked evaluation dimensions.

## Limitations & Future Work

- **High computational cost**: Sampling $n$ responses per forget question incurs significant inference overhead for large models (approximately 5 hours per configuration on Phi-1.5).
- **Reward function design is scenario-dependent**: The current use of ROUGE-L may require more refined reward designs for real-world applications.
- **Evaluation limitations**: Unlearning evaluation remains an open problem; this paper focuses solely on suppressing specific outputs while preserving utility, without examining robustness to relearning attacks.
- **Forget-question prompts are still required**: Although ground-truth answers are not used, the method still requires knowledge of which questions should be forgotten.
- **Reliability of GPT-4 evaluation**: Some evaluations rely on GPT-4 scoring, which may introduce bias.

## Related Work & Insights

This paper cleverly bridges two seemingly unrelated fields: **model collapse** research (Shumailov et al. 2023; Bertrand et al. 2024) and **machine unlearning** research (NPO, SimNPO, etc.). The theoretical work by Ferbach et al. (2024) on curated self-generated data inducing collapse provides a key theoretical foundation for PMC. Methodologically, employing the Bradley-Terry preference model as a "navigation tool" to guide the direction of collapse offers an interesting contrast with the DPO family of methods—DPO optimizes preferences to enhance capabilities, whereas PMC leverages preferences to achieve unlearning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Transforming model collapse into an unlearning tool; a paradigm breakthrough)
- Experimental Thoroughness: ⭐⭐⭐⭐ (500 runs × 2 models, but limited to a single dataset, TOFU)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear theoretical derivations, strong problem motivation, illuminating side-effect analysis)
- Value: ⭐⭐⭐⭐ (High practical value in privacy-compliance scenarios, though computational cost remains to be addressed)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Easier Painting Than Thinking: Can Text-to-Image Models Set the Stage, but Not Direct the Play?](easier_painting_than_thinking_can_text-to-image_models_set_the_stage_but_not_dir.md)
- [\[ICCV 2025\] MUNBa: Machine Unlearning via Nash Bargaining](../../ICCV2025/image_generation/munba_machine_unlearning_via_nash_bargaining.md)
- [\[ICCV 2025\] Invisible Watermarks, Visible Gains: Steering Machine Unlearning with Bi-Level Watermarking Design](../../ICCV2025/image_generation/invisible_watermarks_visible_gains_steering_machine_unlearning_with_bi-level_wat.md)
- [\[ICLR 2026\] Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning](image_can_bring_your_memory_back_a_novel_multi-modal_guided_attack_against_image.md)
- [\[NeurIPS 2025\] A Closer Look at Model Collapse: From a Generalization-to-Memorization Perspective](../../NeurIPS2025/image_generation/a_closer_look_at_model_collapse_from_a_generalization-to-memorization_perspectiv.md)

</div>

<!-- RELATED:END -->
