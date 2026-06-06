---
title: >-
  [Paper Note] LLMSurgeon: Diagnosing Data Mixture of Large Language Models
description: >-
  [ACL2026][Self-Supervised Learning][Data Mixture Auditing] LLMSurgeon formalizes the question of "what data was this LLM trained on" as Data Mixture Surgery. It utilizes the soft confusion matrix of a proxy classifier to…
tags:
  - "ACL2026"
  - "Self-Supervised Learning"
  - "Data Mixture Auditing"
  - "Training Corpus Composition"
  - "Label Shift"
  - "Confusion Matrix"
  - "Black-box Auditing"
date: 2026-05-08
content_hash: 52cd09842dae49ef
---

# LLMSurgeon: Diagnosing Data Mixture of Large Language Models

**Conference**: ACL2026  
**arXiv**: [2605.30348](https://arxiv.org/abs/2605.30348)  
**Code**: Paper mentions Code & Data: LLMSurgeon; specific URL not provided in cache  
**Area**: LLM Transparency / Training Data Audition / Model Governance  
**Keywords**: Data Mixture Auditing, Training Corpus Composition, Label Shift, Confusion Matrix, Black-box Auditing  

## TL;DR
LLMSurgeon formalizes the question of "what data was this LLM trained on" as Data Mixture Surgery. It utilizes the soft confusion matrix of a proxy classifier to invert the domain distribution of generated text, estimating pre-training data mixture proportions while only accessing model outputs.

## Background & Motivation
**Background**: The behavior, bias, and capabilities of Large Language Models (LLMs) largely stem from their pre-training data composition, yet real data recipes are often not disclosed. Existing transparency tools primarily focus on Membership Inference Attack (MIA), which determines whether a specific sample appeared in the training set.

**Limitations of Prior Work**: MIA can answer "has this sample been seen," but struggles to determine "how much Web, Code, Books, Papers, or Forums content is in the entire training corpus." Directly aggregating MIA results is computationally expensive, suffers from cumulative errors, and introduces systematic bias because different domains vary in inference difficulty.

**Key Challenge**: Auditing training corpora requires macro-level distribution estimation, whereas existing tools mostly provide micro-level sample signals. Closed-source or fixed models do not provide access to training loops, raw corpora, or internal weight states, requiring methods to work on black-box generated text.

**Goal**: The authors propose Data Mixture Surgery (DMS): given a predefined set of domains and generated samples from a target LLM, estimate the implicit effective domain prior $\pi$ of the model. The goal is not open-ended discovery of unknown categories, but the recovery of mixture proportions within a defined taxonomy.

**Key Insight**: The paper adopts the label-shift assumption: while domain proportions change from the training corpus to generated text, the linguistic features within the same domain remain approximately invariant. Thus, the distribution observed after passing generated text through a proxy domain classifier is "blurred" by the classifier's confusion matrix and can be corrected as an inverse problem.

**Core Idea**: First, estimate the soft confusion matrix of a proxy classifier using reference corpora. Then, treat the average classification output of the target model's generated text as a biased observation, recovering the latent training data mixture proportions through constrained linear inversion.

## Method
The LLMSurgeon approach is akin to performing a "data composition CT" on a black-box model: rather than finding a specific training sample, it allows the model to generate text naturally under neutral prompts, observes the distribution of these texts as seen by a predefined domain classifier, and infers the true domain proportions using the classifier's own error patterns.

### Overall Architecture
Inputs include a predefined set of domains $\mathcal{Y}=\{1,\dots,K\}$, reference corpora for each domain, neutral generated text from the target LLM, and ground truth data recipes from public documentation for evaluation. The output is an estimated vector $\hat{\pi}$ on a simplex, representing the domain mixture proportions reflected in the model's behavior.

Before training, the system trains an external domain classifier $f_\phi$ on reference corpora and calculates a soft confusion matrix $C$ on held-out data. During inference, the target model generates a batch of text $X_{gen}$. The classifier outputs domain probabilities for each generated text, which are averaged to obtain an observation vector $\bar{p}$. Finally, it solves $\min_{\pi\in\Delta^{K-1}} \|C^\top\pi-\bar{p}\|_2^2$, subject to $\sum_k\pi_k=1$ and $\pi_k\ge 0$.

### Key Designs
1. **Data Mixture Surgery Problem Definition**:

	- Function: Elevates training data auditing from sample-level membership judgment to domain-level proportion recovery.
	- Mechanism: Assumes the training corpus comes from $p_\alpha(x)=\sum_i \alpha_i p_i(x)$ and the target model's generation distribution can be approximated as $q_\pi(x)=\sum_i \pi_i p_i(x)$; the auditing goal is to estimate $\pi$.
	- Design Motivation: Safety, copyright, and bias governance require macro-level data recipes rather than knowing if a single sample was memorized.

2. **Soft Confusion Matrix Calibration**:

	- Function: Explicitly models systematic confusion of the proxy classifier between similar domains.
	- Mechanism: For a reference sample with true domain $i$, record the average probability of the classifier predicting each domain $j$, obtaining $C_{ij}=\mathbb{E}_{x\sim p_i}[f_\phi(x)_j]$.
	- Design Motivation: Directly summing classifier outputs would treat the confusion between similar domains (e.g., C vs. C++, C4 vs. Common Crawl) as real proportions, leading to skewed estimates.

3. **Constrained Inverse Problem for Domain Prior Recovery**:

	- Function: Restores the latent domain proportions from the "blurred" observed distribution.
	- Mechanism: Based on $\mathbb{E}_{x\sim q_\pi}[f_\phi(x)]=C^\top\pi$, solve for $\hat{\pi}$ using constrained least squares (non-negative, sum-to-one).
	- Design Motivation: The inversion step is the core gain of LLMSurgeon over simple audit-by-aggregation, as it corrects classifier bias without requiring additional internal model access.

### Loss & Training
The proxy classifier is trained on reference domain data. The core estimation objective is not a standard end-to-end loss but the constrained linear inversion $\min_{\pi\in\Delta^{K-1}} \|C^\top\pi-\bar{p}\|_2^2$. In experiments, the Coarse-Grained setting uses SlimPajama-627B-DC with 5,000 documents sampled per domain (6 classes) to train the classifier; Mid-Grained uses 17 classes from The Pile; Fine-Grained uses 87 programming languages from The Stack. Metrics include Overlap Accuracy, MAE, and $R^2$.

## Key Experimental Results

### Main Results
| Setting / Model | Granularity | LLMSurgeon Overlap Accuracy | Strong or Representative Baseline | Note |
|--------|------|-----------------------------|------------------------|------|
| OLMo-1B | 6-class Coarse | 94.46% | Recall 48.05% | Coarse corpus boundaries are clear; inversion has a huge advantage |
| LLaMA1-7B | 6-class Coarse | 95.14% | Neighbor 40.13% | Nearly recovers the public data recipe |
| Amber-13B | 6-class Coarse | 78.87% | Recall 41.55% | Still significantly higher than MIA aggregation methods |
| LLaMA1-65B | 6-class Coarse | 94.26% | GradNorm 46.52% | Remains stable across model scales |
| GPT-Neo-2.7B | 17-class Mid | 61.86% | GradNorm 58.78% | Advantage narrows at medium granularity |
| Pythia-12B | 17-class Mid | 65.98% | Recall 52.63% | Finer taxonomies increase confusion |
| StarCoder-15.5B | 87-class Fine | 30.37% | GradNorm 27.54% | Similar languages like C/C++ make the inverse problem ill-posed |

### Ablation Study
| Ablation Item | Configuration | Key Result | Conclusion |
|------|------|---------|------|
| Classifier Backbone | DistilBERT vs Transformer / TF-IDF / MLP | DistilBERT 95.14%, Transformer 90.22%, TF-IDF 86.83%, MLP 82.97% on LLaMA1-7B | Proxy classifier quality directly affects final recovery |
| Sample Size | 100 / 1,000 / 5,000 / 10,000 per domain | StarCoder: 20.15 / 25.62 / 30.37 / 29.51; LLaMA1-7B: 85.78 / 93.68 / 95.14 / 92.44 | 5,000 is a good trade-off between accuracy and cost |
| Inverse Correction | w/o Inverse Correction vs LLMSurgeon | StarCoder: 26.47% → 30.37%; OLMo: 92.77% → 94.46% | Soft confusion matrix inversion provides real gains |
| Merging Similar Categories | Separate C4&CC vs Merge C4&CC | LLaMA1-7B: 42.42% → 99.14% | Semantically inseparable sources should be merged, otherwise estimates are unstable |
| Held-out OLMo-3 | Transfer with fixed early protocol | OLMo-3 overlap accuracy 86.41%, Web 76.88 → 75.37 | Method shows some out-of-protocol generalization |
| Toxicity Injection Audit | GPT-2 5% / 10% / 20% toxic | Estimated 7.90% / 12.00% / 22.73%, Toxic Est. Accuracy 97.10% / 98.00% / 97.27% | Can serve as a low-cost safety triage signal |

### Key Findings
- DMS and MIA have different objectives: MIA is suited for asking if a sample appeared, while DMS is suited for domain proportion composition.
- LLMSurgeon performs strongest on coarse-grained, semantically separable domains; once categories overlap heavily, the inversion matrix becomes ill-posed, and accuracy drops.
- Neutral sampling is most stable for general-purpose models, e.g., reaching 95.14% for LLaMA1-7B; however, for specialized models like StarCoder, neutral prompts may not sufficiently trigger the target distribution.
- There is a strong positive correlation between classifier accuracy and final estimation accuracy; the paper reports an average correlation greater than 0.9, and notes a Pearson coefficient exceeding 0.85 in another analysis.

## Highlights & Insights
- The strongest aspect of the paper is transforming black-box data auditing into a clear statistical inverse problem rather than stacking membership inference scores. This formalization clarifies the problem, assumptions, and failure boundaries.
- The soft confusion matrix is a practical design: it acknowledges that proxy classifiers will inevitably make mistakes and incorporates the error structure into the estimation rather than taking the classifier output as ground truth.
- The value of LLMScan lies not just in evaluating LLMSurgeon, but also in providing a "recipe-verifiable" data auditing benchmark, avoiding proving the method only on synthetic mixtures.
- The point that "categories must be semantically separable" is crucial. It reminds future work that taxonomy design is not a trivial preprocessing step; the definition of domains determines whether the audit is solvable.

## Limitations & Future Work
- The method relies on the label-shift assumption, meaning neutral generation reflects the pre-training prior; models subjected to RLHF, instruction tuning, or strong system prompts may deviate from this assumption.
- The method uses a closed-world taxonomy, making it unable to discover new domains outside predefined categories or automatically identify gaps in the taxonomy.
- Fine-grained, semantically overlapping categories lead to ill-conditioned confusion matrices (e.g., C4 vs. Common Crawl, C vs. C++), limiting interpretable resolution.
- Generation sampling styles affect estimation stability; neutral prompts work for general models but may be insufficient for specialized ones.
- Future research could investigate hierarchical taxonomies, non-linear transport, inverse alignment correction, and verification across languages, multi-modality, and more closed-source models.

## Related Work & Insights
- **vs Membership Inference Attack**: MIA determines if a single sample is in the training set; LLMSurgeon estimates macro domain proportions. The former is a micro-privacy tool; the latter is a macro-transparency tool.
- **vs DUCI**: DUCI estimates the usage proportion of specific candidate datasets; LLMSurgeon recovers the global mixture of multiple domains without requiring access to the original training sets.
- **vs Data Mixture Optimization**: Data mixture optimization selects or re-weights corpora before training; LLMSurgeon performs post-hoc auditing on trained models.
- **vs Direct Classifier Aggregation**: Direct aggregation of $\bar{p}$ retains classifier bias; LLMSurgeon uses $C^\top\pi$ inversion to correct this bias.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of the DMS problem setting and soft confusion matrix inversion is clear and represents a useful advancement in transparency.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 8 models with public recipes, three levels of granularity, sampling styles, sample sizes, held-out data, and toxicity injection, though it still relies on closed-world taxonomy.
- Writing Quality: ⭐⭐⭐⭐☆ Formulas and experimental designs are easy to follow; strengths and boundaries are clearly stated.
- Value: ⭐⭐⭐⭐⭐ Highly practical for model governance, training data transparency, and safety auditing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Improving Large Vision and Language Models by Learning from a Panel of Peers](../../ICCV2025/self_supervised/improving_large_vision_and_language_models_by_learning_from_a_panel_of_peers.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[ICLR 2026\] PonderLM: Pretraining Language Models to Ponder in Continuous Space](../../ICLR2026/self_supervised/ponderlm_pretraining_language_models_to_ponder_in_continuous_space.md)
- [\[ICML 2026\] Scaling Continual Learning to 300+ Tasks with Bi-Level Routing Mixture-of-Experts](../../ICML2026/self_supervised/scaling_continual_learning_to_300_tasks_with_bi-level_routing_mixture-of-experts.md)
- [\[ICLR 2026\] Why Prototypes Collapse: Diagnosing and Preventing Partial Collapse in Prototypical Self-Supervised Learning](../../ICLR2026/self_supervised/why_prototypes_collapse_diagnosing_and_preventing_partial_collapse_in_prototypic.md)

</div>

<!-- RELATED:END -->
