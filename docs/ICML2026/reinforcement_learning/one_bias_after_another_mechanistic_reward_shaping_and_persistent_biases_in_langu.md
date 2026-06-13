---
title: >-
  [Paper Note] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models
description: >-
  [ICML 2026][Reinforcement Learning][Reward Models] This paper systematically measures five types of biases—length, uncertainty, position, sycophancy…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Reward Models"
  - "Reward Hacking"
  - "Linear Probing"
  - "Null-Space Projection"
  - "RM Bias"
date: 2026-05-08
content_hash: 050a0d50f4d57f4b
---

# One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models

**Conference**: ICML 2026  
**arXiv**: [2603.03291](https://arxiv.org/abs/2603.03291)  
**Code**: https://github.com/drfein/OneBiasAfterAnother (Available)  
**Area**: RLHF Alignment / AI Safety  
**Keywords**: Reward Models, Reward Hacking, Linear Probing, Null-Space Projection, RM Bias

## TL;DR
This paper systematically measures five types of biases—length, uncertainty, position, sycophancy, and model style—in five high-quality RMs (including the SOTA Skywork-Reward-V2). It classifies these into "low-complexity" (linearly fixable) and "high-complexity" (linearly unfixable) groups. The authors propose mechanistic reward shaping—using DiffMean linear probes for null-space projection on the final-layer hidden states—to significantly mitigate the first three biases without compromising RewardBench2 accuracy, while demonstrating OOD generalization to Best-of-N.

## Background & Motivation

**Background**: RLHF is the mainstream approach for aligning LMs, but RMs acting as proxy rewards are easily exploited by policy-learned reward hacking. Biases such as length, position, overconfidence, and sycophancy have been frequently documented. Existing remedies either modify training data, add length penalties, or train robust RMs, most of which treat biases as linear spurious correlations.

**Limitations of Prior Work**: (1) Even recent SOTA RMs (Skywork-Reward-V2 series, AllenAI-Llama-8B) continue to exhibit old biases, and RMs trained to fix length bias often show "verbosity penalty" over-correction—ranking concise incorrect answers higher than correct long ones. (2) Current post-hoc fixes (e.g., length penalty) rely on explicit modeling of the bias function form and fail in prompt-conditioned scenarios (Best-of-N). (3) There is no systematic distinction between biases that are indeed linear spurious correlations and those that are entangled and require deeper intervention, leading to one-size-fits-all methods being wasted on unresolvable high-complexity biases.

**Key Challenge**: Biases may be co-linear with useful signals in the RM activation space; unidirectional intervention either fails to fix the bias or erases beneficial signals.

**Goal**: (i) Re-audit known biases and uncover new ones in the latest RMs; (ii) Provide an empirical classification of "linearly fixable vs. unfixable"; (iii) Design an intervention method that is effective for low-complexity biases, data-efficient, in-model (no change to policy optimizer), and OOD-generalizable.

**Key Insight**: Driven by the linear representation hypothesis from Park et al. (2024a)—that high-level concepts are approximately linear directions in the representation space. If a bias is primarily carried by a single linear direction, nulling that direction can achieve local debiasing; if the bias and the true signal are already entangled in the same subspace, linear nulling will naturally be ineffective, which itself serves as a useful diagnostic signal.

**Core Idea**: Construct a difference-of-mean probe using pairs of "biased vs. unbiased" samples, and perform null-space projection of this probe direction on the final layer hidden states (mechanistic reward shaping). This both fixes the bias and identifies which biases are inherently unfixable.

## Method

### Overall Architecture
The method consists of three steps: (1) Bias Auditing—systematically measuring five bias types in five RMs (Skywork-Llama-8B, Skywork-Qwen-8B/0.6B, AllenAI-Llama-8B, DeBERTa-large-v2) across PlausibleQA, BigBench, GSM8K-MC, and MMLU. (2) Constructing contrastive datasets for each bias type, extracting the RM's final layer non-padding token hidden states, and calculating the linear probe $\mathbf{p}$ using DiffMean. (3) During inference, the hidden state $\mathbf{h}$ of each new input is projected: $\mathbf{h}_{\text{null}} = \mathbf{h} - \sum_k \alpha (\mathbf{p}_k^{\top}\mathbf{h})\mathbf{p}_k$, then fed into the reward head to get the debiased reward. For multiple probes, Gram-Schmidt orthogonalization is performed before joint nulling.

Input: Prompt-completion pair; Output: Debiased scalar reward. The entire intervention occurs internally within the RM, requiring no RM retraining or changes to the policy optimization algorithm, making it naturally compatible with Best-of-N, red-teaming, data filtering, and other alignment techniques using RMs as a foundation.

### Key Designs

1.  **Empirical Taxonomy (Bias Complexity Classification)**:
    - **Function**: Empirically divides RM biases into low-complexity (linearly fixable, e.g., length, uncertainty, position) and high-complexity (linearly unfixable, e.g., sycophancy, model-style sensitivity) as a diagnostic criterion for applying probe-nulling.
    - **Mechanism**: The authors define "mechanistic" in the narrow sense of Saphra & Wiegreffe (2024)—focusing on whether identifying and removing a direction in the activation space leads to measurable causal changes in reward behavior, rather than circuit-level explanations. If the dominant signal of a bias can be approximated by a single linear direction, nulling it will significantly narrow the target bias without harming baseline accuracy. Otherwise, the bias is co-linear with quality signals, requiring deeper solutions. Iterative Nullspace Projection (INLP) provides independent representation-level evidence (Appendix C.9).
    - **Design Motivation**: In real-world deployment where resources are limited, low-cost intervention can be used for low-complexity biases, allowing research effort to be directed toward truly difficult high-complexity biases; meanwhile, "unfixability" itself becomes a publishable empirical conclusion.

2.  **Mechanistic Reward Shaping (DiffMean Probe + Null-Space Projection)**:
    - **Function**: Uses minimal labeled data to construct a single direction and subtracts the component of this direction from the RM hidden states to achieve targeted bias removal.
    - **Mechanism**: For each bias type, positive and negative contrastive sets $\{\mathbf{h}_i^+\}, \{\mathbf{h}_j^-\}$ are constructed (e.g., for length bias, using GSM8K verbose-correct vs. concise-correct). The hidden state of the last non-padding token before the reward head is taken, and the probe is calculated via DiffMean, $\mathbf{p} = \text{normalize}(\frac{1}{n_+}\sum_i \mathbf{h}_i^+ - \frac{1}{n_-}\sum_j \mathbf{h}_j^-)$, which AxBench verifies as a strong method. During inference, activations are projected onto the orthogonal complement: $\mathbf{h}_{\text{null}} = \mathbf{h} - \sum_k \alpha (\mathbf{p}_k^{\top}\mathbf{h})\mathbf{p}_k$, where $\alpha$ controls intensity ($\alpha=1$ except for confidence calibration).
    - **Design Motivation**: Unlike global post-processing such as length penalty or ensemble methods, this approach does not require assuming a functional form for the bias and can perform "surgical" removal within the RM latent space. It is plug-and-play for any RM-based technology. It is highly data-efficient (a length probe from GSM8K can transfer OOD to RewardBench2 and AlpacaEval BoN).

3.  **Contrastive Data Construction Paradigm for Five Bias Types**:
    - **Function**: Creates a reproducible paradigm for mining and quantifying RM biases, providing a complete pipeline of "diagnosis + probe construction + intervention evaluation" for each bias.
    - **Mechanism**: (a) **Length**: Constructs (concise-correct, incorrect, verbose-correct) triplets on GSM8K (verbose ~477 words vs. concise ~171 words) to see if RMs prefer incorrect but verbose answers. (b) **Uncertainty**: Adds prefixes like "I'm not exactly sure..." to check if the RM satisfies $r(C) \geq r(C+U) \geq r(I+U) \geq r(I)$ normative ranking. (c) **Calibration**: Appends `confidence: {low, medium, high}` to check if Spearman(confidence, correctness) improves. (d) **Position**: Rotates correct answers A-D in MCQA and measures preference differences for start/end positions in free-form. (e) **Model Style**: Uses 10 LMs to calculate per-byte cross-entropy and measures the Spearman correlation between RM rewards and panel-relative $\Delta s_m$. Any non-zero correlation indicates a systematic preference for a specific "familiar style."
    - **Design Motivation**: Allows future researchers to apply the same paradigm to newly released RMs and extend mechanistic reward shaping to new biases at low cost.

### Loss & Training
Ours is **completely training-free**. All interventions are inference-time linear projections. The only "parameters" are the sample size for probe construction and the projection strength $\alpha \in \{0.5, 1.0, 1.5\}$. Calibration experiments show that $\alpha$ can be adjusted; for RMs with already low bias, like the Llama8B series, $\alpha=0.5$ performs better than $\alpha=1.0$.

## Key Experimental Results

### Main Results

| Bias Type | Baseline Performance | After Intervention | Significant Mitigation? |
| :--- | :--- | :--- | :--- |
| Length (DeBERTa prefers verbose) | Classic length bias, Spearman(reward, length) = 0.611 | 0.067 (95% CI does not overlap) | Yes |
| Length (SOTA RM over-correction) | Prefers concise-wrong > verbose-right | Reduced to no longer favoring wrong answers, no accuracy drop | Yes |
| Uncertainty | Correct answers with "I'm not sure" → RM accuracy drops by 22.6% on average | Prefers uncertainty in wrong answers, remains direct in correct ones | Yes |
| Calibration (Skywork-Qwen-8B) | Spearman(confidence, correctness) = 0.182 | 0.386 at $\alpha=1.0$ (doubled), becomes strongest calibrated RM | Yes |
| Position (MCQA A-D) | Cross-position deviation 2-28% | Significant reduction in position variance for three models | Partial |
| RewardBench2 Overall Accuracy | 70.1% | 69.3% / 69.3% / 70.1% / 69.3% after Length / Position / Uncertainty / Combined | All pass 5pp non-inferiority test ($p < 0.001$) |

### Ablation Study

| Experimental Setting | Key Finding |
| :--- | :--- |
| Best-of-N on AlpacaEval (5 RM, 512 prompts × 64 cand) | Average within-prompt $\|r,L\|$ correlation from 0.10 → 0.04; 4/5 RMs improved length-controlled win rate, better than global calibration in Huang et al. (2025). |
| Best-of-N on GSM8K (5 RM, 64 generations) | Within-prompt correlation from 0.076 → 0.007; mean BoN accuracy 62.1% → 62.8%. |
| Sycophancy (regressive) | All 5 RMs significantly sycophantic; even Skywork-Qwen-8B followed wrong answers 23.7% of the time. |
| Model-Style Sensitivity | All 5 RMs show statistically significant reward-style correlation; average abs correlation ≈0.1, up to ±0.2~0.4 for individual LMs. |
| Calibration $\alpha$ Scan | Skywork-Qwen peaks at $\alpha=1.0$; Llama8B series better at $\alpha=0.5$—indicating models with less bias need lighter intervention. |

### Key Findings
- **Crucial "Aha" Moment**: SOTA RMs do not lack length bias; rather, they **penalize** verbosity because specialized data designed to eliminate length bias introduced an opposite bias. This forces RMs to prefer "concise but wrong" answers on GSM8K.
- **Strong OOD Transfer**: A length probe constructed only from GSM8K math problems successfully drives reward-length correlation to near zero on RewardBench2 and AlpacaEval BoN without losing ranking accuracy.
- **Actionable Taxonomy**: Sycophancy and model-style biases cannot be eradicated even with iterative INLP projection, indicating a need for solutions beyond activation space (e.g., SAEs or behavioral interventions).
- **Style Bias Impact**: Mainstream RMs may systematically reward "familiar writing styles" rather than true quality, which has direct implications for RM-policy pairing and data filtering.

## Highlights & Insights
- **Framing "unfixability" as a main result**: Instead of hiding failure, the authors present high-complexity biases as a formal contribution with empirical and representation-level (INLP) evidence.
- **Engineering-friendly inference-time intervention**: No changes to RM weights or policy algorithms; new biases can be added with just a pair of contrastive samples.
- **Transferable Paradigm**: The DiffMean + null-space projection framework can be applied to other single-head scalar output models (discriminators, toxicity classifiers, etc.).
- **Revealing Model-Family Bias**: Quantifying RM preference for specific LM styles via panel-relative cross-entropy is a new, repeatable diagnostic paradigm.

## Limitations & Future Work
- **Limitations**: (1) Linear intervention remains ineffective for high-complexity biases. (2) The length probe introduces a slight positive correlation in some SOTA RMs, suggesting $\alpha$ needs model-specific tuning. (3) Evaluation is limited to reasoning/knowledge benchmarks and lacks long-tail domains like safety or chat.
- **Ours**: (a) Probe construction is sensitive to contrastive sample selection; if "verbose" samples contain more reasoning traces, the projection might inadvertently nullify "reasoning" directions. (b) All evaluated RMs are single-output scalar architectures; transferability to generative RMs or critics remains unproven. (c) Projection strength $\alpha$ requires manual tuning without an automated selection criterion.
- **Future Work**: Upgrading DiffMean to SAE feature nulling; attempting intervention in deeper layers for sycophancy; expanding probe construction to safety and chat domains.

## Related Work & Insights
- **vs. Park et al. (2024b) / Huang et al. (2025) length penalty**: These use explicit functional forms of length-reward; ours does not assume form and outperforms global calibration in BoN experiments.
- **vs. Ravfogel et al. (2020) INLP**: INLP projects until an attribute is unpredicted; ours uses a lighter weight single-step DiffMean.
- **vs. Casademunt et al. (2025) SAE-based steering**: They use SAEs to extract directions; ours uses simpler DiffMean, suggesting a future merger of these methods for finer bias decomposition.
- **vs. Sharma et al. (2024) sycophancy evaluation**: We filter cases where the RM was already wrong, providing a cleaner measure of sycophancy and proving its linear unfixability.

## Rating
- Novelty: ⭐⭐⭐⭐ — The "Complexity Taxonomy + Mechanistic Reward Shaping" is a clear new framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Extensive testing across RMs, benchmarks, and OOD scenarios with representation-level evidence.
- Writing Quality: ⭐⭐⭐⭐ — Honest reporting of failures; clear taxonomy; although tables are dense and some details are in appendices.
- Value: ⭐⭐⭐⭐⭐ — Directly applicable to RLHF pipelines and provides a roadmap for bias research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2026\] CAMEL: Confidence-Gated Reflection for Reward Modeling](camel_confidence-gated_reflection_for_reward_modeling.md)
- [\[NeurIPS 2025\] Checklists Are Better Than Reward Models For Aligning Language Models](../../NeurIPS2025/reinforcement_learning/checklists_are_better_than_reward_models_for_aligning_langua.md)
- [\[ICML 2026\] From Reward-Free Representations to Preferences: Rethinking Offline Preference-Based Reinforcement Learning](from_reward-free_representations_to_preferences_rethinking_offline_preference-ba.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
