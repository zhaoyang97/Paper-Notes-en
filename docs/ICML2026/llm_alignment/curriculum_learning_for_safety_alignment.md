---
title: >-
  [Paper Note] Curriculum Learning for Safety Alignment
description: >-
  [ICML 2026][LLM Alignment][DPO] This paper proposes Staged-Competence—a DPO safety alignment framework utilizing "model-specific preference alignment margin" as difficulty scores. By combining dual curricula of "staged r…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "DPO"
  - "Safety Alignment"
  - "Curriculum Learning"
  - "OOD Robustness"
  - "Jailbreak Attacks"
date: 2026-05-08
content_hash: 4cdbf3057d2db0f3
---

# Curriculum Learning for Safety Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.26315](https://arxiv.org/abs/2605.26315)  
**Code**: https://github.com/Sandeep5500/curriculum-learning-for-safety  
**Area**: RLHF Alignment / LLM Safety  
**Keywords**: DPO, Safety Alignment, Curriculum Learning, OOD Robustness, Jailbreak Attacks

## TL;DR
This paper proposes Staged-Competence—a DPO safety alignment framework utilizing "model-specific preference alignment margin" as difficulty scores. By combining dual curricula of "staged reference model updates + intra-stage competence-based sampling," it reduces the OOD harmful response rate by an average of 16% and jailbreak attack success rate by 20% across three 8B-scale LLMs, while maintaining general capabilities and avoiding over-refusal.

## Background & Motivation

**Background**: The mainstream approach for LLM safety alignment involves fine-tuning with DPO on "safe/unsafe" preference pairs $(x, y^+, y^-)$, bypassing the cost of training reward models.

**Limitations of Prior Work**: DPO safety alignment has been proven to be "shallow"—safe behaviors are predominantly concentrated in the first few tokens of a response. Jailbreak attacks like prefill or GCG can bypass the beginning to induce harmful output. Furthermore, generalization to out-of-distribution (OOD) harmful prompts remains poor.

**Key Challenge**: Standard DPO treats all preference pairs as equally difficult via random sampling. However, the "difficulty" of a preference pair is not based on linguistic complexity but on the **unaligned base model's** ability to distinguish safe from unsafe responses. Ignoring this model-dependent difficulty wastefully applies gradient signals to "easy" samples the model can already distinguish.

**Goal**: (1) Design a model-specific, globally comparable difficulty score; (2) Develop a training algorithm that utilizes this difficulty within the DPO framework; (3) Significantly improve OOD and jailbreak robustness without modifying the DPO loss or introducing new hyperparameter families.

**Key Insight**: The authors leverage the "easy-to-hard" principle of curriculum learning (Bengio 2009). They identify flaws in existing approaches: competence-based curriculum (Sqrt-Competence) lacks reference model updates, while Curri-DPO fails to maintain curriculum order within its stages. These approaches should be integrated rather than treated as alternatives.

**Core Idea**: The "difference in cosine similarity between a base model's zero-shot response and $y^+$ vs $y^-$" is used as a global difficulty score. The dataset is sorted and divided into $K=3$ buckets. Between buckets, the reference model $\pi_\text{ref}$ is updated; within buckets, the sampling pool is gradually expanded using a $\sqrt{\cdot}$ competence scheduler. The curriculum operates simultaneously at the "macro-stage" and "micro-step" scales.

## Method

### Overall Architecture
Staged-Competence is a two-phase pipeline wrapped around standard DPO, leaving the DPO loss itself unchanged.

**Phase 1 (Scoring)**: The base model $\pi_0$ generates zero-shot responses $\hat y_i$ for each prompt. A lightweight sentence encoder (all-MiniLM-L6-v2) encodes $\hat y_i, y_i^+, y_i^-$ to compute global difficulty scores. The entire dataset is sorted from easy to hard.

**Phase 2 (Training)**: The sorted data is split into $K=3$ buckets of increasing difficulty ($\mathcal B_1, \mathcal B_2, \mathcal B_3$), trained in $K$ stages. Within each stage, rather than random shuffling, a competence function $c(t)$ dynamically expands the "qualified pool." Initially, only the easiest samples in the bucket are sampled, with harder samples added at a $\sqrt{\cdot}$ rate as steps increase. After each stage, the policy $\pi^{(k)}$ becomes the reference model $\pi_\text{ref}^{(k+1)}$ for the next stage.

Input: Preference dataset $\mathcal D = \{(x_i, y_i^+, y_i^-)\}$, unaligned base $\pi_0$; Output: Aligned policy $\pi^{(K)}$.

### Key Designs

1.  **Preference Alignment Margin (Model-dependent Global Difficulty Score)**:
    - **Function**: Quantifies "how difficult a preference pair is for the current model" into a scalar for global sorting.
    - **Mechanism**: The unaligned base model generates a zero-shot response $\hat y_i$ for prompt $x_i$. The encoder computes $m_i = \cos(e_{\hat y_i}, e_{y_i^+}) - \cos(e_{\hat y_i}, e_{y_i^-})$. A larger $m_i$ indicates the base model is naturally closer to the safe response (easy), whereas a smaller $m_i$ indicates difficulty. Global order is determined by descending margin values.
    - **Design Motivation**: Curri-DPO difficulty is based on pairwise quality differences among four candidates per prompt, allowing only local sorting. The proposed margin enables cross-prompt comparison, which is essential for migrating competence-based sampling to DPO without relying on external GPT-4 judges.

2.  **Staged Reference Update (Inter-stage Reference Propagation)**:
    - **Function**: Anchors the optimization objective of each stage to the achievements of the previous stage, preventing the model from re-learning easy samples.
    - **Mechanism**: Sorted data is cut into $K=3$ equal buckets. Stage $k$ runs DPO on $\mathcal B_k$ for $E$ epochs using $\pi_\text{ref}^{(k)}$, after which $\pi_\text{ref}^{(k+1)} \leftarrow \pi^{(k)}$.
    - **Design Motivation**: Fixed references in standard DPO dilute late-stage gradients with already-learned easy pairs. Iterative updates, as evidenced by reward margin "jumps" at stage transitions (Fig. 2), inject fresh effective gradients.

3.  **Within-Stage Competence Sampling (Intra-stage Qualified Pool Expansion)**:
    - **Function**: Utilizes the curriculum order within a single stage to avoid regression to uniform sampling within buckets.
    - **Mechanism**: Normalized difficulty $d_i \in [0,1]$ is assigned to samples within bucket $\mathcal B_k$. At step $t$, the competence function $c(t) = \sqrt{(1-c_0^2)\,t/T + c_0^2}$ (with $c_0=0.01$) determines the difficulty threshold. Mini-batches are sampled from the dynamic pool $\{i \in \mathcal B_k : d_i \le c(t)\}$. The $\sqrt{\cdot}$ shape allows harder samples to be added at a decreasing rate, giving the model time to "digest."
    - **Design Motivation**: Sqrt-Competence alone lacks reference updates, and Curri-DPO discards intra-stage signals. Combining them ensures consistency across both macro and micro scales.

### Loss & Training
The DPO loss remains standard: $\mathcal L_\text{DPO} = -\mathbb E\,[\log \sigma(\beta(\log\frac{\pi_\theta(y^+|x)}{\pi_\text{ref}(y^+|x)} - \log\frac{\pi_\theta(y^-|x)}{\pi_\text{ref}(y^-|x)}))]$ with $\beta=0.1$. Training utilizes LoRA ($r{=}16, \alpha{=}32$, q/v projections), lr $5{\times}10^{-5}$, effective batch size 32, and sequence length 1024. The staged method uses $K{=}3$ stages with 5 epochs each (4 epochs for Yi-1.5-9B to avoid over-optimization). Training is feasible on a single A6000 (48GB).

The authors also cleaned the data, discovering that 82.2% of "chosen" responses in PKU-SafeRLHF and 87.2% of "rejected" responses in HH-RLHF were incorrectly labeled. They released the Cleaned-PKU-HH-SafeRLHF dataset filtered by GPT-4o-mini.

## Key Experimental Results

### Main Results: OOD Safety and Jailbreak Attacks

| Model | Metric | Standard DPO | Curri-DPO | Staged-Competence | Gain (vs DPO) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| LLaMA-3-8B | Avg OOD Harmful Rate ↓ | 23.6% | 17.1% | 11.4% | -12.2 pp |
| Qwen3-8B | Avg OOD Harmful Rate ↓ | 32.9% | 23.0% | 4.0% | -28.9 pp |
| Yi-1.5-9B | Avg OOD Harmful Rate ↓ | 8.8% | 4.5% | 1.7% | -7.1 pp |
| LLaMA-3-8B | Avg Prefill/GCG Attack ↓ | 35.1% | 27.0% | 16.3% | -18.8 pp |
| Qwen3-8B | Avg Prefill/GCG Attack ↓ | 39.3% | 27.3% | 12.3% | -27.1 pp |
| Yi-1.5-9B | Avg Prefill/GCG Attack ↓ | 19.2% | 13.8% | 5.4% | -13.8 pp |

Average across three models: OOD harmful rate decreased by 16%, and attack success rate decreased by 20%. General capabilities (MMLU/HellaSwag) remained stable, and the XSTest over-refusal rate was near zero.

### Ablation Study

| Configuration | Key Effect | Description |
| :--- | :--- | :--- |
| Standard DPO | Baseline | Random sampling, single stage, fixed reference |
| Sequential | OOD -5~11 pp | Difficulty-sorted feeding, single stage, fixed reference |
| Sqrt-Competence | +0.5 pp on Qwen3 (Worse) | Single-stage competence sampling without reference updates |
| Curri-DPO | OOD -4~10 pp | Multi-stage + reference updates, but random intra-stage sampling |
| **Staged-Competence** | OOD -7~29 pp, Attack -14~27 pp | Intra-stage competence + inter-stage reference updates |
| Efficiency: 75% Data | Matches/Exceeds 100% DPO | Saves 25% of preference data for same safety level |
| Scaling: Qwen3 1.7B→8B | Advantage grows 1.5pp→29pp | Larger models have worse baselines, increasing curriculum value |

### Key Findings
- **Reward margin vs. accuracy**: While ID accuracy for Staged-Competence is similar to the baseline (88–91%), the reward margin increases to roughly 3× the baseline, showing significant jumps at stage transitions. This validates the effectiveness of reference model updates.
- **"Deeper" safety alignment**: Token-level suppression analysis $\delta(t) = \log\pi_\text{unaligned}(y_t|\cdot) - \log\pi_\text{aligned}(y_t|\cdot)$ shows that Staged-Competence suppresses unsafe tokens significantly harder across the first 128 tokens (approx. 3× the cumulative suppression of standard DPO). This explains the sharp drop in prefill attack success—resistance persists deep into the response.
- **Scaling curriculum dividends**: For the Qwen3 series (1.7B to 8B), standard DPO OOD harmfulness worsened from 6.5% to 32.9%, whereas Staged-Competence remained stable at 2–8%. The marginal value of curriculum learning increases with model size as "alignment failure" risks grow.

## Highlights & Insights
- **Model-dependent difficulty is crucial**: Using the zero-shot embedding margin as a global scalar allows for a cheap, universal, and effective migration of competence-based curricula to preference optimization.
- **Two-scale progression**: Inter-stage updates handle macro progression, while intra-stage competence handles micro-level feeding. The empirical evidence showing that Sqrt-Competence alone can fail while the combined approach excels is a valuable design pattern.
- **Dataset cleaning as a contribution**: Identifying high noise levels in PKU-SafeRLHF and HH-RLHF reveals that previous DPO safety work was likely hindered by label noise. Cleaned-PKU-HH-SafeRLHF serves as a new high-quality benchmark.
- **Orthogonality**: Since the loss function is not modified, this approach is orthogonal to KTO, IPO, and others, allowing for potential stacked benefits.

## Limitations & Future Work
- **Scale and Tuning**: Evaluation was limited to 8B models and LoRA. Full-parameter fine-tuning and larger scales (70B/MoE) remain unexplored.
- **Judge Bias**: Reliance on GPT-4o-mini for cleaning and evaluation may introduce specific biases, particularly in sensitive categories like biosecurity.
- **Encoder Dependency**: Generic semantic encoders like all-MiniLM-L6-v2 may miss subtle safety-related nuances. Safety-specific encoders or LM hidden states might improve sorting.
- **Hyperparameter Sensitivity**: The number of stages $K$ and epochs was not systematically swept. Automated scheduling based on margin change rates is a potential future direction.

## Related Work & Insights
- **vs. Curri-DPO (Pattnaik 2024)**: Both use $K=3$ stages and reference propagation. **Ours** differs by using model-dependent global margins and intra-stage competence sampling, outperforming Curri-DPO by 9–11pp in OOD and attack metrics.
- **vs. Sqrt-Competence (Platanios 2019)**: **Ours** adapts the scheduling function to the LLM DPO context. In isolation, this method was found ineffective (0.5pp worse than baseline on Qwen3), requiring staged reference updates for synergy.
- **vs. Qi et al. 2024 (Shallow safety alignment)**: This work directly addresses the observation that standard alignment only affects initial tokens. Staged training extends suppression deep into responses.
- **vs. Loss Modification Works**: Methods like KTO/Safe-DPO focus on the objective function, whereas **Ours** focuses on data order and reference policies.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First systematic curriculum learning for DPO safety alignment; innovations lie in the fusion of methods and model-dependent margins.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation across three model families, two attack types, three OOD benchmarks, and token-level mechanistic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear narrative flow; Fig. 2 (stage jumps) and Fig. 3 (token suppression) are highlights.
- **Value**: ⭐⭐⭐⭐⭐ High ROI for the safety community due to its plug-and-play nature, standard loss usage, and released dataset.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Implicit Safety Alignment from Crowd Preferences](implicit_safety_alignment_from_crowd_preferences.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](../../ICLR2026/llm_alignment/superficial_safety_alignment_hypothesis.md)
- [\[ICML 2026\] MESA: Improving MoE Safety Alignment via Decentralized Expertise](mesa_improving_moe_safety_alignment_via_decentralized_expertise.md)
- [\[AAAI 2026\] Differentiated Directional Intervention: A Framework for Evading LLM Safety Alignment](../../AAAI2026/llm_alignment/differentiated_directional_intervention_a_framework_for_evading_llm_safety_align.md)
- [\[ICML 2026\] Quantifying the Salience of Geo-Cultural Values for Pluralistic Safety Alignment](quantifying_the_salience_of_geo-cultural_values_for_pluralistic_safety_alignment.md)

</div>

<!-- RELATED:END -->
