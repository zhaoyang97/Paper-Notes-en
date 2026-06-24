---
title: >-
  [Paper Note] Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling
description: >-
  [ICLR2026][LLM Pretraining][Distilled Pretraining] This paper systematically deconstructs the gains and losses of "Distilled Pretraining (DPT)" under the modern LLM paradigm. It finds that distillation significantly enhances test-time scaling (pass@k diversity) but simultaneously impairs in-context learning (weakening induction heads). Using a bigram sandbox, the authors prove that these opposing effects stem from the same mechanism: distillation only benefits high-entropy di…
tags:
  - "ICLR2026"
  - "LLM Pretraining"
  - "Distilled Pretraining"
  - "Test-Time Scaling"
  - "In-Context Learning"
  - "Induction Heads"
  - "Bigram Sandbox"
date: 2026-05-08
content_hash: f216ca761d9e3943
---

# Distilled Pretraining: A modern lens of Data, In-Context Learning and Test-Time Scaling

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=PNm2dl7HcY](https://openreview.net/forum?id=PNm2dl7HcY)  
**Code**: To be confirmed  
**Area**: LLM Pretraining / Knowledge Distillation  
**Keywords**: Distilled Pretraining, Test-Time Scaling, In-Context Learning, Induction Heads, Bigram Sandbox

## TL;DR
This paper systematically deconstructs the gains and losses of "Distilled Pretraining (DPT)" under the modern LLM paradigm. It finds that distillation significantly enhances test-time scaling (pass@k diversity) but simultaneously impairs in-context learning (weakening induction heads). Using a bigram sandbox, the authors prove that these opposing effects stem from the same mechanism: distillation only benefits high-entropy distributions and is unhelpful or even harmful for low-entropy deterministic mappings. Finally, practical pretraining suggestions such as token routing are provided.

## Background & Motivation
**Background**: Knowledge distillation (training a student with teacher soft labels) was previously used mainly for post-training and model compression. Early LLMs (GPT-2/3, Llama 1/2) rarely used it during pretraining. However, it has made a strong comeback recently—Llama-3.2 and Gemma-3 both distilled small models during pretraining, and Llama-4-Maverick was distilled entirely from Llama-4-Behemoth. The reality is that deploying ultra-large models is too expensive; they may serve only as "teachers" in the future, while production models are likely to rely entirely on distilled pretraining.

**Limitations of Prior Work**: Despite its growing importance, the "science" of distillation in pretraining is under-researched. While Gemma-3 and Llama-3.2 show gains on standard benchmarks, these teachers typically see much more data than the students. A fundamental question remains: are the gains from distillation due to the teacher's "extra data" or inherent to distillation itself? Will distillation still be effective after hitting the "data wall" (where teacher and student see the same amount of data)?

**Key Challenge**: More importantly, modern LLM capabilities have evolved beyond "standard language modeling." Test-time scaling (sampling multiple times to find the best) and in-context learning (ICL, learning on the fly from prompts) are the current frontiers, yet the impact of distilled pretraining on these paradigms is unknown. The authors suspect that the same distillation that boosts standard benchmarks may not be equally friendly to these new capabilities.

**Goal**: To address three sub-problems: (1) Does distillation remain effective under data-controlled (IsoData, equal data for teacher and student) settings? (2) What are the respective impacts of distillation on test-time scaling and ICL? (3) If the impacts are contradictory, do they stem from the same underlying mechanism?

**Key Insight**: The authors analyze distillation through the lens of "distribution entropy." The value of soft labels lies in spreading probability across multiple plausible answers. Thus, it naturally provides information increments only for high-entropy cases with multiple valid continuations. For low-entropy deterministic mappings where the answer is unique, it offers little to no benefit. Induction heads, which drive ICL, represent low-entropy copy operations, whereas the diversity required for test-time scaling relies on high-entropy distributions.

**Core Idea**: Use a single axis of "high-entropy vs. low-entropy distributions" to unify the explanation of why distillation helps test-time scaling while harming ICL, and design selective distillation (token routing) based on this finding.

## Method

### Overall Architecture
This is an analytical paper following an empirical and theoretical investigation chain rather than proposing a new model architecture. The chain consists of: **IsoData experiments** (a student 1B trained on the same 1T tokens as an 8B teacher) to eliminate the "extra data" confounder; evaluating these models from **two modern perspectives**—ICL (measuring induction-head-driven copying) and test-time scaling (measuring pass@k diversity); reducing the phenomena to a **tractable bigram sandbox** to prove that learning differences between high/low-entropy rows are the root cause with sample complexity propositions; and finally, proposing **token routing** to mitigate the negative effects.

### Key Designs

**1. IsoData Distillation Experiment: Removing the "Extra Teacher Data" Confounder**

Existing distillation gains are often suspected of being results of the teacher having seen more data. The authors construct an "equal data" control: an 8B teacher is trained on 1T tokens, then a 1B student is trained on the **exact same 1T tokens**, with and without distillation. Results show that even when data is strictly controlled, distilled students outperform training-from-scratch on benchmarks like HellaSwag, NaturalQuestions, and MBPP. This confirms that distillation remains valuable even when hitting "data walls."

**2. Modern Dual-Perspective Metrics: ICL (Induction Heads) vs. Test-Time Scaling (pass@k)**

The authors separate evaluation into two critical but opposing capabilities. For **In-Context Learning**, they use three types of tasks measuring "copying from context": Contextual QA (DROP, RACE), Needle In A Haystack (babilong), and Counterfactual Contextual QA (where the correct answer conflicts with parametric knowledge). These rely on "induction heads" that copy previous tokens—a low-entropy deterministic mapping. For **Test-Time Scaling**, they use pass@k (giving $k$ attempts per question). While pass@1 only requires the top-1 to be correct, pass@k requires spreading probability mass across multiple plausible answers, testing "generative diversity." The authors find a sharp contrast: as data scales to 1T, distillation's advantage on ICL disappears and even becomes a disadvantage in counterfactual tasks; conversely, for pass@k, DPT-90 significantly outperforms standard pretraining (SPT-2x) using only half the data on GSM/MATH—the diversity gain is equivalent to seeing twice the data.

**3. Bigram Sandbox + Sample Complexity Analysis: Unifying Contrasting Phenomena**

The authors use a bigram model (first-order Markov chain) with a transition matrix $\pi \in \mathbb{R}^{k\times k}$ to explain the cause. Rows are categorized by entropy: high-entropy rows like "I go to" (office/gym/restaurant) vs. low-entropy rows like "2+3=" (answer fixed at 5). Experiments show distilled students approach the true distribution faster on high-entropy rows but show no difference on low-entropy rows. Theoretically, given at most $p$ non-zero entries per row, distillation sample complexity is $S_{\text{distill}} = O(k\log k)$, while standard training is $S_{\text{standard}} \approx \frac{p}{\epsilon^2} S_{\text{distill}}$. For high-entropy rows where $p=O(k)$, distillation is vastly more efficient. For low-entropy rows where $p$ is constant, both are $O(k\log k)$. Induction heads are modeled as low-entropy rows:

$$\tilde\pi_{ji} = \begin{cases} \pi_{ji} & i \neq t \\ \mathbb{I}(j=c) & i = t \end{cases}$$

Here, a trigger token $t$ indicates the next token is a deterministic copy $c$. Distillation provides no new information here; in fact, an imperfect teacher introduces noise into the soft labels, slowing the formation of induction heads.

**4. Token Routing: Selective Distillation Based on Teacher Entropy**

To mitigate ICL degradation, the authors propose avoiding distillation on low-entropy positions. The standard distillation objective:

$$h^\dagger \in \arg\min_{h}\ \frac{1}{n}\Big[(1-\alpha)\sum_{i}\ell\big(y_i,\sigma(h(x_i))\big) + \alpha\sum_{i}\ell\big(s_i,\sigma(h(x_i))\big) \Big],\quad s_i=\sigma\!\big(h_{\text{teacher}}(x_i)/T\big)$$

In token routing, the teacher's entropy is calculated for each input. At the lowest $x\%$ entropy positions, the distillation term is dropped, and the model reverts to **pure ground-truth supervision**; high-entropy positions are distilled normally. At $x=15\%$, ICL performance recovers (matching standard pretraining) without losing gains on standard benchmarks.

### Loss & Training
The core objective is Hinton-style soft-label distillation. Weight $\alpha$ controls intensity (DPT-50 and DPT-90). Token routing acts as a **position-level gate** on this loss, switching back to hard labels for the lowest $x\%$ entropy positions. Main experiments use 1B students and 8B teachers (or Llama-3.1-8B) with data scaled from 125B to 1T tokens.

## Key Experimental Results

### Main Results

| Perspective | Task/Metric | Key Phenomenon |
|------|-----------|----------|
| Standard LM (IsoData) | Avg. HellaSwag / NaturalQA / MBPP | Distillation outperforms scratch even with equal 1T data. |
| Test-Time Scaling | GSM8K pass@16 | DPT 27–28% vs SPT 23%, despite similar or lower pass@1. |
| Test-Time Scaling | GSM/MATH/MBPP pass@k | DPT-90 (half data) exceeds SPT-2x (double data) in pass@16. |
| In-Context Learning | Needle / Counterfactual QA | Distillation advantage vanishes or becomes negative at 1T tokens. |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| SPT (Standard) | ICL / pass@k baseline | No distillation. |
| DPT (Vanilla) | ICL↓, pass@k↑ | Distillation on all positions; shows primary trade-off. |
| DPT + Token Routing (15%) | ICL recovers, standard tasks intact | Lowest 15% tokens revert to hard labels. |
| DPT + Token Routing (30%) | ICL gain stalls, standard benchmark drops | Skip ratio too high. |

### Key Findings
- Distillation helps pass@k but hurts ICL due to the same mechanism: soft labels contain information for high-entropy distributions but introduce noise for low-entropy deterministic mappings.
- Bigram sample complexity quantifies this: distillation reduces complexity from $O(k^2\log k)$ to $O(k\log k)$ for high-entropy rows but provides no gain for low-entropy rows.
- Token routing at 15% is the "sweet spot"—it recovers ICL without hurting standard performance.

## Highlights & Insights
- **Unifying contrasting phenomena with "entropy"**: Attributing the pros and cons of distillation to entropy is a powerful insight that explains both pass@k gains and induction head damage.
- **IsoData design**: Isolating the "extra data" confounder ensures the causal attribution is robust, a good experimental hygiene practice for analytical work.
- **Tractable Bigram Sandbox**: Reducing complex ICL phenomena to a first-order Markov chain with a copy structure allows for both empirical testing and theoretical proof.
- **Engineering value of Token Routing**: A cost-free position-level gate allows practitioners to mitigate ICL side effects of distillation directly.

## Limitations & Future Work
- The scale is limited to 1B students / 8B teachers; validation on larger models or longer training is needed.
- Token routing is a preliminary "concept validation"; optimal thresholds and synergy with other data filtering require more exploration.
- The bigram sandbox is an analogy; while it captures the essence of induction heads, the formal rigor relies on appendix proofs.
- pass@k is a proxy for diversity; whether downstream tasks like RLVR benefit similarly requires further study.

## Related Work & Insights
- **vs. Llama-3.2 / Gemma-3**: These report standard benchmark gains with extra teacher data. This paper proves distillation is effective even in IsoData settings but reveals trade-offs in ICL/test-time scaling.
- **vs. Cha & Cho (2025)**: They study hard-label distillation (synthetic data) and find diversity issues. This paper focuses on soft-labels, concluding that they actually *improve* diversity by smoothing high-entropy distributions.
- **vs. Busbridge et al. (2025)**: They argue distillation is not always useful in compute-matched settings. This paper suggests evaluating in data-constrained settings where teacher logit costs are secondary to data quality.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying the impact on two modern paradigms via entropy is insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐ IsoData, dual perspectives, and routing form a closed loop, though limited in scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Very clear progression from phenomenon to mechanism to mitigation.
- Value: ⭐⭐⭐⭐⭐ Provides direct practical insights for teams pretraining small models via distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TNT: Improving Chunkwise Training for Test-Time Memorization](tnt_improving_chunkwise_training_for_test-time_memorization.md)
- [\[ICLR 2026\] Beyond Length: Quantifying Long-Range Information for Long-Context LLM Pretraining Data](beyond_length_quantifying_long-range_information_for_long-context_llm_pretrainin.md)
- [\[ICCV 2025\] ETA: Energy-based Test-time Adaptation for Depth Completion](../../ICCV2025/llm_pretraining/eta_energy-based_test-time_adaptation_for_depth_completion.md)
- [\[ICLR 2026\] Scaling Laws Revisited: Modeling the Role of Data Quality in Language Model Pretraining](scaling_laws_revisited_modeling_the_role_of_data_quality_in_language_model_pretr.md)
- [\[ICLR 2026\] Reformulation for Pretraining Data Augmentation](reformulation_for_pretraining_data_augmentation.md)

</div>

<!-- RELATED:END -->
