---
title: >-
  [Paper Note] Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook
description: >-
  [ICML 2026][AIGC Detection][Cultural Alignment] DOVE employs rate-distortion variational optimization to automatically construct a compact "Value Codebook" from 10…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "Cultural Alignment"
  - "Value Evaluation"
  - "Value Codebook"
  - "Rate-Distortion"
  - "Unbalanced Optimal Transport"
date: 2026-05-08
content_hash: 098ed6074457c089
---

# Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook

**Conference**: ICML 2026  
**arXiv**: [2604.06210](https://arxiv.org/abs/2604.06210)  
**Code**: None  
**Area**: LLM Alignment / Cultural Value Evaluation  
**Keywords**: Cultural Alignment, Value Evaluation, Value Codebook, Rate-Distortion, Unbalanced Optimal Transport  

## TL;DR
DOVE employs rate-distortion variational optimization to automatically construct a compact "Value Codebook" from 10,000 human texts. It then uses Unbalanced Optimal Transport (UOT) to measure the distribution difference between human and LLM long-form texts in the value space, improving the "Evaluation-Downstream Task" correlation from $\le 24\%$ in baselines to $31.56\%$ across 12 LLMs.

## Background & Motivation

**Background**: Existing LLM cultural value evaluations either directly apply social science surveys (WVS, Hofstede) or use human/LLM-authored multiple-choice questions (MCQs) to let models select the option closest to a specific culture. A few generative approaches merely extract keywords or use LLMs as judges to score open-ended responses.

**Limitations of Prior Work**: The authors summarize these works into a unified issue — the **C³ challenge** (Construct / Composition / Context gaps): (1) Construct gap: Discriminative MCQs only test "value knowledge"; a correct answer does not imply a true value orientation and is sensitive to framing and social desirability bias. (2) Composition gap: Averaging item scores into a total score completely erases the heterogeneity of sub-groups within the same culture. (3) Context gap: Constrained choice tasks are severely misaligned with the open-ended long-text generation scenarios of real LLM deployment.

**Key Challenge**: Faithfully characterizing "the value orientation expressed by an LLM when facing a specific culture" is essentially comparing two **long-text distributions** (human-written vs. LLM-written). However, long texts contain both value signals and significant value-irrelevant content; traditional surveys fail here, bag-of-words/rules are inaccurate, and pure LLM-as-a-judge is unstable.

**Goal**: Construct an open-ended distribution-level evaluation framework that does not rely on pre-defined value systems or option framing, filling all three C³ gaps and providing stronger predictive power for downstream real-world tasks.

**Key Insight**: The authors borrow the tradition of "coding" from social sciences — compressing long documents into a set of discrete "value codes" — viewing it as a lossy compression problem. Thus, **rate-distortion theory** + **VQ-VAE style variational optimization** can be used to automatically learn a value codebook. Distribution comparison is performed using **Unbalanced Optimal Transport (UOT)**, which preserves geometric structures while tolerating quality mismatches caused by sub-groups.

**Core Idea**: Rewrite "evaluating LLM cultural alignment" as "calculating the UOT distance between two distributions over an automatically learned value codebook."

## Method

### Overall Architecture
Given a target culture $\bm{g}$ (e.g., Japan) and an LLM under evaluation $p_{\bm{\theta}}$: (1) Collect human long-texts $\hat p^{\bm g}(\bm x)$ from that culture and LLM-generated long-texts $p_{\bm{\theta}}(\bm x|\bm o)$ on the same topics $\bm o$; (2) On a compact **Value Codebook** $\mathcal{\bm C}=(\bm c_1,\dots,\bm c_K)$ obtained via rate-distortion variational optimization, use a value recognizer $q_{\bm\omega}(z|\bm x,\mathcal{\bm C})$ to project each document into a $K$-dimensional value probability vector; (3) Measure the distance between human and LLM distributions $\bm a^{\bm g}, \bm a^{\bm\theta}$ using **Unbalanced Optimal Transport**, then rescale it into a final alignment score. The process does not fine-tune LLM parameters; $q_{\bm\omega}$ and $p_{\bm\phi}$ are black-box calls optimized via ICL + Variational EM.

### Key Designs

1.  **Rate–Distortion Value Codebook**:
    - **Function**: Automatically extracts a compact, informative, and low-redundancy set of discrete value codes $\mathcal{\bm C}$ from unlabeled long texts to serve as a common "coordinate system," filling the Construct gap.
    - **Mechanism**: Views "document $\bm x \to$ value code sequence $\bm s$" as lossy compression, treating the codebook as discrete latents of VQ-VAE. It derives an ELBO: $\mathbb E_{\hat p(\bm x)}[\log p(\bm x|\mathcal{\bm C})] \ge \mathbb E_{\hat p(\bm x)}\{\mathbb E_{q_{\bm\omega}}[\log p(\bm x|\bm s,\mathcal{\bm C})] - \mathrm{KL}[q_{\bm\omega}\|p(\bm s|\mathcal{\bm C})]\}$ and adds rate–distortion regularization for the final objective (Eq. 3): an information retention term $-\log p_{\bm\phi}(\bm x|\bm s,\mathcal{\bm C})$, a single-document code entropy term $-\beta_1 H_q(\bm s|\bm x,\mathcal{\bm C})$ (encouraging multiple codes to avoid monopoly), and a prior entropy term $\beta_2 H_q(\bm s|\mathcal{\bm C})$ (encouraging uniform code usage). Optimization uses Variational EM-style Bipartite Black-box Optimization (BBO): each round samples $N_1$ code sets $\bm s_j$ to estimate scores $\mathcal S(\mathcal{\bm C}^{t-1})$, followed by (i) **Extension** (splitting high-usage codes with persistent distortion), (ii) **Merge** (merging low-usage codes with nearest neighbors), and (iii) **re-creation** (re-clustering to generate new codes) (Algorithm 1). The value recognizer $q_{\bm\omega}$ extracts $M'$ natural language value phrases $\bm v$ from the document, then performs soft assignment via $q_{\bm\omega}(z=k|\bm x,\mathcal{\bm C})=\frac{1}{M'}\sum_j \mathrm{softmax}_{\mathcal{\bm C}}[\mathrm{sim}(\bm e_{\bm v_j},\bm e_{\bm c_k})/\sigma^2]$.
    - **Design Motivation**: Traditional value evaluations are tied to prior systems like Schwartz/Hofstede (introducing researcher bias) or rely on noisy LLM keyword extraction. Rate-distortion regularization explicitly incorporates "information retention" and "redundancy reduction" while allowing black-box ICL optimization, resulting in a data-driven value coordinate system tailored for evaluation.

2.  **Value-based Unbalanced OT**:
    - **Function**: Compares the value distributions $\bm a^{\bm g}$ and $\bm a^{\bm\theta}$ on the learned codebook to provide an alignment score that reflects sub-group heterogeneity and is robust to sample imbalance, filling the Composition gap.
    - **Mechanism**: Defines the $K$ value codes as centers in the transport space. The UOT objective is $\mathcal D_{\mathrm{UOT}}(\hat p^{\bm g},p_{\bm\theta})=\min_{\bm\pi\ge 0}\sum_{i,j}[D_{i,j}\bm\pi_{i,j}+\epsilon\bm\pi_{i,j}(\log\bm\pi_{i,j}-1)]+\gamma\mathrm{KL}[\bm\pi\bm 1\|\bm a^{\bm g}]+\gamma\mathrm{KL}[\bm\pi^T\bm 1\|\bm a^{\bm\theta}]$. The cost matrix considers not only semantic distance $\rho(\bm c_i,\bm c_j)$ but also a **co-occurrence discount** $1-\mathbb E[\min(\bm a_i,\bm a_j)]/(\mathbb E[\max(\bm a_i,\bm a_j)]+\epsilon_2)$ — transport costs are lower between values that frequently co-occur in human documents. This is solved via Unbalanced Sinkhorn iterations followed by debiasing: $\mathcal D_{\mathrm{UOT}}\leftarrow \hat{\mathcal D}_{\mathrm{UOT}}(\hat p^{\bm g},p_{\bm\theta})-\tfrac12\hat{\mathcal D}_{\mathrm{UOT}}(\hat p^{\bm g},\hat p^{\bm g})-\tfrac12\hat{\mathcal D}_{\mathrm{UOT}}(p_{\bm\theta},p_{\bm\theta})$, and rescaled as $r=(0.1-\mathcal D_{\mathrm{UOT}})\times 10$.
    - **Design Motivation**: Averaging scores (WVS/CDEval) collapses the value disagreements of different sub-groups into a single mean, losing distributional nuances. KL divergence is sensitive to zero-mass terms. UOT allows for total mass mismatch, fitting scenarios where LLMs and humans might have structural gaps in certain value codes while retaining Wasserstein geometric properties.

3.  **Distributional Long-text Evaluation**:
    - **Function**: Abandons MCQs/Likert scales in favor of comparing distributions of long texts written by humans vs. LLMs on the same topics $\bm o$, addressing the Context gap at its root.
    - **Mechanism**: LLMs generate essays/blogs $\bm x\sim p_{\bm\theta}(\bm x|\bm o)$ based on topics $\bm o$ (e.g., "The role of money in life"). Aligning with the psychological observation that "writing reflects personality," long texts carry more stable value signals than short answers. To support this, **Ours** constructs **DOVE Set**: 824 value-oriented topics across KR/JP/CN/US cultures, containing 15,213 human long documents (average 1,034 tokens).
    - **Design Motivation**: Constrained options + averaged scoring naturally limit the richness of value expression and misalign with actual LLM deployment (open generation). Long texts + automatic codebooks align evaluation signals, expression media, and deployment scenarios for the first time.

### Loss & Training
DOVE does not update LLM parameters. $q_{\bm\omega}$ uses GPT-5.2, $p_{\bm\phi}$ uses GPT-4.1 nano, and embeddings use OpenAI text-embedding-3-large. Codebook optimization is a Variational EM iteration: fix $\mathcal{\bm C}^{t-1}$ to estimate $\mathcal S(\mathcal{\bm C}^{t-1})$, then apply split/merge/recluster actions to get $\mathcal{\bm C}^t$ until convergence or $T=10$ rounds. Hyperparameters: $N_1=3, N_2=1, \beta_1=0.3, \beta_2=0.08, \tau_1=1.0$. Training corpus includes $N=10,676$ documents from humans and GPT-4o/DeepSeek-v3.1/Llama-4-Maverick.

## Key Experimental Results

### Main Results

| Method | $\Delta^{\bm g}\uparrow$ Value Priming | $\delta_{\text{con}}$ Convergent | $\delta_{\text{dis}}\uparrow$ Discriminant | Avg. Downstream Corr. $\uparrow$ |
| :--- | ---: | ---: | ---: | ---: |
| WVS | 0.08% | -9.76% | 0.98% | 16.20% |
| GOQA | -1.56% | -17.95% | -2.05% | -13.05% |
| CDEval | 0.76% | -14.40% | 1.79% | 23.56% |
| NormAd | 4.25% | -1.57% | -23.70% | 0.90% |
| NaVAB | -1.15% | 4.43% | -88.00% | -20.77% |
| **Ours (DOVE)** | **5.60%** | **6.00%** | 0.89% | **31.56%** |

Across 12 LLMs $\times$ 4 cultures, DOVE's correlation with the downstream "cultural harmful content detection" task reaches 31.56%, 1.3x that of the runner-up CDEval. Several baselines show negative correlation, suggesting their results have little predictive value for real deployment.

### Ablation Study

| Dimension | Ours (DOVE) Performance | Notes |
| :--- | :--- | :--- |
| Sampling Reliability (Cronbach α) | High | Stable with 500 documents per culture |
| Test–retest Stability | High | Consistent scores across three independent runs |
| Template Invariance | High | Scores remain stable when changing prompt templates |
| Topic Count Robustness | Over 300 topics | More efficient than typical massive LLM benchmarks |
| Codebook Size Sensitivity | $\mathcal S(\mathcal{\bm C})$ correlated with validity | Small codebooks lack capacity; large ones introduce redundancy |

### Key Findings
- All constrained methods (WVS/GOQA/CDEval/NormAd) exhibit negative convergent validity — their scores contradict each other. DOVE is the only one to move this into positive territory.
- NaVAB's discriminant validity drops to -88%, which the authors attribute to its reliance on human-written reference statements, failing to distinguish cultural similarity structures.
- In value priming experiments, only DOVE and NormAd show significant positive $\Delta^{\bm g}$; DOVE also yields the largest negative $\Delta^{\bm g^-} = -5.38\%$ for opposing cultures, demonstrating the cleanest directionality.

## Highlights & Insights
- Reformulating the "evaluation design" problem as "rate-distortion lossy compression + Optimal Transport" is the core "aha" moment — it leverages robust tools from representation learning to bypass debates over "which value system" to use.
- The "co-occurrence discount" in the cost matrix is ingenious: OT based only on semantic similarity overestimates cultural differences. Integrating data-driven co-occurrence makes the measure closer to the real "cultural semantic geometry."
- Learning the codebook via ICL + Variational EM without fine-tuning allows the pipeline to switch underlying LLMs seamlessly. As LLMs improve, the "evaluator" upgrades automatically without redesigning the benchmark.

## Limitations & Future Work
- Evaluation relies heavily on the GPT-5.2 / GPT-4.1 / OpenAI embedding toolchain; the "value perspective" in the codebook may inherit the biases of these closed-source models, creating a circular risk.
- Only four national-level cultures (KR/JP/CN/US) are covered; the ability to capture sub-cultures or cross-regional subgroups is not directly verified.
- Downstream "predictive validity" uses hate/harmful text detection as a proxy task, which is not equivalent to "value alignment." Whether the 31.56% correlation holds for creative preference or ethical persuasion remains to be seen.
- Future Work: Extend codebooks to multiple levels (universal → cultural → sub-group) and use contrastive objectives for cross-cultural code distinctiveness; replace $q_{\bm\omega}$ with open-source models to break the circularity.

## Related Work & Insights
- **vs. WVS / GOQA / CDEval**: These measure "value knowledge" via surveys, while **Ours** measures "value orientation" via long-text distributions. DOVE's downstream correlation is 8–44 percentage points higher.
- **vs. NaVAB**: Both use open-ended generation, but NaVAB relies on human references, leading to high reference bias (discriminant validity -88%). DOVE uses self-learned codes, abstracting "references" into data-driven code sets.
- **vs. LLM-as-a-judge (Shi 2024, Mushtaq 2025)**: Those methods use direct LLM scoring, which is affected by judge model mood/framing. DOVE decomposes the "judgment" into "identifying value codes + calculating distribution distance," with both steps constrained by rate-distortion objectives for better reproducibility.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Uses rate-distortion + UOT to effectively address the three C³ gaps.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 12 LLMs $\times$ 4 cultures $\times$ 5 baselines with 15K documents. Lacks some human blind evaluation of codebook quality.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The C³ framework is well-defined, and the derivation of formulas and algorithms is coherent.
- **Value**: ⭐⭐⭐⭐⭐ Provides a truly scalable framework for LLM cultural alignment; DOVE Set is a valuable resource for the alignment community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](../../ACL2026/aigc_detection/temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[AAAI 2026\] Optimized Algorithms for Text Clustering with LLM-Generated Constraints](../../AAAI2026/aigc_detection/optimized_algorithms_for_text_clustering_with_llm-generated_constraints.md)
- [\[ICML 2026\] Generating Robust Portfolios of Optimization Models using Large Language Models](generating_robust_portfolios_of_optimization_models_using_large_language_models.md)

</div>

<!-- RELATED:END -->
