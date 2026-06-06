---
title: >-
  [Paper Note] Causal Fine-Tuning under Latent Confounded Shift
description: >-
  [ICML 2026][NLP Understanding][Causal Fine-tuning] This paper proposes Causal Fine-Tuning (CFT): embedding an SCM-inspired decomposition of "high-level stable features $C$ + low-level confounder-sensitive features $\Phi$…
tags:
  - "ICML 2026"
  - "NLP Understanding"
  - "Causal Fine-tuning"
  - "Latent Confounded Shift"
  - "front-door adjustment"
  - "single-domain generalization"
  - "BERT"
date: 2026-05-08
content_hash: 9f84b1c4e4f11bdb
---

# Causal Fine-Tuning under Latent Confounded Shift

**Conference**: ICML 2026  
**arXiv**: [2410.14375](https://arxiv.org/abs/2410.14375)  
**Code**: https://github.com/jialin-yu/CausalFineTuning (Available)  
**Area**: NLP Understanding / Causal Representation Learning / Out-of-Distribution Generalization  
**Keywords**: Causal Fine-tuning, Latent Confounded Shift, front-door adjustment, single-domain generalization, BERT

## TL;DR
This paper proposes Causal Fine-Tuning (CFT): embedding an SCM-inspired decomposition of "high-level stable features $C$ + low-level confounder-sensitive features $\Phi$" into standard BERT fine-tuning. Predictions are made via a front-door style do-calculus adjustment formula, significantly outperforming single-domain generalization baselines like SFT, SWA, and WISE under spurious correlation injection attacks in text.

## Background & Motivation
**Background**: Downstream adaptation of foundation models (BERT/GPT/CLIP) mostly follows a black-box route using "full-parameter SFT or LoRA + ERM," treating all input features equally.

**Limitations of Prior Work**: When training data contains spurious correlations driven by latent variables (e.g., an "Amazon" tag strongly correlated with positive sentiment), models learn shortcuts. During deployment, if these spurious correlations flip (e.g., Amazon reviews become predominantly negative), the model fails sharply. Traditional invariance methods like IRM require multi-domain labels or environment annotations, making them inapplicable to single-domain data.

**Key Challenge**: In standard fine-tuning $p(y\mid x;\sigma)=\sum_u p(y\mid u,x)\,p(u\mid x;\sigma)$, $p(u\mid x;\sigma)$ changes with the environment. However, single-domain observations can neither identify $u$ nor provide environment labels. Minimax robust optimization is either unidentifiable or overly conservative under such "unseen confounding."

**Goal**: Under single-domain fine-tuning, decompose input representations into (i) cross-environment stable causal components $C$; and (ii) environment-sensitive low-level local components $\Phi$, then "strip away" the dependence on $\Phi$ through do-calculus adjustment.

**Key Insight**: Treat the pre-trained LM itself as "another implicit environment"—the frozen model provides $R_0$, and the fine-tuned model provides $R_1$. The discrepancy between these two views naturally exposes which dimensions are sensitive to the training domain and which are stable across domains.

**Core Idea**: Use an SCM as an inductive bias, stipulating that $R_0$ and $R_1$ explain consistency through a shared stable causal latent variable $C$ while carrying spurious correlations via low-level local features $\Phi$. Front-door adjustment $p(y\mid \mathrm{do}(x))=\sum_{\Phi',x'} p(y\mid\Phi',C)\,p(\Phi'\mid x')\,p(x')$ is then estimated via Monte-Carlo by shuffling $\Phi$ within the mini-batch.

## Method

### Overall Architecture
The training phase maintains two BERT models simultaneously: a frozen pre-trained model $p(r_0\mid x)$ and a fine-tuned model $p(r_1\mid x)$. Each sample passes through three heads: (1) An SFT head for supervised classification on $R_1$; (2) A Causal head that learns a mapping from $(R_0, R_1)$ to the stable causal representation $C$; (3) A Local head that extracts $\Phi$ from the embedding layer of $R_1$. The final predictor $p(y\mid C, \Phi)$ performs do-calculus Monte-Carlo adjustment by shuffling $\Phi$ within the mini-batch $K=20$ times. At inference, the frozen model is discarded, and $C$ is estimated solely from $R_1$, maintaining the same model scale as standard SFT.

### Key Designs

1.  **SCM-based Causal Identification Scaffold**:
    - **Function**: Combines "high-level stable semantics $C$ + low-level confounding features $\Phi$ + unobserved confounders $U_S, U_\Phi$ + environment $\sigma$" into a graph (Fig. 2(b)), demonstrating that $p(y\mid \mathrm{do}(x))$ can be derived from $p(y\mid\Phi, C)$ and the marginal $p(x)$ using single-domain data.
    - **Mechanism**: Assumes $\sigma$ only affects $R_1$ via $S_1$, and the effect of $\Phi$ on $Y$ passes entirely through $C$ (front-door structure). It leverages Von Kügelgen’s identifiability theorem to define $C$ as an invariant projection $p(C\mid R_0) \approx p(C\mid R_1)$.
    - **Design Motivation**: Uses the causal graph to explicitly state that "train-test distribution shift is equivalent to changes in $p(u\mid x;\sigma)$," converting the problem of being "robust to all $\sigma$" into "training under the max-entropy default environment of $\mathrm{do}(x)$," thus avoiding the over-conservatism and non-identifiability of minimax approaches.

2.  **Dual-view Causal Representation Alignment $\mathcal{L}_C$**:
    - **Function**: Projects the sentence vector $R_0$ from the frozen model and $R_1$ from the fine-tuned model into the same stable causal space $C$.
    - **Mechanism**: Minimizes $\mathcal{L}_C = \mathbb{E} \|p(c\mid r_0) - p(c\mid r_1)\|_2^2 - H(p(c\mid r_0)) - H(p(c\mid r_1))$, where the first term enforces cross-view invariance and the negative entropy terms prevent representation collapse into a constant.
    - **Design Motivation**: Treats the "pre-trained vs. fine-tuned" pair as a natural surrogate for IRM multi-domain signals, extracting cross-domain stable components using only single-domain data. If $(R_1, R_1)$ identical views are used instead (failure mode), the method degrades to SFT.

3.  **Local Patch Features $\Phi$ + Front-door Batch Shuffle**:
    - **Function**: Extracts local low-level features from the embedding layer of the fine-tuned model as "proxies" for spurious correlations and severs the causal path of $\Phi$ via do-calculus adjustment.
    - **Mechanism**: Segments token sequences into 10 non-overlapping patches, applies mean pooling, and passes them through an MLP to obtain $\Phi = \text{MLP}(\frac{1}{10}\sum_i p_i)$. During prediction, $\Phi$ is randomly shuffled within the mini-batch $\Phi' \sim \hat{p}_B(\Phi)$ to compute $\mathbb{E}_{\Phi'}[p(y\mid C, \Phi')]$ via 20 Monte-Carlo averages.
    - **Design Motivation**: The embedding layer is closest to low-level spurious clues like word forms or data sources. Shuffling $\Phi$ is equivalent to breaking the active collider path $\sigma \to S_1 \to R_1 \leftrightarrow \Phi \leftrightarrow Y$, pulling the prediction from $p(y\mid x)$ back to $p(y\mid \mathrm{do}(x))$.

### Loss & Training
The total objective is $\mathcal{L} = \mathcal{L}_{\text{SFT}} + \mathcal{L}_C + \mathcal{L}_{\text{adjust}}$, where $\mathcal{L}_{\text{adjust}}$ is the cross-entropy loss based on shuffled $\Phi'$. The optimizer used is AdamW with a learning rate of $5 \times 10^{-5}$ for 10 epochs, initialized with BERT-base. A frozen copy is maintained only for $R_0$ extraction and is discarded after training.

## Key Experimental Results

### Main Results

| Dataset | Test Spurious 10% | SFT | SWA | WISE | CFT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Yelp (Exp1, stop-word attack) | F1 | 49.24 | 62.92 | 55.91 | **58.40** (+9.16 vs SFT) |
| Amazon (Exp1) | F1 | 49.33 | 59.75 | 50.40 | **56.40** (+7.07) |
| Amazon (Exp2, data-source attack) | F1 | 37.78 | 47.41 | 31.83 | **49.22** (+11.44) |

On In-Distribution (ID, 90% spurious) data, CFT is nearly tied with SWA and SFT. However, as OOD conditions become more extreme (spurious ratio dropping from 70% to 10%), the advantage of CFT increases. Under 4x and 8x noise scaling, the gap between CFT and SWA widens further.

### Ablation Study

| Configuration | Spurious 10% F1 (Amazon) | Note |
| :--- | :--- | :--- |
| Full CFT | 56.40 | Causal decomposition + do-calculus |
| CFT-N (No do-shuffle, conditioning on $\Phi$) | 48.00 | Active collider remains; OOD degrades to SFT levels |
| CFT-C (Predict using only $C$) | 53.40 | Stronger than SFT but weaker than full, indicating $\Phi$ adjustment contributes |
| CFT-$\Phi$ (Predict using only $\Phi$) | 12.40 | Near random; confirms $\Phi$ captures spurious correlations |
| CFT (identical view $R_1, R_1$) | 37.24 | Failure mode: degrades to SFT without dual-view signals |

### Key Findings
- Predictions using $\Phi$ alone are near-random on OOD data (19/12 F1), confirming it captures spurious correlations. Adjustment ensures the model remains robust against distribution flips.
- Removing cross-view signals (identical view) causes the method to immediately revert to SFT levels, proving that "dual-view + invariance constraints" drive the representation decomposition.
- CFT's lead over SWA increases with shift intensity. While SWA and CFT are comparable under default noise, CFT takes a definitive lead under 4x/8x amplified noise, suggesting structural causal adjustment is more stable than general regularization under severe shifts.
- Layer sensitivity studies (Table 6) show that using the embedding layer for $\Phi$ is most stable under strong shifts, while higher layers remain competitive under weak shifts, aligning with the expectation that lower layers capture shift-sensitive clues.

## Highlights & Insights
- **Pre-trained Models as "Free Environments"**: Traditional causal invariance methods require multiple real environments. This work uses the frozen vs. fine-tuned views to naturally form a pair, applying Theorem 4.4 (Von Kügelgen) to obtain invariant representations. This eliminates multi-domain data collection costs, which is highly beneficial for NLP scenarios where environment labels are hard to define.
- **Simplifying do-calculus with front-door + batch shuffle**: Implements the abstract $p(y\mid \mathrm{do}(x))$ as a simple batch shuffle operation. This breaks implicit collider paths with almost no added training cost.
- **SCM as Inductive Bias, Not Truth Identification**: $C$ and $\Phi$ are empirical estimates rather than ground-truth variables from a data-generating graph. The method's minimum requirement is that the two views can distinguish some information; the failure mode experiment provides a clear diagnostic signal.

## Limitations & Future Work
- The study is limited to text sentiment classification with injected spurious cues; it does not cover real-world multi-source confounding (e.g., hospitals, platforms, languages).
- It assumes a valid front-door structure (the effect of $\Phi$ on $Y$ is mediated by $C$). Identification fails if there are direct causal paths from $\Phi$ to $Y$.
- Batch Monte-Carlo with $K=20$ shuffles only models the intra-batch distribution; estimating $p(\Phi)$ across batches still relies on i.i.d. sampling. The number of patches (10) and the layer for $\Phi$ extraction are empirical choices that require validation for long-form text or multimodal data.
- Future work intends to extend this to multimodality, investigating confounders that live in one modality but interact with another.

## Related Work & Insights
- **vs. IRM/V-REx**: IRM-style methods require multi-domain environment labels to learn invariant features. Ours uses the pre-train/fine-tune pair as a "pseudo-environment" alignment, making it usable with single-domain data.
- **vs. SWA/WISE**: SWA/WISE are general flat-minima or parameter-interpolation regularizers. While they perform similarly to CFT under moderate shifts, CFT outperforms them as shifts become more severe, highlighting the advantage of structural causal adjustment over geometric regularization.
- **vs. back-door causal attention (Yue 2020, Zhang 2020)**: Back-door methods require observed confounders. This work uses the front-door approach for "latent confounding + text" scenarios, following Mao 2022 but making do-calculus adjustment a plug-and-play module for fine-tuning.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining front-door adjustment with dual-view pre-train/fine-tune alignment for BERT is a creative approach.
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple datasets, shift intensities, ablations, and failure modes, though restricted to synthetic spurious injections.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from causal graphs and theorems to algorithms, distinguishing between the identification scaffold and practical proxies.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play causal robustness solution for single-domain NLP fine-tuning, relevant for handling dataset artifacts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Weak-to-Strong Generalization under Distribution Shifts](../../NeurIPS2025/nlp_understanding/weak-to-strong_generalization_under_distribution_shifts.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](../../ACL2026/nlp_understanding/lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[ICML 2026\] Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting](controlling_the_risk_of_corrupted_contexts_for_language_models_via_early-exiting.md)
- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](../../ACL2026/nlp_understanding/lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)
- [\[ACL 2026\] MetFuse: Figurative Fusion between Metonymy and Metaphor](../../ACL2026/nlp_understanding/metfuse_figurative_fusion_between_metonymy_and_metaphor.md)

</div>

<!-- RELATED:END -->
