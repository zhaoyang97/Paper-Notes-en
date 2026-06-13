---
title: >-
  [Paper Note] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm
description: >-
  [ICML 2026][Optimization][Continual Pre-training] The authors derive closed-form training dynamics for simplified single-layer linear attention Transformers…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Continual Pre-training"
  - "Catastrophic Forgetting"
  - "Transformer Training Dynamics"
  - "Data Replay"
  - "Attention Attribution"
date: 2026-05-08
content_hash: 8be3aa4d187648dd
---

# Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm

**Conference**: ICML 2026  
**arXiv**: [2605.10640](https://arxiv.org/abs/2605.10640)  
**Code**: https://github.com/WhyDwelledOnAi/continual_Factual_Knowledge_Acquision (Available)  
**Area**: Continual Pre-training / Language Model Theory / Catastrophic Forgetting  
**Keywords**: Continual Pre-training, Catastrophic Forgetting, Transformer Training Dynamics, Data Replay, Attention Attribution

## TL;DR
The authors derive closed-form training dynamics for simplified single-layer linear attention Transformers, proving that regularization methods only alter convergence speed without shifting the convergence point (thus almost certain to fail in cFKA scenarios). In contrast, data replay directly shifts the convergence point and increases oscillation amplitude to stabilize old knowledge. Based on this, they propose STOC, which clips snippets based on token attention contribution to guide pre-trained models in generating replay corpora. STOC consistently suppresses forgetting better than LAMOL across synthetic, KnowEdit, and IndustryCorpus legal datasets.

## Background & Motivation

**Background**: LLMs accumulate massive factual knowledge during open-domain pre-training (PT), but industrial deployment often requires Continual Pre-training (CPT) to inject domain knowledge (e.g., legal corpora) or new facts. Like traditional continual learning, CPT suffers from catastrophic forgetting where old knowledge is overwritten by new data.

**Limitations of Prior Work**: Existing CPT mitigation strategies are mainly divided into regularization-based (e.g., EWC) and data replay-based (e.g., replay / LAMOL). Experiments on LLMs generally find that regularization has limited effects, whereas replay can significantly mitigate forgetting even at small ratios. However, the community lacks a unified theoretical explanation for this, leading to heuristic tuning of ratios in engineering practice.

**Key Challenge**: Continual Factual Knowledge Acquisition (cFKA) is essentially "pulling the output distribution of the same token towards new facts under a shared next-token-prediction objective, without collapsing the probability of old facts." However, there is a lack of a transformer-specific dynamical description of how the relative magnitudes of learning rate, token frequency, and attention allocation determine the degree of forgetting.

**Goal**: (a) Formulate a tractable training dynamics framework for cFKA to characterize the evolution of $\mathbf{Y}$ (FFN-like knowledge storage) and $\mathbf{Z}$ (attention) parameters; (b) use this theory to explain why regularization fails and why replay succeeds; (c) derive STOC, a transformer-native generative replay method based on token-level attention attribution, and validate it in synthetic and real-world scenarios.

**Key Insight**: Following the "Physics of Language Models" series by Allen-Zhu & Li, facts are represented as (subject, relation, object) triplets and fed into a single-layer linear attention Transformer. Assuming $\eta_Y \gg \eta_Z$ treats $\mathbf{Z}$ as slowly changing, simplifying multi-body nonlinear optimization into controllable Taylor expansions.

**Core Idea**: Instead of patching existing CPT algorithms, first prove through transformer training dynamics that regularization is a method that "cannot move the convergence point," while replay is a method that "can both shift the point and amplify oscillations to preserve the old." Then, use token attention attribution to select replay sources, enabling generative replay to produce samples that truly contain old knowledge.

## Method

### Overall Architecture
Analytical framework: Reparameterize the model as $\mathbf{Y} := \mathbf{E}\mathbf{W}_O\mathbf{W}_V^\top \mathbf{E}^\top$ (FFN-like knowledge storage) and $\mathbf{Z} := \mathbf{E}\mathbf{W}_K\mathbf{W}_Q^\top \mathbf{E}^\top / \sqrt{d}$ (attention). Under cross-entropy optimization, use SGD to derive the evolution theorem for $\mathbf{Y}$ and conservation laws for $\mathbf{Z}$. Subsequently, incorporate regularization and replay into the gradient equations to compare their impacts on convergence points, convergence speeds, and oscillation amplitudes. Finally, based on the inference that "tokens with higher attention scores carry more factual information," design STOC: perform a forward pass on each CPT sample to obtain token-level attention scores $\to$ average across layers $\to$ extract fixed-length snippets with the highest attention via a sliding window $\to$ feed these as prompts to the pre-trained LM to generate replay $\to$ deduplicate and filter using MinHash $\to$ mix with new data at ratio $\alpha$ for the CPT process.

### Key Designs

1.  **cFKA Training Dynamics Theorem for Single-layer Transformer**:

    - **Function**: Expresses the evolution of $\mathbf{Y}$ and $\mathbf{Z}$ in analytic Taylor forms, providing explicit expressions for the convergence point, convergence speed, and oscillation amplitude.
    - **Mechanism**: Under the assumption $\eta_Y \gg \eta_Z$, $\mathbf{Y}$ is convex relative to the loss. Its reference optimal solution is $\mathbf{U}=\sum_{o,s}\frac{1}{a_s}[\ln \Pr(s\mid o) + \frac{1}{L}\ln \Pr(o)]\,\mathbf{x}_o\mathbf{x}_s^\top$ (Bayes optimal prediction). The error $\mathbf{e}_s(T) = \mathbf{y}_s(T)-\mathbf{u}_s$ satisfies $\mathbf{e}_s(T) \approx [\prod_{t=1}^T (\mathbf{I}-\eta_Y z_s \delta_s(t)\tilde{\mathbf{H}}(t))]\mathbf{e}_s(0) + \sum_t \eta_Y z_s \delta_s(t) [\prod (\cdot)]\bm{\xi}(t)$. The first term decays exponentially, determining convergence speed (controlled by the maximum eigenvalue of $\tilde{\mathbf{H}}$), while the second term represents fixed-amplitude oscillation (controlled by the minimum positive eigenvalue of $\tilde{\mathbf{H}}$). Simultaneously, $\mathbf{Z}$ satisfies a conservation law $\frac{d}{dt}[(\frac{z_s}{\eta_Z})^2 - \sum_o(\frac{y_{o,s}}{\eta_Y})^2] = 0$, leading to the conclusion that the attention of token $s$ is determined by its Diversity Index $\mathrm{DI}(\overline{\mathbf{x}}_s) \propto -\sqrt{\eta_Z/\eta_Y}\sqrt[4]{\sum_o[\ln\Pr(s\mid o)+L^{-1}\ln\Pr(o)]^2}+C$—tokens with narrower distributions and more exclusive information receive higher attention.
    - **Design Motivation**: Previous transformer optimization theories either focused on ICL or ignored multi-token knowledge structures. This paper specifically addresses "how facts are distributed and stored across tokens," decomposing training dynamics to the token level to allow precise adjustments during CPT.

2.  **Comparison of Regularization / Data Replay Mechanisms**:

    - **Function**: Substitutes both CPT methods into the aforementioned dynamics to prove that "regularization only modifies speed, while replay modifies convergence points and amplifies oscillations."
    - **Mechanism**: For EWC-style objectives $\mathcal{L}=\mathcal{L}_{\text{new}}+\frac{k}{2}\sum_i w_i (\theta_i-\theta_i^*)^2$, $\mathbf{e}_s(T)$ gains an extra term $-\sum_t k\eta_Y[\prod(\cdot)]\,\tilde{\mathbf{u}}$, but it is restricted by $\lambda^+_{\min}(\mathrm{diag}(\mathbf{w}_s))=\min_o w_{o,s}$. When factual knowledge is carried by only a few dimensions of a token, this minimum eigenvalue is nearly zero, meaning the convergence point barely changes and only the speed slows down. For replay, the frequency distribution becomes $\Pr(\mathbf{x}_s)=\frac{1-\alpha}{|\mathcal{O}_s^{\mathrm{old}}|}\sum_{o\in\mathcal{O}_s^{\mathrm{old}}}\mathbf{x}_o + \frac{\alpha}{|\mathcal{O}_s^{\mathrm{new}}|}\sum_{o\in\mathcal{O}_s^{\mathrm{new}}}\mathbf{x}_o$, where the first term directly writes old knowledge back into the convergence point. Meanwhile, the oscillation term $\lambda^+_{\min}(\tilde{\mathbf{H}})$ is amplified after mixing new and old data, providing a "reminder" effect for old knowledge.
    - **Design Motivation**: Empirical evidence has long shown that "regularization is ineffective, while replay is effective even at 10%," but an explanation was missing. This theory clearly reveals which term in the dynamics each method affects, predicting that solving forgetting requires shifting the convergence point through replay.

3.  **STOC: Selecting Tokens via attentiOn Contribution**:

    - **Function**: Automatically selects the "most informative and old-knowledge-triggering" snippets from CPT samples to let the pre-trained LM generate replay without storing original PT data.
    - **Mechanism**: (i) Perform a forward pass on each CPT sample and average attention scores across all layers and heads to get token-level importance $a_t$. (ii) Use a sliding window to find a fixed-length snippet with the highest $\sum_t a_t$. (iii) Use this snippet as a prompt for the "original pre-trained model" to generate completion. According to the dynamics, high-attention tokens are those with low Diversity Index that "lock" onto a set of old facts, so the completion likely covers old knowledge. (iv) Ensure replay diversity via MinHash deduplication. (v) Mix with CPT corpora at ratio $\alpha\in\{0.5, 0.67, 0.8, 0.9\}$ for training.
    - **Design Motivation**: Methods like LAMOL use special tokens as prompts for generation, which does not utilize the transformer's attention structure and may generate content far from what the model actually "knows." STOC uses tokens the model actually "cares about" as seeds, equivalent to letting the model recall in its most familiar directions, leading to significantly higher replay quality.

### Loss & Training
Basic CPT uses cross-entropy $\mathcal{L} = -\mathrm{logit}(x_{T+2}\mid \mathbf{X}) + \log\sum_o \exp(\mathrm{logit}(x_o\mid \mathbf{X}))$. In synthetic Biography experiments, SGD is performed assuming $\eta_Y \gg \eta_Z$. For real-world LLM experiments, Pythia-160M / Qwen2.5-0.5B-1.7B are used with three update strategies: full parameter, rank-128 LoRA, and freezing the first 6 layers. EWC estimates parameter importance via Fisher Information; STOC and LAMOL both apply next-token loss to mixed new and old data at ratio $\alpha$.

## Key Experimental Results

### Main Results
Comparison on Pythia-160M synthetic Biography data. "Original" indicates retention of old (PT stage) knowledge, and "Continual" indicates acquisition of new (CPT stage) knowledge. $\alpha$ is the CPT data mixture ratio; higher is better.

| Config ($\alpha$) | Replay | Update | Original sFTA | Original EM | Continual sFTA |
|---|---|---|---|---|---|
| 0.5 | Random | Full | 17.68 | 3.14 | 90.37 |
| 0.5 | LAMOL | Full | 19.90 | 5.95 | 92.58 |
| **0.5** | **STOC** | **Full** | **51.54** | **29.84** | 90.47 |
| 0.67 | Random | Freeze | 21.02 | 6.43 | 91.67 |
| 0.67 | LAMOL | Freeze | 21.69 | 9.47 | 92.62 |
| **0.67** | **STOC** | **Freeze** | **53.80** | **32.83** | 92.04 |
| 0.9 | LAMOL | Freeze | 18.88 | 7.58 | 92.06 |
| **0.9** | **STOC** | **Freeze** | **40.54** | **21.62** | 91.96 |

### Ablation Study
Average soft token accuracy on KnowEdit (ZSRE / Wiki_Bio / Wiki_Recent) for Qwen2.5-0.5B (higher is better):

| Method | ZSRE Orig | ZSRE Cont | Wiki_Bio Orig | Wiki_Bio Cont | Wiki_Recent Orig | Wiki_Recent Cont |
|---|---|---|---|---|---|---|
| Naive | 34.58 | 63.28 | 32.33 | 35.50 | 19.28 | 28.42 |
| LAMOL ($\alpha{=}0.5$) | 37.54 | 58.37 | 31.29 | 34.49 | 20.48 | 27.19 |
| **STOC ($\alpha{=}0.5$)** | **37.12** | **62.26** | **35.57** | 35.46 | **21.40** | **28.75** |
| STOC ($\alpha{=}0.8$) | 37.47 | 62.59 | 35.28 | 33.16 | 20.12 | 27.34 |

In the IndustryCorpus2 legal domain 1B token CPT evaluation (MMLU / MMLU-Redux-2.0 / SuperGPQA), STOC outperformed LAMOL by 1–4 percentage points on 0.6B and 1.7B models. On SuperGPQA, the Continual subset performance improved from 13.35% (LAMOL) to 15.85%.

### Key Findings
- Even with a replay ratio of only 10%, the model significantly retains old knowledge—matching the theory that "replay directly modifies the frequency distribution and shifts the convergence point."
- Among replay selection strategies, "one biography per individual" performed better than "two biographies for half the individuals," suggesting replay should favor **broad coverage** over local depth.
- STOC using attention-clipped snippets as prompts outperforms random snippets (Ablation), confirming that attention-based selection is a true causal factor rather than a heuristic accident.
- In the synthetic 1-Aug setting (1 biography per person), the set EM was 87% on the training set but only 8.85% on the test set, highlighting the critical role of data augmentation in generalization. This also validates the Diversity Index theory: augmentation makes the $\overline{\mathbf{x}}_s$ of relation tokens more uniform $\to$ decreasing attention $\to$ forcing the model to rely more on subject tokens $\to$ better cross-template generalization.
- LoRA performed worst at low $\alpha$, while full parameter and freezing the first 6 layers showed little difference, suggesting cFKA is more sensitive to parameter count than to "which layers" are moved.

## Highlights & Insights
- The long-standing engineering experience that "regularization is ineffective on LLMs" is explained for the first time via a clean eigenvalue argument: $\lambda^+_{\min}(\mathrm{diag}(\mathbf{w}_s))$ being close to zero means the regularization term cannot shift the convergence position of $\mathbf{y}_s$. This perspective of "explaining failed methods with dynamics" applies to other continual learning scenarios.
- The Diversity Index quantifies the role of tokens in factual expression into a closed form $\sqrt[4]{\sum_o[\ln\Pr(s\mid o)+L^{-1}\ln\Pr(o)]^2}$, which correlates strongly with measured attention in multi-layer LMs (Pearson $<-0.8$), proving valuable for attention probing and interpretability research.
- STOC is a "zero-extra-training-cost" engineering component—it only requires collecting attention during the original forward pass to add a high-quality replay source to existing CPT pipelines. It is orthogonal to freezing/LoRA and can be easily combined.

## Limitations & Future Work
- The theory is built on single-layer linear attention + structured-input assumptions; softmax/multi-head/multi-layer/multi-stage training only find empirical echoes without formal expansion.
- Facts are modeled only as (subject, relation, object) triplets, which has limited coverage for common sense, chain-of-thought, and long-context knowledge.
- Snippet selection in STOC currently uses sliding windows in the sequence dimension and layer-averaging; layer-specific or head-specific selection strategies remain unexplored.
- The maximum model size in real-world experiments is Qwen2.5-1.7B. Scaling to 10B+ and investigating the scaling laws of replay ratios relative to model size remain open questions.

## Related Work & Insights
- **vs LAMOL (Sun 2020)**: Both are generative replay; however, LAMOL uses special tokens for free generation, while STOC uses attention attribution to find "knowledge-dense" snippets, producing content closer to the old distribution.
- **vs EWC (Kirkpatrick 2017)**: A classic regularization method; this paper proves from an eigenvalue perspective that it is destined to "slow down" rather than "inhibit" forgetting under cFKA.
- **vs Allen-Zhu & Li "Physics of LM" series**: This paper adopts the same synthetic Biography tasks and hFTA/sFTA/EM metrics but for the first time pushes the training dynamics into the CPT stage to design a new algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides both a closed-form dynamics for cFKA and an attention-based replay based on it; the theory-to-algorithm chain is very complete.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic + KnowEdit + IndustryCorpus legal data + multiple scales of Pythia/Qwen + three parameter update strategies; well-covered, though maximum scale is small.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical derivations; the roadmap in Figure 1 (PT validate $\to$ CPT analyze $\to$ algorithm propose) makes the long paper easy to read.
- Value: ⭐⭐⭐⭐ STOC is a ready-to-use tool for industrial CPT teams. Simultaneously provides a citable argument for "why EWC doesn't work," holding both theoretical and engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Trilemma of Truth in Large Language Models](../../NeurIPS2025/optimization/the_trilemma_of_truth_in_large_language_models.md)
- [\[ICML 2026\] Adaptive Sharpness-Aware Minimization with a Polyak-type Step size: A Theory-Grounded Scheduler](adaptive_sharpness-aware_minimization_with_a_polyak-type_step_size_a_theory-grou.md)
- [\[ICML 2026\] Towards Understanding Adam Convergence on Highly Degenerate Polynomials](towards_understanding_adam_convergence_on_highly_degenerate_polynomials.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)
- [\[ICCV 2025\] Federated Continual Instruction Tuning](../../ICCV2025/optimization/federated_continual_instruction_tuning.md)

</div>

<!-- RELATED:END -->
