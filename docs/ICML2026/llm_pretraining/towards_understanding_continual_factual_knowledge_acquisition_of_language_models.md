---
title: >-
  [Paper Note] Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm
description: >-
  [ICML 2026][LLM Pretraining][Continual Pretraining] The authors derive closed-form training dynamics on a simplified single-layer linear attention Transformer…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "Continual Pretraining"
  - "Catastrophic Forgetting"
  - "Transformer Training Dynamics"
  - "Data Replay"
  - "Attention Attribution"
date: 2026-05-08
content_hash: 960f237d946eaa6b
---

# Towards Understanding Continual Factual Knowledge Acquisition of Language Models: From Theory to Algorithm

**Conference**: ICML 2026  
**arXiv**: [2605.10640](https://arxiv.org/abs/2605.10640)  
**Code**: https://github.com/WhyDwelledOnAi/continual_Factual_Knowledge_Acquision (available)  
**Area**: Continual Pretraining / Language Model Theory / Catastrophic Forgetting  
**Keywords**: Continual Pretraining, Catastrophic Forgetting, Transformer Training Dynamics, Data Replay, Attention Attribution

## TL;DR
The authors derive closed-form training dynamics on a simplified single-layer linear attention Transformer, proving that regularization methods can only alter convergence speed but cannot shift the convergence point (thus are almost doomed to fail in cFKA scenarios), while data replay can directly shift the convergence point and amplify oscillations to stabilize old knowledge. They further propose STOC, which prunes fragments based on token attention contribution and guides the pretrained model to generate replay corpora. STOC consistently suppresses forgetting better than LAMOL on synthetic + KnowEdit + IndustryCorpus legal corpora.

## Background & Motivation

**Background**: LLMs have accumulated massive factual knowledge in open-domain pretraining (PT), but industrial deployment often requires continual pretraining (CPT) to inject domain knowledge (e.g., legal corpora) or new facts. Like traditional continual learning, CPT suffers from catastrophic forgetting: old knowledge is overwritten by new data.

**Limitations of Prior Work**: Existing CPT mitigation approaches mainly fall into two categories—regularization-based (EWC, etc.) and data replay-based (replay / LAMOL). Experiments on LLMs generally find regularization to be limited in effect, while even a small proportion of replay can significantly alleviate forgetting. However, the community lacks a unified theoretical explanation for this—so engineering relies on heuristically tuning the ratio.

**Key Challenge**: Continual Factual Knowledge Acquisition (cFKA) is essentially "under a shared next-token-prediction objective, pulling the output distribution of the same token toward new facts, but without collapsing the probability on the old facts." However, how learning rate, token frequency, and attention allocation jointly determine forgetting remains unclear, lacking a transformer-specific dynamical description.

**Goal**: (a) Develop an analytically tractable training dynamics framework for cFKA, characterizing the evolution of $\mathbf{Y}$ (FFN-like knowledge storage) and $\mathbf{Z}$ (attention) parameters; (b) Use this theory to explain why regularization fails and why replay works; (c) Derive a native transformer, token-level attention-based generative replay method (STOC) from the analysis, and validate it on both synthetic and real scenarios.

**Key Insight**: Inspired by Allen-Zhu & Li's "Physics of Language Models" series, represent facts as (subject, relation, object) triples and feed them into a single-layer linear attention Transformer. Assume $\eta_Y \gg \eta_Z$ to treat $\mathbf{Z}$ as slowly varying, simplifying the multi-body nonlinear optimization into a controllable Taylor expansion.

**Core Idea**: Rather than patching existing CPT algorithms, first prove from transformer training dynamics that regularization is a "cannot move the convergence point" method, while replay is a "can both shift the point and amplify oscillations to preserve the old" method. Then use token attention attribution to select replay material → enabling generative replay to actually produce samples containing old knowledge.

## Method

### Overall Architecture
Analytical framework: Reparameterize the model as $\mathbf{Y} := \mathbf{E}\mathbf{W}_O\mathbf{W}_V^\top \mathbf{E}^\top$ (FFN-like knowledge storage) and $\mathbf{Z} := \mathbf{E}\mathbf{W}_K\mathbf{W}_Q^\top \mathbf{E}^\top / \sqrt{d}$ (attention). Under cross-entropy optimization, use SGD to derive the evolution theorem for $\mathbf{Y}$ and conservation law for $\mathbf{Z}$. Then, incorporate regularization and replay into the gradient equations, comparing their effects on convergence point, speed, and oscillation amplitude. Finally, based on the inference that "tokens with higher attention scores carry more factual information," design STOC: for each CPT sample, perform a forward pass to obtain token-level attention scores → average across layers → use a sliding window to extract the fixed-length snippet with highest attention → use it as a prompt to the pretrained LM to generate replay → MinHash deduplication → mix with new data at ratio $\alpha$ for CPT.

### Key Designs

1. **cFKA Training Dynamics Theorem for Single-layer Transformer**:

    - **Function**: Expresses the evolution of $\mathbf{Y}$ and $\mathbf{Z}$ in analytically tractable Taylor form, providing explicit expressions for convergence point, speed, and oscillation amplitude.
    - **Mechanism**: Under the assumption $\eta_Y \gg \eta_Z$, $\mathbf{Y}$ is convex to the loss, with reference optimal solution $\mathbf{U}=\sum_{o,s}\frac{1}{a_s}[\ln \Pr(s\mid o) + \frac{1}{L}\ln \Pr(o)]\,\mathbf{x}_o\mathbf{x}_s^\top$ (i.e., Bayes optimal prediction). The error $\mathbf{e}_s(T) = \mathbf{y}_s(T)-\mathbf{u}_s$ satisfies $\mathbf{e}_s(T) \approx [\prod_{t=1}^T (\mathbf{I}-\eta_Y z_s \delta_s(t)\tilde{\mathbf{H}}(t))]\mathbf{e}_s(0) + \sum_t \eta_Y z_s \delta_s(t) [\prod (\cdot)]\bm{\xi}(t)$, where the first term's exponential decay determines convergence speed (controlled by the largest eigenvalue of $\tilde{\mathbf{H}}$), and the second term is fixed-amplitude oscillation (controlled by the smallest positive eigenvalue of $\tilde{\mathbf{H}}$). Meanwhile, $\mathbf{Z}$ satisfies the conservation law $\frac{d}{dt}[(\frac{z_s}{\eta_Z})^2 - \sum_o(\frac{y_{o,s}}{\eta_Y})^2] = 0$, leading to token $s$'s attention being determined by its Diversity Index $\mathrm{DI}(\overline{\mathbf{x}}_s) \propto -\sqrt{\eta_Z/\eta_Y}\sqrt[4]{\sum_o[\ln\Pr(s\mid o)+L^{-1}\ln\Pr(o)]^2}+C$—tokens with narrower distributions and more exclusive information have higher attention.
    - **Design Motivation**: Previous transformer optimization theory either focused on ICL or ignored multi-token knowledge structure; this work specifically addresses "how facts are distributed among tokens," decomposing training dynamics to the token level, enabling targeted CPT interventions.

2. **Mechanism Comparison: Regularization vs. Data Replay**:

    - **Function**: Substitutes both CPT methods into the above dynamics, yielding the provable conclusion that "regularization only changes speed, replay shifts convergence point and amplifies oscillations."
    - **Mechanism**: For EWC-style objective $\mathcal{L}=\mathcal{L}_{\text{new}}+\frac{k}{2}\sum_i w_i (\theta_i-\theta_i^*)^2$, $\mathbf{e}_s(T)$ gains an extra term $-\sum_t k\eta_Y[\prod(\cdot)]\,\tilde{\mathbf{u}}$, but it is limited by $\lambda^+_{\min}(\mathrm{diag}(\mathbf{w}_s))=\min_o w_{o,s}$—when factual knowledge is carried by only a few token dimensions, this minimum eigenvalue is nearly zero, so the convergence point is almost unchanged, only the speed slows. For replay, the frequency distribution becomes $\Pr(\mathbf{x}_s)=\frac{1-\alpha}{|\mathcal{O}_s^{\mathrm{old}}|}\sum_{o\in\mathcal{O}_s^{\mathrm{old}}}\mathbf{x}_o + \frac{\alpha}{|\mathcal{O}_s^{\mathrm{new}}|}\sum_{o\in\mathcal{O}_s^{\mathrm{new}}}\mathbf{x}_o$, with the first term directly writing old knowledge into the convergence point; meanwhile, the oscillation term $\lambda^+_{\min}(\tilde{\mathbf{H}})$ is amplified after mixing, serving as a "reminder" for old knowledge.
    - **Design Motivation**: Empirically, it is long known that "regularization is ineffective, replay works even at 10%," but lacked an explanation. This theory clarifies which term in the dynamics each method affects, predicting: to truly solve forgetting, the convergence point must be shifted → replay is necessary.

3. **STOC: Selecting Tokens via attentiOn Contribution**:

    - **Function**: Without storing original PT data, automatically selects "the most informative, most likely to trigger old knowledge" snippets from CPT samples, letting the pretrained LM generate replay.
    - **Mechanism**: (i) For each CPT sample, perform a forward pass, average attention scores across all layers × heads to obtain token-level importance $a_t$; (ii) use a sliding window to find the fixed-length snippet (typically 16–32 tokens) with the highest $\sum_t a_t$; (iii) use this snippet as a prompt to the "original pretrained model" for continuation generation—according to the dynamics, high-attention tokens are those with low Diversity Index, most likely to lock onto a set of old facts, so the generated content likely covers old knowledge; (iv) MinHash deduplication ensures replay diversity; (v) mix with CPT corpus at ratio $\alpha\in\{0.5, 0.67, 0.8, 0.9\}$ for training.
    - **Design Motivation**: LAMOL and similar methods use special tokens as prompts for generation, not leveraging the transformer's attention structure, so generated content may drift from the model's actual knowledge. STOC uses attention to directly select "tokens the model actually cares about" as seeds, effectively prompting the model to recall along its most familiar directions, yielding higher-quality replay.

### Loss & Training
Basic CPT uses cross-entropy $\mathcal{L} = -\mathrm{logit}(x_{T+2}\mid \mathbf{X}) + \log\sum_o \exp(\mathrm{logit}(x_o\mid \mathbf{X}))$. In synthetic Biography experiments, assume $\eta_Y \gg \eta_Z$ and use SGD; in real LLM experiments, use Pythia-160M / Qwen2.5-0.5B-1.7B, trying full-parameter, rank-128 LoRA, and freezing the first 6 layers as update strategies. EWC estimates parameter importance via Fisher Information; both STOC and LAMOL mix new and old data at ratio $\alpha$ for next-token loss.

## Key Experimental Results

### Main Results
Comparison on Pythia-160M with synthetic Biography data. "Original" indicates retention of old (PT phase) knowledge, "Continual" indicates absorption of new (CPT phase) knowledge, $\alpha$ is the CPT data mixing ratio; higher is better.

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
On KnowEdit (ZSRE / Wiki_Bio / Wiki_Recent), average soft token accuracy of Qwen2.5-0.5B (higher is better):

| Method | ZSRE Orig | ZSRE Cont | Wiki_Bio Orig | Wiki_Bio Cont | Wiki_Recent Orig | Wiki_Recent Cont |
|---|---|---|---|---|---|---|
| Naive | 34.58 | 63.28 | 32.33 | 35.50 | 19.28 | 28.42 |
| LAMOL ($\alpha{=}0.5$) | 37.54 | 58.37 | 31.29 | 34.49 | 20.48 | 27.19 |
| **STOC ($\alpha{=}0.5$)** | **37.12** | **62.26** | **35.57** | 35.46 | **21.40** | **28.75** |
| STOC ($\alpha{=}0.8$) | 37.47 | 62.59 | 35.28 | 33.16 | 20.12 | 27.34 |

On legal domain IndustryCorpus2 1B token real CPT evaluation (MMLU / MMLU-Redux-2.0 / SuperGPQA), STOC outperforms LAMOL by 1–4 percentage points on both 0.6B and 1.7B models; on SuperGPQA, Continual subset improves from LAMOL's 13.35% to 15.85%.

### Key Findings
- Even with replay ratio as low as 10%, the model can retain significant old knowledge—matching the theory that "replay directly alters frequency distribution and shifts the convergence point."
- Among two replay selection strategies, "each individual retains one biography" outperforms "half the individuals retain two," indicating replay data should have **broad coverage** rather than local deepening.
- STOC's attention-pruned snippets as prompts outperform random snippets (ablation), confirming attention-based selection is truly causal rather than a heuristic coincidence.
- In the synthetic 1-Aug setting (each person generates only 1 biography), LM achieves 87% EM on the training set but only 8.85% on the test set, highlighting the key role of data augmentation for generalization—also validating the Diversity Index theory: augmentation makes relation token $\overline{\mathbf{x}}_s$ more uniform → attention decreases → model relies more on subject token → better cross-template generalization.
- LoRA performs worst at low $\alpha$, while full-parameter and freezing the first 6 layers show little difference, suggesting that for cFKA, parameter count is more critical than "which layers are updated."

## Highlights & Insights
- The long-standing engineering observation that "regularization does not work on LLMs" is, for the first time, explained by a clean eigenvalue argument: $\lambda^+_{\min}(\mathrm{diag}(\mathbf{w}_s))$ approaching zero means the regularization term cannot shift the convergence position of $\mathbf{y}_s$ at all. This "explaining failed methods via dynamics" perspective is also applicable to other continual learning scenarios.
- The Diversity Index quantifies the role of tokens in factual representation in a closed form $\sqrt[4]{\sum_o[\ln\Pr(s\mid o)+L^{-1}\ln\Pr(o)]^2}$, and is strongly correlated with empirical attention in multi-layer LMs (Pearson $<-0.8$), making it valuable for attention probing/interpretability research.
- STOC is a "zero extra training cost" engineering component—just collect attention once during the original forward pass to provide a high-quality replay source for existing CPT pipelines; it is orthogonal and combinable with freezing/LoRA.

## Limitations & Future Work
- The theory is built on single-layer linear attention + structured-input assumptions; softmax/multi-head/multi-layer/multi-stage training are only empirically echoed, lacking formal extension.
- Facts are modeled only as (subject, relation, object) triples, with limited coverage of commonsense, chain reasoning, or long-context knowledge forms.
- STOC selects snippets by sliding window in the sequence dimension and averaging across layers, without exploring layer-specific or head-specific selection strategies—potentially more precise attention attribution methods exist.
- Real-world experiments use models up to Qwen2.5-1.7B; scalability to 10B+ models and the scaling law between replay ratio and model size remain open questions.

## Related Work & Insights
- **vs LAMOL (Sun 2020)**: Both are generative replay, but LAMOL uses special tokens as prompts for free-form generation; STOC uses attention attribution to find truly "knowledge-dense" snippets, generating content closer to the old distribution.
- **vs EWC (Kirkpatrick 2017)**: Classic regularization method; this work proves from an eigenvalue perspective that under cFKA, it can only "slow forgetting" but not "suppress forgetting."
- **vs Allen-Zhu & Li "Physics of LM" series**: This work adopts the same synthetic Biography task and hFTA/sFTA/EM metric system, but for the first time extends training dynamics to the CPT stage and designs a new algorithm accordingly.

## Rating
- Novelty: ⭐⭐⭐⭐ Provides closed-form dynamics for cFKA and designs attention-based replay accordingly, forming a complete theory → algorithm chain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers synthetic + KnowEdit + IndustryCorpus real legal corpora + Pythia/Qwen at multiple scales + three parameter update strategies; coverage is solid but maximum scale is small.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear; Figure 1's "PT validate → CPT analyze → algorithm propose" roadmap makes the long paper readable.
- Value: ⭐⭐⭐⭐ Directly usable tool for industrial CPT teams (STOC is plug-and-play), also provides a citable argument for "why EWC doesn't work," with both theoretical and engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] FictionalQA: A Dataset for Studying Memorization and Knowledge Acquisition](../../ICLR2026/llm_pretraining/fictionalqa_a_dataset_for_studying_memorization_and_knowledge_acquisition.md)
- [\[ICML 2026\] Understanding Catastrophic Forgetting In LoRA via Mean-Field Attention Dynamics](understanding_catastrophic_forgetting_in_lora_via_mean-field_attention_dynamics.md)
- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Edit-Based Refinement for Parallel Masked Diffusion Language Models](edit-based_refinement_for_parallel_masked_diffusion_language_models.md)
- [\[ACL 2026\] FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning](../../ACL2026/llm_pretraining/forever_forgetting_curve-inspired_memory_replay_for_language_model_continual_lea.md)

</div>

<!-- RELATED:END -->
