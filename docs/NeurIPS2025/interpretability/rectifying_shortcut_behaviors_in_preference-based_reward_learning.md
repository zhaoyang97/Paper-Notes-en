---
title: >-
  [Paper Note] Rectifying Shortcut Behaviors in Preference-based Reward Learning
description: >-
  [NeurIPS 2025][reward hacking] This paper proposes PRISM (Preference-based Reward Invariance for Shortcut Mitigation), which unifies reward hacking as a shortcut learning problem and employs group-invariant kernels approximated via random feature maps to simultaneously mitigate multiple spurious correlations (verbosity, sycophancy, tone, etc.), achieving consistent improvements on out-of-distribution preference data and downstream policy models.
tags:
  - NeurIPS 2025
  - reward hacking
  - shortcut learning
  - group-invariant kernel
  - RLHF
  - preference alignment
date: 2026-05-08
content_hash: ce3694c3539e914c
---

# Rectifying Shortcut Behaviors in Preference-based Reward Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.19050](https://arxiv.org/abs/2510.19050)
**Code**: To be confirmed
**Area**: Interpretability
**Keywords**: reward hacking, shortcut learning, group-invariant kernel, RLHF, preference alignment

## TL;DR

This paper proposes PRISM (Preference-based Reward Invariance for Shortcut Mitigation), which unifies reward hacking as a shortcut learning problem and employs group-invariant kernels approximated via random feature maps to simultaneously mitigate multiple spurious correlations (verbosity, sycophancy, tone, etc.), achieving consistent improvements on out-of-distribution preference data and downstream policy models.

## Background & Motivation

The core challenge facing reward models in RLHF is **reward hacking** — models exploit spurious correlations in training data (e.g., longer responses, flattering tone, sycophantic attitudes) rather than features genuinely aligned with human intent. Specific manifestations include:

- **Verbosity**: Reward models favor longer responses regardless of content quality.
- **Sycophancy**: Models prefer responses that defer to user opinions, even when the user is incorrect.
- **Concept correlation**: Irrelevant textual concepts (e.g., "food") are erroneously associated with target attributes (e.g., positive sentiment).

Limitations of prior work:

**Piecemeal treatment**: Methods such as ODIN address only verbosity, while RRM targets only length bias — neither can handle multiple shortcuts simultaneously.

**Reliance on attribute annotations**: Multi-objective approaches require fine-grained annotations (e.g., HelpSteer), which are difficult to obtain in practice.

**Lack of a unified theory**: Different biases are treated as independent problems with no overarching framework.

The paper's core insight is that **reward hacking is fundamentally shortcut learning** — the same phenomenon as models exploiting background or texture features in classification tasks — and can therefore be addressed uniformly through invariance theory.

## Method

### Overall Architecture

The core mechanism of PRISM (illustrated in Figure 2) proceeds as follows:

1. **Model shortcuts as group actions**: Verbosity, sycophancy, and similar shortcuts are treated as transformations of response $y$ under a group $\mathcal{G}$ (e.g., increasing/decreasing length, altering tone).
2. **Learn group-invariant kernels**: Ensure that the reward is invariant to these transformations.
3. **Approximate kernels via random feature maps**: Decouple the reward margin from spurious features.
4. **Modify the BT loss**: Subtract the shortcut kernel distance from the reward margin, forcing the model to rely on generalizable features.

### Key Designs

**Group-invariant kernel (Definition 1)**:

For a kernel $\kappa$ on the response space $\mathcal{Y}$, the group-invariant kernel is defined as:

$$\mathcal{K}(y_w, y_l | x) = \int_{g \in \mathcal{G}} \int_{g' \in \mathcal{G}} \kappa(gy_w, g'y_l | x) \, d\mu(g) \, d\mu(g')$$

satisfying $\mathcal{K}(gy_w, g'y_l | x) = \mathcal{K}(y_w, y_l | x)$, i.e., invariance to arbitrary group transformations.

**Random feature map approximation (Proposition 1)**:

Since direct computation of the Haar integral is intractable, random feature maps $\Phi$ are used for approximation:

$$\Phi(y) = \left[\phi\left(y, t_j, \frac{sk}{n}\right)\right]_{j=1\dots m, k=-n\dots n} \in \mathbb{R}^{(2n+1) \times m}$$

where $\phi$ is a normalized empirical CDF. As the number of bins $n \to \infty$, $\langle \Phi(y_w), \Phi(y_l) \rangle \to \mathcal{K}_s(y_w, y_l | x)$.

**Orbit distance (Theorem 1)**: The inner product of the feature maps accurately reflects the orbit distance $d_{\mathcal{G}}$ between two responses in the shortcut space.

**Practical kernel**: For $m$ shortcuts, a convex combination of RBF kernels is used:

$$\mathcal{K}_{\text{inv}} = \sum_{j=1}^m \alpha_j \exp\left(-\frac{\|\Phi_j(y_w, x) - \Phi_j(y_l, x)\|^2}{\omega_j^2}\right)$$

**PRISM full loss function**:

$$\mathcal{L}_{\text{PRISM}}(\theta) = -\frac{1}{N}\sum_{i=1}^N \log\sigma\left(\Delta_{r_\theta}(y_w, y_l | x) - \lambda_1 \mathcal{K}_{\text{inv}}(y_w, y_l | x)\right) + \lambda_2 \mathcal{R}_{\text{global}}(\theta)$$

- The first term **subtracts** the shortcut kernel value from the reward margin, discouraging reliance on spurious differences.
- The second term is a global decorrelation regularizer that penalizes batch-level correlation between the reward and shortcut features:

$$\mathcal{R}_{\text{global}}(\theta) = \sum_{j=1}^m \left(\frac{\text{Cov}_{\mathcal{B}}(r_\theta, \Phi_j)}{\sigma_{\mathcal{B},r_\theta} \cdot \sigma_{\mathcal{B},\Phi_j}}\right)^2$$

### Shortcut Feature Extraction

- **Rule-based**: Response length (character count), lexical diversity (TTR = unique tokens / total tokens).
- **LLM-as-Judge**: Sycophancy, creativity, and helpfulness scores (0–10) extracted via GPT-4o + LangChain API.
- LRU cache (10K entries) + batch parallel processing + heuristic fallback.

### Loss & Training

- $\lambda_1, \lambda_2$ follow a curriculum schedule: linearly increasing from 0.01 to 0.1 (first half), then decreasing to 0.06 (second half).
- Learning rate $2 \times 10^{-6}$, cosine annealing with 3% warmup.
- Weights $\alpha_j$ are learned via a learnable softmax layer.
- Hardware: 8 × NVIDIA A6000.
- Implementation based on HuggingFace + DeepSpeed.

## Key Experimental Results

### Main Results

**RewardBench**:

| Method | Base Model | Chat | Chat Hard | Safety | Reasoning | Score |
|--------|-----------|------|-----------|--------|-----------|-------|
| Bradley-Terry | Llama-3 8B | 99.4 | 65.1 | 87.8 | 86.4 | 83.6 |
| RLHFlow | Llama-3 8B | 99.4 | 65.1 | 87.8 | 86.4 | 84.7 |
| GRM | Llama-3 8B | 98.6 | 67.8 | 89.4 | 92.3 | 87.0 |
| **PRISM** | Llama-3 8B | **98.7** | **68.3** | **91.1** | **93.1** | **87.8** |

PRISM achieves improvements across all three difficult categories (Chat Hard, Safety, Reasoning), with an overall score of 87.8 — the best among all compared methods.

**RM-Bench (more challenging benchmark)**:

| Method | Chat | Math | Code | Safety | Easy | Normal | Hard | Avg |
|--------|------|------|------|--------|------|--------|------|-----|
| Skywork-8B | 69.5 | 60.6 | 54.5 | 95.7 | 89.0 | 74.7 | 46.6 | 70.1 |
| URM-8B | 71.2 | 61.8 | 54.1 | 93.1 | 84.0 | 73.2 | 53.0 | 70.0 |
| **PRISM (8B)** | 70.6 | **70.8** | **57.0** | 94.1 | **90.6** | **76.3** | 46.9 | **71.0** |

Gains are particularly pronounced on Math (+9.0) and Code (+2.9), categories in which shortcuts are harder to exploit.

### Ablation Study

**Downstream policy model evaluation (AlpacaEval-2)**:

Gemma-9B policy models trained with different reward models show that PRISM-induced policies achieve higher win rates with moderate response lengths, whereas BT, RRM, and ODIN baselines either yield lower win rates or produce excessively long responses.

**Shortcut correlation analysis (Figure 4)**:

| Shortcut | BT (PCC) | PRISM (PCC) |
|----------|----------|-------------|
| Response Length | Strong positive correlation | ≈ 0 |
| Tone | Non-trivial correlation | ≈ 0 |
| Sycophancy | Non-trivial correlation | ≈ 0 |

PRISM achieves near-zero Pearson correlation coefficients across all three shortcut dimensions, directly demonstrating the effectiveness of shortcut mitigation.

### Key Findings

1. Joint multi-shortcut regularization outperforms piecemeal treatment — consistent improvements on RewardBench stem from simultaneous multi-dimensional regularization.
2. The learned weights $\alpha_j$ reveal the relative importance of different shortcuts.
3. Curriculum scheduling of $\lambda$ prevents underfitting caused by premature penalization.
4. Rule-based and LLM-Judge features are complementary — the former provides fast coverage of simple shortcuts, while the latter handles semantic-level biases.

## Highlights & Insights

1. **Elegance of the unified framework**: Multiple manifestations of reward hacking (verbosity, sycophancy, concept correlation, etc.) are unified under a single shortcut learning framework, addressed through invariance theory.
2. **Theoretical guarantee (Theorem 2)**: A generalization bound is provided showing that empirical risk converges to optimal risk as the number of shortcut features $m$, bins $n$, group elements $|\mathcal{G}|$, and training samples $N$ increase.
3. **Flexible feature interface**: The framework supports features ranging from simple length counting to complex LLM-as-Judge scores, and is easily extensible to new shortcuts.
4. **Intuitive validation via near-zero correlation**: The PCC analysis in Figure 4 provides highly convincing empirical evidence at a glance.
5. **No attribute annotations required**: Unlike methods relying on fine-grained annotations (e.g., HelpSteer2 RM), PRISM requires no manual attribute labeling.

## Limitations & Future Work

1. **GPT-4o API cost**: LLM-as-Judge feature extraction requires substantial API calls, increasing training overhead.
2. **Prior knowledge dependency**: The set of shortcuts to mitigate must be specified in advance; unspecified shortcuts may still be exploited.
3. **Kernel function selection**: Only RBF kernels are explored; alternative kernels (e.g., polynomial, Matérn) remain uninvestigated.
4. **Evaluation bias**: RewardBench and RM-Bench themselves may carry unknown biases, potentially affecting evaluation fairness.
5. **Scalability**: As the number of shortcut types grows, learning $\alpha_j$ and computing features may become computational bottlenecks.
6. **Theory–practice gap**: The applicability of Theorem 2's RKHS assumptions to LLM-based reward models warrants further verification.

## Related Work & Insights

- **ODIN (Chen et al.)**: Addresses verbosity bias via length penalties; representative of single-shortcut methods.
- **RRM (Liu et al.)**: Reduces reward model dependence on length through regularization, but handles length only.
- **GRM (Yang et al.)**: Improves reward model generalization via regularization; PRISM advances further upon this foundation.
- **Invariant Risk Minimization (Arjovsky et al.)**: Foundational work in invariance theory; PRISM adapts its core ideas to RLHF.
- **Shortcut Learning (Geirhos et al.)**: A systematic survey of shortcut problems in classification; this paper generalizes the concept to preference learning.

**Implications**: The group-invariant kernel framework has broad applicability and can be extended to direct preference optimization algorithms such as DPO, multimodal reward models, and shortcut problems in code generation (e.g., logical correctness vs. surface formatting).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Reframing reward hacking as shortcut learning is a highly original perspective; the introduction of group-invariant kernel theory is refreshing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation, downstream policy assessment, and correlation analysis together provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are complete and method motivation is clear, though some mathematical details of the kernel formulation may pose a high barrier for readers.
- Value: ⭐⭐⭐⭐⭐ Addresses one of the most central challenges in RLHF; the framework's generality makes it directly applicable to practical alignment work.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Probing Preference Representations: A Multi-Dimensional Evaluation and Analysis Method for Reward Models](../../AAAI2026/interpretability/probing_preference_representations_a_multi-dimensional_evaluation_and_analysis_m.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[NeurIPS 2025\] FastDINOv2: Frequency Based Curriculum Learning Improves Robustness and Training Speed](fastdinov2_frequency_based_curriculum_learning_improves_robustness_and_training_.md)
- [\[NeurIPS 2025\] SynBrain: Enhancing Visual-to-fMRI Synthesis via Probabilistic Representation Learning](synbrain_enhancing_visual-to-fmri_synthesis_via_probabilistic_representation_lea.md)
- [\[NeurIPS 2025\] Learning to Focus: Causal Attention Distillation via Gradient-Guided Token Pruning](learning_to_focus_causal_attention_distillation_via_gradient-guided_token_prunin.md)

<!-- RELATED:END -->
