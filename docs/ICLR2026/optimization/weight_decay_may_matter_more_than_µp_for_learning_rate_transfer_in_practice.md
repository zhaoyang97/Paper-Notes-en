---
title: >-
  [Paper Note] Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice
description: >-
  [ICLR 2026][Optimization & Theory][µP] This paper revisits learning rate transfer in large model training through a unified "relative update" framework. It discovers that the alignment assumptions upon which µP relies quickly fail during actual training. What truly stabilizes cross-width feature learning and enables learning rate transfer throughout most of
tags:
  - ICLR 2026
  - Optimization & Theory
  - µP
  - weight decay
  - AdamW
date: 2026-05-08
content_hash: e0e0782b59b209f5
---
# Weight Decay May Matter More Than µP for Learning Rate Transfer in Practice

**Conference**: ICLR 2026  
**Paper**: Published as a conference paper at ICLR 2026  
**Code**: None (Cache not provided)  
**Area**: Optimization / LLM Pre-training  
**Keywords**: µP, Learning Rate Transfer, weight decay, AdamW, Feature Learning

## TL;DR
This paper revisits learning rate transfer in large model training through a unified "relative update" framework. It discovers that the alignment assumptions upon which µP relies quickly fail during actual training. What truly stabilizes cross-width feature learning and enables learning rate transfer throughout most of the training process is **independent weight decay**. The learning rate scaling of µP essentially serves only as an "implicit learning rate warmup," which can be largely replaced by a stronger explicit warmup.

## Background & Motivation
**Background**: Training ultra-large-scale models (especially LLMs) makes tuning hyperparameters for each model size prohibitively expensive. Maximal Update Parameterization (µP) provides a shortcut—tuning the optimal learning rate on a small model and scaling it to larger models according to specific rules, which theoretically keeps the "optimal learning rate" invariant across widths. µP has become a cornerstone of many open-source LLM training recipes.

**Limitations of Prior Work**: However, a confusing phenomenon has repeatedly surfaced in practice (observed by Wortsman 2024, Wang & Aitchison 2025, Bergsma 2025, etc.)—µP only enables good learning rate transfer when paired with **independent weight decay** (as seen in the original Paper Figure 1: optimal learning rates overlap across widths under independent WD, but scatter under standard WD). This is counter-intuitive: standard weight decay is the one that "follows µP theoretical scaling," whereas independent WD violates µP yet yields better results.

**Key Challenge**: The scaling rules of µP depend on a set of strong geometric assumptions, centered on the **alignment** between layer inputs $X$, weights $W$, and gradient updates $\Delta W$. µP assumes weight alignment $\alpha_W = \Theta(1/\sqrt{C})$ and update alignment $\alpha_{\Delta W}=\Theta(1)$. The problem is: do these assumptions actually hold during real training? If not, what is truly making the learning rate transfer work correctly?

**Goal**: (1) Construct a unified framework characterizing both µP and weight decay; (2) Measure the evolution of µP alignment assumptions throughout training; (3) Clarify the actual roles of µP scaling and weight decay; (4) Verify if explicit warmup can replace µP.

**Key Insight**: Instead of focusing on "absolute update amounts," the authors use **relative updates** (relative weight update $\|\Delta W\|/\|W\|$ and relative representation change $\|\Delta Y\|/\|Y\|$) as measures. The reason is that neural networks are naturally invariant to a class of rescaling transformations (normalization, learnable gains, homogeneous activations). Fixed absolute update amounts have varying effects under these equivalent states, whereas fixed relative updates have consistent impacts—relative amounts better reflect "how much influence this update step actually has," and they naturally incorporate weight decay.

**Core Idea**: By using the goal of "maintaining constant relative representation change across widths," the authors put µP and weight decay on the same scale. It is found that µP's alignment assumptions only hold in the first few steps of training; thereafter, independent weight decay takes over, and the only remaining practical value of µP scaling is "implicit warmup."

## Method
This is an analysis and large-scale empirical paper without a system architecture for a pipeline. The core is an inference chain: first unifying the two with a relative update framework, then measuring the failure of alignment assumptions, explaining how independent WD takes over, and finally reducing µP to warmup and verifying its replaceable nature.

### Overall Architecture
The entire paper revolves around a **single-layer linear transformation** $Y = WX$ ($X\in\mathbb{R}^{C\times B}$ input, $W\in\mathbb{R}^{K\times C}$ weights, $Y\in\mathbb{R}^{K\times B}$ output). An update $W\mapsto W+\Delta W$ causes an output change $\Delta Y = \Delta W X$. Optimizers (like Adam) easily control the absolute size of updates $\|\Delta W\|$, but what truly determines the impact of a step is the representation change $\|\Delta Y\|$, which is harder to control—precisely what µP learning rate scaling aims to achieve.

The bridge linking the two is **alignment**. Update alignment is defined as:

$$\alpha_{\Delta W} := \frac{\|\Delta Y\|}{\|\Delta W\|\,\|X\|} = \frac{\|\Delta W X\|}{\|\Delta W\|\,\|X\|}\in[0,1]$$

This is essentially the weighted root-mean-square of the cosine similarity between input samples $x_b$ and update rows $\Delta w_k$: higher alignment means the same size $\|\Delta W\|$ brings a larger $\|\Delta Y\|$. Similarly, weight alignment is defined as $\alpha_W := \|Y\|/(\|W\|\,\|X\|)$. Thus, relative representation change can be decomposed as:

$$\frac{\|\Delta Y\|}{\|Y\|} = \frac{\alpha_{\Delta W}}{\alpha_W}\cdot\frac{\|\Delta W\|}{\|W\|}$$

Where $\alpha_{\Delta W}/\alpha_W$ is called the **alignment ratio**. To keep $\|\Delta Y\|/\|Y\|$ constant across widths, µP must use the learning rate to precisely offset the change in the alignment ratio as width varies. All conclusions in this paper are based on observing "which term changes and who compensates" in this equation.

µP's theoretical hypothesis is: at initialization, $\alpha_W\approx 1/\sqrt{C}$ and $\alpha_{\Delta W}=\Theta(1)$, so the alignment ratio $\propto \sqrt{C}$. When width $C\mapsto mC$, to offset this, µP specifies that relative updates $\propto 1/\sqrt{m}$, corresponding to Adam learning rate $\eta=\eta_{\text{base}}/m$.

### Key Designs

**1. Unified Relative Update Framework: Putting µP and weight decay on the same scale**

Studies on µP traditionally use "absolute representation change $\|\Delta Y\|_{\text{RMS}}=\Theta(1)$," while weight decay research traditionally uses "relative updates." The first step of this paper is to rewrite µP in relative terms (the $\|\Delta Y\|/\|Y\| = (\alpha_{\Delta W}/\alpha_W)\cdot\|\Delta W\|/\|W\|$ above) to make them comparable. The key fact is: weight decay does not primarily act as a regularizer, but as a "second learning rate," modulating the steady-state size of relative updates. Kosson et al. show that under AdamW steady state, the weight norm converges to $\|W\|\approx\sqrt{KC\,\eta/\lambda}$. Combined with AdamW's update normalization $\|\Delta W\|\propto\eta\sqrt{KC}$, we get the core relationship:

$$\|W\|\propto\sqrt{KC\,\eta/\lambda},\qquad \frac{\|\Delta W\|}{\|W\|}\propto\sqrt{\eta\lambda}$$

This formula reveals the key: in steady state, the influence of learning rate $\eta$ and weight decay $\lambda$ on relative updates is **completely symmetric; only the product $\eta\lambda$ matters**. This sets the stage for "weight decay can replace/neutralize µP scaling"—since only $\eta\lambda$ matters, tuning $\lambda$ is equivalent to tuning $\eta$ in steady state.

**2. Independent weight decay ultimately neutralizes µP scaling**

AdamW has two weight decay implementations: PyTorch's default multiplies weights by $1-\eta\lambda$ each step, while the original version by Loshchilov & Hutter multiplies by $1-\lambda$. When µP scales the learning rate to $\eta\mapsto\eta/m$, these two methods of handling $\lambda$ yield completely different consequences:

$$(\eta,\lambda)\mapsto(\eta/m,\ \lambda)\quad\text{(standard scaling)}$$
$$(\eta,\lambda)\mapsto(\eta/m,\ m\lambda)\quad\text{(independent scaling)}$$

Independent scaling keeps the product $\eta\lambda$ constant, so according to the conclusion of Design 1, the **steady-state relative update is not scaled at all**. This contradicts µP's requirement that "relative updates $\propto 1/\sqrt{m}$." In other words, independent weight decay **directly overrides µP's update scaling** in later stages of training; whereas standard weight decay strictly follows µP. Measurements in Figure 3 of the original paper confirm: under independent WD, relative representation changes for different widths overlap, whereas they diverge significantly under standard WD later in training. This explains the counter-intuitive phenomenon—"µP-violating" independent WD transfers better because µP's theoretical scaling is incorrect from the mid-training stage onwards and needs to be neutralized.

**3. Weight decay fills the gap left by failing µP alignment assumptions**

Why is neutralization necessary? Because µP's alignment assumptions collapse quickly. Figure 4 in the original paper shows that the alignment ratio drops from its initial "width-dependent" value to approximately $1$ rapidly, violating the $\Theta(\sqrt{C})$ assumption of µP. Once the alignment ratio $\approx 1$, the relative representation change equation dictates that to keep $\|\Delta Y\|/\|Y\|$ constant across widths, the relative weight update $\|\Delta W\|/\|W\|$ must be constant across widths—which is exactly what independent weight decay (keeping $\eta\lambda$ constant) does, whereas µP's standard scaling suppresses the relative updates of large networks, leading to mismatch in later stages.

Why does update alignment become width-dependent? The authors provide a Mechanism via SGD on a single neuron (Paper Eq. 9): output change $\Delta y_i$ contains one "self-contribution term" and $B-1$ "interference terms." Random alignment makes each interference term $1/\sqrt{C}$ weaker, but the random summation of $B-1$ terms amplifies it by $\approx \sqrt{B}$. Consequently, when **batch size $B\gg C$** (width), the interference terms dominate, and update alignment depends on $C$ again, breaking the µP assumption. This is extremely common in practice: in LLaMA experiments, the effective "number of samples" per batch is the total token count 1,048,576, far exceeding the network width $C\le 3\times 2048$. This also explains why µP-like scaling became needed only after Transformers became popular—well-normalized CNNs at reasonable scales might not need it (see ResNet experiments).

**4. Re-characterizing µP as "implicit learning rate warmup" and replacing it with explicit warmup**

Since actual training is dominated by "alignment ratio $\approx 1$" and relative updates should be constant across width, µP scaling is redundant or even harmful later on. Its only remaining benefit occurs **early** in training. Independent scaling $(\eta,\lambda)\mapsto(\eta/m,m\lambda)$ uses a "lower learning rate + higher weight decay" to achieve the same $\eta\lambda$, making early relative updates smaller and gradually recovering as weight norms approach steady state—this matches the shape of a warmup (Paper Figure 5, scaling factor $s_t$ approximates $1$ exponentially from $1/m$). Under a simplified constant learning rate setting, the authors provide a closed-form (Paper Eq. 11):

$$s_t = \sqrt{\frac{1+(\rho_0^2/\rho_\infty^2-1)a^{2t}}{1+(m^2\rho_0^2/\rho_\infty^2-1)a^{2t}}}\ \xrightarrow{\ \rho_0=\rho_\infty\ }\ \frac{1}{\sqrt{1+(m^2-1)a^{2t}}}$$

where $a:=1-\eta\lambda$ is the decay multiplier per step, $\rho_t$ is the weight RMS, and $\rho_\infty=\sqrt{\eta/(2\lambda)}$ is the predicted steady-state value. Based on this insight, the authors construct two multiplicative warmup factors to replace µP (exponentially increasing Eq. 12, decay-away type Eq. 13), both suppressing initial learning rate to $1/m$ before returning to $1$. Figure 6 shows: overlaying this extra warmup on top of a 10% linear warmup achieves learning rate transfer similar to, or even more stable than, µP+independent WD—confirming that warmup is the primary practical benefit of µP.

### Loss & Training
No new loss functions involved. The objective is standard next-token prediction (LLaMA pre-training on DCLM, widths 128–2048, ~20B tokens; ResNet/ImageNet as supporting evidence). The core actionable conclusion is the hyperparameter scaling strategy: use independent weight decay (keeping $\eta\lambda$ constant), or use stronger explicit warmup (exponential/decay-away) without µP scaling to stabilize relative feature updates across widths.

## Key Experimental Results

### Main Results: Learning Rate Transfer Quality

| Setting | Weight Decay Form | LR Transfer | Description |
|------|------|------|------|
| µP (LLaMA Pre-training, 20B tokens, width to 2048) | Independent WD | Good (Optimal LR overlaps) | The only combination where µP truly works (Fig.1) |
| µP | Standard WD | Poor (Divergence in late stages) | Following µP theoretical scaling leads to transfer failure |
| µP | No WD | Intermediate | Scale-invariant weights grow indefinitely; relative updates lose LR dependence |
| No µP, 10% / 50% Linear warmup | — | Poor | Long linear warmup cannot recover µP's gains (Fig.6 center) |
| No µP, Extra Exponential Warmup (Eq.12/13) | — | Good | Explicit warmup largely replaces µP scaling; slightly worse at high LR (Fig.6 right) |

### Mechanism Analysis: Do Alignment Hypotheses Hold?

| Metric (LLaMA, Width 128 vs 2048) | µP Hypothesis | Measured Result |
|------|------|------|
| Weight Alignment $\alpha_W$ | $\Theta(1/\sqrt{C})$ | Matches only at very early training, diverges thereafter (more evident in ResNet) |
| Update Alignment $\alpha_{\Delta W}$ | $\Theta(1)$ (Width-independent) | Changes significantly over time; becomes width-dependent (caused by $B\gg C$) |
| Alignment Ratio $\alpha_{\Delta W}/\alpha_W$ | $\Theta(\sqrt{C})$ | Drops rapidly to $\approx 1$, losing width dependence (Fig.4) |
| Relative Representation Change $\|\Delta Y\|/\|Y\|$ | Should be constant | Maintained by Independent WD; diverges later with Standard WD (Fig.3) |

### Key Findings
- **What enables LR transfer**: Throughout most of the training, independent weight decay (maintaining $\eta\lambda$)—not µP scaling—stabilizes cross-width feature learning. µP's theoretical scaling is only correct in the initial steps.
- **When hypotheses break**: The root cause of µP alignment failure is large batch size $B$ relative to width $C$ ($B\gg C$ makes update alignment width-dependent again), which is standard in LLM training (effective token count $\approx 10^6 \gg C$).
- **True value of µP**: Equivalent to an implicit warmup starting at $1/m$ and exponentially recovering to $1$; for long training, this is the only practical benefit remaining from µP LR scaling.
- **Replaceability**: Explicit exponential warmup can largely replace µP; however, independent scaling is slightly more stable at very high learning rates. For ResNet, this extra warmup is largely unnecessary (possibly due to superior normalization).

## Highlights & Insights
- **The Unified Scale is Clever**: By translating both µP (traditionally absolute representation change) and weight decay (traditionally relative updates) into "relative amounts," the single equation $\|\Delta Y\|/\|Y\|=(\alpha_{\Delta W}/\alpha_W)\cdot\|\Delta W\|/\|W\|$ makes "who is compensating for whom" crystal clear. This is the pivot of the paper's reasoning.
- **Explaining a Long-standing Anti-intuitive Phenomenon**: Why "µP-violating" independent weight decay transfers better—because µP theoretical scaling is wrong from the mid-point of training, and independent WD happens to neutralize it back to the correct track.
- **The $B\gg C$ Criterion is Transferable**: Reducing "when µP fails" to a calculable condition (batch/width ratio) allows practitioners to predict whether their setup falls within the µP validity range and explains why µP was specifically needed in the Transformer era.
- **Pointing Towards Better Alternatives**: The discussion suggests that matrix-level optimizers like Muon/Scion can maintain constant low update alignment, potentially bypassing these complexities naturally and reducing the need for warmup—an insightful direction for future work.

## Limitations & Future Work
- **Limited to AdamW**: The authors explicitly only cover AdamW (as it is the LLM mainstream and SGD-µP does not scale hidden layer LRs, making differences harder to expose). Dynamics under SGD or matrix-level optimizers (Muon/Scion) might differ significantly.
- **Simplified Model Accuracy**: The closed-form warmup shape in Eq. 11 overestimated the time to reach steady state in measurements, which the authors attribute to momentum and gradient temporal correlations not modeled in the simplification.
- **Tuning for Substitutes**: Replacing µP with explicit warmup is slightly less stable at high LRs, and the length of decay-away/exponential warmup requires tuning, rather than being "plug-and-play."
- **Confounding Factors in ResNet**: Regularization gains from large LRs on ImageNet interfere with determining "train loss vs val loss optimal relative change," somewhat diluting the universality of the conclusions.
- **Missing arXiv ID**: The cache did not provide an arXiv number; please refer to the original paper when citing.

## Related Work & Insights
- **vs Wang & Aitchison (2025)**: They also argue that µP needs independent WD but use the perspective that AdamW approximates an EMA of historical updates whose time scale should be cross-width invariant, without formalizing the argument. Ours points out the EMA view ignores "subsequent updates depending on earlier ones" (which prevents one-step optimization) and instead uses "feature change rate" to directly link WD and µP goals.
- **vs Everett et al. (2024)**: Ours follows their layer-wise LR scaling and alignment definitions but whereas Everett only considers weight alignment, Ours characterizes both update and weight alignment to see the evolution of the alignment ratio.
- **vs Yang et al. (2023)**: Borrows their more accessible µP formulation and "local change" perspective to isolate core alignment hypotheses but moves from "infinite width + initialization" to "finite width + full training" empirical analysis.
- **vs Kosson et al. (2024a,b)**: Directly builds upon their weight decay framework and "relative update / warmup" observations, providing the literal foundation for our unified relative update framework.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Overturns the mainstream belief that "µP is responsible for LR transfer," providing a unified explanation where weight decay is the protagonist.
- Experimental Thoroughness: ⭐⭐⭐⭐ Systematic comparison of multi-width LLaMA + ResNet and multiple WD forms, though focused on these two architectures.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear inference chain; formulas and measurements correspond one-to-one; explains anti-intuitive phenomena thoroughly.
- Value: ⭐⭐⭐⭐⭐ Direct, actionable guidance for hyperparameter scaling in large model training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cautious Weight Decay](cautious_weight_decay.md)
- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate](convex_dominance_in_deep_learning_i_a_scaling_law_of_loss_and_learning_rate.md)
- [\[ICLR 2026\] Seesaw: Accelerating Training by Balancing Learning Rate and Batch Size Scheduling](seesaw_accelerating_training_by_balancing_batch_size_and_learning_rate_schedulin.md)
- [\[ICML 2026\] Limits of Convergence-Rate Control for Open-Weight Safety](../../ICML2026/optimization/limits_of_convergence-rate_control_for_open-weight_safety.md)

</div>

<!-- RELATED:END -->
