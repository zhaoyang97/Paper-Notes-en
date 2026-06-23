---
title: >-
  [Paper Note] Quantitative Bounds for Length Generalization in Transformers
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper provides the **first quantitative upper bound on the minimum training sequence length** required for "length generalization" (LG)—where a Transformer trained on short sequences maintains performance on arbitrarily long ones. The core argument is a simulation: as long as the internal behavior of a Transformer
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: ac3c6a24623e78e3
---
# Quantitative Bounds for Length Generalization in Transformers

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=TLSUIyBIfs](https://openreview.net/forum?id=TLSUIyBIfs)  
**Code**: TBD  
**Area**: Learning Theory / Transformer Theory / Length Generalization  
**Keywords**: Length Generalization, Limit Transformers, Simulation Argument, Finite-Precision Attention, Training Length Lower Bound

## TL;DR
This paper provides the **first quantitative upper bound on the minimum training sequence length** required for "length generalization" (LG)—where a Transformer trained on short sequences maintains performance on arbitrarily long ones. The core argument is a simulation: as long as the internal behavior of a Transformer on a long sequence can be "simulated" by a short sequence of bounded length, LG occurs. This "short sequence upper bound $N$" grows polynomially (or exponentially) with the parameter norm, positional period $\Delta$, locality $\tau$, vocabulary size $|\Sigma|$, and inverse error $\varepsilon^{-1}$.

## Background & Motivation
**Background**: Length generalization (LG) is a core challenge in LLM training—models are trained on short sequences but are expected to extrapolate to much longer inputs. Extensive empirical work has found that the success of LG is highly task-dependent: some algorithmic tasks extrapolate easily, while others (such as string copying with repetitive tokens) fail completely. On the theoretical side, Zhou et al. (2023) proposed the RASP-L conjecture (tasks expressible by "simple" RASP-L programs can achieve LG), and Huang et al. (2025) further partially proved this using "limit transformers," showing that tasks expressible by C-RASP can reach LG after **some finite training length**.

**Limitations of Prior Work**: The conclusions of Huang et al. (2025) are **asymptotic**, using "identification in the limit" type arguments—they only state that "there exists a finite threshold $N$ such that LG occurs," but provide no information on how large $N$ is. For practitioners, this offers little guidance: how long should the training context be? What is its relationship with task complexity and model scale?

**Key Challenge**: There is a quantitative gap between asymptotic "existence" proofs and the practical question of "how long to train." To bridge this, one must find a metric that simultaneously characterizes **model complexity** and **task structure**, and explicitly link it to the minimum training length $N$.

**Goal**: Following the limit transformer framework of Huang et al. (2025), this paper aims to provide quantitative bounds—finding the minimum $N$ such that "two limit transformers $f,g$ being approximately consistent on all inputs of length $\le N$" implies "they are approximately consistent on inputs of **arbitrary length**."

**Key Insight**: The authors observe that the output of a Transformer's forward pass essentially depends only on certain **sufficient statistics** of the sequence (e.g., empirical frequencies of tokens in attention, local patterns of $\tau$-suffixes). If a short sequence $z$ can approximately reproduce these statistics of a long sequence $x$, then $f(x)\approx f(z)$ and $g(x)\approx g(z)$. Combined with the assumption of $f,g$ consistency on short sequences, this implies $f(x)\approx g(x)$.

**Core Idea**: Reduce "length generalization" to "**simulating the forward pass of a long string using a short string of bounded length**," and explicitly calculate the upper bound of the short string length required for this simulation under different settings (finite/infinite precision, single/double layer, worst-case/average error).

## Method

### Overall Architecture
This paper does not propose a new model but rather provides quantitative bounds for a **purely theoretical proposition**; thus, the "method" is the proof framework. All results are built upon the **limit transformers** of Huang et al. (2025): which have infinite context length and generalized positional encodings but are constrained by three assumptions—$\Delta$-periodicity ($p_i = p_{i+\Delta}$), translation invariance ($\phi_{l,h}(j,i)=\phi_{l,h}(j+t,i+t)$), and $\tau$-locality ($\phi_{l,h}(j,i)=0$ when $i>j+\tau$). These factors allow an "infinitely long model" to be characterized by finite information.

The unified logic of the paper is a high-level **simulation argument**: given two limit transformers $f,g$ and an arbitrary length input $x$, construct a short string $z$ of length $\le N$ such that $f(x)\approx f(z)$ and $g(x)\approx g(z)$. If it is known that $f,g$ are consistent on all inputs of length $\le N$, then $f(z)\approx g(z)$, leading to $f(x)\approx g(x)$ via a three-step squeeze. The technical core is **constructing this $z$ and calculating its required length for different settings**. The authors split the problem into two precision regimes and further distinguish error types and layer counts within each:

- **Finite Precision (Section 4)**: Matches Huang et al. (2025). When the sequence is long enough, softmax degenerates into hardmax, and attention becomes a uniform average over argmax tokens. Bounds are provided for single-layer Transformers under $\ell_\infty$ error (Theorem 4.1) and distributional average error (Theorem 4.2).
- **Infinite Precision (Section 5)**: Internal computations use infinite precision, which is more suitable for multi-layer Transformers (where deep token representations are continuous mixtures of discrete tokens). Bounds are provided for two-layer Transformers (Theorem 5.2).

The technical details of the two regimes differ significantly, but the fundamental idea of "using short strings to simulate long strings" is consistent, and the qualitative conclusions regarding data requirements versus parameters match and are verified by experiments.

### Key Designs

**1. Limit Transformer + Simulation Argument: Reducing LG to Short-String Simulation**

The most difficult part of length generalization is "arbitrary length"—one cannot verify every length individually. The authors break this by converting it into a **constructible reduction**: if for any long string $x$ one can create a short string $z$ of length $\le N$ such that $f(x)\approx f(z)$ and $g(x)\approx g(z)$, then "consistency on $\le N$" is sufficient to prove consistency on arbitrary lengths.

$$f(x)\approx f(z)\approx g(z)\approx g(x)$$

The middle step $f(z)\approx g(z)$ comes from the "consistency on short strings" assumption, while the ends come from the fidelity of the simulation. The key to constructing $z$ is ensuring it **approximately preserves the sufficient statistics** that the forward pass actually depends on: in finite precision, these are empirical frequency proportions of token types in hardmax attention; in infinite precision, it is the joint distribution of $(\tau\text{-suffix}, \text{empirical histogram})$. Theorem 4.1 uses a **deterministic** construction (explicitly making $z$ approach the empirical frequency of each token), while Theorem 5.2 uses a **random** construction (sampling $z$ from a specialized distribution and proving existence via probabilistic methods). This reduction is the "skeleton" reused throughout the paper, compressing a universal proposition about infinite lengths into an existence problem regarding finite-length short strings.

**2. Hardmax Degeneracy and Logit Margin in Finite Precision: Single-layer bound $N\sim \tau\Delta L^2/\varepsilon^2$**

Finite precision ($p$ bits, values $\le 2^{-p}$ are rounded to 0) brings a key simplification: when the sequence is long enough, softmax behavior is equivalent to hardmax. The authors define the **logit margin** $\gamma(f)$ of a Transformer as the "minimum non-zero gap between the maximum attention logit and any non-maximum logit":

$$\gamma(f) := \min_{y\in\Sigma,\, i\in\mathbb{Z}/\Delta}\ \min_{\substack{a,a'\in\mathcal{A}_{(y,i)}\\ a-a'>0}} (a-a')$$

where $\mathcal{A}_{(y,i)}$ collects all possible attention logits observed when processing token $y$. When $N=|x|\ge 2^{p}/\gamma(f)$, any strictly sub-maximal logit after exponentiation satisfies $s_j\le \exp(-p\log 2 / \gamma\cdot\gamma)=2^{-p}$ and is rounded to 0. Thus, attention only performs a uniform average over argmax tokens—this transforms continuous softmax into a **discrete "token frequency proportion" problem**, where the simulation $z$ only needs to reproduce these proportions.

Theorem 4.1 ($\ell_\infty$ error) gives:
$$N = O\!\left(\max\Big\{2^{p}/\gamma,\ \tfrac{L^2\Delta^7|\Sigma|^6\tau^2}{\varepsilon^2}\Big\}\right),$$
where $L$ is a composite of the parameter norms of $f,g$. Once the threshold of $2^p/\gamma$ to "enter the hardmax regime" is crossed, the required training length grows **polynomially** with the period $\Delta$, parameter norm $L$, vocabulary $|\Sigma|$, and inverse precision $\varepsilon^{-1}$. The paper also notes that hardmax behavior might occur at smaller lengths, where $N$ only grows with $\tau\Delta L^2/\varepsilon^2$. Theorem 4.2 replaces $\ell_\infty$ with the more practical **distributional average error**: taking the expectation over tokens drawn from a Dirichlet prior $\mathrm{Dir}((\alpha_s))$, yielding $N_0=\tilde O(\varepsilon^{-2-2\alpha_0^{-1}})$ ($\alpha_0=\min_s\alpha_s$). Interestingly, controlling the average error comes with a cost—the conclusion is weakened from $O(\varepsilon)$ to $O(\varepsilon^{1/2})$. The authors emphasize that this average setting **must** have some distributional regularity assumption: if short sequences only support $\Sigma_{\text{short}}\subsetneq\Sigma$ and long sequences support the complement, the switching point can be arbitrarily large, making a bound non-existent.

**3. Log-Scaling of $\tau$-Suffixes and Complexity Metrics in Infinite Precision: Two-layer bound $N\lesssim C(f)\,\varepsilon^{-\max(\gamma^{-1},3)}$**

Infinite precision is more suitable for multi-layer Transformers, but it introduces a new problem: under infinite precision and bounded weights, the impact of the $\tau$-suffix (local information) on computation decays to 0 as the sequence length diverges. This would exclude functions like induction heads that Transformers can empirically learn. The authors' fix is to **multiply only the $\tau$-suffix logits by a logarithmic factor** $\log i$:

$$a_{i,j}^{(l,h)} = (y_j^{(l-1)})^\top K_{l,h}^\top Q_{l,h}\, y_i^{(l-1)} + \log i\cdot \phi_{l,h}(j,i)$$

This scaling allows the relative importance of local versus global information to be adjustable, creating three regimes based on the relationship between $\max_{t\le\tau}\phi_{1,h}(i-t,i)$ and 1: **Token-dominated** ($<1$, suffix contribution decays as $i\to\infty$), **Balanced** ($=1$, suffix and global contributions are of the same order), and **Position-dominated** ($>1$, suffix dominates, attention concentrates on local tokens that maximize $\phi$). The paper provides a **unified analysis** covering all three regimes.

The core quantity for the two-layer bound is the **complexity metric** $C(f)$ (Definition 5.1), which grows **exponentially** with the first-layer weight operator norm and polynomially with other quantities. This exponential dependence is unavoidable because, when treating the first-layer softmax as a function on the space of probability measures, its Lipschitz constant itself grows exponentially with $\|K_{1,h}^\top Q_{1,h}\|_{\mathrm{op}}$, while LG requires a uniform bound independent of sequence length $T$. They also define a **position margin** $\gamma(f)=\min_h(\max P_h - \text{second largest})$ ($P_h=\{\phi_{1,h}(i-t,i)\}\cup\{1\}$). Theorem 5.2 gives:
$$N\lesssim \max(C(f),C(g))\,\varepsilon^{-\max(\gamma(f)^{-1},\gamma(g)^{-1},3)}.$$
The technical heart of the proof is the **Key Simulation Lemma** (Lemma 5.3): the first-layer output $Y_i^{(1)}$ can be written as a Lipschitz-bounded function of the $\tau$-suffix $x_{i-\tau:i}$ and the empirical histogram $\mu(x_{\le i})=\frac1i\sum_{j\le i}e_{x_j}$. Thus, $z$ only needs to approximately preserve the joint distribution of $\{(x_{i-\tau:i},\mu(x_{\le i}))\}$. The authors use a Markov chain on $\{0,1\}$ to sample a subset of indices $I$ for $z$ (ensuring $z$ contains large contiguous segments of $x$ to preserve the suffix distribution), proving that there exists an $I$ where $|z|$ is close to the target length and the statistic error is $\lesssim (G+L)(\tau+1)/n^{1/3}$. As an application, an in-context $k$-gram task (a generalization of induction heads) can be approximated by a two-layer Transformer with complexity $C(f)=\varepsilon^{-\Theta(k^2)}$ and $\gamma(f)\ge 1$, so a training length of $\varepsilon^{-\Theta(k^2)}$ is sufficient for LG—indicating that complex tasks indeed require richer (longer) training data.

## Key Experimental Results

The experiments aim not to break SOTA but to **qualitatively verify the scaling relations predicted by the theory**: as training length $N$ increases, the "plateau value" of the test loss decreases; as the task becomes more complex ($\omega$/$\Delta$/$|\Sigma|$/$k$ increases), the required $N$ increases.

### Single-layer Transformer (Verifying Theorem 4.1 / 4.2)

| Task | Expressibility | Varying Complexity Parameter | Observed Scaling |
|------|----------------|------------------------------|------------------|
| SimpleTask ($f^*=\sin(\omega\cdot\frac{c_0-c_1}{c_0+c_1})$) | Single-layer, no PE, $L=\Theta(\omega)$ | Frequency $\omega$ | Plateau loss decreases monotonically with training length and increases with $\omega$ (i.e., $L$) |
| ModPTask (Mean of tokens where pos $\equiv k\bmod p$) | Single-layer, $\Delta=p$, $L=\Theta(1)$ | Period $\Delta$ | Plateau loss decreases with training length and increases with $\Delta$ |

The left plots for both tasks show: when training length is fixed, **test loss tends toward a finite plateau as test length increases** (rather than diverging), confirming that "a finite $N$ exists to achieve $\varepsilon$ error at all lengths." The right plots quantitatively confirm that $N$ grows with parameter norm $L$ (SimpleTask) and period $\Delta$ (ModPTask).

### Two-layer Transformer (Verifying Theorem 5.2)

Training a two-layer Transformer on an in-context $k$-gram task: the test loss similarly plateaus as test length increases. For a fixed $k=2$, the plateau loss increases with vocabulary size $S$; for a fixed $S$, it increases with locality $k$—consistent with the qualitative dependence of complexity $C(f)$ on $S$ and $\tau$.

### Key Findings
- **Hardmax assumption confirmed empirically**: After training on ModPTask ($p=5$), the attention probability for tokens where position $\not\equiv k\bmod p$ is nearly 0, while tokens where position $\equiv k\bmod p$ receive almost identical probabilities (near uniform). This suggests that with sufficient training length, the model indeed enters the hardmax regime relied upon in Theorem 4.1.
- **"Complex tasks require longer training data" verified quantitatively**: In all tasks, plateau loss increases monotonically with complexity parameters ($\omega,\Delta,S,k$), consistent with the theory that $N$ grows with these values.
- **Margin dependence remains an open question**: While $N$ shows an exponential dependence on the inverse margin $1/\gamma$ (e.g., $\exp(\gamma^{-1})$ in Theorem 4.1, $\varepsilon^{-\gamma^{-1}}$ in Theorem 5.2), whether the margin is a true bottleneck for LG or an artifact of the analysis is left for future work.

## Highlights & Insights
- **Compressing the "any length" universal proposition into a "short string" existence proposition**: The simulation argument is the most elegant move—it transforms the impossible task of verifying infinite lengths into "constructing a short string of length $\le N$," making quantitative bounds possible. The idea of "simulating sufficient statistics of long strings using short ones" could be transferred to generalization analysis for other infinite-length inputs (e.g., SSMs, GNNs).
- **Logit margin / Positional margin as unified "difficulty knobs"**: Both regimes use a margin metric to characterize "how well the attention separates," and both exhibit exponential dependence on $1/\gamma$, providing a coordinate for difficulty across settings.
- **$\tau$-suffix log-scaling is a clever modeling fix**: In infinite precision, local information would otherwise be diluted. Multiplying by $\log i$ recovers the expressibility of local-dependent functions like induction heads and naturally derives the three regimes (token/balanced/position-dominated) under a unified analysis.
- **Theoretical scaling verified step-by-step by synthetic experiments**: Every parameter—$L,\Delta,|\Sigma|,k$—has a corresponding loss curve for verification. This "parameter-by-parameter reconciliation" between theory and experiment is highly persuasive.

## Limitations & Future Work
- **Covers only 1 and 2 layers**: Theorem 4.x is single-layer, and Theorem 5.2 is two-layer. The authors explicitly list "extending to greater depth" as the most important future direction; quantitative bounds for LG in deep Transformers remain an open problem.
- **Bounds may not be tight**: Whether the exponential dependence of $N$ on $1/\gamma$ and the exponential dependence of complexity on the first-layer weights in Theorem 5.2 reflect real bottlenecks is unclear. The authors admit margin dependence might be an analytical artifact.
- **Strong distributional assumptions**: Average error results (Theorem 4.2) depend on regularity assumptions like the Dirichlet prior, and the error weakens from $O(\varepsilon)$ to $O(\varepsilon^{1/2})$ in the average setting. The authors point out that finding the **minimal** distributional conditions for LG (e.g., stationary distribution concentration of Markov chains) is an interesting open problem.
- **Idealization of Limit Transformers**: The objects analyzed are idealized models with infinite context, $\Delta$-periodicity, $\tau$-locality, and translation invariance, leaving a gap between these and real architectures with finite context and RoPE/ALiBi.

## Related Work & Insights
- **vs. Huang et al. (2025)**: This paper directly adopts their limit transformer framework but upgrades the **asymptotic existence** of "some finite $N$" to an **explicit quantitative upper bound** of $N$ scaling with parameters—a substantial advancement from "whether" to "how much."
- **vs. RASP-L Conjecture (Zhou et al. 2023) / Yang et al. (2025)**: RASP-L uses program length to characterize which tasks can achieve LG (qualitative stratification). This paper uses margin and complexity metrics to provide a "difficulty continuum" and quantitative data requirements within LG tasks, refining this line of research.
- **vs. Chen et al. (2025)**: Superficially similar (providing non-asymptotic bounds for LG), but they target **general computational models** for variable-length inputs rather than Transformers, providing a complementary perspective. This paper focuses tightly on the attention mechanism itself.
- **vs. Wang et al. (2024) / Golowich et al. (2025)**: These works analyze specific tasks (sparse token selection) or abstractions of self-attention heads for LG. This paper's simulation argument framework is more general, covering finite/infinite precision, 1/2 layers, and worst-case/average error settings.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first quantitative upper bound for training length in Transformer length generalization, moving beyond asymptotic existence to explicit scaling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic tasks verify the theoretical scaling point-by-point, though it remains on "toy" tasks and hasn't touched real LLMs.
- Writing Quality: ⭐⭐⭐⭐ The main simulation argument is clear, and the two-regime comparison is well-structured, though the theoretical density is high and metrics like margins require careful reading.
- Value: ⭐⭐⭐⭐⭐ Provides theoretical guidance for "how long the training context should be" and establishes a quantifiable coordinate for LG difficulty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Queue Length Regret Bounds for Contextual Queueing Bandits](queue_length_regret_bounds_for_contextual_queueing_bandits.md)
- [\[ICLR 2026\] Transformers Are Inherently Succinct](transformers_are_inherently_succinct.md)
- [\[ICLR 2026\] Efficient Turing Machine Simulation with Transformers](efficient_turing_machine_simulation_with_transformers.md)
- [\[ICLR 2026\] Probability Distributions Computed by Autoregressive Transformers](probability_distributions_computed_by_autoregressive_transformers.md)
- [\[ICLR 2026\] Understanding and Relaxing the Limitations of Transformers for Linear Algebra](understanding_and_relaxing_the_limitations_of_transformers_for_linear_algebra.md)

</div>

<!-- RELATED:END -->
