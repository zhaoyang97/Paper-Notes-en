---
title: >-
  [Paper Note] Random Spiking Neural Networks are Stable and Spectrally Simple
description: >-
  [ICLR 2026][learning_theory][Paper Note] This paper treats discrete-time LIF Spiking Neural Network (SNN) classifiers as compositions of boolean functions. Using the analysis of boolean functions, it proves that wide SNNs at random initialization are "stable on average"—outputting the same prediction with high probability when input perturbations affect $O(\s
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: ed534cdeb2135584
---
# Random Spiking Neural Networks are Stable and Spectrally Simple

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Ochp5HHp46](https://openreview.net/forum?id=Ochp5HHp46)  
**Code**: To be confirmed  
**Area**: Learning Theory / Spiking Neural Networks  
**Keywords**: Spiking Neural Networks, LIF Neurons, Analysis of Boolean Functions, Noise Stability, Simplicity Bias

## TL;DR
This paper treats discrete-time LIF Spiking Neural Network (SNN) classifiers as compositions of boolean functions. Using the analysis of boolean functions, it proves that wide SNNs at random initialization are "stable on average"—outputting the same prediction with high probability when input perturbations affect $O(\sqrt{n})$ coordinates. Based on this, the authors propose the concept of "spectral simplicity," proving that random SNNs are biased toward simple functions with Fourier spectra concentrated in low frequencies. Experiments further show that training enhances this stability.

## Background & Motivation
**Background**: Spiking Neural Networks utilize event-driven sparse communication and are regarded as energy-efficient alternatives to traditional Artificial Neural Networks (ANNs), suitable for neuromorphic hardware (Loihi 2, SpiNNaker 2). However, compared to ANNs, the theoretical foundation of SNNs remains weak. While progress has been made in training algorithms and hardware implementation, core properties such as stability, robustness, and generalization have rarely been characterized theoretically.

**Limitations of Prior Work**: "Stability" in SNNs lacks a unified definition—it can refer to the algorithmic stability of learning, dynamical system stability, or "sensitivity to input perturbations," which this paper focuses on. Existing work from a dynamical systems perspective (Ding et al. 2024) only analyzes single neurons and employs a simplified reset-to-zero mechanism. Such work focuses on the difference in output spike trains, failing to address the practical question of whether classification predictions will change.

**Key Challenge**: SNN neurons only fire when the membrane potential exceeds a threshold, a binary event naturally described by boolean functions. However, real LIF models involve "reset-by-subtraction" and temporal auto-regressive decay, introducing complex probabilistic dependencies across time steps. This prevents the direct application of classical boolean function analysis.

**Goal**: (1) Provide quantitative stability bounds for wide LIF-SNN classifiers under random initialization; (2) Formalize this stability as a form of "simplicity"; (3) Theoretically verify the findings with experiments and examine the effects of training.

**Key Insight**: The authors represent the first integration of boolean function analysis (O'Donnell's framework of noise sensitivity and Fourier-Walsh expansion) into SNNs. By restricting the scope to "random networks at initialization," the authors bypass the immature state of SNN training theory and isolate the model's intrinsic stability. Furthermore, random networks hold independent value as priors in PAC-Bayes generalization bounds.

**Core Idea**: Multi-layer sign-LIF SNN classifiers are formulated as iterative compositions of boolean functions. Stability is quantified via noise sensitivity. By leveraging the principle that "noise stability $\implies$ Fourier spectrum concentration in low frequencies," stability is upgraded to "spectral simplicity," connecting SNNs to the simplicity bias narrative in deep networks.

## Method

### Overall Architecture
The paper presents a theoretical chain: "Modeling $\to$ Single-neuron bounds $\to$ Multi-layer bounds $\to$ Spectral simplicity $\to$ Experiments." As there is no trainable pipeline, a flowchart is omitted.

The input is a binary sequence $(x_t)_{t\in[T]}\in(\{-1,1\}^n)^T$ over $T$ time steps, and the output is a classification label. The authors define a single **sign-LIF (sLIF) neuron** as a computational unit evolving recursively: membrane potential $u_t=\beta u_{t-1}+w^\top x_t-\frac{\theta}{2}(s_{t-1}+1)$, and spike $s_t=\mathrm{sign}(u_t-\theta)$, where $\beta\in[0,1]$ is the leak coefficient, $\theta>0$ is the threshold, and weights are initialized as $w\sim N(0,I_n/n)$. Multiple sLIF neurons are stacked into an $L$-layer fully connected network. The classifier predicts the class based on the neuron in the output layer with the maximum spike count: $f^{L,T}=\arg\max_{i\in[n_L]}\sum_{t=1}^T s^{(L)}_{t,i}$.

The key observation is that with fixed weights, each spike $s_t:\{-1,1\}^n\to\{-1,1\}$ is a boolean function, and the collective classifier is a composition of these functions. Stability is then translated into the **noise sensitivity** of boolean functions, $\mathrm{NS}_\nu(f)=\Pr_{x,\xi}[f(x)\neq f(x\odot\xi)]$. By taking the expectation over the random weight family, the ENS (Expected Noise Sensitivity) is obtained. Finally, using the link between low noise sensitivity and low-frequency Fourier concentration, stability conclusions are extended to spectral simplicity.

### Key Designs

**1. Translating SNNs into Compositions of Boolean Functions: sign-LIF Neurons + reset-by-subtraction**

To apply boolean analysis, the continuous LIF dynamics must be discretized into a boolean mapping from binary input sequences to binary output spikes. The authors use sign activation ($s_t=\mathrm{sign}(u_t-\theta) \in \{-1,1\}$) instead of the classic Heaviside function to simplify the subsequent Fourier analysis, as $\{-1,1\}$ is the natural domain for Fourier-Walsh expansions. Weight initialization $w\sim N(0,I_n/n)$ ensures $w^\top x=O(1)$, avoiding degenerate regions of "never firing" or "over-firing."

The primary technical difficulty lies in **reset-by-subtraction**: after each firing, the membrane potential is reduced by $\theta$ (the $-\frac{\theta}{2}(s_{t-1}+1)$ term in the recurrence), rather than being cleared to zero. This makes the threshold adaptively dynamic and creates non-trivial probabilistic dependencies between time steps. Dealing with this mechanism is the core distinction from the simplified models in Ding et al. (2024). To maintain analytical tractability, the main theory focuses on $\beta=1$ (leakless IF) and static inputs (the same sample presented repeatedly over $T$ steps, a common encoding for datasets like MNIST), while generalizations for $\beta\neq1$ are deferred to the appendix.

**2. Single-Neuron Stability Bound: Gaussian Decomposition of Random Linear Threshold Functions (Theorem 1)**

The authors provide the first quantitative bound for whether a single neuron's output flips under input perturbation. Given two input sequences with a relative Hamming distance $\nu_t=d_H(x_t,y_t)/n$ and mean $\bar\nu_t$, when $\max_t\nu_t=O(1/\sqrt n)$, then for all $t$:

$$\Pr_w\!\big[s_t(x_1,\dots,x_t)\neq s_t(y_1,\dots,y_t)\big]\le C(1+\theta)\,t^2\sqrt{\bar\nu_t}\,\log n,$$

The $\log n$ factor can be removed under static inputs. The proof begins at $t=1$, where the problem reduces to whether the signs of two Gaussian variables $X=w^\top x_1$ and $Y=w^\top y_1$ match. Using classical Gaussian decomposition $Y=\rho X+\sqrt{1-\rho^2}\,Z$ (where $\rho=1-\nu_1$), the flip event is expressed via the bivariate Gaussian CDF $\Phi_2$ with a correlation coefficient of $2\nu_1-1$, yielding $\Pr[\cdot]\le C_\theta\sqrt{\nu_1}$. For $t\ge2$, recursion and union bounds are used to handle temporal dependence, introducing the $t^2$ factor. The dynamic thresholding from the reset mechanism is the main obstacle to tightening this temporal dependence. The intuition is that wide neurons are stable on average, with outputs only likely to change when $O(\sqrt n)$ coordinates are perturbed.

**3. Multi-layer Classifier Stability Bound: Absorbing Markov Chains + Chernoff (Theorem 2)**

Extending the single-neuron bound to an $L$-layer classifier provides the primary result. Following the logic of Jonasson et al. (2023), the authors model the "spike difference between two inputs at layer $l$" as a Markov chain $D^{(l)}_1(x,y)=\frac14\|s^{(l)}_1(x)-s^{(l)}_1(y)\|^2$, where $0$ is an absorbing state among $n+1$ states. Conditioned on $D^{(l-1)}_1=\lfloor\nu_1 n\rfloor$, the next layer's difference $D^{(l)}_1\sim\mathrm{Bin}(n,p_{\nu_1})$ where $p_{\nu_1}\le C_\theta\sqrt{\nu_1}$ (from Theorem 1). By applying Chernoff bounds layer by layer, it is shown that for $\nu=O(1/\sqrt n)$ and sufficiently large $n$:

$$\Pr_W\!\big[f^{L,T}((x_t))\neq f^{L,T}((y_t))\big]\le n_L T^4 C(1+\theta)\,\nu^{\frac{1}{2^{2L+1}}}\log^{3/2}n+(L-1)e^{-c\,\nu^{\frac{1}{2^{2L-1}}}n}.$$

The bound loosens as depth $L$, delay $T$, and threshold $\theta$ increase, consistent with the principle that deeper compositions of boolean functions are more sensitive. The authors suggest the $\theta$ dependence and $\log^{3/2}n$ are likely proof artifacts, while the intrinsic nature of the $L,T$ dependence remains an open question explored in experiments.

**4. Spectral Simplicity: From Noise Stability to Fourier Concentration (Corollary 1)**

This step upgrades "stability" to "simplicity." Any $f:\{-1,1\}^n\to\mathbb R$ has a unique Fourier-Walsh expansion $f(x)=\sum_{S\subseteq[n]}\hat f(S)\chi_S(x)$, where low-order terms (small $|S|$) correspond to low frequencies. The authors define **expected spectral concentration**: a function family is "spectrally $\epsilon$-concentrated to degree $k$" in expectation if $\mathbb E_{w\sim\mu}\big[\sum_{|S|>k}\hat f_w^2(S)\big]\le\epsilon$. Using the classical proposition that sets $\epsilon=3\,\mathrm{NS}_\nu(f)$ to imply spectral $\epsilon$-concentration to degree $1/\nu$, the stability bound from Theorem 2 is translated directly. Corollary 1 states that binary sLIF-SNNs are spectrally $\epsilon$-concentrated to degree $1/\nu'$, with $\epsilon=C_{T,\theta}\,\nu'^{\,1/2^{2L+1}}\log^{3/2}n$. By setting $\nu'=\frac{1}{\sqrt n\log n}$, the network is $O(n^{1/2^{2(L+1)}})$-concentrated to degree $O(\sqrt n\log n)$. Only a vanishing fraction of high-frequency terms contribute to the spectrum, signifying "spectral simplicity." Notably, the **maximum degree** of concentration is independent of architectural parameters, while the **degree of concentration** degrades with $L, T, \theta$.

### Loss & Training
The theoretical part focuses on randomly initialized networks without training. In experiments, sLIF/IF SNNs are trained using ADAM + surrogate gradients (e.g., a 3-layer network on MNIST achieving 98% accuracy) to compare noise sensitivity before and after training.

## Key Experimental Results

### Main Results (Verification of ENS)
The authors estimate $\mathrm{ENS}_{1/\sqrt n}$ using Monte Carlo methods to verify Theorems 1 and 2 and to observe the impact of training.

| Setting | Network | Key Findings |
|------|------|----------|
| Single Neuron | sIF / IF, $n=100/1000/10000$, $\theta=0.5, T=10$ | Sensitivity is very low across all $t$; Theorem 1 holds for both sIF and IF. |
| 5-layer Network | sIF / IF, width = input dimension | Depth affects sensitivity more strongly than delay, but Theorem 2 **overestimates** this impact. |
| Before/After Training (MNIST, 3-layer) | sLIF / IF, $n=784$ | Sensitivity drops significantly after training (when final accuracy is high). |
| Before/After Training (CIFAR-10) | $n=3072$ | Training also reduces sensitivity, but less so than MNIST (CIFAR-10 accuracy was only 84.38%). |
| Neuromorphic Data (NMNIST) | Conv SNN, $n=2312$ | ENS is very small both before and after training; training's impact is less pronounced than on static data. |

### Ablation Study / Perturbation Comparison (5-layer, $n=1000, \beta=0.5, \theta=1$)

| Model / Perturbation | Random Flip | Dropout (5%) |
|------------|---------|------------------------|
| sLIF | 0.19 | **0.16** |
| LIF | 0.28 | **0.16** |

### Key Findings
- **Stability is intrinsic and scales with width**: Randomly initialized wide SNNs already exhibit low sensitivity, confirming the "wide networks are spectrally simple" theory.
- **Training enhances stability**: Training on static data (MNIST/CIFAR-10) significantly reduces noise sensitivity, with higher accuracy correlating to lower sensitivity. This effect is weaker on event-based data like NMNIST.
- **Theoretical bounds are loose, especially regarding depth**: 5-layer experiments show Theorem 2 overestimates the negative impact of depth. The authors leave the intrinsic nature of $L, T, \theta$ dependence as an open problem.
- **Dropout is more robust than random flips**: Since dropout does not change zero components in $\{0,1\}^n$ inputs, the perturbation is milder.

## Highlights & Insights
- **Cross-disciplinary Tool Transfer**: This is the first work to bring mature boolean function analysis (noise sensitivity + Fourier-Walsh spectra) to SNNs, connecting a theoretically starved field to the "simplicity bias" narrative of ANNs.
- **Tackling reset-by-subtraction**: Unlike previous works that simplify to reset-to-zero, this paper directly addresses the temporal dependence caused by subtraction-based resets, using "conditional binomials + absorbing Markov chains + Chernoff" to propagate errors.
- **"Stability $\implies$ Spectral Concentration $\implies$ Simplicity"**: This logic chain transforms a robustness conclusion into a representation theory conclusion, which can be reused for analyzing other binary or threshold-based networks.
- **Random Networks as PAC-Bayes Priors**: By focusing analysis on the initialized network, the authors isolate intrinsic model properties while providing an interface for future PAC-Bayes generalization bounds.

## Limitations & Future Work
- **Strong Theoretical Assumptions**: The main results are limited to $\beta=1$ (leakless IF), static inputs, sign activation, and large width. $\beta\neq1$ and dynamic inputs are only addressed in the appendix/experiments; dynamic inputs pose technical hurdles as sums may fall outside the hypercube.
- **Loose Bounds**: The $t^2$, $T^4$, $\log^{3/2}n$, and $\theta$ dependencies are admitted to be proof artifacts. Whether $L, T$ dependencies are intrinsic remains unresolved.
- **Initialization Only**: The observation that training increases stability is purely experimental and lacks theoretical characterization—a gap the authors explicitly leave for future work.
- **Weak Concept of Simplicity**: "Spectral simplicity" is a weaker definition than others (e.g., De Palma et al.'s average Hamming distance to the nearest class boundary); its ability to yield stronger generalization bounds is unclear.

## Related Work & Insights
- **vs. Ding et al. (2024) (Dynamical Systems Perspective)**: Ding et al. analyze single neurons with reset-to-zero based on spike train differences. This paper analyzes multi-layer classifiers (where predictions can remain stable even if spikes change) using boolean analysis rather than Lyapunov methods.
- **vs. Jonasson et al. (2023) (Boolean Threshold Networks)**: This paper uses a similar Markov chain + Chernoff framework but introduces SNN reset dynamics and temporal evolution, which create probabilistic dependencies not present in the earlier work.
- **vs. De Palma et al. (2019) (Simplicity Bias in Random ANNs)**: While both study the bias toward simple functions, De Palma et al. use Gaussian processes (difficult to adapt to SNNs). This paper uses boolean analysis which naturally fits the spiking setting, though "spectral simplicity" is a weaker metric than "average Hamming distance."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First application of boolean analysis to SNN stability and the proposal of spectral simplicity.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers single/deep layers, static/event data, and pre/post-training, though experiments are mainly small-scale ENS verifications.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical chain with honest notation regarding bound artifacts.
- Value: ⭐⭐⭐⭐ Provides a rigorous characterization for SNN stability/simplicity, linking them to simplicity bias and PAC-Bayes.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Overparametrization bends the landscape: BBP transitions at initialization in simple Neural Networks](overparametrization_bends_the_landscape_bbp_transitions_at_initialization_in_sim.md)
- [\[ICLR 2026\] Random Label Prediction Heads for Studying Memorization in Deep Neural Networks](random_label_prediction_heads_for_studying_memorization_in_deep_neural_networks.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)
- [\[ICLR 2026\] Reducing Symmetry Increase in Equivariant Neural Networks](reducing_symmetry_increase_in_equivariant_neural_networks.md)
- [\[ICLR 2026\] Proper Velocity Neural Networks](proper_velocity_neural_networks.md)

</div>

<!-- RELATED:END -->
