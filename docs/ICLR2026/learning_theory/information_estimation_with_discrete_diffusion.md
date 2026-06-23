---
title: >-
  [Paper Note] Information Estimation with Discrete Diffusion
description: >-
  [ICLR 2026][learning_theory][Paper Note] The paper proposes INFO-SEDD, which connects the score function of discrete diffusion (Continuous-Time Markov Chains) to the Dynkin's formula. This allows for the direct estimation of KL divergence, mutual information, and entropy on discrete data, bypassing the conventional "embedding into continuous space" approach.
tags:
  - ICLR 2026
  - learning_theory
date: 2026-05-08
content_hash: 584f6f208afa56da
---
# Information Estimation with Discrete Diffusion

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=m18MXVdrV9](https://openreview.net/forum?id=m18MXVdrV9)  
**Code**: [https://github.com/AlbertoForesti/mutinfo-diffusion](https://github.com/AlbertoForesti/mutinfo-diffusion)  
**Area**: Learning Theory / Information Theoretic Estimation / Discrete Diffusion  
**Keywords**: Mutual Information Estimation, Entropy Estimation, Discrete Diffusion, Continuous-Time Markov Chains, KL Divergence  

## TL;DR
The paper proposes INFO-SEDD, which connects the score function of discrete diffusion (Continuous-Time Markov Chains) to the Dynkin's formula. This allows for the direct estimation of KL divergence, mutual information, and entropy on discrete data, bypassing the conventional "embedding into continuous space" approach. It is significantly more accurate and stable in high-dimensional and high-mutual information scenarios.

## Background & Motivation
**Background**: Information theoretic measures such as mutual information (MI) and entropy are core tools for characterizing non-linear relationships between variables. They are widely used in machine learning training objectives, model selection, neuroscience, and genomics. Neural estimators (e.g., MINE, NWJ, SMILE, F-DIME) have recently replaced classical parametric/non-parametric methods, but most are designed for **continuous distributions**.

**Limitations of Prior Work**: A large amount of real-world data is **discrete and high-dimensional** (DNA sequences, text tokens, Ising spins). The common practice for handling such data is the "embedding trick"—projecting discrete data into a continuous space and then applying continuous estimators. This path has three major flaws: (1) It requires significant engineering for the embedding model and estimator architecture; (2) Embeddings may lose the inherent discrete structure of the data; (3) Estimators based on variational lower bounds fail in **high-MI regions**, where the required sample size grows exponentially with the true MI (the famous negative result by McAllester & Stratos). In practice, MI estimation is capped by the logarithm of the batch size $\log(\text{batch size})$.

**Key Challenge**: There is a gap between the application value of information theoretic measures (genome sequencing, text summarization, neuroscience) and the lack of scalable high-dimensional estimators capable of directly processing discrete data. Strong estimators from the continuous domain (such as MINDE) often fail when applied to discrete data.

**Goal**: To create a MI/entropy estimator that **works directly in discrete space**, is scalable, statistically consistent, and can reuse pre-trained models.

**Core Idea**: **Information estimation is a byproduct of generative modeling**. The score function trained by discrete diffusion models (SEDD/masked diffusion) already encodes information about the probability distribution at different time steps. By substituting this into the KL integral formula derived from the Dynkin's formula, MI and entropy can be calculated. By **training a generative model, one inherently obtains an estimator**.

## Method

### Overall Architecture
INFO-SEDD bases information estimation entirely on the time-reversal framework of Continuous-Time Markov Chains (CTMC). First, two CTMCs are constructed that differ only in their initial distributions. Using the Dynkin's formula, $\mathrm{KL}[\vec{p}_0\|\vec{q}_0]$ is expressed as an expectation of an integral over time. The probability ratio appearing in the integral kernel can be approximated by the score model $s_\theta$ of discrete diffusion. Thus, KL, MI, and entropy are all reduced to Monte Carlo estimation involving "sampling time + simulating the forward process + querying the score." For implementation, the **absorbing state transition matrix** is used to compress the requirement of training two score models into a single joint model, supporting the direct use of pre-trained masked diffusion models.

```mermaid
flowchart TD
    A[Discrete Data X, Y] --> B[Construct two CTMCs<br/>Diff only in initial distributions p0 vs q0]
    B --> C[Dynkin's formula + Backward operator<br/>Write KL as time integral expectation]
    C --> D[Score model s_theta approximation<br/>Prob ratio p_t·x/p_t·Xt]
    D --> E{Absorbing state transition matrix}
    E -->|Single model for marginal score| F[INFO-SEDD-J (Joint)<br/>KL pXY‖pX⊗pY]
    E -->|Conditional distribution modeling| G[INFO-SEDD-C (Conditional)<br/>E·KL pY|X‖pY]
    F --> H[Monte Carlo Integral → MI / Entropy Estimation]
    G --> H
```

### Key Designs

**1. Transforming KL divergence into an estimable time integral via Dynkin's formula**: This is the mathematical foundation of the method. Given two distributions $\vec{p}_0,\vec{q}_0$ on the same support, two CTMCs are constructed sharing the same generator but with different initial values. Utilizing the property that the time-reversal processes both converge to the same reference distribution $\pi$ at the endpoint, the KL can be written as the expectation over reversal trajectories. Applying the Dynkin's formula (for a function $f$, $\mathbb{E}[f(\overleftarrow{X}_T,T)|\overleftarrow{X}_0]-f(\overleftarrow{X}_0,0)=\mathbb{E}[\int_0^T \partial_t f + \mathcal{B}[f]\,dt]$, where the backward operator $\mathcal{B}[f](a,t)=\sum_{b\neq a}\overleftarrow{Q}_t(b,a)(f(b)-f(a))$), the KL is finally formulated as an integral that depends only on the **probability ratio** $\frac{\vec{p}_t(x)}{\vec{p}_t(\vec{X}_t)}$, with the kernel function $K(a)=a(\log a-1)$. The elegance of this step is that KL no longer requires explicit knowledge of the distributions, only the probability ratios of adjacent states—which is precisely the definition of the discrete diffusion score function.

**2. Replacing unknown probability ratios with discrete diffusion scores**: Since the true ratio $\frac{\vec{p}_t(x)}{\vec{p}_t(\vec{X}_t)}$ in the integral is unknown, it is replaced by the parameterized score $s_\theta^p(\vec{X}_t)_x$ of SEDD, yielding the computable estimator:
$$\mathrm{KL}[\vec{p}_0\|\vec{q}_0]\approx\mathbb{E}\Big[\int_0^T\!\!\sum_{x\neq\vec{X}_t}\!\vec{Q}_t(\vec{X}_t,x)\big(K(s_\theta^p)+s_\varphi^q - s_\theta^p\log s_\varphi^q\big)dt\Big].$$
The scores are trained via the native DWDSE (Diffusion Weighted Denoising Score Entropy) loss of SEDD. The Monte Carlo implementation is straightforward: sample time $t$ uniformly from $[0,T]$, simulate the forward process $\vec{X}_t$, and query the score. MI has two variants: the **Joint method (INFO-SEDD-J)** ($I(X,Y)=\mathrm{KL}[p_{XY}\|p_X\otimes p_Y]$) and the **Conditional method (INFO-SEDD-C)** ($I(X,Y)=\mathbb{E}[\mathrm{KL}[p_{Y|X}\|p_Y]]$); the latter is easier to optimize when the label dimension is much lower than the sequence dimension (e.g., DNA to label).

**3. Absorbing state transition matrix: One model for all marginal scores**: In a naive implementation, the scale of the transition matrix $\vec{Q}_t$ explodes with the state space $|\chi|^2$. Leveraging the structure where a sequence can be decomposed into $D$ sub-components, the CTMC is constrained to change only one component per step (unit Hamming distance), with transitions determined by a shared local matrix $\vec{Q}^{tok}$, greatly reducing complexity. Crucially, by choosing an **absorbing state** matrix $\vec{Q}^{tok}_t=\sigma(t)\vec{Q}^{tok}_{absorb}$, sub-components can only transition into the absorbing state $\varnothing$. This choice allows a single score model trained on the joint distribution to directly yield marginal scores (when $Y$ is fully absorbed into $\varnothing$, the joint score ratio automatically reduces to the marginal score ratio of $X$), thereby compressing MI estimation into a single model and ensuring compatibility with pre-trained masked diffusion models (MDLM, Caduceus, MD4, LLaDA).

**4. Theoretical guarantees of consistency and error decomposition**: Under the mild assumptions that the scores are bounded (constants $C_1,C_2$) and network approximation errors are $\epsilon_p,\epsilon_q$, the estimation bias is decomposed into two terms:
$$\big|\mathbb{E}\,\mathcal{E}(s_\theta^p,s_\varphi^q)-\mathrm{KL}[p\|q]\big|\le\underbrace{\bar\sigma(T)D|\chi|(1+\tfrac{C_2}{C_1})(\epsilon_p+\epsilon_q)}_{\text{估计误差}}+\underbrace{(1-\vec{p}_T(\varnothing^D))DC_2\log|\chi|}_{\text{截断偏差}}.$$
The estimation error grows linearly with the score error; the truncation bias stems from the finite time $T$ and **decays exponentially** as the absorbing state probability $\vec{p}_T(\varnothing^D)\to1$. Thus, INFO-SEDD is a consistent estimator (up to an exponentially small truncation term). Critically, it **lacks the variance that explodes exponentially with MI**, which is found in importance sampling estimators—this is the fundamental reason it outperforms variational methods in high-MI regions. Entropy estimation is integrated into the same framework: $H(\vec{p}_0)=\log N-\mathrm{KL}[\vec{p}_0\|\vec{u}_0]$ by calculating the KL against a uniform distribution.

## Key Experimental Results

### Main Results: High-dimensional synthetic benchmark (Known Ground Truth MI)
All methods use the same backbone network, $10^5$ samples, batch size 1024, trained for $10^5$ steps, averaged over 10 seeds (mean ± std).

| Estimator | MI=10, D=10 | MI=20, D=20 | MI=30, D=30 | MI=40, D=40 | MI=50, D=50 |
|-----------|-----------|-----------|-----------|-----------|-----------|
| **INFO-SEDD** | **9.92±0.12** | **20.02±0.21** | **29.83±0.54** | **39.11±0.65** | **47.77±1.18** |
| GAN-DIME | 12.15±0.89 | 22.09±1.75 | 20.74±1.75 | 19.64±1.33 | 17.27±1.46 |
| MINDE | 14.01±2.91 | 26.98±3.16 | 31.08±4.33 | 33.97±3.32 | 32.60±3.93 |
| SMILE | 12.83±0.95 | 23.11±1.41 | 21.79±1.08 | 20.13±1.27 | 18.97±1.05 |
| MINE | 10.21±6.33 | 8.82±0.80 | 7.41±1.23 | 6.91±0.66 | 7.21±1.14 |
| KL-DIME | 8.38±0.90 | 7.51±0.56 | 7.02±0.43 | 6.52±0.32 | 6.41±0.62 |

As MI and dimensionality increase simultaneously, all competitors either severely underestimate (variational methods capped by $\log(\text{batch})$) or show drastically high variance (MINDE/MINE). INFO-SEDD remains close to the ground truth with minimal standard deviation.

### Downstream Application: Alignment with human indicators for text summarization (Pearson correlation)
Estimating the MI between model summaries and original texts on SUMMEVAL and its correlation with human scores:

| Method | Coherence (COH) | Consistency (CON) | Fluency (FLU) | Relevance (REL) | Overall (OVR) |
|------|----------|----------|----------|----------|---------|
| **INFO-SEDD-C** | 0.209 | **0.740** | **0.679** | **0.411** | **0.568** |
| INFO-SEDD-J | -0.091 | 0.550 | 0.455 | 0.288 | 0.322 |
| KL-DIME | 0.170 | 0.214 | 0.194 | 0.076 | 0.193 |
| HD-DIME | -0.243 | 0.331 | 0.281 | -0.145 | 0.063 |
| SMILE | -0.367 | -0.074 | -0.162 | -0.149 | -0.221 |

MI has the highest correlation with "Consistency" (0.740), which aligns with intuition as consistency reflects the amount of shared information between the text and the summary.

### Key Findings
- **Consistency Tests (Text/Genome)**: By pairing BART summaries with original texts with probability $\rho$ and random texts with $1-\rho$, MI should theoretically grow linearly with $\rho$; both INFO-SEDD variants follow the empirical derivation (256–303 nats range) most closely. Variational methods severely underestimate due to the $\log(\text{batch})$ limit, while MINDE fails completely due to high-dimensional embeddings.
- **Genomic Motif Discovery**: On *Arabidopsis thaliana* promoter sequences, using a sliding window and masking to estimate MI curves, INFO-SEDD-J precisely locates the TATA-BOX motif (MI significantly rises in the -39 to -26 range). A single training allows estimation for any subset of sequences, showing robustness to related motifs.
- **Sample Efficiency and Convergence**: Ablations on synthetic experiments show INFO-SEDD is accurate with only $10^3$ samples, is robust to the support size $|\chi|$, and converges faster than GAN-DIME/SMILE.

## Highlights & Insights
- **Paradigm Unification**: It fuses "information theoretic estimation" and "generative modeling"—training a discrete diffusion model inherently produces the estimator. The score function serves a dual purpose, eliminating extra architectural engineering.
- **Overcoming the Curse of Dimensionality**: The exponential sample complexity of variational lower bound estimators in high-MI regions is a structural flaw. INFO-SEDD uses the KL integral and consistent estimation, where variance does not explode with MI, providing a cleaner theoretical bound.
- **Absorbing State Selection as Engineering Excellence**: A mathematical trick (making marginals a special case of the joint via the absorbing state) reduces the need for two models to one and allows seamless inheritance of pre-trained masked diffusion models.
- **Practical Wisdom on C vs J Variants**: When the dimension of one side is much lower than the other (e.g., DNA to binary labels), the conditional method only needs to model the marginal/conditional scores of low-dimensional labels, drastically reducing optimization difficulty.

## Limitations & Future Work
- **Truncation Bias Dependence on Absorbing State Convergence**: The exponential decay term in the consistency guarantee requires $\vec{p}_T(\varnothing^D)$ to be sufficiently close to 1, meaning the time horizon $T$ must be long enough for full absorption, otherwise bias remains non-negligible.
- **Score Model Quality as the Ceiling**: Errors depend linearly on the network approximation errors $\epsilon_p,\epsilon_q$. Estimation accuracy is essentially limited by the quality of the discrete diffusion training.
- **Training/Fine-tuning Required**: While pre-trained models can be reused, most scenarios still require training or fine-tuning the score model on target data, incurring higher computational costs than plug-and-play classical estimators.
- **Future Work**: The authors suggest the Generator Matching framework could extend the method to **mixed continuous/discrete data** and integrate stronger masked diffusion backbones like MD4 or LLaDA for scientific discovery (genomics, neuroscience).

## Related Work & Insights
- **Discrete Diffusion / SEDD Lineage**: The method builds directly on Lou et al. (SEDD), Sahoo et al. (MDLM), and the masked diffusion/CTMC work of Campbell/Austin, repurposing generative score functions as estimation tools.
- **Diffusion Estimators**: It extends the logic of continuous diffusion estimators like Franzese et al. (MINDE) and Kong et al. to the discrete domain, directly addressing MINDE's failure on discrete data.
- **Critique of Variational MI Estimation**: The criticisms by McAllester & Stratos and Song & Ermon concerning variational lower bound failures at high MI serve as the theoretical motivation and benchmark for this consistent estimator design.
- **Insight**: This work demonstrates a general strategy of converting generative model training side-products into statistical estimators. For any field with strong existing generative models (language, single-cell RNA, protein sequences), this provides an information-theoretic analysis path with almost zero additional costs.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The first MI/entropy estimator that works directly on high-dimensional discrete data with consistency guarantees and reuses pre-trained masked diffusion models. The combination of Dynkin + absorbing states is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic benchmarks, text summarization, genomic motifs, and Ising entropy. Comparisons against 8 strong baselines are provided, though real tasks mostly use indirect validation (correlation), lacking more downstream evaluations with hard ground truths.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear; motivations and error analyses are thorough. Explanations for the trade-offs between variants are well-reasoned, though the high density of formulas may be steep for readers unfamiliar with CTMC.
- **Value**: ⭐⭐⭐⭐⭐ Solves a long-standing pain point in discrete data information estimation. It has direct value for discrete-heavy fields like genomics and NLP evaluation. Open-source code is a plus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InfoBridge: Mutual Information Estimation via Bridge Matching](infobridge_mutual_information_estimation_via_bridge_matching.md)
- [\[ICLR 2026\] A Unification of Discrete, Gaussian, and Simplicial Diffusion](a_unification_of_discrete_gaussian_and_simplicial_diffusion.md)
- [\[ICLR 2026\] Characterizing the Discrete Geometry of ReLU Networks](characterizing_the_discrete_geometry_of_relu_networks.md)
- [\[ICLR 2026\] Polynomial Convergence of Riemannian Diffusion Models](polynomial_convergence_of_riemannian_diffusion_models.md)
- [\[ICLR 2026\] On the Interpolation Effect of Score Smoothing in Diffusion Models](on_the_interpolation_effect_of_score_smoothing_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
