---
title: >-
  [Paper Note] Private Frequency Estimation via Residue Number Systems
description: >-
  [AAAI 2026][Local Differential Privacy] This paper proposes ModularSubsetSelection (MSS), a local differential privacy frequency estimation protocol based on the Residue Number System (RNS). MSS achieves estimation accur…
tags:
  - "AAAI 2026"
  - "Local Differential Privacy"
  - "Frequency Estimation"
  - "Residue Number Systems"
  - "Communication Efficiency"
  - "Data Reconstruction Attack"
date: 2026-05-08
content_hash: 5d5b086217bf78cc
---

# Private Frequency Estimation via Residue Number Systems

**Conference**: AAAI 2026
**arXiv**: [2511.11569](https://arxiv.org/abs/2511.11569)  
**Code**: [GitHub](https://github.com/hharcolezi/private-frequency-oracle-rns)  
**Area**: Other
**Keywords**: Local Differential Privacy, Frequency Estimation, Residue Number Systems, Communication Efficiency, Data Reconstruction Attack

## TL;DR

This paper proposes ModularSubsetSelection (MSS), a local differential privacy frequency estimation protocol based on the Residue Number System (RNS). MSS achieves estimation accuracy comparable to SubsetSelection and PGR while significantly reducing communication overhead (up to 50% less than SS), substantially accelerating server-side decoding (11–448× faster than PGR), and attaining the lowest data reconstruction attack success rate.

## Background & Motivation

Local Differential Privacy (LDP) is a core paradigm for user privacy protection in federated analytics, widely deployed in billion-device scenarios such as Apple keyboard prediction, Google Chrome RAPPOR, and Microsoft telemetry systems. The central task of LDP frequency estimation is: each user holds a private input $x_i \in [k]$, sends a perturbed message to an untrusted server via a local randomization mechanism $\mathcal{M}$, and the server recovers the population histogram $\hat{\mathbf{f}}$.

Evaluating LDP protocols involves four key dimensions that constitute a **multi-constraint regime**:

**Utility**: Estimation accuracy measured by MSE

**Communication**: Number of bits transmitted per user

**Server Runtime**: Decoding complexity

**Attackability**: Success rate of an adversary reconstructing the original input from a single message

Existing protocols make trade-offs across these four dimensions:
- **RandomizedResponse (GRR)**: Low communication ($\lceil\log_2 k\rceil$ bits), but MSE gap of $\Theta(k/e^\varepsilon)$ and highest attackability
- **SubsetSelection (SS)**: Achieves optimal MSE, but higher communication overhead $\Theta(\omega \log_2(k/\omega))$
- **RAPPOR/OUE**: Optimal MSE, but message length $O(k)$ and server cost $O(nk)$
- **ProjectiveGeometryResponse (PGR)**: Optimal MSE and communication $\lceil\log_2 k\rceil$, but requires domain sizes matching projective geometry constraints, relies on finite field arithmetic, and incurs dynamic programming decoding cost $O(n + ke^\varepsilon \log k)$

Core Problem: **Can a protocol be designed that simultaneously achieves a favorable balance across accuracy, bandwidth, computation, and attack resistance?**

## Method

### Overall Architecture

MSS adopts a **divide-and-conquer** design. On the user side, the input is decomposed into residues over multiple small domains via RNS, one of which is randomly transmitted. On the server side, a weighted least-squares solver reconstructs the full histogram.

### Key Designs

#### 1. User-Side "Divide" Perturbation (Algorithm 1)

**Residue Number System Encoding**: Given $\ell$ pairwise coprime moduli $m_0, \ldots, m_{\ell-1}$, the input $x$ is mapped to a residue vector:

$$\mathbf{r}(x) = (x \bmod m_0, \ldots, x \bmod m_{\ell-1})$$

By the **Chinese Remainder Theorem (CRT)**, this mapping is injective when $\prod_j m_j \geq k$.

**Modular Sampling**: Rather than transmitting all residues (which would split the privacy budget), each user:
1. Uniformly samples a modulus index $J \sim \text{Uniform}([\ell])$
2. Computes the residue $r = x \bmod m_J$
3. Applies SubsetSelection with the full privacy budget $\varepsilon$ to $r$ over domain $[m_J]$
4. Sends $(J, Z)$, where $Z \subseteq [m_J]$ is a perturbed subset of size $\omega_J$

Communication cost is only $\lceil\log_2 \ell\rceil + \lceil\log_2 \binom{m_J}{\omega_J}\rceil$ bits, far less than standard SS.

**Privacy Guarantee**: Since $J$ is independent of $x$ and uniformly distributed, the privacy loss is entirely determined by the $\varepsilon$-LDP of the single SS mechanism, so MSS satisfies $\varepsilon$-LDP (Theorem 1).

#### 2. Server-Side "Conquer" Estimation

**Design Matrix Construction**: For each modulus $m_j$, a mapping matrix $A_j \in \{0,1\}^{m_j \times k}$ is constructed where $A_j[r,x] = \mathbf{1}\{x \bmod m_j = r\}$. Stacking all $A_j$ yields a sparse design matrix $A \in \{0,1\}^{T \times k}$ with $T = \sum_j m_j$.

**Variance-Optimal Row Weights**: Based on the true and false inclusion probabilities $p_j, q_j$ of SS, the optimal GLS weight matrix $W^{1/2}$ is computed, yielding the weighted design matrix $A_w = W^{1/2}A$.

**Debiasing and Least-Squares Solving**: After debiasing the residual counts of each block, the following regularized least-squares problem is solved:

$$\hat{\mathbf{f}} = \arg\min_{\mathbf{z}} \|A_w \mathbf{z} - \tilde{\mathbf{s}}\|_2^2 + \lambda\|\mathbf{z}\|_2^2$$

This is solved efficiently using the LSMR iterative algorithm, exploiting the structured sparsity of $A_w$, with each iteration requiring only $O(k\ell)$.

#### 3. Modulus Selection Optimization

A combination of theoretical $\kappa$ bounds (Theorem 4) and random sampling is used:
1. Derive a condition number upper bound $\kappa \leq \alpha \cdot \frac{\ell + T^\star}{\ell - T^\star}$ via the Gershgorin circle theorem
2. Randomly sample $\ell$ coprime primes from the prime band $[k/(\beta\ell), \beta k/\ell]$
3. Verify CRT coverage, full rank, and $\kappa \leq \kappa_{\max}$ conditions
4. Select the combination with the minimum exact MSE among all qualified candidates

### Loss & Training

This paper proposes a privacy mechanism rather than a learning algorithm; the core optimization objective is MSE minimization. Key theoretical results include:

- **Unbiasedness (Theorem 2)**: When $\lambda=0$, $\mathbb{E}[\hat{\mathbf{f}}] = \mathbf{f}$
- **Asymptotic Unbiasedness (Corollary 1)**: When $\lambda > 0$, the bias is $O(\lambda)$
- **MSE Upper Bound (Theorem 3)**: $\text{MSE}_{\text{MSS}} \leq \frac{4\kappa e^\varepsilon}{n(e^\varepsilon - 1)^2}$, i.e., $\kappa$ times the optimal MSE of standard SS
- **DRA Upper Bound (Eq. 7)**: $\widehat{\mathbb{E}}[\text{DRA}]_{\text{MSS}} = \frac{1}{\ell}\sum_{j=0}^{\ell-1} \frac{p_j}{\omega_j \lceil k/m_j \rceil}$

## Key Experimental Results

### Main Results

Experimental setup: $n=10000$ users, $k \in \{1024, 22000\}$, $\varepsilon \in \{0.5, \ldots, 5.0\}$, 300 independent trials.

| Protocol | MSE (vs. SS) | Communication (bits) | Server Time | DRA |
|------|-------------|-------------|-----------|-----|
| GRR | Highest (gap $\Theta(k/e^\varepsilon)$) | $\lceil\log_2 k\rceil$ | $O(n+k)$ | Highest |
| SS | Optimal | High | Moderate | Moderate |
| OUE/RAPPOR | Optimal | $O(k)$ | $O(nk)$ | Moderate |
| PGR | Optimal | $\lceil\log_2 k\rceil$ | High (see below) | Moderate-high |
| **MSS** | **≤1.3× SS** | **Up to 50% less than SS** | **Fastest** | **Lowest** |

### Server Runtime Comparison ($k=22000, n=10000$)

| $\varepsilon$ | MSS (sec) | PGR (sec) | MSS Speedup |
|---|---|---|---|
| 2.0 | 0.160±0.027 | 2.897±0.220 | 18.1× |
| 3.0 | 0.272±0.086 | 9.618±0.679 | 35.4× |
| 4.0 | 0.168±0.056 | 11.461±0.702 | 68.3× |
| 4.5 | 0.127±0.047 | 56.906±3.570 | **447.8×** |
| 5.0 | 0.152±0.054 | 3.208±0.198 | 21.1× |

MSS substantially outperforms PGR in runtime across all privacy budgets, with speedups ranging from 11× to 448×.

### Key Findings

1. **Near-Optimal Utility**: The empirical MSE ratio of MSS to SS/PGR consistently remains $\leq 1.3$, a finding that holds under both Zipf and Spike distributions
2. **Substantially Reduced Communication**: Particularly in the high-privacy (small $\varepsilon$) regime, MSS communication overhead can be half that of SS
3. **Strongest Attack Resistance**: MSS achieves the lowest DRA across all $\varepsilon$ values, as modular randomization disperses probability mass across multiple residue classes
4. **Anomalous DRA in PGR**: When the domain size $k$ is smaller than the natural size $K$ required by projective geometry, PGR's truncation mismatch breaks combinatorial symmetry, causing a sharp increase in DRA
5. **Controllable Condition Number $\kappa$**: In practice, $\kappa$ never exceeds approximately 1.3, far below the theoretical bound $\kappa_{\max}=10$

## Highlights & Insights

1. **Elegant Integration of Number Theory and Privacy**: Leveraging the Chinese Remainder Theorem to decompose a large-domain problem into multiple small-domain problems is a natural and elegant divide-and-conquer strategy
2. **Four-Dimensional Joint Optimization**: Unlike prior work focused on binary utility–communication or utility–privacy trade-offs, MSS simultaneously optimizes accuracy, bandwidth, computation, and attack resistance
3. **No Algebraic Prerequisites**: In contrast to PGR, which requires finite field arithmetic and projective geometry constraints, MSS uses only integer arithmetic, accepts arbitrary $k$ and $\varepsilon$, and has an extremely low barrier to practical deployment
4. **Tunable Parameter $\ell$**: Practitioners can flexibly navigate the communication–accuracy spectrum; PGR does not offer this flexibility
5. **Complete Theoretical Analysis**: A comprehensive theoretical framework is provided, including closed-form MSE expressions, condition number bounds, and DRA upper bounds

## Limitations & Future Work

1. The $\kappa$-factor constant in MSE, while small in practice ($\leq 1.3$), is theoretically not information-theoretically optimal
2. The modulus selection procedure (Algorithms 2–3) may require substantial sampling trials for very large domains, though it can be precomputed and cached offline
3. Only the basic task of frequency estimation is addressed; the authors note future extensions to heavy hitters and multi-dimensional estimation
4. Experiments rely solely on synthetic distributions (Zipf and Spike), lacking validation in real-world federated analytics deployment scenarios
5. The choice of regularization parameter $\lambda$ (e.g., $1/\varepsilon^2$) may introduce non-negligible bias in the strong-privacy (small $\varepsilon$) regime

## Related Work & Insights

- **SubsetSelection (SS)**: A classical statistically optimal method with high communication overhead; MSS's "block SS" can be viewed as its modular extension
- **PGR**: An algebraic coding approach with optimal MSE but high deployment complexity; MSS trades a slight theoretical precision loss for substantial practical gains
- **RAPPOR**: Google's industrial-grade LDP system, where the $O(k)$ message length limits large-domain applicability
- Insight: Number-theoretic tools (CRT, coprime moduli) hold significant potential in privacy-preserving computation, and the divide-and-conquer paradigm is broadly applicable to large-domain statistical estimation problems

## Rating

| Dimension | Score (1–5) | Notes |
|------|-----------|------|
| Novelty | 5 | The combination of RNS and LDP is highly novel and opens a new research direction |
| Technical Depth | 5 | Complete theoretical framework: unbiasedness, variance bounds, condition number bounds, DRA bounds |
| Experimental Thoroughness | 4 | Comprehensive comparison across four dimensions, but limited to synthetic data |
| Value | 4 | Simple deployment, no algebraic constraints, suitable for industrial-grade federated systems |
| Writing Quality | 5 | Clear structure; the four-dimensional comparison in Table 1 is immediately interpretable |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OmniFood8K: Single-Image Nutrition Estimation via Hierarchical Frequency-Aligned Fusion](../../CVPR2026/others/omnifood8k_nutrition_estimation.md)
- [\[AAAI 2026\] Improved Differentially Private Algorithms for Rank Aggregation](improved_differentially_private_algorithms_for_rank_aggregation.md)
- [\[AAAI 2026\] Designing Incident Reporting Systems for Harms from General-Purpose AI](designing_incident_reporting_systems_for_harms_from_general-purpose_ai.md)
- [\[AAAI 2026\] Scalable Vision-Guided Crop Yield Estimation](scalable_vision-guided_crop_yield_estimation.md)
- [\[AAAI 2026\] Reward Redistribution via Gaussian Process Likelihood Estimation](reward_redistribution_via_gaussian_process_likelihood_estimation.md)

</div>

<!-- RELATED:END -->
