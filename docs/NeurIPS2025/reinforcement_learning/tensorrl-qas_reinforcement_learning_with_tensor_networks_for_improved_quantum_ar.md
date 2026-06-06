---
title: >-
  [Paper Note] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search
description: >-
  [NeurIPS 2025][Reinforcement Learning][Quantum Architecture Search] This work proposes TensorRL-QAS, a framework that warm-starts reinforcement learning-based quantum architecture search (RL-QAS) using tensor networks (M…
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Quantum Architecture Search"
  - "Tensor Networks"
  - "DMRG"
  - "Variational Quantum Algorithms"
date: 2026-05-08
content_hash: 3ec1578d8000b3ac
---

# TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search

**Conference**: NeurIPS 2025  
**arXiv**: [2505.09371](https://arxiv.org/abs/2505.09371)  
**Code**: [https://github.com/Aqasch/TensorRL-QAS](https://github.com/Aqasch/TensorRL-QAS)  
**Area**: Quantum Computing / Reinforcement Learning  
**Keywords**: Quantum Architecture Search, Reinforcement Learning, Tensor Networks, DMRG, Variational Quantum Algorithms

## TL;DR

This work proposes TensorRL-QAS, a framework that warm-starts reinforcement learning-based quantum architecture search (RL-QAS) using tensor networks (MPS/DMRG), achieving up to 10× reduction in circuit depth and CNOT gate count, and up to 98% acceleration in training time, thereby effectively addressing the scalability bottleneck of RL-QAS on large-scale quantum systems.

## Background & Motivation

Variational quantum algorithms (VQAs) represent the dominant paradigm in the NISQ era, where the central challenge is designing parameterized quantum circuits (PQCs) to minimize the expectation value of a target Hamiltonian. PQC architecture design faces an inherent dilemma: shallow circuits lack expressibility, while deep circuits are severely affected by noise.

Quantum architecture search (QAS) aims to automatically discover optimal PQC structures. RL-based QAS has shown considerable promise, with RL agents incrementally constructing circuits by selecting gates and their placements. However, **RL-QAS suffers from serious scalability issues**: most existing methods have only been validated on up to 8 qubits in noiseless settings and 4 qubits under noise. The root causes are: (1) the action space grows exponentially with the number of qubits; (2) episodes require more steps and increasingly expensive quantum simulation.

**Core Idea**: Tensor network methods (DMRG) are employed to obtain an approximate solution of the target state as a warm-start, constraining the search space to physically meaningful circuit regions and substantially accelerating RL-QAS convergence.

## Method

### Overall Architecture

TensorRL-QAS consists of three stages:

1. **DMRG Solving**: Given a target Hamiltonian $H$, DMRG is applied with a maximum bond dimension $\chi$ to find an MPS approximation of the ground state $|\Psi\rangle$.
2. **MPS → PQC Mapping**: The MPS is converted into a brickwork-structured quantum circuit $U|0\rangle \approx |\Psi\rangle$ via Riemannian optimization on the Stiefel manifold.
3. **RL-QAS Refinement**: Starting from the warm-start circuit, the RL agent continues appending gates $VU|0\rangle$ to further reduce the energy.

### Key Designs

1. **MPS-to-Circuit Mapping**: The MPS overlap $|\langle\Psi|U|0\rangle|$ is expressed as a tensor network contraction, with gradients computed via automatic differentiation. The Cayley transform is used to implement Riemannian Adam optimization on the Stiefel manifold, ensuring that 2-qubit unitary matrices $U_k \in U(4)$ remain unitary throughout optimization. The brickwork structure naturally accommodates linearly connected quantum hardware.

2. **TensorRL (trainable TN-init)**: The structure and parameters of the warm-start circuit are encoded into the RL state via binary encoding, allowing the agent to modify TN parameters during training. The RL state has size $(D_{\text{MPS}} + D) \times N \times (N + N_{\text{1-qubit}})$. This provides full information visibility but results in a larger state and slower training.

3. **TensorRL (fixed TN-init)**: The warm-start circuit is not encoded into the RL state; instead, it serves as a fixed initial statevector. The RL state size is reduced to $D \times N \times (N + N_{\text{1-qubit}})$. This yields a triple speedup: (i) faster statevector simulation; (ii) smaller neural network input; (iii) fewer trainable parameters, reducing classical optimizer function calls.

### Loss & Training

- DDQN (Double Deep Q-Network) with 5-step trajectory roll-outs
- Discount factor $\gamma = 0.88$, $\epsilon$-greedy exploration ($\epsilon$ decays from 1 to 0.05)
- Reward function tied to chemical accuracy relative to the target energy
- Action space: $\{RX, RY, RZ, CNOT\}$
- Default bond dimension $\chi = 2$
- Maximum steps per episode set to half that of the baseline

## Key Experimental Results

### Main Results (Noiseless Setting, Table 1)

| Molecular System | Method | Error | CNOT | Depth |
|------------------|--------|-------|------|-------|
| 8-H₂O | TensorRL (fixed) | 8.9×10⁻⁴ | **9** | **6** |
| 8-H₂O | CRLQAS | 1.8×10⁻⁴ | 105 | 75 |
| 8-H₂O | Vanilla RL | 1.7×10⁻⁴ | 117 | 96 |
| 10-H₂O | TensorRL (fixed) | 4.1×10⁻⁴ | **15** | **17** |
| 10-H₂O | Vanilla RL | 2.5×10⁻⁴ | 96 | 73 |
| 12-LiH | TensorRL (trainable) | **1.0×10⁻²** | **37** | **31** |
| 12-LiH | Vanilla RL | 2.2×10⁻² | 321 | 140 |

### Ablation Study (Noisy Setting, Table 2, 8-H₂O)

| Noise Type | Method | Error | CNOT | Success Rate |
|------------|--------|-------|------|--------------|
| Depolarizing | TensorRL (fixed) | **9.0×10⁻⁴** | 5 | **100%** |
| Depolarizing | CRLQAS (rerun) | 1.3×10⁻³ | 11 | 30% |
| Shot noise | TensorRL (trainable) | **8.7×10⁻⁵** | 28 | 100% |

### Key Findings

- TensorRL (fixed) reduces function evaluation counts by **100×** and per-episode execution time by **98%**.
- As system size increases, CNOT gate reduction scales from 5–9× at 6 qubits to 10–13× at 12 qubits.
- TensorRL achieves chemical accuracy in 100% of random seeds, whereas Vanilla RL succeeds in only 70% of trials on 10-H₂O.
- Effectiveness is also demonstrated on a 20-qubit TFIM model (Appendix experiments).

## Highlights & Insights

- **Complementarity of TN and RL**: DMRG at low bond dimension provides a sufficiently good initialization, while RL subsequently discovers more compact circuits — an outcome neither approach achieves independently.
- **Unexpected advantage of fixed TN-init**: Despite carrying less information, the smaller state and fewer parameters yield superior practical performance and the most compact circuits.
- **CPU-only feasibility**: TensorRL (fixed) can be efficiently trained on CPU for systems up to 8 qubits, lowering the hardware barrier for quantum architecture search.

## Limitations & Future Work

- The action space is restricted to $\{RX, RY, RZ, CNOT\}$; larger systems may require richer gate sets.
- Only the brickwork warm-start structure is explored; alternative tensor network topologies (e.g., MERA) remain uninvestigated.
- Training on actual quantum hardware has not been conducted; the current evaluation relies on noise simulation.
- The default bond dimension $\chi=2$ may be insufficient for more complex molecules.

## Related Work & Insights

- Builds upon the TN–VQA co-design idea of Rudolph et al. (2023), but is the first to integrate it with RL-QAS.
- Complements non-RL approaches such as CRLQAS (curriculum learning) and SA-QAS (simulated annealing).
- The TN warm-start strategy is generalizable to other RL-based quantum optimization problems.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of TN and RL is natural yet rich in implementation detail; the fixed/trainable mode comparison yields meaningful insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 6–12 qubit systems across multiple molecules, noisy and noiseless settings, multiple baselines, and success rate statistics.
- **Writing Quality**: ⭐⭐⭐⭐ Method descriptions are clear with abundant illustrations, though the main text is somewhat lengthy.
- **Value**: ⭐⭐⭐⭐ Substantially advances the scalability of RL-QAS and represents a state-of-the-art framework for quantum circuit design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Note 5: ReSearch — Learning to Reason with Search](research_learning_to_reason_with_search_for_llms_via_reinforcement_learning.md)
- [\[NeurIPS 2025\] A Theory of Multi-Agent Generative Flow Networks](a_theory_of_multi-agent_generative_flow_networks.md)
- [\[NeurIPS 2025\] DeepDiver: Adaptive Search Intensity Scaling via Open-Web Reinforcement Learning](deepdiver_adaptive_search_intensity_scaling_via_open-web_reinforcement_learning.md)
- [\[NeurIPS 2025\] Meta-World+: An Improved, Standardized, RL Benchmark](meta-world_an_improved_standardized_rl_benchmark.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)

</div>

<!-- RELATED:END -->
