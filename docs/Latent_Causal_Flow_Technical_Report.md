# Latent Causal Flow: Continuous Confounder Disentanglement for Time Series via Monte Carlo Flow Matching

**Author:** Changchen Song  
**Date:** January 7, 2026  
**Status:** Technical Report / Preprint

---

## Abstract

We introduce **Latent Causal Flow (LCF)**, a novel generative framework for causal time series analysis. LCF addresses the limitations of existing methods that rely on explicit causal graphs (DAGs) or discrete environmental approximations. By identifying a theoretical isomorphism between Pearl's Backdoor Adjustment and the marginal vector field property in Flow Matching, we derive a method to perform exact causal interventions in continuous latent spaces. We prove that the interventional vector field is the posterior-weighted linear superposition of conditional vector fields, enabling inference-time confounder disentanglement via Monte Carlo integration without model retraining.

---

## 1. Introduction

Causal inference in time series is challenging due to the presence of unobserved confounders. While recent approaches like *DoFlow* (Wu et al., 2025) integrate causality with flow-based models, they assume causal sufficiency—requiring explicit Directed Acyclic Graphs (DAGs). Others, such as *CaTSG* (2025), rely on discrete approximations of the latent environment, which become insufficient when environmental dynamics are continuous.

We propose **Latent Causal Flow (LCF)**, which treats the latent confounding environment $E$ as a continuous random variable ($E \in \mathbb{R}^d$). By leveraging the linearity of the **Continuity Equation**, we demonstrate that complex causal integration (Backdoor Adjustment) can be solved efficiently via vector field superposition, providing an exact, graph-agnostic framework for generating interventional distributions $P(X|do(C))$.

---

## 2. Comparison with Existing Methods

| Feature | DoFlow (Wu et al., 2025) | CaTSG (2025) | **Latent Causal Flow (Ours)** |
| :--- | :--- | :--- | :--- |
| **Problem Setting** | Explicit Causal Graph | Latent Confounding | **Latent Confounding** |
| **Latent Space** | N/A (Assumes Sufficiency) | Discrete ($K$ clusters) | **Continuous ($E \in \mathbb{R}^d$)** |
| **Mathematical Basis** | Conditioning | Approx. Summation | **Exact Integration** |
| **Inference Method** | Point Estimation | Score Guidance | **Monte Carlo Integration** |

---

## 3. Theoretical Framework

### 3.1 Problem Definition

We consider a Structural Causal Model (SCM) where:
- $C$: Observed condition / treatment.
- $E$: Unobserved continuous confounder ($E \in \mathbb{R}^d$).
- $X$: Target time series trajectory.

Our goal is to sample from the interventional distribution:

$$ p(x|do(c)) = \int p(x|c, e) p(e) de $$

### 3.2 Derivation of the Causal Vector Field

The evolution of probability density $\rho_t(x)$ in Flow Matching is governed by the Continuity Equation:

$$ \frac{\partial \rho_t}{\partial t} + \nabla \cdot (\rho_t v_t) = 0 $$

**Theorem 1 (Continuous Causal Flow).**  
The interventional vector field required to sample from $P(X|do(C))$ is the posterior expectation of the environment-specific vector fields:

$$ v_{do}(x, t, c) = \mathbb{E}_{e \sim p(e|x, t, c)} [v_\theta(x, t, c, e)] $$

**Proof:**

1. **Micro-Dynamics:** For a fixed environment $e$, the conditional density path $\rho_t(x|c, e)$ satisfies the Continuity Equation:

$$ \frac{\partial \rho_t(\cdot|e)}{\partial t} + \nabla \cdot (\rho_t(\cdot|e) v_t(\cdot|e)) = 0 $$

2. **Macro-Intervention:** The target interventional distribution is defined by marginalizing out $E$ (Backdoor Adjustment):

$$ \rho_{target}(x, t) = \int p(e) \rho_t(x|c, e) de $$

3. **Linear Superposition:** Multiply the Micro-Dynamics equation by $p(e)$ and integrate over $e$. Due to the linearity of $\frac{\partial}{\partial t}$ and $\nabla \cdot$:

$$ \frac{\partial}{\partial t} \left[ \int p(e)\rho_t(\cdot|e) de \right] + \nabla \cdot \left[ \int p(e)\rho_t(\cdot|e) v_t(\cdot|e) de \right] = 0 $$

4. **Matching the Target:** The target field $v_{do}$ must satisfy:

$$ \frac{\partial \rho_{target}}{\partial t} + \nabla \cdot (\rho_{target} v_{do}) = 0 $$

5. **Identification:** Comparing the flux terms (inside $\nabla \cdot$):

$$ \rho_{target} v_{do} = \int p(e) \rho_t(\cdot|e) v_t(\cdot|e) de $$

6. **Final Result:** Solving for $v_{do}$ and applying Bayes' Rule ($p(e|x) = \frac{p(e)\rho_t(\cdot|e)}{\rho_{target}}$):

$$ v_{do}(x, t, c) = \int \frac{p(e)\rho_t(\cdot|e)}{\rho_{target}} v_t(\cdot|e) de = \int p(e|x, t, c) v_t(x, t, c, e) de $$

This proves that causal intervention corresponds to posterior-weighted velocity averaging in Flow Matching. $\square$

---

## 4. Algorithm Implementation

### 4.1 Training Objective

The model jointly trains an **Environment Encoder** (for abduction) and a **Conditional Vector Field** (for prediction):

$$ \mathcal{L} = \mathcal{L}_{FM} + \lambda \mathcal{L}_{Info} $$

- **Encoder $q_\phi(e | x_1, c)$**: Infers the latent environment from observed data.
- **Flow Matching Loss**: $\mathbb{E}_{t, x_t, e} || v_\theta(x_t, t, c, e) - (x_1 - (1-\sigma_{min})x_0) ||^2$

### 4.2 Inference: Monte Carlo Flow Integration

At inference time, we approximate the integral via Monte Carlo sampling:

```python
import torch

def sample_interventional(x_noise, c_target, num_mc_samples=10, num_steps=100):
    """
    Generates X | do(C=c_target) using Monte Carlo Flow Integration.
    
    Args:
        x_noise: Initial Gaussian noise x_0.
        c_target: The intervention value for condition C.
        num_mc_samples: Number of samples for MC integration of latent E.
        num_steps: Number of Euler integration steps.
    """
    x_t = x_noise
    dt = 1.0 / num_steps
    
    for t in range(num_steps):
        # 1. Abduction: Infer continuous latent environment distribution
        mu, logvar = env_encoder(x_t, t, c_target)
        
        # 2. Monte Carlo Sampling (Reparameterization Trick)
        batch_size = x_t.shape[0]
        e_dim = mu.shape[-1]
        eps = torch.randn(batch_size, num_mc_samples, e_dim).to(x_t.device)
        e_samples = mu.unsqueeze(1) + torch.exp(0.5 * logvar.unsqueeze(1)) * eps
        
        # 3. Action: Compute conditional velocities for each environment sample
        v_cond = velocity_net(x_t.unsqueeze(1), t, c_target.unsqueeze(1), e_samples) 
        
        # 4. Integration: Compute the expectation (mean) over the MC samples
        v_marginal = v_cond.mean(dim=1) 
        
        # 5. Prediction: Euler Step
        x_t = x_t + v_marginal * dt
        
    return x_t
```

---

## 5. Conclusion

Latent Causal Flow provides a rigorous framework for causal interventions in time series with unobserved confounders. By exploiting the linearity of the Continuity Equation, we establish that the interventional vector field is the posterior expectation of conditional vector fields. This eliminates the need for explicit causal graphs and discrete approximations, enabling exact inference-time confounder disentanglement via Monte Carlo integration.

---

## References

1. Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow Matching for Generative Modeling. *ICLR*.
2. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.
3. Wu, et al. (2025). DoFlow: Causal Flow Matching. *Preprint*.
4. CaTSG (2025). Causal Time Series Generation with Discrete Environments. *Preprint*.

