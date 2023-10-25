# Lecture 5 [Shubham Thorat] [SeqGAN: Sequence Generative Adversarial Nets with Policy Gradient]
---

Here is a link to the paper: [https://arxiv.org/pdf/1609.05473.pdf](https://arxiv.org/pdf/1609.05473.pdf)

---

### Background Knowledge: Generative Adversarial Networks (GANs)
GANs consist of two networks:

- Generator ($G$): Tries to generate data.
- Discriminator ($D$): Tries to distinguish between real and generated data.

Objective: $G$ wants to fool $D$ while $D$ wants to correctly classify real vs. generated data.

#### Disadvantages of GANs in the Context of SeqGAN:

- **Discrete Data Challenge**: The discrete nature of sequence data (like text) poses a challenge for GANs. Traditional GANs rely on the gradient of the loss from $D$ w.r.t. the outputs by $G$ to update the generator. This gradient is ill-defined for discrete data, making the straightforward application of GANs to sequence generation non-trivial.
- **Partial Feedback**: In SeqGAN, the discriminator only provides feedback once an entire sequence is generated, which may not provide granular insight for early tokens in the sequence.

---

### Background Knowledge: Markov Decision Process (MDP) & SeqGAN
In this paper, the decision-making is modeled as a Markov Decision Process (MDP):

- **States $S$**: Represent different situations or configurations of the environment.
- **Actions $A$**: Actions that can be taken in each state.
- **Transition Probabilities**: Probability of transitioning from one state to another after taking a specific action.
- **Rewards $R$**: Rewards received after taking an action in a state and transitioning to a new state.

---

### In the context of SeqGAN:
SeqGAN is a novel approach to generating sequences using Generative Adversarial Networks (GANs) combined with policy gradient methods from reinforcement learning. This combination addresses the non-differentiability issue that arises when generating discrete tokens in sequences.

- **State**: The generated tokens so far. In timestep $t$, the state $s$ is the current produced tokens $(y_1, \dots, y_{t-1})$
- **Action**: The next token to be generated. the action $a$ is the next token $y_t$ to select.
- **Agent $G_{\theta}$**: The generator is treated as the agent in reinforcement learning.
- **Reward $D_{\phi}$**: Discriminator provides reward signal for generated sequences.

Unlike other methods that require task-specific sequence scores (e.g., BLEU in machine translation), SeqGAN employs a discriminator to evaluate sequences and guide the learning of the generative model.

---

### SeqGAN: A Deep Dive - Mathematical Foundations
For the generator $($G_\theta$)$ with parameters $($\theta$)$, the objective is:
$\[ J(\theta) = \mathbb{E}[R_T | s_0, \theta] = \sum_{y_1 \in \mathcal{Y}} G_\theta(y_1|s_0) \cdot {Q_{G_\theta}^{ D_\phi}}(s_0, y_1) \]$

$Q_{G_\theta}^{D_\phi}(s, a)$ is the action-value function of a sequence.

$\[ Q_{G_\theta}^{D_\phi}(a = y_T, s = Y_{1:T-1}) = D_\phi(Y_{1:T}) \]$

However, the discriminator only provides a reward value for a finished sequence. Thus, to evaluate the action-value for an intermediate state, we apply Monte Carlo search with a roll-out policy $G_\beta$ to sample the unknown last $T - t$ tokens.

To reduce the variance and get a more accurate assessment of the action value, we run the roll-out policy starting from the current state until the end of the sequence for $N$ times to get a batch of output samples. Thus, we have:

$$
Q_{G_\theta}^{D_\phi}(s = Y_{1:t-1}, a = y_t) = 
\begin{cases} 
\frac{1}{N} \sum_{n=1}^{N} D_\phi(Y^n_{1:T}), & Y^n_{1:T} \in MC_{G_\beta}(Y_{1:t}; N) \text{ for } t < T \\
D_\phi(Y_{1:t}), & \text{for } t = T 
\end{cases}
$$


We represent an ($N$)-time Monte Carlo search as:
$\[ {{ Y^1_{1:T}, \dots, Y^N_{1:T} }} = MC_{G_\beta}(Y_{1:t}; N) \]$

---

### The illustration of SeqGAN
![SeqGAN Flow](https://github.com/shubham4104/SeqGAN_Review/blob/main/SeqGAN_Flow.png)

---

### SeqGAN via Policy Gradient

- For the generator ($G_\theta$) with parameters ($\theta$), the objective is:
$\[ J(\theta) = \mathbb{E}[R_T | s_0, \theta] = \sum_{y_1 \in \mathcal{Y}} G_\theta(y_1|s_0) \cdot {Q_{G_\theta}^{ D_\phi}}(s_0, y_1) \]$
    
- Optimizes a parametrized policy for long-term reward:
-  $
{{\nabla_\theta J(\theta) = \sum_{t=1}^{T} E_{Y_{1:t-1} \sim G_\theta} \sum_{y_t \in Y} \nabla_\theta G_\theta(y_t|Y_{1:t-1}) \cdot Q^{G_\theta}_{D_\phi}(Y_{1:t-1}, y_t)}
  $ 
-   $$
\nabla_\theta J(\theta) \approx \sum_{t=1}^{T} E_{y_t \sim G_\theta(y_t|Y_{1:t-1})}[\nabla_\theta \log G_\theta(y_t|Y_{1:t-1}) \cdot Q^{G_\theta}_{D\phi}(Y_{1:t-1}, y_t)]}
$$

- Update based on the derived gradient:
$\[ \theta \leftarrow \theta + \alpha_h \nabla_\theta J(\theta) \]$

Advanced gradient algorithms like Adam and RMSprop can be used.

- Train Discriminator model as follows (Same as GAN):
$\[ \min_\phi -E_{Y \sim p_{\text{data}}} [\log D_\phi(Y)] - E_{Y \sim G_\theta} [\log(1 - D_\phi(Y))] \]$

---

### SeqGAN: Evaluation on Synthetic Data and Real-world Scenarios

- **Synthetic Data Experiments**:

    - Utilized a randomly initialized LSTM as the true model (oracle) to simulate real-world structured sequences.
    - Generated sequences have a fixed length $\(T\)$.
    - Oracle provides both the training dataset and evaluates the generative models.

- **Evaluation Metric**:

    - Uses Negative Log-Likelihood with respect to the oracle $\(NLL_{oracle}\)$.
    - Perfect evaluation metric:
$\[ NLL_{\text{oracle}} = -E_{Y_{1:T} \sim G_\theta} \left[ \sum_{t=1}^{T} \log G_{\text{oracle}}(y_t|Y_{1:t-1}) \right] \]$

- **Training Setting**:

    - Used LSTM as the oracle to generate 10,000 sequences of length 20.
    - Training set for discriminator comprises generated examples (label 0) and instances from training set $( S )$ (label 1).

- **Results**:

    - SeqGAN outperforms other baselines significantly in $\(NLL_{oracle}\)$.
    - Demonstrates the potential of adversarial training strategies for discrete sequence generative models.

- **Real-world Scenarios**:

    - Applied to generate Chinese poems, Obama political speeches, and music.
    - Evaluated using BLEU score and human judgment for poem generation.

**Evaluation Table**

**Chinese poem generation performance comparison:**

| Algorithm | MLE | p-value | BLEU-2 | p-value |
| --- | --- | --- | --- | --- |
| MLE | 0.4165 | 0.0034 | 0.6670 | $<\(10^{-6}\)$ |
| SeqGAN | **0.5356** | 0.0034 | **0.73898** | $<\(10^{-6}\)$ |
| Real Data | 0.6011 |  | 0.746 |  |

**Obama political speech generation performance:**

| Algorithm | BLEU-3 | p-value | BLEU-4 | p-value |
| --- | --- | --- | --- | --- |
| MLE | 0.519 | $<\(10^{-6}\)$ | 0.416 | 0.00014 |
| SeqGAN | **0.556** | $<\(10^{-6}\)$ | **0.427** | 0.00014 |

**Music generation performance comparison:**

| Algorithm | BLEU-4 | p-value | MSE | p-value |
| --- | --- | --- | --- | --- |
| MLE | 0.9210 | $<\(10^{-6}\)$ | 22.38 | 0.00034 |
| SeqGAN | **0.9406** | $<\(10^{-6}\)$ | **20.62** | 0.00034 |

---

### Impact of Training Strategies on SeqGAN’s Convergence and Stability

![SeqGAN Param](https://github.com/shubham4104/SeqGAN_Review/blob/main/SeqGAN_Param.png)

**Caption**: Negative log-likelihood convergence performance of SeqGAN with different training strategies. The vertical dashed line represents the beginning of adversarial training.

---

### Strengths and Weaknesses of SeqGAN

- **Strengths**:

    - Addresses the non-differentiability problem in sequence generation using GANs.
    - Combines advantages of both GANs (high-quality samples) and policy gradient (sequence-level training signal).
    - Demonstrates improved performance over MLE-based sequence generation methods.

- **Weaknesses**:

    - Training instability, common to many GAN architectures.
    - Requires careful tuning of hyperparameters.
    - Partial observability of the reward signals can lead to suboptimal policies.

---

### Suggestions to Overcome Weaknesses

- Employ techniques to stabilize GAN training, e.g., gradient penalties.
- Use automated hyper parameter search to fine-tune model parameters.
- Integrate with other reinforcement learning methods like MCTS, TD learning to better estimate long-term rewards.
- Subtracting a state-dependent baseline from the reward, such as the value function, can significantly reduce the variance of gradient estimates, offering a direction for more stable and efficient training in SeqGAN.

---

### Citation

- [Goodfellow and others 2014] Goodfellow, I., et al. 2014. Generative adversarial nets. In NIPS, 2672–2680.
- [https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/](https://www.reddit.com/r/MachineLearning/comments/40ldq6/generative_adversarial_networks_for_text/)
- ChatGPT for latex code

---
