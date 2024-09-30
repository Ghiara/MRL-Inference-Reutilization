# VAE
This submodule defines abstract classes for encoder & decoder networks. It also provides implementations of ELBO computations for VAE ELBOs [1,3] and NP ELBOs [2].

## Motivation
We train an encoder and a decoder network to predict transitions and rewards in our MDP. This objective can be expressed as follows: 
    $$ \max_\theta \log p_\theta(s^\prime, r \vert s, a, c^\mathcal{T}) $$
Here, $(s, a, r, s^\prime)$ forms a transition (a "*target*") which we would like to predict based on the *context* $c^\mathcal{T}$ of recent transitions. To do so, the encoder network (parameters $\phi$) can provide compact representations of the task $\mathcal{T}$ based on $c^\mathcal{T}$. The decoder (parameters $\theta$) can use this information, as well as the current state $s$ and action $a$, to determine a new state and reward. This information can be provided to a policy later on.

The ELBO is a surrogate objective which provides a sort of lower bound to our objective. This is necessary because the objective above is usually infeasible. For more details, please rely on the references below and on the extensive literature on variational autoencoders.

## Base classes for encoders and decoders
The file *mdpvae.py* defines the base classes ``MdpEncoder`` and ``MdpDecoder`` which extend the abstract classes ``Encoder`` and ``Decoder`` to MDP prediction and encoding. Most importantly, ``MdpEncoder`` instances implement the following functions:
- ``forward()``: For a sequence of transitions, this function returns a distribution over the latent space
- ``get_encoding()``: For a sequence of trainsitions, this function returns an encoding which is computed from the latent space posterior (``forward()``). A function for batched data (``get_encodings()``) exists as well. These functions are important for providing task information to a policy.

## ELBO types and implementations
Also in *mdpvae.py*, we provide implemenations of encoder-decoder wrappers which represent a VAE or NP. They differentiate themselves by the way that the ELBO is computed (function ``elbo()``). 

These two types of classes and ELBOs are especially relevant:
- ``MdpVAE``: Following the approach of Rakelly et al. [1], the ELBO is computed by estimating
    $$ 	\mathcal{L}_{ELBO} = \mathbb{E}_{z \sim q_\phi(z \vert c^\mathcal{T})} \left[ \log p_\theta(r, s^\prime \vert z, s, a) \right] - \beta \mathcal{D}_{KL}(q_\phi(z \vert c^\mathcal{T}) \Vert p(z)) $$ 
    with Monte-Carlo samples. $\theta$ are the parameters of the decoder network which approximates the transition distribution and reward function and $\phi$ are the parameters of the encoder network which approximates the posterior distribution.
- ``NeuralProcess``: Similarly, the approach by Garnelo et al. [2] can be laveraged to find an alternative optimization objective:
    $$ \mathcal{L}_{ELBO} = 	\mathbb{E}_{z \sim q(\cdot \vert c^\mathcal{T}, s, a, s^\prime, r)} \left[ \log p(s^\prime, r \vert z, s, a) + \log \frac{p(z \vert c^\mathcal{T})}{q(z \vert c^\mathcal{T}, s, a, s^\prime, r)}\right] $$


## Implementations of encoders and decoders
The directory *encoder_networks* provides different types of encoder implementations:
- ``MlpEncoder``: Transition sequences are stacked to be passed through a multilayer perceptron. Since the inputs to the first layer of the network need to have a fixed length, only sequences of a particular length can be processed. As a consequence, this class of encoder cannot be used with ``NeuralProcess``.
- ``GRUEncoder``: This encoder architecture relies on GRU [4] cells which can parse the inputs sequentially. 
- ``AttentionEncoder``: This encoder architecture relies on attention layers to parse input sequences of arbitrary length (in a parallel fashion). 

We only provide much simpler decoder architectures since they do not need to process sequences. Most importantly, we implement ``SeparateMlpDecoder`` - a decoder network with a multilayer perceptron architecture. This decoder treats the reward and next state as two independent random variables and provides separate pathways for computing their likelihoods.

## Encoder decorators
In an attempt to allow more advanced encoder usage and handling, we use the decorator pattern (see e.g. [wikipedia.org](https://en.wikipedia.org/wiki/Decorator_pattern)) to extend the functionality of encoders.

Encoder decorators can be useful for a variety of use-cases, including
- Input modifications (*encoder_decorators/io_modification.py*, *encoder_decorators/striding.py*)
- Loading pretrained encoder weights (*encoder_decorators/pretrained_encoder.py*)


---

## References
[1] Rakelly, K. et al. (2019) ‘Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables’, Proceedings of the 36th International Conference on Machine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA: PMLR, pp. 5331–5340. Available at: http://proceedings.mlr.press/v97/rakelly19a.html.

[2] Garnelo, M. et al. (2018) ‘Neural Processes’, in ICML Workshop on Theoretical Foundations and
Applications of Deep Generative Models. Available at: https://arxiv.org/abs/1807.01622.

[3] Kingma, D.P. and Welling, M. (2014) ‘Auto-Encoding Variational Bayes’, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings. Available at: http://arxiv.org/abs/1312.6114v10.

[4] Cho, K. et al. (2014) ‘Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation’, Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). Doha, Qatar: Association for Computational Linguistics.

[5] Vaswani, A. et al. (2017) ‘Attention is All you Need’, Advances in Neural Information Processing Systems: Curran Associates, Inc. Available at: https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.