import torch

from smrl.vae.mdpvae import MdpEncoder
from smrl.vae.encoder_decorators import PretrainedEncoder

class Toy1dForToy1dContinuousDecorator(PretrainedEncoder):
    """DEPRECATED: Use a more general approach of combining pretrained encoder
    with input-output modification. See 'experiments/encoder_transfer' and 
    'smrl/vae/encoder_deorators'.
    
    A decorator for pretrained Toy1D encoders.

    The pretrained encoder should be trained on Toy1D and accept observations of 
    shape (1,).
    The decorated encoder can be used for Toy1dDiscretized and accepts observations
    of shape (2,).

    Parameters
    ----------
    encoder_class : Type[MdpEncoder]
        The class of the wrapped encoder 
    path_to_weights : str
        Path to the pretrained encoder's weights 
    state_dict_keyword : str, optional
        Keyword for the state_dict from which the pretrained encoder weights
        are loaded. Set to None to indicate that the state_dict contains only
        the encoder weights.
        By default 'trainer/Inference trainer/encoder'
    trainable : bool, optional
        Set to ``True`` to train the wrapped encoder further,
        by default False 
    *args, **kwargs
        Instantiation arguments for the wrapped encoder object
    """
    def __init__(
        self, 
        encoder: MdpEncoder, 
        path_to_weights: str, 
        state_dict_keyword: str = 'trainer/Inference trainer/encoder', 
        trainable: bool = False, 
        *args, **kwargs
    ) -> None:
        super().__init__(encoder, 
                         path_to_weights=path_to_weights, 
                         state_dict_keyword=state_dict_keyword,
                         trainable=trainable, 
                         *args, **kwargs)

    def forward(
        self, 
        observations: torch.Tensor, 
        actions: torch.Tensor, 
        rewards: torch.Tensor, 
        next_observations: torch.Tensor, 
        terminals: torch.Tensor
    ) -> torch.distributions.Distribution:

        # Map Toy1dDiscretized inputs to Toy1d inputs
        observations = observations[..., :1]
        next_observations = next_observations[..., :1]
        actions = next_observations - observations  # Required to not confuse the encoder!

        # Let the wrapped encoder encode the modified inputs
        if self.trainable:
            return self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)
        else:
            with torch.no_grad():
                return self._wrapped_encoder.forward(observations, actions, rewards, next_observations, terminals)

Toy1dForToy1dDiscretizedDecorator = Toy1dForToy1dContinuousDecorator    # TODO: Legacy support, remove in future commit
