import pygame
import numpy as np


class PygameRenderer():
    """This class helps to render images with ``pygame``.
    """
    def __init__(self) -> None:
        pygame.init()
        self.screen = None
        self.clock = pygame.time.Clock()

    def render_image(self, image: np.ndarray, title: str = ""):
        assert image.ndim == 3 or image.ndim == 2, "Image can either have three dimensions (color) or two dimensions (greyscale)"
        width, height = image.shape[0], image.shape[1]
        if self.screen is None:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(title)

        image = pygame.surfarray.make_surface(image)
        self.screen.blit(image, (0,0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise SystemExit
