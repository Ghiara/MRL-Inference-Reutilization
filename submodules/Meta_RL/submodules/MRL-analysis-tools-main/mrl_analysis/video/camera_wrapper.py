"""
This module contains the ``CameraWrapper`` class which allows headless rendering
of mujoco environments.

Author(s): 
    Julius Durmann, based on Kate Rakelly's PEARL (see https://github.com/katerakelly/oyster/blob/master/rlkit/envs/wrappers.py)
Contact: 
    julius.durmann@tum.de
Date: 
    2022-12-30
"""

import mujoco_py


class CameraWrapper(object):
    """Wrapper class for environments which allows headless rendering of images.

    Based on the equally named class in Kate Rakelly's PEARL code
    (see https://github.com/katerakelly/oyster/blob/master/rlkit/envs/wrappers.py).

    Parameters
    ----------
    env
        Environment
    """

    def __init__(self, env,  *args, **kwargs):
        self._wrapped_env = env
        self.initialize_camera()

    def get_image(self, width=256, height=256, camera_name=None):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        return self.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )

    def initialize_camera(self):
        # set camera parameters for viewing
        sim = self.sim
        viewer = mujoco_py.MjRenderContextOffscreen(sim)
        camera = viewer.cam
        camera.type = 1
        camera.trackbodyid = 0
        camera.elevation = -20
        sim.add_render_context(viewer)

    def render(self, mode, width=256, height=256, camera_name=None):
        assert mode == 'rgb_array', "CameraWrapper environments can only render RGB images!"
        return self.get_image(width, height, camera_name)

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)