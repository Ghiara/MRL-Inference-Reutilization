import mujoco
import mujoco_viewer
import imageio

def save_snapshot(xml_path, save_path="snapshot.png"):
    # Load the MuJoCo model from the XML file
    model = mujoco.MjModel.from_xml_path(xml_path)
    # Create a data structure to hold the simulation state
    data = mujoco.MjData(model)
    
    # Initialize the viewer (for rendering purposes)
    viewer = mujoco_viewer.MujocoViewer(model, data)
    
    # Step the simulation to get an initial state
    for _ in range(100):  # Run some steps to avoid blank snapshot
        mujoco.mj_step(model, data)
        viewer.render()
    
    # Capture the snapshot of the current frame
    image = viewer.read_pixels()
    
    # Save the snapshot using imageio
    imageio.imwrite(save_path, image)
    
    # Close the viewer
    viewer.close()

# Specify the path to the XML model
xml_file = "path_to_your_model.xml"
save_snapshot(xml_file, save_path="env_snapshot.png")