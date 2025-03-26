import open3d as o3d
import numpy as np
import cv2
import os

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
stl_dir = os.path.join(BASE_DIR, "3dantlers")  
output_dir = os.path.join(BASE_DIR, "depth_maps")  
os.makedirs(output_dir, exist_ok=True)  

angles = [0, 45, 90, 135, 180, 225, 270, 315]

def render_depth_image(mesh, angle):
    R = mesh.get_rotation_matrix_from_xyz((0, np.radians(angle), 0))
    mesh.rotate(R, center=mesh.get_center())
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(mesh)
    vis.poll_events()
    vis.update_renderer()
    depth = vis.capture_depth_float_buffer()
    vis.destroy_window()

    depth = np.asarray(depth) * 255
    depth = depth.astype(np.uint8)
    return depth

for i in range(1, 48):
    stl_file = os.path.join(stl_dir, f"Antler{i}F.stl")

    if not os.path.exists(stl_file):
        continue 

    antler_folder = os.path.join(output_dir, f"Antler{i}F")
    os.makedirs(antler_folder, exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(stl_file)
    if mesh.is_empty():
        print(f"[ERROR] Mesh is empty or failed to load: {stl_file}")
        continue
    for angle in angles:
        depth_img = render_depth_image(mesh, angle)
        output_path = os.path.join(antler_folder, f"antler{i}F_angle{angle}.png")
        cv2.imwrite(output_path, depth_img)

print("done")
