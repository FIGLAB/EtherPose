import trimesh
import pyrender
import numpy as np

def main(obj_trimesh):
    # print(obj_trimesh.visual)
    # exit()
    radii = np.linalg.norm(obj_trimesh.vertices - obj_trimesh.center_mass, axis=1)
    # obj_trimesh.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')
    c = np.repeat(np.array([[255,255,0,255]]),100,axis=0)
    obj_trimesh.visual.vertex_colors[0:100,:] = c

    mesh = pyrender.Mesh.from_trimesh(obj_trimesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    pyrender.Viewer(scene, use_raymond_lighting=True)

if __name__ == "__main__":

    gtobj = trimesh.load('gt.obj')
    estobj = trimesh.load('est.obj')

    main(gtobj)
    main(estobj)


# while True:
#     pose = np.eye(4)
#     pose[:3,3] = [i, 0, 0]
#     v.render_lock.acquire()
#     scene.set_pose(mesh_node, pose)
#     v.render_lock.release()
#     i += 0.01