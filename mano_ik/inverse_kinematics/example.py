from solver import *
from armatures import *
from models import *
import numpy as np
import config
import argparse
import pyrender
import trimesh

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d') # Axe3D object


parser = argparse.ArgumentParser()
parser.add_argument('--file_name', type=str,
                    help='input pose file path')
args = parser.parse_args()

pose_data = np.load(args.file_name)

np.random.seed(20160923)
pose_glb = np.zeros([1, 3]) # global rotation


########################## mano settings #########################
# n_pose = 12 # number of pose pca coefficients, in mano the maximum is 45
# n_shape = 10 # number of shape pca coefficients
# pose_pca = np.random.normal(size=n_pose)
# shape = np.random.normal(size=n_shape)
# mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=1000)


########################## smpl settings ##########################
# note that in smpl and smpl-h no pca for pose is provided
# therefore in the model we fake an identity matrix as the pca coefficients
# to make the code compatible

n_pose = 23 * 3 # degrees of freedom, (n_joints - 1) * 3
pose_abs = pose_data['poses'][0][:72]
n_shape = 10
betas = pose_data['betas'][:n_shape]
mesh = KinematicModel(config.SMPL_MODEL_PATH, SMPLArmature, scale=10)


########################## smpl-h settings ##########################
# n_pose = 51 * 3
# n_shape = 16
# pose_pca = np.random.uniform(-0.2, 0.2, size=n_pose)
# shape = np.random.normal(size=n_shape)
# mesh = KinematicModel(config.SMPLH_MODEL_PATH, SMPLHArmature, scale=10)


########################## solving example ############################

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = Solver(verbose=True)

# _, keypoints = mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
vert, keypoints = mesh.set_params(pose_abs=pose_abs, pose_glb=pose_glb, shape=betas)

m = trimesh.Trimesh(vertices=vert,faces=mesh.faces)
m = pyrender.Mesh.from_trimesh(m)
scene = pyrender.Scene()
scene.add(m)
pyrender.Viewer(scene, use_raymond_lighting=True)

ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.set_zlim(-5,5)

ax.scatter(keypoints[:,0], keypoints[:,1], keypoints[:,2])
plt.show()

# exit()

params_est = solver.solve(wrapper, keypoints, u=2.5, v=0.9)


shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

print('----------------------------------------------------------------------')
print('ground truth parameters')
print('pose pca coefficients:', pose_pca)
print('pose global rotation:', pose_glb)
print('shape: pca coefficients:', betas)

print('----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)

mesh.set_params(pose_abs=pose_abs, pose_glb=pose_glb, shape=betas)
mesh.save_obj('./gt.obj')
mesh.set_params(pose_pca=pose_pca_est, pose_glb=pose_glb, shape=betas)
mesh.save_obj('./est.obj')

m = trimesh.Trimesh(vertices=mesh.verts,faces=mesh.faces)
m = pyrender.Mesh.from_trimesh(m)
scene = pyrender.Scene()
scene.add(m)
pyrender.Viewer(scene, use_raymond_lighting=True)

print('ground truth and estimated meshes are saved into gt.obj and est.obj')
