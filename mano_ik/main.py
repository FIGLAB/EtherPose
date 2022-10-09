import numpy as np
import argparse
try:
    from mano_ik.etherpose_viewer.visualizer import PoseVisualizer
    from mano_ik.inverse_kinematics.solver import *
    from mano_ik.inverse_kinematics.armatures import *
    from mano_ik.inverse_kinematics.models import *
    import mano_ik.inverse_kinematics.config
except:
    from etherpose_viewer.visualizer import PoseVisualizer
    from inverse_kinematics.solver import *
    from inverse_kinematics.armatures import *
    from inverse_kinematics.models import *
    import inverse_kinematics.config
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import glob
# from etherpose_viewer.visualizer import PoseVisualizer

n_pose = 17

class Keypoints2Mano():
    def __init__(self,model_path='./mano_ik/MANO_RIGHT.pkl'):
        mesh = KinematicModel(model_path, MANOArmature, scale=1000)
        self.wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
        self.solver = Solver(max_iter=3,verbose=True)

    def get_mano_params(self, keypoints):
        keypoints = self.mediapipe2mano_joints(keypoints)
        keypoints = self.set_default_rotation(keypoints)
        params_est = self.solver.solve(self.wrapper, keypoints, u=35, v=10)
        shape_est, pose_pca_est, pose_glb_est = self.wrapper.decode(params_est)
        return pose_pca_est

    def mediapipe2mano_joints(self,keypoints):
        mapping = [0,13,14,15,20,1,2,3,16,4,5,6,17,10,11,12,19,7,8,9,18]
        mano_keypoints = np.zeros((21,3))
        for i, m in enumerate(mapping):
            mano_keypoints[m] = keypoints[i]
        return mano_keypoints

    def set_default_rotation(self,keypoints):
        keypoints = keypoints*1.65*500 # 1.65? 1.85?
        # keypoints = keypoints*1.85*500 # 1.65? 1.85?
        r_mat = R.from_euler('xyz', [180,190,0], degrees=True).as_matrix()
        origin = np.array([95.66993092,6.38342886,6.18630528])
        for i in range(keypoints.shape[0]):
            k = np.array([keypoints[i]]).T
            keypoints[i] = np.matmul(r_mat,k).T[0]
        keypoints = keypoints - (keypoints[0]-origin)
        return keypoints

def mediapipe2mano_joints(keypoints):
    mapping = [0,13,14,15,20,1,2,3,16,4,5,6,17,10,11,12,19,7,8,9,18]
    mano_keypoints = np.zeros((21,3))
    for i, m in enumerate(mapping):
        mano_keypoints[m] = keypoints[i]
    return mano_keypoints

def set_default_rotation(keypoints):
    keypoints = keypoints*1.85*500 # 1.68? 1.85?
    r_mat = R.from_euler('xyz', [180,190,0], degrees=True).as_matrix()
    origin = np.array([95.66993092,6.38342886,6.18630528])
    for i in range(keypoints.shape[0]):
        k = np.array([keypoints[i]]).T
        keypoints[i] = np.matmul(r_mat,k).T[0]
    keypoints = keypoints - (keypoints[0]-origin)
    return keypoints

def remove_invalid(data):
    keypoints = np.array(data['right_hand'].tolist())
    idx = np.where(np.abs(keypoints[:,0,0])>200)[0]
    data = data.drop(idx)
    data.reset_index(drop=True, inplace=True)
    return data

def amass_to_mano(filename,vert_q,face_q):
    data = np.load(filename)

    n_pose = 20*3 # num of joints X 3
    n_hand = 16*3
    n_beta = 10
    poses = data['poses'][:,:n_pose]
    hands = data['poses'][:,n_pose+n_hand:n_pose+n_hand*2]
    betas = data['betas'][:n_beta]
    trans = data['trans']

    mesh = KinematicModel('./MANO_RIGHT.pkl', MANOArmature, scale=1000)
    wrapper = KinematicPCAWrapper(mesh, n_pose=15*3)
    poses = hands
    solver = Solver(verbose=True)

    if vert_q is None:
        data2 = pd.read_pickle('../data/2022-02-08-03-50.pkl')
        data2 = remove_invalid(data2)
        keypoints_all = np.array(data2['right_hand'].tolist())
        isvalid_all = np.array(data2['is_right_hand'].tolist())
        ax = plt.axes(projection='3d')
    for i, pose in enumerate(poses):
        # if i < 100:
        #     continue
        rot = np.array([1,1,1]).flatten()
        _, keypoints = mesh.set_params(
                                        pose_abs=np.zeros(16*3), \
                                        # pose_abs=pose, \
                                        # pose_abs=np.concatenate([np.zeros(15*3),rot]), \
                                        pose_glb=np.zeros((1,3)), \
                                        shape=np.zeros(10))
                                        # shape=np.random.normal(size=10))

        if vert_q is None:
            ax.cla()
            ax.set_xlim(-80,80)
            ax.set_ylim(-80,80)
            ax.set_zlim(-80,80)

            isvalid = isvalid_all[i]
            # print(isvalid)
            print(keypoints_all[i])
            keypoints_leap = mediapipe2mano_joints(keypoints_all[i])
            keypoints_leap = set_default_rotation(keypoints_leap)
            print(keypoints_leap)
            print("---")
            # keypoints_leap = keypoints_all[i]*1.3
            # r_mat = R.from_euler('y', 90, degrees=True).as_matrix()
            # for i in range(keypoints_leap.shape[0]):
            #     k = np.array([keypoints_leap[i]]).T
            #     keypoints_leap[i] = np.matmul(r_mat,k).T[0]
            # keypoints_leap = keypoints_leap - (keypoints_leap[0]-keypoints[0])

            ax.scatter3D(keypoints[:,0],keypoints[:,1],keypoints[:,2],c='blue')
            ax.scatter3D(keypoints_leap[:,0],keypoints_leap[:,1],keypoints_leap[:,2],c='black')
            idx = 1
            ax.scatter3D(keypoints_leap[idx:idx+1,0],keypoints_leap[idx:idx+1,1],keypoints_leap[idx:idx+1,2],c='red',s=50)
            ax.scatter3D(keypoints[idx:idx+1,0],keypoints[idx:idx+1,1],keypoints[idx:idx+1,2],c='red',s=50)
            # print(keypoints.shape, keypoints_leap.shape)
            plt.pause(0.1)

        if vert_q is not None:
            params_est = solver.solve(wrapper, keypoints, u=2.5, v=0.9)
            shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)
            _, keyoints = mesh.set_params(pose_pca=pose_pca_est, pose_glb=pose_glb_est, shape=shape_est)
            vert_q.put(mesh.verts)
            face_q.put(mesh.faces)
    plt.show()

def keypoints_to_mano(filename,vert_q=None,face_q=None):
    data = pd.read_pickle(filename)
    data = remove_invalid(data)
    keypoints_all = np.array(data['right_hand'].tolist())
    isvalid_all = np.array(data['is_right_hand'].tolist())
    # orientation = np.array(data['orientation'].tolist())

    mesh = KinematicModel('./MANO_RIGHT.pkl', MANOArmature, scale=1000)
    wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
    solver = Solver(verbose=False)

    pca_ik= []
    for i, (keypoints, isvalid) in enumerate(zip(tqdm(keypoints_all),isvalid_all)):
        # if i%1000!=0:
        #     continue

        keypoints = mediapipe2mano_joints(keypoints)
        keypoints = set_default_rotation(keypoints)
        params_est = solver.solve(wrapper, keypoints, u=35, v=10)
        shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)
        # pose_glb_est = np.random.normal(size=3)
        # print(orientation[i]*180/np.pi)
        pose_glb_est = np.array([0,0,0])
        mesh.set_params(pose_pca=pose_pca_est, pose_glb=pose_glb_est, shape=shape_est)
        if vert_q is not None and isvalid:
            vert_q.put(mesh.verts)
            face_q.put(mesh.faces)
        pose_pca_est = np.concatenate([pose_pca_est,pose_glb_est])
        pca_ik.append(pose_pca_est)
    data['pca'] = pca_ik
    foldername = filename.split('/2022')[0]
    savename = filename.split('.pkl')[0] + '_pca_est.pkl'
    data.to_pickle(savename)

def main(arg):
    # PoseVisualizer(filename=args.name,\
    #                 update=amass_to_mano,\
    #                 model_filename='./model.pkl')
    # amass_to_mano(args.name,None,None)

    inputfiles = []
    for i in range(2,3):
        # fnames = glob.glob("../data/user_study/p{}/hand_pose/2022*.pkl".format(i))
        fnames = glob.glob("../data/video_data/2022*.pkl".format(i))
        # fnames = [fn.split('_')[0]+'.pkl' for fn in fnames]
        fnames = np.unique(fnames)
        inputfiles.append(fnames.tolist())
    inputfiles = sum(inputfiles, [])

    for fname in inputfiles:
        keypoints_to_mano(fname)
        # PoseVisualizer(filename=fname,\
        #                 update=keypoints_to_mano,\
        #                 model_filename='./model.pkl', isTrain=False)

    # PoseVisualizer(filename="../regression/eval.pkl",\
    #                 update=keypoints_to_mano,\
    #                 model_filename='./model.pkl')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, help='input pose file path')

    args = parser.parse_args()

    main(args)