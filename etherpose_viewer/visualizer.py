import trimesh
import pyrender
import numpy as np
import argparse
from etherpose_viewer.smpl_np import SMPLModel
from threading import Thread
from queue import Queue
import time
from etherpose_viewer.model_visual_config import joint_name_to_num

class PoseVisualizer():
    def __init__(self,filename,model_filename='./model.pkl',update=None,cam=[1.4, 1, 0, 0],isTrain=True):

        # camera
        sx, sy, tx, ty = cam
        camera_pose = np.array([[1,0,0,50],
                                [0,1,0,25],
                                [0,0,-1,-300],
                                [0,0,0,1]])

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.414, znear=0.01, zfar=10000)

        # scene and viewer
        scene = pyrender.Scene(ambient_light=[0.02, 0.02, 0.02], bg_color=[1.0, 1.0, 1.0])
        cam_node = scene.add(camera, pose=camera_pose)
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True)

        # add queue
        viewer_q = Queue()
        scene_q = Queue()
        scene_q.put(scene)
        viewer_q.put(viewer)

        # smpl
        self.smpl = SMPLModel(model_filename)
        self.joint_name_to_num = joint_name_to_num

        # In macos, ```run_in_thread``` cannot be used.
        # So, we create a thread for update function.
        if update is None:
            print("AMASS file read mode")
            proc = Thread(target=self.update, args=(viewer_q,scene_q,filename,), daemon=True)
            proc.start()
        else:
            print("IK visualization mode")
            vert_q, face_q = Queue(), Queue()
            proc1 = Thread(target=update, args=(filename,vert_q,face_q,), daemon=True)
            proc2 = Thread(target=self.update_IK_mode, args=(viewer_q,scene_q,vert_q,face_q,isTrain,), daemon=True)
            proc1.start()
            proc2.start()

        viewer.etherposevis_run()

    def _load_amass(self,file_name):
        data = np.load(file_name)

        print("\nLoad {}.".format(file_name.split('/')[-1]))
        print("Formatted as {}\n".format(data.files))

        n_pose = 24*3 # num of joints X 3
        n_beta = 10
        poses = data['poses'][:,:n_pose]
        betas = data['betas'][:n_beta]
        trans = data['trans']
        return poses, betas, trans

    def np_wrapper(self, beta, pose, trans):
        result, global_joint_pos = self.smpl.set_params(pose=pose, beta=beta, trans=trans)
        self.joint_pos = global_joint_pos
        return result, self.smpl.faces

    def update_IK_mode(self,viewer_q,scene_q,vert_q,face_q,isTrain):
        viewer = viewer_q.get()
        scene = scene_q.get()
        node = []
        self.prev_t = time.time()
        while True:
            # print("{} FPS".format(np.around(1/(time.time()-self.prev_t),1)))
            self.prev_t = time.time()

            verts, faces = vert_q.get(), face_q.get()

            mesh = self._to_triesh(verts,faces)
            if isTrain:
                if len(node) == 1:
                    self.coloring_hand(mesh,[100,100,255,220])
                else:
                    self.coloring_hand(mesh,[255,100,100,220])
            mesh = self._from_trimesh_to_pyrender_mesh(mesh)

            node = self._update_scene(viewer,scene,mesh,node,isTrain)

    def update(self,viewer_q,scene_q,file_name):
        poses, betas, trans = self._load_amass(file_name)
        viewer = viewer_q.get()
        scene = scene_q.get()
        node = None
        self.prev_t = time.time()
        for i, (pose, tran) in enumerate(zip(poses, trans)):
            # print(next(iter(scene.camera_nodes)).camera.yfov)
            print("Frame #: {}   {} FPS".format(i,\
                                                np.around(1/(time.time()-self.prev_t),1)))
            self.prev_t = time.time()

            verts, faces = self.np_wrapper(betas, pose, tran)

            mesh = self._to_triesh(verts,faces)
            self.coloring(mesh)
            mesh = self._from_trimesh_to_pyrender_mesh(mesh)

            node = self._update_scene(viewer,scene,mesh,node)
        print('END!')

    def _to_triesh(self,verts,faces):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        return mesh

    def _from_trimesh_to_pyrender_mesh(self,mesh):
        mesh = pyrender.Mesh.from_trimesh(mesh)
        return mesh

    def _update_scene(self,viewer,scene,mesh,node,isTrain):
        viewer.render_lock.acquire()
        if not isTrain:
            while len(node) > 0:
                nd = node.pop(0)
                scene.remove_node(nd)
        else:
            if len(node)>1:
                while len(node) > 0:
                    nd = node.pop(0)
                    scene.remove_node(nd)
        nd = scene.add(mesh)
        node.append(nd)
        viewer.render_lock.release()
        return node

    def coloring_hand(self,mesh,color):
        mesh.visual.vertex_colors[:,:] = color

    def coloring(self,mesh):
        i = self.joint_pos[self.joint_name_to_num['left armroot']]
        j = self.joint_pos[self.joint_name_to_num['left elbow']]
        v = (j - i)
        v_unit = (j - i)/np.linalg.norm(j - i)
        u = mesh.vertices - i
        u_j = mesh.vertices - j

        length2 = np.abs(v_unit[0]*u[:,0]+v_unit[1]*u[:,1]+v_unit[2]*u[:,2])
        length3 = np.abs(v_unit[0]*u_j[:,0]+v_unit[1]*u_j[:,1]+v_unit[2]*u_j[:,2])
        length = np.sqrt(np.linalg.norm(u,axis=1)**2-np.power(length2,2))
        bound = np.linalg.norm(j - i)

        idx = np.all([length2<bound,length3<bound],axis=0)

        length2_ = length2[idx]
        length3_ = length3[idx]
        u_ = u[idx,:]
        u_j_ = u_j[idx,:]

        # length1 = np.sqrt(np.linalg.norm(u_,axis=1)**2-np.power(length2_,2))
        # length0 = np.sqrt(np.linalg.norm(u_j_,axis=1)**2-np.power(length3_,2))
        # bound1 = np.max([np.min(length1),np.min(length0)])
        # print(np.max(length1),np.max(length0))

        in_vert = mesh.vertices[idx,:]
        length1 = np.linalg.norm(i-in_vert,axis=1)
        length0 = np.linalg.norm(j-in_vert,axis=1)
        bound1 = np.max([np.min(length1),np.min(length0)])+0.015

        idx = np.all([length2<bound,length3<bound,length<bound1],axis=0)

        # radii = np.linalg.norm(mesh.vertices - self.joint_pos[self.joint_name_to_num['lowerneck']], axis=1)
        # idx = radii < 0.06

        # mesh.visual.vertex_colors = trimesh.visual.interpolate(radii, color_map='viridis')

        mesh.visual.vertex_colors[idx,:] = [255,0,0,255]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, help='input pose file path')

    args = parser.parse_args()

    PoseVisualizer(args.file_name)