from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn import preprocessing
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from etherpose_viewer.visualizer import PoseVisualizer
from scipy.spatial.transform import Rotation as R
# from mano_ik.inverse_kinematics.solver import *
from mano_ik.inverse_kinematics.armatures import *
from mano_ik.inverse_kinematics.models import *
import time
from tqdm import tqdm
import glob
from data.combiner import combine_dataset
from scipy import signal
import pickle5 as pickle

class SimpleRegression():
    def __init__(self):
        self.n_pose = 17
        self.ts_size = 3
        self.prev_chunk = []
        self.ismano = False
        self.predmode = 0 # 0: both, 1: only hand pose, 2: only rotation
        self.error = []

    def phase_convert(self, phase):
        phase = phase - np.pi
        phase[phase <= -np.pi] = phase[phase <= -np.pi] + 2*np.pi
        return phase

    def phase(self, x):
        lst = []
        for x_ in x:
            l = np.angle(x_)
            # l = np.rad2deg(l)
            lst.append(l)
        return np.array(lst)

    def logmag(self,x):
        lst = []
        for x_ in x:
            l = 20*np.log10(np.abs(x_))
            lst.append(l)
        return np.array(lst)

    def subtract_static_data(self, data, label):
        # static data as calibration data
        cal_data = np.mean(data[0:10],axis=0)
        cal_data_new = []
        flag = False
        for i in range(10,len(data)):
            if label[i] != -1:
                cal_data_new.append(np.array(data[i],copy=True))
                data[i] = data[i] - cal_data
                flag = True
            else:
                if flag:
                    cal_data_new = np.array(cal_data_new)
                    cal_data = np.mean(cal_data_new,axis=0)
                    cal_data_new = []
                    flag = False
                data[i] = data[i] - cal_data
        return data

    def to_impedance(self, mag, phase):
        impedance = np.power(10,mag/20)*np.exp(1j*np.deg2rad(phase))
        Re, Im = np.real(impedance), np.imag(impedance)
        return Re, Im

    def flipped_hand_filter(self, data):
        isRhand = np.array(data['is_right_hand'].tolist())
        isLhand = np.array(data['is_left_hand'].tolist())
        filter_idx = np.all([isRhand,~isLhand],axis=0)
        return filter_idx

    def gesture_filter(self, data):
        only_keep = ['stretch', 'spider_man', 'ulnar_flexion', 'radial_flexion',
                    'left_flexion','right_flexion', 'pointing', 'gun', 'fist','neutral',
                    'thumbs_up', 'index_pinch','middle_pinch', 'ring_pinch', 'little_pinch']
        filter_idx = []

        label_name = np.array(data['class_name'].tolist(),dtype='str').reshape(-1,1)


    def load(self, train_fname, eval_fname, isname=True):
        # load data
        if isname:
            # self.train_data = pd.read_pickle(train_fname)
            # self.eval_data = pd.read_pickle(eval_fname)
            self.train_data = pickle.load(open(train_fname, "rb"))
            self.eval_data = pickle.load(open(eval_fname, "rb"))
        else:
            self.train_data = train_fname
            self.eval_data = eval_fname
        self.train_data = self.train_data[self.flipped_hand_filter(self.train_data)]
        self.eval_data = self.eval_data[self.flipped_hand_filter(self.eval_data)]

        # info
        device_num = np.array(self.train_data['s11'].tolist()).shape[1]
        # loop per device
        train_feat, eval_feat = [], []
        train_label, eval_label = [], []
        train_rot, eval_rot = [], []
        train_cname_lst, eval_cname_lst  = [], []
        for i in range(device_num):
            istrain = True
            for data, feat_lst in zip([self.train_data, self.eval_data], [train_feat, eval_feat]):
                for flag in [False,True]:
                    if not flag:
                        mag = self.logmag(np.real(np.array(data['s_raw'].tolist()))[:,i,:])
                        phase = self.phase(np.array(data['s_raw'].tolist()))[:,i,:]
                        # cal_phase = self.phase(np.array(data['calibration_s_raw'].tolist())[:,i,:])
                        # cal_mag = self.logmag(np.array(data['calibration_s_raw'].tolist())[:,i,:])
                        # if i == 1:
                        #     phase  = self.phase_convert(phase)

                    else:
                        mag = self.logmag(np.array(data['s_raw'].tolist()))[:,i,:]
                        phase = self.phase(np.array(data['s_raw'].tolist()))[:,i,:]
                        mag, phase = self.to_impedance(mag, phase)


                    # mag = self.time_serial_chunk(mag)
                    # phase = self.time_serial_chunk(phase)
                    # mag = np.mean(mag,axis=1)
                    # phase = np.mean(phase,axis=1)


                    # mag = mag - cal_mag
                    # phase = phase - cal_phase
                    
                    # label = np.array(data['class_label'].tolist())
                    # mag = self.subtract_static_data(mag, label)
                    # phase = self.subtract_static_data(phase, label)

                    # mag = self.time_serial_chunk(mag)
                    # phase = self.time_serial_chunk(phase)
                    # # temporal calibration
                    # mag_t = mag - mag[:,0:1,:]
                    # mag_t = mag[:,1:,:]
                    # phase_t = phase - phase[:,0:1,:]
                    # phase_t = phase[:,1:,:]
                    # mag_t, phase_t = mag.reshape(mag.shape[0],-1), phase.reshape(phase.shape[0],-1)
                    # mean
                    # mag = np.mean(mag,axis=1)
                    # phase = np.mean(phase,axis=1)

                    # sample
                    # if istrain:
                    #     istrain = False
                    #     # mag = mag - mag[0,:]
                    #     # phase = phase - phase[0,:]
                    #     num = int(mag.shape[0]*0.25)
                    #     mag, phase = mag[0:-num], phase[0:-num]
                    #     mag21, phase21 = mag21[0:-num], phase21[0:-num]
                    # else:
                    #     # mag = mag - mag[0,:]
                    #     # phase = phase - phase[0,:]
                    #     num = int(mag.shape[0]*0.25)
                    #     mag, phase = mag[-num:], phase[-num:]
                    #     mag21, phase21 = mag21[-num:], phase21[-num:]

                    if not flag:
                        mag_f = self.extract_features(mag)
                        phase_f = self.extract_features(phase)
                    else:
                        mag_f = self.extract_features2(mag)
                        phase_f = self.extract_features2(phase)


                    mag_f = self.time_serial_chunk(mag_f)
                    phase_f = self.time_serial_chunk(phase_f)
                    mag_f = mag_f.reshape(mag_f.shape[0],-1)
                    phase_f = phase_f.reshape(phase_f.shape[0],-1)
                    # # cut window
                    # indices = self.center_window_cut(mag)
                    # mag_w = self.selector(mag, indices)
                    # phase_w = self.selector(phase, indices)
                    # concatenate
                    size = mag_f.shape[0]
                    mag_f, phase_f = mag_f.reshape(size,-1), phase_f.reshape(size,-1)
                    # mag_t, phase_t = mag_t.reshape(size,-1), phase_t.reshape(size,-1)
                    # mag_w, phase_w = mag_w.reshape(size,-1), phase_w.reshape(size,-1)
                    # mag21_f, phase21_f = mag_f.reshape(size,-1), phase_f.reshape(size,-1)
                    # feat = np.concatenate([mag_f, phase_f, mag_w, phase_w, mag_t, phase_t],axis=1)
                    feat = np.concatenate([mag_f, phase_f],axis=1)
                    feat_lst.append(feat)


        istrain = True
        for data, label_lst, rot_lst, class_name_lst in zip([self.train_data, self.eval_data], [train_label, eval_label], [train_rot, eval_rot], [train_cname_lst, eval_cname_lst]):
            if self.ismano:
                label = np.array(data['pca'].tolist())[:,:-3]
            else:
                label = np.array(data['right_hand'].tolist())*500*1.65
            # if not self.ismano and self.predmode == 0:
            #     rot = np.array(data['axis'].tolist())
            #     for i in range(len(rot)):
            #         label[i] = self.to_global_coordinate(label[i], rot[i])
            label = self.time_serial_chunk(label)
            # label = np.mean(label,axis=1)
            label = label[:,-1,:]
            label = label.reshape(label.shape[0],-1)
            rot = np.array(data['axis'].tolist())
            rot = self.time_serial_chunk(rot)
            rot = rot[:,-1,:]
            class_name = np.array(data['class_name'].tolist())
            # sample
            # if istrain:
            #     istrain = False
            #     num = int(label.shape[0]*0.25)
            #     rot, label = rot[:-num], label[:-num]
            # else:
            #     num = int(label.shape[0]*0.25)
            #     rot, label = rot[-num:], label[-num:]
            label_lst.append(label)
            rot_lst.append(rot)
            class_name_lst.append(class_name)

        # train
        self.train_feature = np.concatenate(train_feat,axis=1)
        self.train_label = np.concatenate(train_label,axis=1)
        self.train_rot = np.concatenate(train_rot,axis=1)
        self.train_rot = self.mat2vec(self.train_rot)
        self.train_rot = self.rvec2mat(self.train_rot)
        self.train_rot = self.train_rot.reshape(-1,9)
        # self.train_rot = self.rvec2quat(self.train_rot)
        if self.ismano:
            if self.predmode == 0:
                self.train_label = np.concatenate([self.train_label,self.train_rot],axis=1)
            elif self.predmode == 1:
                self.train_label = np.concatenate([self.train_label],axis=1)
            elif self.predmode == 2:
                self.train_label = np.concatenate([self.train_rot],axis=1)
        else:
            if self.predmode == 0:
                # self.train_label = np.concatenate([self.train_label],axis=1)
                self.train_label = np.concatenate([self.train_label, self.train_rot],axis=1)
            elif self.predmode == 1:
                self.train_label = np.concatenate([self.train_label],axis=1)
            elif self.predmode == 2:
                self.train_label = np.concatenate([self.train_rot],axis=1)
        self.train_cname_lst = np.concatenate(train_cname_lst)

        # eval
        self.eval_feature = np.concatenate(eval_feat,axis=1)
        self.eval_label = np.concatenate(eval_label,axis=1)
        # test = self.eval_label.reshape(-1,21,3)
        # length = np.linalg.norm(test[0,5]-test[0,17])
        # length = np.linalg.norm(test[0,5]-test[0,6])+np.linalg.norm(test[0,6]-test[0,7])+np.linalg.norm(test[0,7]-test[0,8])
        # print(length)
        # exit()
        self.eval_rot = np.concatenate(eval_rot,axis=1)
        self.eval_rot = self.mat2vec(self.eval_rot)
        self.eval_rot = self.rvec2mat(self.eval_rot)
        self.eval_rot = self.eval_rot.reshape(-1,9)
        # self.eval_rot = self.rvec2quat(self.eval_rot)
        if self.ismano:
            if self.predmode == 0:
                self.eval_label = np.concatenate([self.eval_label,self.eval_rot],axis=1)
            elif self.predmode == 1:
                self.eval_label = np.concatenate([self.eval_label],axis=1)
            elif self.predmode == 2:
                self.eval_label = np.concatenate([self.eval_rot],axis=1)
        else:
            if self.predmode == 0:
                self.eval_label = np.concatenate([self.eval_label,self.eval_rot],axis=1)
            elif self.predmode == 1:
                self.eval_label = np.concatenate([self.eval_label],axis=1)
            elif self.predmode == 2:
                self.eval_label = np.concatenate([self.eval_rot],axis=1)
        self.eval_cname_lst = np.concatenate(eval_cname_lst)

        # save
        self.train_feature = self.train_feature.reshape(self.train_feature.shape[0],-1)
        self.eval_feature = self.eval_feature.reshape(self.eval_feature.shape[0],-1)

        # print(self.train_feature.shape, self.train_label.shape)
        # print(self.eval_feature.shape, self.eval_label.shape)


    def to_global_coordinate(self, joints, axis):
        mat_inv = np.linalg.inv(axis)
        for i in range(len(joints)):
            v = np.copy(joints[i])
            joints[i, 0] = np.dot(v,mat_inv[0,:])
            joints[i, 1] = np.dot(v,mat_inv[1,:])
            joints[i, 2] = np.dot(v,mat_inv[2,:])
        return joints

    def rvec2quat(self, rot):
        quat = []
        for i in range(len(rot)):
            r = R.from_rotvec(rot[i])
            quat.append(r.as_quat())
        return np.array(quat)

    def rvec2mat(self, rot):
        matrix = []
        for i in range(len(rot)):
            r = R.from_rotvec(rot[i])
            matrix.append(r.as_matrix())
        return np.array(matrix)


    def load_data(self, feat, label, device_num, isTrain=True):
        feat, label = np.array(feat), np.array(label)
        num_points = int(feat.shape[2]/2)
        num_timestamp = feat.shape[0]
        train_s11 = feat[:,:,:num_points]
        train_s11_phase = feat[:,:,num_points:]
        train_feature = []
        for i in range(device_num):
            for flag in [False, True]:
                # mag = np.array(data['s_raw'].tolist())[:,i,:]
                # phase = np.array(data['s_raw'].tolist())[:,i,:]
                mag = train_s11[:,i,:]
                phase = train_s11_phase[:,i,:]
                if not flag:
                    mag = self.logmag(np.real(mag))
                    phase = self.phase(phase)
                    # extract feature
                    mag_f = self.extract_features(mag)
                    phase_f = self.extract_features(phase)
                else:
                    mag = self.logmag(mag)
                    phase = self.phase(phase)
                    mag, phase = self.to_impedance(mag, phase)
                    mag_f = self.extract_features2(mag)
                    phase_f = self.extract_features2(phase)
                # cut window
                indices = self.center_window_cut(train_s11[:,i,:])
                mag_w = self.selector(train_s11[:,i,:], indices)
                phase_w = self.selector(train_s11_phase[:,i,:], indices)
                feat = np.concatenate([mag_f, phase_f],axis=1)
                train_feature.append(feat)
        train_feature = np.concatenate(train_feature,axis=1)
        if isTrain:
            train_feature = self.time_serial_chunk(train_feature)
            label = self.time_serial_chunk(label)
            self.train_feature = train_feature.reshape(train_feature.shape[0],-1)
            self.train_label = label[:,-1,:]
        else:
            self.prev_chunk.append(train_feature)
            if len(self.prev_chunk) < self.ts_size:
                return None, None
            if len(self.prev_chunk) > self.ts_size:
                self.prev_chunk.pop(0)
            train_feature = np.array(self.prev_chunk)
            train_feature = train_feature.reshape(1,self.ts_size,-1)
            train_feature = train_feature.reshape(1,-1)
        return train_feature, label

    def load_model(self):
        reg = pickle.load(open('reg.pkl', 'rb'))
        scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
        return reg, scaler

    def time_serial_chunk(self, data):
        data_ts = []
        for i in range(self.ts_size,len(data)):
            data_ts.append(data[i-self.ts_size:i])
        data_ts = np.array(data_ts)
        return data_ts

    def cal_window(self, data):
        data_1st = data[:,0:1,:]
        data_cal = data[:,1:,:]-data_1st
        return data_cal
        # data = np.concatenate([data, data_cal],axis=1)
        # return data

    def center_window_cut(self, data):
        winsize = int(data.shape[1]/4)
        idx = np.argmin(data,axis=1)
        indices = []
        for i in idx:
            if i-winsize < 0:
                sel = np.arange(0,2*winsize+1)
            elif i+winsize >= data.shape[1]:
                sel = np.arange(data.shape[1]-2*winsize-1,data.shape[1],1)
            else:
                sel = np.arange(i-winsize, i+winsize+1,1)
            indices.append(sel)
        indices = np.array(indices)
        return indices

    def selector(self, data, indices):
        data_new = []
        for d, i in zip(data, indices):
            data_new.append(d[i])
        data_new = np.array(data_new)
        return data_new

    def normalize(self):
        # self.minmaxscaler = preprocessing.MinMaxScaler()
        self.minmaxscaler = preprocessing.StandardScaler()
        self.train_feature_norm = self.minmaxscaler.fit_transform(self.train_feature)
        pickle.dump(self.minmaxscaler,open('minmaxscaler.pkl','wb'))
        return self.minmaxscaler

    def train(self):
        reg = ExtraTreesRegressor(n_estimators=300, n_jobs=-1)
        # reg = GradientBoostingRegressor(random_state=0)
        label = self.train_label.reshape(self.train_label.shape[0], -1)
        reg.fit(self.train_feature_norm, label)
        pickle.dump(reg, open('reg.pkl', 'wb'))
        importances = np.around(reg.feature_importances_,4)
        # plt.bar(np.arange(len(importances))-0.5,importances)
        # plt.xticks(np.arange(len(importances))-0.5, self.xticks)
        # plt.xticks(rotation=90, fontsize=5)
        # plt.axvspan(-1,63, alpha=0.2, color="red")
        # plt.axvspan(63,128, alpha=0.2, color="yellow")
        # plt.show()
        self.reg = reg
        return reg

    def eval(self):
        self.eval_feature_norm = self.minmaxscaler.transform(self.eval_feature)
        # self.pred = self.reg.predict(self.eval_feature_norm).reshape(-1,21,3)
        # self.pred = self.reg.predict(self.eval_feature_norm).reshape(-1,self.n_pose)
        self.pred = self.reg.predict(self.eval_feature_norm)
        if self.ismano:
            mesh = KinematicModel('./mano_ik/MANO_RIGHT.pkl', MANOArmature, scale=1000)
            wrapper = KinematicPCAWrapper(mesh, n_pose=self.n_pose)
            keypionts_pred, keypionts_label = [], []
            for p, l in zip(self.pred,self.eval_label):
                _, pkey = mesh.set_params(pose_pca=p, pose_glb=np.zeros(3), shape=np.zeros(10))
                _, lkey = mesh.set_params(pose_pca=l, pose_glb=np.zeros(3), shape=np.zeros(10))
                keypionts_pred.append(pkey)
                keypionts_label.append(lkey)
            keypionts_pred = np.array(keypionts_pred)
            keypionts_label = np.array(keypionts_label)
            error = keypionts_pred - keypionts_label
            error = np.linalg.norm(error,axis=2)
            self.error.append([np.mean(error),np.std(error)])
            pred, eval_label = keypionts_pred, keypionts_label
        else:
            pred = self.pred
            eval_label = self.eval_label
            if self.predmode == 1:
                pred = pred.reshape(-1,21,3)
                eval_label = eval_label.reshape(-1,21,3)
                error = pred-eval_label
                error = np.linalg.norm(error, axis=2)
                error = np.mean(np.abs(error))
                print(error)
            elif self.predmode == 0:
                pred = pred.reshape(-1,24,3)
                eval_label = eval_label.reshape(-1,24,3)
                error = pred[:,0:21,:]-eval_label[:,0:21,:]
                error = np.linalg.norm(error, axis=2)
                error = np.mean(np.abs(error))
                print(error)
        return pred, eval_label, self.pred, self.eval_label

    def visualize(self, pred=None, label=None):
        if pred is None:
            pred = self.pred
            label = self.eval_label
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for pred, label in zip(pred.reshape(-1,21,3), label.reshape(-1,21,3)):
            plt.cla()
            ax.set_xlim(-100,100)
            ax.set_ylim(-100,100)
            ax.set_zlim(-100,100)
            ax.scatter3D(pred[:,0], pred[:,1], pred[:,2], c='red')
            ax.scatter3D(label[:,0], label[:,1], label[:,2], c='blue')
            plt.pause(0.1)
        plt.show()

    ##############
    ## FEATURE ###
    ##############

    def extract_features(self, data):
        # derv = self.get_first_derivative(data)
        slope = self.get_slope(data)
        peak_fq = np.expand_dims(self.get_peak_freq_idx(data),1)
        val_min = np.expand_dims(np.min(data,axis=1),1)
        val_max = np.expand_dims(np.max(data,axis=1),1)
        val_avg = np.expand_dims(np.mean(data,axis=1),1)
        val_std = np.expand_dims(np.std(data,axis=1),1)
        feat = np.concatenate([data,slope,peak_fq,val_min,val_max,val_avg,val_std],axis=1)
        # feat = np.concatenate([peak_fq,val_std],axis=1)
        # feat = np.concatenate([data[:,0:1]],axis=1)
        xticks = [["s11"]*data.shape[1],["slope"]*slope.shape[1],["peak_fq"]*peak_fq.shape[1],["val_min"]*val_min.shape[1],["val_max"]*val_max.shape[1],
                    ["val_avg"]*val_avg.shape[1],["val_std"]*val_std.shape[1]]
        # plt.plot(range(val_avg.shape[0]), data[:,0].flatten())
        # plt.show()
        # exit()
        self.xticks = [x for sublist in xticks for x in sublist]*2
        return feat


    def extract_features2(self, data):
        # derv = self.get_first_derivative(data)
        slope = self.get_slope(data)
        peak_fq = np.expand_dims(self.get_peak_freq_idx(data),1)
        val_min = np.expand_dims(np.min(data,axis=1),1)
        val_max = np.expand_dims(np.max(data,axis=1),1)
        val_avg = np.expand_dims(np.mean(data,axis=1),1)
        val_std = np.expand_dims(np.std(data,axis=1),1)
        # print(data.shape, slope.shape, peak_fq.shape, val_min.shape, val_max.shape, val_avg.shape, val_std.shape)
        feat = np.concatenate([data,slope,val_avg,val_std],axis=1)
        # feat = np.concatenate([val_min,val_std],axis=1)
        xticks = [["s11"]*data.shape[1],["slope"]*slope.shape[1],["peak_fq"]*peak_fq.shape[1],["val_min"]*val_min.shape[1],["val_max"]*val_max.shape[1],
                    ["val_avg"]*val_avg.shape[1],["val_std"]*val_std.shape[1]]
        self.xticks = [x for sublist in xticks for x in sublist]*4
        return feat

    def get_first_derivative(self, data):
        prev, post = data[:-1], data[1:]
        return post-prev

    def get_slope(self, data):
        prev, post = data[:,:-1], data[:,1:]
        return post-prev

    def get_peak_freq_idx(self, data):
        idx = np.argmin(data,axis=1)
        return idx

    def keypoints_to_mano(self, filename, vert_q=None,face_q=None):
        prediction, groundtuth = filename[0], filename[1]
        # load mano
        mesh = KinematicModel('./mano_ik/MANO_RIGHT.pkl', MANOArmature, scale=1000)
        wrapper = KinematicPCAWrapper(mesh, n_pose=self.n_pose)
        # inference loop
        for i, (pose_pca_est, label) in enumerate(zip(prediction, groundtuth)): # TODO: change it to filename's pred and label
            pose_glb_est = np.array([0,0,0])
            shape_est = np.zeros((10))
            if vert_q is not None:
                print(label)
                print(i)
                # print(self.eval_cname_lst[i])
                if self.predmode == 0:
                    rot_est = R.from_matrix(pose_pca_est[-9:].reshape(3,3))
                    rot_est = rot_est.as_rotvec()
                    mesh.set_params(pose_pca=pose_pca_est[:-9], pose_glb=rot_est, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                    rot = R.from_matrix(label[-9:].reshape(3,3))
                    rot = rot.as_rotvec()
                    mesh.set_params(pose_pca=label[:-9], pose_glb=rot, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                elif self.predmode == 1:
                    mesh.set_params(pose_pca=pose_pca_est, pose_glb=pose_glb_est, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                    mesh.set_params(pose_pca=label, pose_glb=pose_glb_est, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                elif self.predmode == 2:
                    rot_est = R.from_matrix(pose_pca_est.reshape(3,3))
                    # rot_est = R.from_matrix(pose_pca_est)
                    rot_est = rot_est.as_rotvec()
                    mesh.set_params(pose_pca=np.zeros(17), pose_glb=rot_est, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                    rot = R.from_matrix(label.reshape(3,3))
                    # rot = R.from_matrix(label)
                    rot = rot.as_rotvec()
                    mesh.set_params(pose_pca=np.zeros(17), pose_glb=rot, shape=shape_est)
                    vert_q.put(mesh.verts)
                    face_q.put(mesh.faces)
                time.sleep(0.1)

    def mat2vec(self, rot):
        if len(rot.shape) <=2:
            rot_inv = np.linalg.inv(rot)
            r = R.from_matrix(rot_inv)
            vec = r.as_rotvec()
        else:
            lst = []
            for rmat in rot:
                rot_inv = np.linalg.inv(rmat)
                r = R.from_matrix(rot_inv)
                vec = r.as_rotvec()
                lst.append(vec)
            vec = np.array(lst)
        return vec

def main():
    # f = "./data/1antenna_position_test/front/2022-03-22-05-44.pkl"
    # fl = "./data/1antenna_position_test/front-left/2022-03-22-04-16.pkl"
    # l = "./data/1antenna_position_test/left/2022-03-22-04-28.pkl"
    # lb = "./data/1antenna_position_test/left-back/2022-03-22-04-36.pkl"
    # b = "./data/1antenna_position_test/back/2022-03-22-04-46.pkl"
    # br = "./data/1antenna_position_test/back-right/2022-03-22-04-54.pkl"
    # r = "./data/1antenna_position_test/right/2022-03-22-05-01.pkl"
    # rf = "./data/1antenna_position_test/right-front/2022-03-22-05-09.pkl"
    # trainfilename = "./data/final/train_pca_est.pkl"
    # evalfilename = "./data/final/test_pca_est.pkl"
    # # trainfilename = "./data/s21_test/2022-03-25-05-02.pkl"
    # # evalfilename = "./data/s21_test/2022-03-25-05-02.pkl"
    # # trainfilename = "./data/final_test/2antennas_front-side_sync/mano/2022-03-16-00-45_pca_est.pkl"
    # # evalfilename = "./data/final_test/2antennas_front-side_sync/reworn/mano/2022-03-17-23-01_pca_est.pkl"
    # # print(trainfilename.split('/')[-2])

    # simpReg = SimpleRegression()
    # # simpReg.load("./data/train.pkl","./data/test.pkl")
    # simpReg.load(trainfilename, evalfilename)
    # simpReg.normalize()
    # simpReg.train()
    # simpReg.eval()

    ######## folder
    pred_all, label_all = [], []
    for pnum in range(1,10):
        foldername = "./data/user_study/p{}/hand_pose_cloth/".format(pnum)
        # foldername = "./data/video_data/".format(pnum)
        fnames = glob.glob("{}2022*.pkl".format(foldername))
        pred, label = [], []
        mano_pred, mano_label = [], []
        for i in range(len(fnames)):
        # for i in range(0,1):
            print(fnames[i])
            # for j in tqdm(range(i,len(fnames))):

            #     if i+j>=len(fnames):
            #         break
            #     train_data_all = combine_dataset(fnames[:i] + fnames[i+1:i+j] + fnames[i+j+1:])
            #     test_data_all = combine_dataset([fnames[i]] + [fnames[i+j]])
            if i+2>=len(fnames):
                break
            train_data_all = combine_dataset(fnames[:i] + fnames[i+2:])
            test_data_all = combine_dataset([fnames[i]] + [fnames[i+1]])
            # train_data_all = combine_dataset(fnames[:i] + fnames[i+1:])
            # test_data_all = combine_dataset([fnames[i]])
            simpReg = SimpleRegression()
            simpReg.load(train_data_all, test_data_all, False)
            simpReg.normalize()
            simpReg.train()
            p, l, mp, lp = simpReg.eval()
            if simpReg.ismano:
                mano_pred.append(mp)
                mano_label.append(lp)
            pred.append(p)
            label.append(l)
        pred = np.concatenate(pred,axis=0)
        label = np.concatenate(label,axis=0)
        try:
            mano_pred = np.concatenate(mano_pred,axis=0)
            mano_label = np.concatenate(mano_label,axis=0)
        except:
            pass


        if simpReg.predmode == 2:
            pred_all.append(pred.reshape(-1,3,3))
            label_all.append(label.reshape(-1,3,3))
            # if not simpReg.ismano:
            #     mano_pred = pred
            #     mano_label = label

            # rot_pred, rot_label = [], []
            # for i in range(mano_label.shape[0]):
            #     rot = R.from_matrix(mano_pred[i].reshape(3,3))
            #     # rot = R.from_quat(mano_pred[i])
            #     theta = rot.as_euler('zyx', degrees=False)[-1]
            #     inv_roll_mat = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
            #     inv_roll_mat = R.from_matrix(inv_roll_mat)
            #     rot = inv_roll_mat*rot
            #     rot = rot.as_euler('zyx', degrees=True)
            #     rot_pred.append(rot)

            #     rot = R.from_matrix(mano_label[i].reshape(3,3))
            #     # rot = R.from_quat(mano_label[i])
            #     theta = rot.as_euler('zyx', degrees=False)[-1]
            #     inv_roll_mat = np.array([[1,0,0],[0,np.cos(theta),np.sin(theta)],[0,-np.sin(theta),np.cos(theta)]])
            #     inv_roll_mat = R.from_matrix(inv_roll_mat)
            #     rot = inv_roll_mat*rot
            #     rot = rot.as_euler('zyx', degrees=True)
            #     rot_label.append(rot)
            # rot_pred = np.array(rot_pred)
            # rot_label = np.array(rot_label)
            # pred_all.append(rot_pred)
            # label_all.append(rot_label)
            # yaw_error = np.abs(rot_pred[:,1]-rot_label[:,1])
            # pitch_error = np.abs(rot_pred[:,0]-rot_label[:,0])
            # yaw_error[yaw_error>=180] = np.abs(yaw_error[yaw_error>=180]-360)
            # pitch_error[pitch_error>=180] = np.abs(pitch_error[pitch_error>=180]-360)
            # print("P{} -  Pitch Error: {} degrees ({}), Yaw Error {} degrees ({})".format(pnum, np.mean(pitch_error),np.std(pitch_error),np.mean(yaw_error),np.std(yaw_error)))

        else:
            if simpReg.ismano:
                pred_all.append(mano_pred)
                label_all.append(mano_label)
            else:
                pred_all.append(pred)
                label_all.append(label)
            error = pred-label
            error = np.linalg.norm(error, axis=2)
            joint_error = np.mean(np.abs(error),axis=0)
            joint_std = np.std(np.abs(error),axis=0)
            # print("Joint Error: {}mm (SD = {})".format(joint_error, joint_std))

            total_error = np.mean(np.abs(error))
            total_std = np.std(np.abs(error))
            print("Total Error: {} mm (SD = {})".format(total_error, total_std))

            # fig = plt.figure()
            # # ax = fig.add_axes([0,0,1,1])
            # if simpReg.ismano:
            #     langs = [
            #         'W', #0
            #         'I0', 'I1', 'I2', #3
            #         'M0', 'M1', 'M2', #6
            #         'L0', 'L1', 'L2', #9
            #         'R0', 'R1', 'R2', #12
            #         'T0', 'T1', 'T2', #15
            #         # extended
            #         'I3', 'M3', 'L3', 'R3', 'T3' #20
            #       ]
            # else:
            #     langs = ['wrist',
            #             'thumb cmc', 'thumb mcp', 'thumb ip', 'thumb tip',
            #             'index mcp', 'index pip', 'index dip', 'index tip',
            #             'middle mcp', 'middle pip', 'middle dip', 'middle tip',
            #             'ring mcp', 'ring pip', 'ring dip', 'ring tip',
            #             'pinky mcp', 'pinky pip', 'pinky dip', 'pinky tip']
            # plt.bar(langs,joint_error, yerr = joint_std)
            # plt.xticks(rotation=90)
            # # ax.bar(ind, menMeans, width, color='r')
            # # ax.bar(ind, womenMeans, width,bottom=menMeans, color='b')
            # plt.tight_layout()
            # plt.show()

    file = pd.DataFrame()
    file['pred'] = pred_all
    file['label'] = label_all
    filename = "joint"
    # file.to_pickle("./result/regression/hand_pose_blazepose_joint_12vs4.pkl")
    file.to_pickle( "./result/regression/hand_pose_cloth_rotation_12vs4.pkl")

    # if simpReg.ismano:
    #     print(mano_label[200])
    #     PoseVisualizer(filename=[mano_pred, mano_label],\
    #                     update=simpReg.keypoints_to_mano,\
    #                     model_filename='./mano_ik/model.pkl')
    # else:
    #     simpReg.visualize(pred,label)

if __name__ == "__main__":
    main()


# ###########
#         1antenna_front    1.37733533734886
# 2antennas_front-side_sync 0.5308756302291949
# 2antennas_front-back_sync 0.5950652406247892
# 2antennas_front-back_simul 1.3134078295150566
#
# 2antennas_front-side_sync (0.53) <= 2antennas_front-back_sync (0.60) < 2antennas_front-back_simul (1.31) < 1antenna_front (1.378)
 ##########

### Euclidean Distance Error ####
# 2antennas_front-back_sync     1.0013546834735174
# 2antennas_front-side_sync     1.0878843749357112
# 2antennas_front-side_simul    2.4072511606071445
# 2antennas_front-back_simul    2.59295754273418
# 1antenna_front                2.7706431136033034
