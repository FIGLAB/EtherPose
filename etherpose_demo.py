from nanovna import *
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.image as mpimg
import matplotlib.patheffects as path_effects
import time
import skrf as sk
import sys
import dash_daq as daq
from datetime import datetime
import pickle
import serial
import pandas as pd
from mano_ik.main import Keypoints2Mano
from simple_regressor import SimpleRegression
from scipy.spatial.transform import Rotation as R
import random
from joint_collector import JointCollector
from etherpose_viewer.visualizer import PoseVisualizer
from mano_ik.inverse_kinematics.armatures import *
from mano_ik.inverse_kinematics.models import *

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def get_vna_devices(opt):
    nv = []
    ports = getport_all()
    cnt = 0
    for p in ports:
        device = NanoVNAV2(p)
        nv.append(VNAObject(dotdict(opt.__dict__.copy()), device))
    return nv

class VNAObject():
    def __init__(self, opt, device):
        self.device = device
        self.opt = opt
        self.init_vna()

    def init_vna(self, start=None, stop=None, points=None):
        if start is None:
            start = self.opt.start
        if stop is None:
            stop = self.opt.stop
        if points is None:
            points = self.opt.points
        self.opt.start, self.opt.stop, self.opt.points = start, stop, points
        self.device.set_frequencies(start, stop, points)
        self.device.set_sweep(start, stop, points)

    def get_data(self, port):
        s = self.device.scan()
        s = s[port]
        return s

    def get_data_(self):
        s = self.device.scan()
        return s[0], s[1]

class VNAStream():
    def __init__(self, opt, devices):
        self.init_params()
        self.nv = devices
        self.opt = opt
        if self.opt.plot:
            self.init_plot()
        else:
            self.data_push()

    def init_params(self):
        self.fig = None
        self.isRawImp = False
        self.isSpectrogram = False
        self.isTrain = False
        self.isPred = False
        self.mode = "start"
        self.train_data = []
        self.label_data = []
        self.data = []

        self.jc = JointCollector()
        self.k2m = Keypoints2Mano()

    def on_release(self, event):
        mkey = event.key
        if mkey == " ":
            self.isTrain = False
            self.jc.stop()
            print("[train] End to save")

    def on_press(self, event, force_key=None):
        mkey = event.key
        #--------------------------#
        #        Data Setup        #
        #--------------------------#

        # converting to complex number 
        if mkey == 'i':
            self.isRawImp = not self.isRawImp
        # time data
        elif mkey == "t":
            self.opt.timedomain = not self.opt.timedomain
            if not self.opt.timedomain:
                self.isSpectrogram = False
            self.init_plot()
        # calibration
        elif mkey == "c":
            self.calibration()
        elif mkey == "v":
            self.calibration2()
        elif mkey == "enter":
            self.isPred = not self.isPred
            if self.isPred:
                if self.mode == "handpose":
                    self.set_handpose()
                elif self.mode == "trackpad":
                    self.set_trackpad()

        #--------------------------#
        #        Prediction        #
        #--------------------------#

        # hand pose estimation
        if mkey == " ":
            if not self.isTrain:
                self.jc.run()
            self.isTrain = True
            self.jc.get_image()
            print("[train] collecting...")
        elif mkey == "h":
            if self.mode == "start":
                self.mode = "handpose"
            else:
                self.mode = "start"
            self.jc
        # trackpad example
        elif mkey == "p":
            if self.mode == "start":
                self.mode = "trackpad"
            else:
                self.mode = "start"
        # reset train list
        elif mkey == "r":
            print("refresh training data buffer")
            self.train_data = []
            self.label_data = []

        if self.mode == "start":
            self.axes[1].set_title("h: Hand Pose, p: Track Pad",c='white')
        else:
            self.axes[1].set_title(self.mode,c='white')

    def set_handpose(self):
        print("Hand Pose Start")
        self.simReg = SimpleRegression()
        self.simReg.load_data(self.train_data, self.label_data, len(self.nv))
        self.scaler = self.simReg.normalize()
        self.reg = self.simReg.train()
        PoseVisualizer(filename=None,\
            update=self.keypoints_to_mano,\
            model_filename='./mano_ik/model.pkl',isTrain=False)

    def set_trackpad(self):
        print("TrackPad Start")
        self.simReg = SimpleRegression()
        self.simReg.load_data(self.train_data, self.label_data, len(self.nv))
        self.scaler = self.simReg.normalize()
        self.reg = self.simReg.train()
        self.axes[1].cla()
        self.axes[1].set_ylim(-18, 18)
        self.axes[1].set_xlim(-36, 0)
        self.cursor, = self.axes[1].plot([0], [0],'wo',markersize=30)

    def calibration(self):
        for dev_idx, nv in enumerate(self.nv):
            idx = np.argmin(self.s11[dev_idx])
            freq = nv.device.frequencies[idx]
            width = nv.opt.stop - nv.opt.start
            start_new, stop_new = freq-width/2, freq+width/2
            nv.init_vna(start=start_new, stop=stop_new)
            print("new freq: {} Hz".format((start_new+stop_new)/2))

    def calibration2(self):
        for i in range(self.num_lines):
            recent_y = self.line[i].get_ydata()[-1]
            if not recent_y!=recent_y:
                self.default_value[i] += recent_y
        # print(self.default_value)

    def set_interaction_windows(self):
        plt.rcParams["font.family"] = "Chalkduster"
        self.axes[1].cla()
        self.axes[1].tick_params(axis='x', colors='white')
        self.axes[1].tick_params(axis='y', colors='white')
        self.axes[1].spines['bottom'].set_color('#00000000')
        self.axes[1].spines['top'].set_color('#00000000')
        self.axes[1].spines['left'].set_color('#00000000')
        self.axes[1].spines['right'].set_color('#00000000')
        if self.scenario == "slider":
            self.axes[1].set_facecolor('#ffffff')
            self.axes[1].tick_params(axis='x', colors='#155eab')
            self.axes[1].set_ylim(-5, 5)
            self.axes[1].set_xlim(-10, 110)
        elif self.scenario == "button":
            self.axes[1].set_facecolor('#458edb')
            self.axes[1].set_ylim(-50, 50)
            self.axes[1].set_xlim(-10, 110)
        self.axes[1].axis('off')

    def init_plot(self):
        if self.fig is None:
            self.fig, self.axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [1, 1.5]})
            # self.fig.set_size_inches(15, 4.5)
            self.fig.set_size_inches(8, 5)
            self.fig.subplots_adjust(left=0.05, right=0.98, wspace=0.01)
            self.fig.canvas.mpl_connect('key_press_event', self.on_press)
            self.fig.canvas.mpl_connect('key_release_event', self.on_release)
            anim = animation.FuncAnimation(self.fig, self.animate, interval=1, blit=False)
            self.axes[0].set_facecolor((0, 0, 0))
            self.axes[1].set_facecolor((0, 0, 0))
            self.axes[1].set_title("""h: Hand Pose, p: Track Pad""",c='white')
            self.axes[1].cla()
            self.axes[1].set_ylim(-18, 18)
            self.axes[1].set_xlim(-36, 0)
            self.fig.patch.set_facecolor((0, 0, 0))
            self.cursor, = self.axes[1].plot([0], [0],'wo',markersize=30)

        if self.opt.timedomain:
            if self.isRawImp:
                self.ymax, self.ymin = 0.03, -0.03
            else:
                self.ymax, self.ymin = 4, -4
            start, stop = 0, 50
            num = stop - start
            # start, stop = 0, opt.points
            self.num_lines = self.opt.points*len(self.nv)*2
            # self.scenario = "none"
            # self.set_interaction_windows()
        else:
            if self.isRawImp:
                self.ymax, self.ymin = 0.2, -0.2
            else:
                self.ymax, self.ymin = -0, -40
            # start, stop = 0, self.opt.points
            start, stop = self.opt.start, self.opt.stop
            num = self.opt.points
            self.num_lines = 2*len(self.nv)

        cm = plt.get_cmap('gist_ncar') # gist_rainbow
        self.color = [cm(1.*i/self.num_lines) for i in range(self.num_lines)]
        self.color[0] = (0.2,0.2,1)

        self.axes[0].cla()
        self.axes[0].set_xlim(start, stop)
        self.axes[0].set_ylim(self.ymin, self.ymax)
        self.axes[0].set_title("S11 Raw Signal",fontsize=30,c='white')
        self.axes[0].set_xlabel("frequency (Hz)",fontsize=20,c='white')
        self.axes[0].set_ylabel("return loss (dB, °)",fontsize=20,c='white')
        self.line = []
        self.default_value = []
        for i in range(self.num_lines):
            lineobj, = self.axes[0].plot(np.linspace(start, stop, num),
                               np.ones(num).astype(float)*np.nan,
                               lw=4, color=self.color[i])
            self.line.append(lineobj)
            self.default_value.append(0)

        if self.isSpectrogram:
            self.axes[0].set_ylim(0, self.opt.points-1)
            x, y = np.meshgrid(np.arange(stop-start), np.arange(0,self.opt.points))
            self.sptrgrm = self.axes[0].pcolormesh(x, y,np.zeros(x.shape),shading='nearest', vmin=0, vmax=35)
            self.Sxx = np.ones((stop-start,self.opt.points), dtype=np.float)*np.nan
        # for ax in self.axes:
        #     ax.set_xticks([])
        #     ax.set_yticks([])
        self.axes[0].tick_params(colors='white', which='both')
        plt.show()

    def update_signal(self, x, idx):
        old_y1 = self.line[idx].get_ydata()
        x = [x_ - self.default_value[idx] for x_ in x.tolist()]
        new_y1 = np.r_[old_y1[1:], x]
        self.line[idx].set_ydata(new_y1)

    def data_store(self):
        s11_lst, s11_phase_lst = [], []
        s_lst = []
        s21_lst, s21_phase_lst = [], []
        s__lst = []
        starttime = time.time()
        for dev_idx, nv in enumerate(self.nv):
            # s = nv.get_data(port=0)
            s, s_ = nv.get_data_()
            s11 = self.logmag(s)
            s11_phase = self.phase(s)
            s21 = self.logmag(s_)
            s21_phase = self.phase(s_)
            s11_lst.append(s11)
            s11_phase_lst.append(s11_phase)
            s_lst.append(s)
            s21_lst.append(s21)
            s21_phase_lst.append(s21_phase)
            s__lst.append(s_)
        self.s11, self.s11_phase = s11_lst, s11_phase_lst
        self.s_raw = s_lst
        self.s21, self.s21_phase = s21_lst, s21_phase_lst
        self.s21_raw = s21_lst

    def plot_it(self):
        if self.isRawImp:
            val1, val2 = np.real(self.s_raw), np.imag(self.s_raw)
            val3, val4 = np.real(self.s21_raw), np.imag(self.s21_raw)
        else:
            val1, val2 = self.s11,self.s11_phase
            val3, val4 = self.s21,self.s21_phase
        for dev_idx, (s11, s11_phase, s21, s21_phase) in enumerate(zip(val1, val2, val3, val4)):
            if self.opt.timedomain:
                chunk_size = int(self.num_lines/len(self.nv)/2)
                for i in range(chunk_size):
                    self.update_signal(np.array([s11[i]]), dev_idx*chunk_size*2+i)
                    self.update_signal(np.array([s11_phase[i]]), dev_idx*chunk_size*2+i+chunk_size)
                if self.isSpectrogram:
                    idx = 0 if self.dev_sel is None else self.dev_sel-1
                    self.Sxx = np.r_[self.Sxx[1:], np.array([np.abs(self.s11[idx])])]
                    self.sptrgrm.set_array(self.Sxx.T)
            else:
                self.line[dev_idx*len(self.nv)+0].set_ydata(s11)
                self.line[dev_idx*len(self.nv)+1].set_ydata(s11_phase)

    def interaction_mode(self):
        if self.isTrain and self.mode == "handpose":
            Rhand, _, isRhand, _ = self.jc.get_hands()
            rot = self.jc.get_axis()
            if isRhand:
                # feat = np.concatenate([np.real(self.s_raw),np.imag(self.s_raw)],axis=1)
                feat = np.concatenate([self.s_raw,self.s_raw],axis=1)
                cal_data = np.array(self.default_value).reshape(len(self.nv),-1)
                label = self.k2m.get_mano_params(Rhand)
                # label = np.concatenate([label,rot.flatten()])
                label = np.concatenate([label])
                self.train_data.append(feat)
                self.label_data.append(label)
        elif self.isTrain and self.mode == "trackpad":
            Rhand, _, isRhand, _ = self.jc.get_hands()
            if isRhand:
                feat = np.concatenate([self.s_raw,self.s_raw],axis=1)
                pos = self.jc.get_cursor()[0,0:2]
                self.train_data.append(feat)
                self.label_data.append(pos)
                self.cursor.set_xdata([-pos[1]*500])
                self.cursor.set_ydata([pos[0]*500])

        if self.isPred and self.mode == "trackpad" :
            feat = np.concatenate([self.s_raw,self.s_raw],axis=1)
            feat, _ = self.simReg.load_data(np.array([feat]), np.zeros((1,1)), len(self.nv), isTrain=False)
            if feat is not None:
                feat = self.scaler.transform(feat)
                result = self.reg.predict(feat)[0]
                self.cursor.set_xdata([-result[1]*500])
                self.cursor.set_ydata([result[0]*500])
                # print(result)

    def keypoints_to_mano(self, filename, vert_q=None,face_q=None):
        mesh = KinematicModel('./mano_ik/MANO_RIGHT.pkl', MANOArmature, scale=1000)
        pose_glb_est = np.array([0,0,-np.pi/2])
        shape_est = np.zeros((10))
        if vert_q is not None:
            while True:
                # feat = np.concatenate([np.real(self.s_raw),np.imag(self.s_raw)],axis=1)
                # feat = np.concatenate([self.s11,self.s11_phase],axis=1)
                feat = np.concatenate([self.s_raw,self.s_raw],axis=1)
                cal_data = np.array(self.default_value).reshape(len(self.nv),-1)
                feat, _ = self.simReg.load_data(np.array([feat]), np.zeros((1,1)), len(self.nv), isTrain=False)
                if feat is None:
                    continue
                feat = self.scaler.transform(feat)
                result = self.reg.predict(feat)[0]
                rot = result[-9:].reshape(3,3)
                rot = R.from_matrix(rot).as_rotvec()
                pose = result[:-9]
                # print(pose.shape)
                # print(rot, pose)
                # mesh.set_params(pose_pca=self.predLabel[-1], pose_glb=pose_glb_est, shape=shape_est)
                # pose = np.array([-1.86549734, 0.51486007, -0.00557345, 0.06798197, 0.44848555, -0.86959741,
                #   1.27118212, 2.70635774, 0.43205655, -0.49142564, 1.40765928, -0.13424986,
                #  -0.59755769, -0.44440632,  0.20177723,  0.21892481, -0.18556081, -0.02667932,
                #   0.05261153, -0.9982586,0.32948741, -0.94234772, -0.05847068, -0.94378295,
                #  -0.3304736,0.00780635])
                mesh.set_params(pose_pca=pose, pose_glb=pose_glb_est, shape=shape_est)
                mesh.set_params(pose_pca=result, pose_glb=pose_glb_est, shape=shape_est)
                # mesh.set_params(pose_pca=np.ones(17), pose_glb=rot, shape=shape_est)
                vert_q.put(mesh.verts)
                face_q.put(mesh.faces)
                time.sleep(0.01)
                if not self.isPred:
                    print("Hand Pose Stop")
                    exit()

    def animate(self, frame):
        # starttime = time.time()
        self.data_store()
        self.plot_it()
        self.interaction_mode()
        # print(1/(time.time()-starttime))

    def get_data(self, port):
        s = self.nv.scan()
        s = s[port]
        return s

    def phase(self, x):
        a = np.angle(x)
        a = np.rad2deg(a)
        a = a / 360 * abs(self.ymax-self.ymin)
        a = a + (self.ymin+self.ymax)/2
        return a

    def logmag(self,x):
        return 20*np.log10(np.abs(x))
        # return x

    def linmag(self, x):
        return np.abs(x)

    def groupdelay(self, x):
        gd = np.convolve(np.unwrap(np.angle(x)), [1,-1], mode='same')
        return gd

    def vswr(self, x):
        vswr = (1+np.abs(x))/(1-np.abs(x))
        return vswr

    def polar(self, x):
        return np.angle(x), np.abs(x)

    def tdr(self, x):
        window = np.blackman(len(x))
        NFFT = 256
        td = np.abs(np.fft.ifft(window * x, NFFT))
        t_axis = np.linspace(0, time, NFFT)
        return t_axis, td

    def skrf_network(self, x):
        n = sk.Network()
        n.frequency = sk.Frequency.from_f(self.nv.frequencies / 1e6, unit='mhz')
        n.s = x
        return n

    def smith(self, x):
        n = self.skrf_network(x)
        n.plot_s_smith()
        return n


def main(opt):
    nano_vna = get_vna_devices(opt)
    VNAStream(opt, devices=nano_vna)


if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog: [options]")
    parser.add_option("-V", "--vna", dest="numvna",
                      type="int",default=1,
                      help="number of VNA", metavar="NUMVNA")
    parser.add_option("-t", "--timedomain", dest="timedomain",
                      action="store_true",default=False,
                      help="time domain visualization", metavar="TIMEDOAIM")
    parser.add_option("-r", "--raw", dest="rawwave",
                      type="int", default=None,
                      help="plot raw waveform", metavar="RAWWAVE")
    parser.add_option("-p", "--plot", dest="plot",
                      action="store_true", default=False,
                      help="plot rectanglar", metavar="PLOT")
    parser.add_option("-s", "--smith", dest="smith",
                      action="store_true", default=False,
                      help="plot smith chart", metavar="SMITH")
    parser.add_option("-L", "--polar", dest="polar",
                      action="store_true", default=False,
                      help="plot polar chart", metavar="POLAR")
    parser.add_option("-D", "--delay", dest="delay",
                      action="store_true", default=False,
                      help="plot delay", metavar="DELAY")
    parser.add_option("-G", "--groupdelay", dest="groupdelay",
                      action="store_true", default=False,
                      help="plot groupdelay", metavar="GROUPDELAY")
    parser.add_option("-W", "--vswr", dest="vswr",
                      action="store_true", default=False,
                      help="plot VSWR", metavar="VSWR")
    parser.add_option("-H", "--phase", dest="phase",
                      action="store_true", default=False,
                      help="plot phase", metavar="PHASE")
    parser.add_option("-U", "--unwrapphase", dest="unwrapphase",
                      action="store_true", default=False,
                      help="plot unwrapped phase", metavar="UNWRAPPHASE")
    parser.add_option("-T", "--timedomain2", dest="tdr",
                      action="store_true", default=False,
                      help="plot TDR", metavar="TDR")
    parser.add_option("-c", "--scan", dest="scan",
                      action="store_true", default=False,
                      help="scan by script", metavar="SCAN")
    parser.add_option("-S", "--start", dest="start",
                      type="float", default=1e6,
                      help="start frequency", metavar="START")
    parser.add_option("-E", "--stop", dest="stop",
                      type="float", default=900e6,
                      help="stop frequency", metavar="STOP")
    parser.add_option("-N", "--points", dest="points",
                      type="int", default=101,
                      help="scan points", metavar="POINTS")
    parser.add_option("-P", "--port", type="int", dest="port",
                      help="port", metavar="PORT")
    parser.add_option("-d", "--dev", dest="device",
                      help="device node", metavar="DEV")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="verbose output")
    parser.add_option("-C", "--capture", dest="capture",
                      help="capture current display to FILE", metavar="FILE")
    parser.add_option("-e", dest="command", action="append",
                      help="send raw command", metavar="COMMAND")
    parser.add_option("-o", dest="save",
                      help="write touch stone file", metavar="SAVE")
    (opt, args) = parser.parse_args()

    main(opt)