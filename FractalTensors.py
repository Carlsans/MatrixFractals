import datetime
from collections import namedtuple
from os.path import dirname, abspath
from typing import Tuple

import numpy as np
import torch
from torchcodec.encoders import VideoEncoder
from TimeFunctions import TimeFunctions as tf
from MatrixFormula import MatrixFormula
import matplotlib.pyplot as plt
from matplotlib import colormaps

from src.SizeTrackingBuffer import SizeTrackingBuffer


class FractalTensors:
    def __init__(self,dimensions: Tuple[int,int,int],center: Tuple[float,float],range: float):
        self.max_iter = 100
        Dimension = namedtuple("Dimension", ["x", "y", "t"])
        self.dimensions = Dimension(dimensions[0],dimensions[1],dimensions[2])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bailout = torch.zeros(self.dimensions, dtype=torch.int16, device=self.device, requires_grad=False)
        self.bailoutexitvalues = torch.zeros(self.dimensions, dtype=torch.complex128, device=self.device, requires_grad=False)

        # Fractal shape is [dimensions.x,dimensions.y,,dimensions.z,2] the last dimension account for variable C and D. Then reduced to [dimensions.x,dimensions.y,,dimensions.z] in Z
        self.Z = torch.zeros([self.dimensions.x,self.dimensions.y,self.dimensions.t], dtype=torch.complex128, device=self.device, requires_grad=False)
        self.C = torch.zeros([self.dimensions.x,self.dimensions.y,self.dimensions.t,2], dtype=torch.complex128, device=self.device, requires_grad=False)
        rangereal = (center[0] - range, center[0] + range)
        rangeimag = (center[1] - range, center[1] + range)
        #real_linspace = torch.linspace(rangereal[0], rangereal[1], self.dimensions.y, device=self.device) * torch.ones(self.dimensions.x, 1, 1, device=self.device)
        real_linspace = torch.linspace(rangereal[0], rangereal[1], self.dimensions.y, device=self.device).reshape(1,-1,1)
        #print("real_linspace.shape",real_linspace.shape)
        self.C[:,:,:,0] += real_linspace

        imag_linspace = torch.linspace(rangeimag[0], rangeimag[1], self.dimensions.x, device=self.device).reshape(-1, 1, 1) * torch.ones(
            1, self.dimensions.y, 1, device=self.device) * 1j
        self.C[:,:,:,0] += imag_linspace
        #print(self.C)
        time = tf.time_from_sin(-3,3,self.dimensions.t) * torch.ones(1, 1, self.dimensions.t, device=self.device)
        # Note Pas certain que la forme est bonne. Le tenseur est trop homogene
        #print( "time ",time.shape)
        self.C[:, :, :, 1] += time
        #print(self.C)
    def compute_fractal(self, matrix):

        for i in range(2,self.max_iter+1):
            self.Z = matrix.process_matrix(self)
            result = self.bailouttest(i)
            # Something wrong here
            if not result:
                break
            self.addnotbailedout()

        print("bailout at iteration",self.bailout)
        print("bailout shape",self.bailout.shape)
        print("bailout exit values", self.bailoutexitvalues)

        #for layer in range(self.dimensions.t):
        #    self.drawFractal(filename,layer)

    def bailouttest(self, iteration):
        if self.Z is None or not torch.is_tensor(self.Z):
            return False
        if self.Z.shape == torch.Size([1]) or self.Z.shape == torch.Size([]):
            return False

        isbailingout = self.Z.real + self.Z.imag > 4
        self.bailout[isbailingout] = iteration
        self.bailoutexitvalues[isbailingout] = self.Z[isbailingout]
        # Here, Z might only be a complex number
        if torch.is_tensor(self.Z):
            self.Z[isbailingout] = float('nan') + float('nan') * 1j
        return True
    def addnotbailedout(self):
        if self.Z == None:
            return
        nonbailedout = self.Z.real + self.Z.imag > 4
        if isinstance(nonbailedout,bool):
            return
        print("nonbailedout.shape",nonbailedout.shape)
        if not(nonbailedout.shape == self.bailoutexitvalues.shape and self.bailoutexitvalues.shape == self.Z.shape):
            return
        self.bailoutexitvalues[nonbailedout] = self.Z[nonbailedout]

    def videoFractal(self):
        video_name = str(datetime.datetime.now().timestamp()).replace(".","")

        filepath = dirname(abspath(__file__)) + "/matrix_images/"
        filepathandname = filepath + video_name + str(self.dimensions.x) + "x" + str(self.dimensions.y) + "F" + str(
            str(self.dimensions.t)) + ".mp4"

        encoder = VideoEncoder(self.prepare_tensor_for_video(),frame_rate=10)
        encoder.to_file(filepathandname)

    def evaluate_video_size(self):
        encoder = VideoEncoder(self.prepare_tensor_for_video(),frame_rate=10)
        buf = SizeTrackingBuffer()
        encoder.to_file_like(buf, format="mp4", codec="libx264", crf=23)
        #print(f"Encoded file size: {buf.bytes_written} bytes")
        #print(f"Encoded file size: {buf.bytes_written / 1024:.2f} KB")
        tensor_parameters_count = self.dimensions.x*self.dimensions.y*self.dimensions.t
        return buf.bytes_written / tensor_parameters_count

    def prepare_tensor_for_video(self):
        to_video = (self.bailout - self.bailout.min()) / (self.bailout.max() - self.bailout.min())
        colormap = colormaps.get_cmap("viridis")
        to_video = colormap(to_video.cpu().numpy())
        to_video = torch.from_numpy((to_video[..., :3] * 255).astype(np.uint8))  # (32, 32, 32, 3)
        return to_video.permute(2, 3, 0, 1)

    def drawFractal(self,image_name,layer,saveonly = True,saveblackandwhitetoo= False):
        filepath = dirname(abspath(__file__)) + "/matrix_images/"
        normalizedbailout = self.bailout / torch.max(self.bailout)

        colormap = cm.get_cmap('gist_earth')
        fractalimage = colormap(normalizedbailout[:,:,layer].cpu())

        if saveonly:
            filepathandname = filepath+image_name+str(self.dimensions.x)+"x"+str(self.dimensions.y)+"l"+str(layer)+".png"
            print("filepathandname",filepathandname)
            plt.imsave(filepathandname,fractalimage)
        else:
            plt.imshow(fractalimage)
            plt.show()
        # TODO
        if saveblackandwhitetoo:
            colormap = cm.get_cmap('binary')
            fractalimage = colormap(normalizedbailout.cpu())
            plt.imsave(self.conf.generateTimestamp(prefix="bin",suffix="r"+str(self.resolution[0])+"x"+str(self.resolution[1])),fractalimage)


    def get_empty_Z(self):
        return torch.zeros([self.dimensions.x, self.dimensions.y, self.dimensions.t], dtype=torch.complex128,
                    device=self.device, requires_grad=False)
    def get_C_var(self):
        return self.C[:,:,:,0]

    def get_D_var(self):
        return self.C[:,:,:,1]

while(True):
    fractal = FractalTensors((32,32,32),(0,0),2)
    matrix = MatrixFormula((4, 4))
    matrix.create_random_matrix()
    fractal.compute_fractal(matrix)
    relative_size = fractal.evaluate_video_size()
    print("relative_size",relative_size)
    # if relative_size > .5:
    #     fractal = FractalTensors((512, 512, 150), (0, 0), 2)
    #     fractal.compute_fractal(matrix)
    #     fractal.videoFractal()


