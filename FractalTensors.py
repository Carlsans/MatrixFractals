import datetime
import gc
import subprocess
from collections import namedtuple
from os.path import dirname, abspath
from typing import Tuple

import numpy as np
import torch
from mpmath.libmp import normalize
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
        self.parameters = dimensions[0]*dimensions[1]*dimensions[2]
        self.window_size = 40000000
        self.current_window = 0
        self.max_window = int((dimensions[0]*dimensions[1]*dimensions[2])/self.window_size)+1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bailout = torch.zeros(self.dimensions, dtype=torch.int16, device=torch.device("cpu"), requires_grad=False)
        self.bailoutexitvalues = torch.zeros(self.dimensions, dtype=torch.complex128, device=torch.device("cpu"), requires_grad=False)

        # Fractal shape is [dimensions.x,dimensions.y,,dimensions.z,2] the last dimension account for variable C and D. Then reduced to [dimensions.x,dimensions.y,,dimensions.z] in Z
        self.Z = torch.zeros([self.dimensions.x,self.dimensions.y,self.dimensions.t], dtype=torch.complex128, device=torch.device("cpu"), requires_grad=False)
        #self.C = torch.zeros([self.dimensions.x,self.dimensions.y,self.dimensions.t], dtype=torch.complex128, device=torch.device("cpu"), requires_grad=False)
        self.D = torch.zeros([self.dimensions.x,self.dimensions.y,self.dimensions.t], dtype=torch.complex128, device=torch.device("cpu"), requires_grad=False)

        rangereal = (center[0] - range, center[0] + range)
        rangeimag = (center[1] - range, center[1] + range)
        #real_linspace = torch.linspace(rangereal[0], rangereal[1], self.dimensions.y, device=self.device) * torch.ones(self.dimensions.x, 1, 1, device=self.device)
        real_linspace = torch.linspace(rangereal[0], rangereal[1], self.dimensions.y, device=torch.device("cpu")).reshape(1,-1,1)
        #print("real_linspace.shape",real_linspace.shape)
        self.C_real = torch.linspace(rangereal[0], rangereal[1], self.dimensions.y,
                                     dtype=torch.complex128, device=torch.device("cpu")
                                     ).reshape(1, -1, 1)  # shape (1, y, 1)

        self.C_imag = (torch.linspace(rangeimag[0], rangeimag[1], self.dimensions.x,
                                      dtype=torch.complex128, device=torch.device("cpu")
                                      ) * 1j).reshape(-1, 1, 1)  # shape (x, 1, 1)
        self.C = self.C_real + self.C_imag

        # self.C[:,:,:] += real_linspace
        #
        # imag_linspace = torch.linspace(rangeimag[0], rangeimag[1], self.dimensions.x, device=torch.device("cpu")).reshape(-1, 1, 1) * torch.ones(
        #     1, self.dimensions.y, 1, device=torch.device("cpu")) * 1j
        # self.C[:,:,:] += imag_linspace



        print("c shape before addition",(self.C_real + self.C_imag).shape)
        print("c shape before added",self.C.shape)

        time = tf.time_from_sin(-3,3,self.dimensions.t,True) * torch.ones(1, 1, self.dimensions.t, device=torch.device("cpu"))
        # Note Pas certain que la forme est bonne. Le tenseur est trop homogene
        #print( "time ",time.shape)
        self.D = time.expand(self.dimensions.x, self.dimensions.y, self.dimensions.t)
        #self.D[:, :, :] += time
        #print(self.C)



    def flatten_all_tensors(self):
        self.bailout = torch.flatten(self.bailout)
        self.bailoutexitvalues = torch.flatten(self.bailoutexitvalues)
        self.Z = torch.flatten(self.Z)
        self.C = torch.flatten(self.C).repeat_interleave(self.dimensions.t)
        print("self.C flatenned",self.C.shape)
        self.D = torch.flatten(self.D)

    def reshape_all_tensors(self,only_bailout = False):
        self.bailout = self.bailout.reshape(self.dimensions)
        self.bailoutexitvalues = self.bailoutexitvalues.reshape(self.dimensions)
        if only_bailout:
            return
        self.Z = self.Z.reshape(self.dimensions)
        self.C = self.C.reshape(self.dimensions)
        self.D = self.D.reshape(self.dimensions)

    # Compute the fractal. First flatten all the tensor so you don't bust memory for larger tensors
    # Split them by windows, C and D window variables are cached since they are constant. The cache is reset each time we open a new window
    # We iterate until max_iter is up. At each iteration, we calculate Z using the MatrixFormula processing matrix
    # At each iteration we do a bailout test
    def compute_fractal(self, matrix):
        fractal.flatten_all_tensors()
        for window_count in range(self.max_window):
            #self.cached_gpu_reset()
            self.current_window = window_count
            window = self.get_window()
            self.window_bailout = self.get_empty_window(dtype = torch.int16)
            self.window_bailout_exit_values = self.get_empty_window(dtype = torch.complex128)
            self.window_Z = self.Z[window['start']:window['stop']].to(device=self.device)
            self.window_C = self.C[window['start']:window['stop']].to(device=self.device)
            self.window_D = self.D[window['start']:window['stop']].to(device=self.device)
            print("window", window)
            for i in range(2, self.max_iter + 1):
                self.window_Z = matrix.process_matrix(self)
                bailout_result = self.bailouttest(i)
                # Something wrong here
                if not bailout_result:
                    break
                self.addnotbailedout()
            self.bailout[window['start']:window['stop']] = self.window_bailout
            self.bailoutexitvalues[window['start']:window['stop']] = self.window_bailout_exit_values
        self.reshape_all_tensors()

        print("bailout at iteration",self.bailout)
        print("bailout shape",self.bailout.shape)
        print("bailout exit values", self.bailoutexitvalues)

        #for layer in range(self.dimensions.t):
        #    self.drawFractal(filename,layer)
    # Handle fractal bailout
    def bailouttest(self, iteration):
        # if self.Z is None or not torch.is_tensor(self.Z):
        #     return False
        # if self.Z.shape == torch.Size([1]) or self.Z.shape == torch.Size([]):
        #     return False
        window = self.get_window()
        isbailingout = self.window_Z.real + self.window_Z.imag > 4
        #print("self.bailout",self.bailout.shape,"isbailingout",isbailingout.shape)
        self.window_bailout[isbailingout] = iteration
        self.window_bailout_exit_values[isbailingout] = self.window_Z[isbailingout]
        # Here, Z might only be a complex number
        if torch.is_tensor(self.window_Z):
            self.window_Z[isbailingout] = float('nan') + float('nan') * 1j
        return True

    # Check values that are bailout and put them in the bailed out exit values tensor
    def addnotbailedout(self):
        if self.Z == None:
            return
        nonbailedout = self.window_Z.real + self.window_Z.imag > 4
        if isinstance(nonbailedout,bool):
            return
        print("nonbailedout.shape",nonbailedout.shape)
        if not(nonbailedout.shape == self.window_bailout_exit_values.shape and self.window_bailout_exit_values.shape == self.window_Z.shape):
            return
        self.window_bailout_exit_values[nonbailedout] = self.window_Z[nonbailedout]

    # Save a 3d tensor as a video using bailout and exit values. A timestamp identify the file uniquely
    def videoFractal(self):
        video_name = str(datetime.datetime.now().timestamp()).replace(".","")

        filepath = dirname(abspath(__file__)) + "/matrix_images/"
        filepathandname = filepath + video_name + str(self.dimensions.x) + "x" + str(self.dimensions.y) + "F" + str(
            str(self.dimensions.t)) + ".mp4"

        encoder = VideoEncoder(self.prepare_tensor_for_video(),frame_rate=10)
        encoder.to_file(filepathandname)

    def encode_video_streaming(self):
        """
        frame_generator: yields one frame at a time as a numpy array (H, W, 3) uint8
        """
        video_name = str(datetime.datetime.now().timestamp()).replace(".","")
        filepath = dirname(abspath(__file__)) + "/matrix_images/"
        filepathandname = filepath + video_name + str(self.dimensions.x) + "x" + str(self.dimensions.y) + "F" + str(
            str(self.dimensions.t)) + ".mp4"
        fps = 10
        np_tensor = self.prepare_tensor_for_streaming()  # [x,y,3,t]
        width = np_tensor.shape[1]
        height = np_tensor.shape[0]
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', 'png',
            '-pix_fmt', 'yuv420p',
            filepathandname
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        print("np tensor shape",np_tensor.shape)
        for t in range(np_tensor.shape[3]):
            print("frame :",t)
            frame = np_tensor[:, :, :, t]  # (x, y, 3)
            print("frame",frame)
            print("frame shape",frame.shape)
            frame = np.ascontiguousarray(frame)  # ensure contiguous memory layout
            process.stdin.write(frame.tobytes())
            # frame: torch tensor (H, W, 3) uint8


        process.stdin.close()
        process.wait()

    # Mesure a video size divided by the multiplication of dimensions, mainly for information content mesurement
    def evaluate_video_size(self):
        encoder = VideoEncoder(self.prepare_tensor_for_video(),frame_rate=10)
        buf = SizeTrackingBuffer()
        encoder.to_file_like(buf, format="mp4", codec="libx264", crf=23)
        #print(f"Encoded file size: {buf.bytes_written} bytes")
        #print(f"Encoded file size: {buf.bytes_written / 1024:.2f} KB")
        tensor_parameters_count = self.dimensions.x*self.dimensions.y*self.dimensions.t
        return buf.bytes_written / tensor_parameters_count

    # Create a tensor corresponding to colors formed from the bailout and the bailout exit values tensor
    # Use a colormap and fix the tensor for the VideoEncoder
    def prepare_tensor_for_video(self):
        # Drop big tensors
        del self.D, self.C, self.Z
        # Drop temporary tensors too
        del self.window_bailout_exit_values,self.window_bailout, self.window_Z,self.window_C,self.window_D
        # Drop misc
        del self.C_real,self.C_imag
        gc.collect()
        torch.cuda.empty_cache()

        magnitude = self.bailoutexitvalues.abs()
        normalize_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        normalized_bailout = (self.bailout - self.bailout.min()) / (self.bailout.max() - self.bailout.min())
        to_video = (normalize_magnitude+normalized_bailout)/2
        colormap = colormaps.get_cmap("gist_earth")
        to_video = colormap(to_video.cpu().numpy())

        to_video = torch.from_numpy((to_video[..., :3] * 255).astype(np.uint8))  # (dim.x, dim.y, dim.t, 3)
        return to_video.permute(2, 3, 0, 1)

    def prepare_tensor_for_streaming(self):
        # Drop big tensors
        del self.D, self.C, self.Z
        # Drop temporary tensors too
        del self.window_bailout_exit_values, self.window_bailout, self.window_Z, self.window_C, self.window_D
        # Drop misc
        del self.C_real, self.C_imag
        gc.collect()
        torch.cuda.empty_cache()
        print("Calculating bailout")
        magnitude = self.bailoutexitvalues.abs()
        #normalize_magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
        normalized_bailout = (self.bailout - self.bailout.min()) / (self.bailout.max() - self.bailout.min())
        print("normalized_bailout",normalized_bailout)
        to_video = normalized_bailout
        colormap = colormaps.get_cmap("gist_earth")
        print("Cleaning memory")
        del magnitude, normalized_bailout
        gc.collect()
        print("Memory cleaned, returning transpose")
        colormaptensor = colormap(to_video.cpu().numpy())*255
        print("colormaptensor",colormaptensor)
        #(128, 128, 8, 4)
        return np.transpose((colormaptensor[:,:,:,:3].astype(np.uint8)),(0,1,3,2))


    # Draw a single layer
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

    # Empty 1D tensor the size of the window
    def get_empty_window(self,dtype=torch.complex128):
        window = self.get_window()
        current_window_size = window['stop']-window['start']
        return torch.zeros([current_window_size], dtype=dtype,
                    device=self.device, requires_grad=False)

    # Splitting tensor by window to minimize memory use, each window has a fixed size
    def get_window(self):
        start = self.current_window*self.window_size
        stop = (self.current_window+1)*self.window_size
        if stop > self.parameters:
            stop = self.parameters
        return {'start' : start,  'stop' : stop}
    # C and D Do not change so they can be cached
    def cached_gpu_reset(self):
        self.cached_C = None
        self.cached_D = None

    def get_C_var(self):

        return self.window_C

    def get_D_var(self):
        return self.window_D

    def get_Z_var(self):
        return self.window_Z

while(True):
    fractal = FractalTensors((128,128,8),(0,0),2)


    matrix = MatrixFormula((8, 8))

    matrix.create_random_matrix()
    fractal.compute_fractal(matrix)
    #fractal.encode_video_streaming()

#fractal.videoFractal()
    relative_size = fractal.evaluate_video_size()
    print("relative_size",relative_size)
    if relative_size > .23:
        fractal = FractalTensors((1024, 1024, 300), (0, 0), 2)
        #fractal = FractalTensors((128, 128, 8), (0, 0), 2)
        fractal.compute_fractal(matrix)
        #fractal.videoFractal()
        fractal.encode_video_streaming()


