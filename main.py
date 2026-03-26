# This is a sample Python script.
import datetime
import os
import random
import socket
import sys
import time
from os.path import dirname, abspath
from pathlib import Path

import torch

from src.Configuration import Configuration
from src.FractalTensors import FractalTensors
from src.GeneticAlgorithm import GeneticAlgorithm
from src.MatrixFormula import MatrixFormula
from src.NormVar import NormVar
from src.voxeltoVR import voxel_to_vr_video, voxel_to_vr_video_hires, compile_fractal_step_real
import coolname

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def find_fractals_cuda():
    normed_relative_size1 = NormVar()
    normed_relative_size2 = NormVar()

    normed_relative_size1.set(.1)
    while (True):
        matrix = MatrixFormula((4, 4))
        matrix.create_random_matrix()
        current_timestamp = datetime.datetime.now().timestamp()
        fractal_name = str(current_timestamp).replace(".", "") + coolname.generate_slug()


        fractal = FractalTensors(matrix,fractal_name,(24, 24, 24), (0, 0), 2)


        fractal.compute_fractal()
        # fractal.encode_video_streaming()

        # fractal.videoFractal()

        relative_size = fractal.evaluate_video_size()
        normed_relative_size1.set(relative_size)
        print("relative_size,normed_relative_size", relative_size, normed_relative_size1.get())
        if normed_relative_size1.get() > .8:
            fractal = FractalTensors(matrix,fractal_name,(256, 256, 64), (0, 0), 2)
            fractal.compute_fractal()
            relative_size = fractal.evaluate_video_size()
            normed_relative_size2.set(relative_size)
            if normed_relative_size2.get() > .7:
                fractal = FractalTensors(matrix,fractal_name,(1024, 1024, 300), (0, 0), 2, False)
                # fractal = FractalTensors((128, 128, 8), (0, 0), 2)
                fractal.save_fractal()
                fractal.compute_fractal()
                # fractal.videoFractal()
                fractal.encode_video_streaming()

def schedule_mode(n):
    if n < 1000:
        return FractalTensors.get_dimensions_by_mode('very_low_def')
    if n < 2000:
        return FractalTensors.get_dimensions_by_mode('low_def')
    if n < 3000:
        return FractalTensors.get_dimensions_by_mode('medium_def')
    return FractalTensors.get_dimensions_by_mode('high_def')

# def apply_random_flip_mutation_entire_pool_no_check(pool,probability):
#     fractal, matrix_note = ga.evaluate_matrix(matrix)
#     ga.add_item(fractal.get_fractal(), matrix_note)
#     # Do not mutate n first items (they might be rendered mutated potentially broken)
#     start_index = min(5,len(pool)-1)
#     for fractal in pool[start_index:]:
#         if random.random() < probability:
#             fractal['fractal']['matrix'].mutate_flip_item()
#
#
def generate_VR_fractal():
    conf = Configuration()
    FT = FractalTensors

    #mode = FT.get_dimensions_by_mode('higher_def')
    #mode = FT.get_dimensions_by_mode('medium_def')
    fractal = FractalTensors(conf, None, FT.generate_fractal_name(), (512,512,512), (0, 0), 2)
    bailout = fractal.load_Normalized_Bailout("test.bailout")
    print(bailout)
    voxel_to_vr_video(bailout, output_path="vr2b.mp4", eye_disparity=.08,duration_sec=30,colormap="gist_earth")
    exit(0)
    #filename = "note 552impressive-porcelain-seal-of-excitement.frac"
    filename = "note 543inquisitive-satisfied-tanuki-of-witchcraft.frac"
    filename = "industrious-innocent-sponge-of-glee.frac"
    fractal.load_fractal(fractal_name="", filefolder="/", filename=filename)
    fractal.compute_fractal()
    fractal.save_Normalized_Bailout("test.bailout")
    normalized_bailout = fractal.get_Normalized_bailout()
    print(normalized_bailout)
    voxel_to_vr_video(normalized_bailout,eye_disparity=.1)
def compute_multiple_fractals_from_files(directory):
    conf = Configuration()
    files_list = list(Path(directory).glob("*.frac"))
    print(files_list)
    for file in files_list:
        compute_fractal_from_frac_filename(file)
def generate_multiple_projection_from_files(directory):
    conf = Configuration()
    FT = FractalTensors
    files_list = list(Path(directory).glob("*.bailout"))
    print(files_list)
    equirectangular_modes = [{'mode' : 'alpha_composite', 'chunk' : 256},{'mode' : 'first_hit', 'chunk' : 256},
                             {'mode' : 'isosurface', 'chunk' : 64},{'mode' : 'depth_color', 'chunk' : 256},
                             {'mode' : 'emission_absorption', 'chunk' : 256},{'mode' : 'max_intensity', 'chunk' : 256}]
    for file in files_list:
        #for equirectangular_mode in equirectangular_modes:
        fractal = FractalTensors(conf, None, FT.generate_fractal_name(), (512, 512, 512), (0, 0), 2)
        bailout = fractal.load_Normalized_Bailout(file)
        print(bailout)
        output_filename = str(file)[0:-8]+"M_pointcloud"+".mp4"
        #voxel_to_vr_video(bailout, output_path=output_filename, eye_disparity=.08, duration_sec=1, colormap="gist_earth",equirectangular_mode= None)
        voxel_to_vr_video_hires(bailout, output_path=output_filename, eye_disparity=.08, duration_sec=30, colormap="gist_earth")
        continue
        file_size_bytes = os.path.getsize(output_filename)
        if file_size_bytes > 500000:
            print("Video selected : ",output_filename)
            voxel_to_vr_video(bailout, output_path=output_filename, eye_disparity=.08, duration_sec=30,
                              colormap="gist_earth", equirectangular_mode=None)
        else:
            print("Video rejected : ",output_filename)

def generate_multiple_projection_from_matrices_ray_cast(directory):
    conf = Configuration()
    FT = FractalTensors
    files_list = list(Path(directory).glob("*.frac"))
    print(files_list)

    for file in files_list:
        before_time = time.time()

        #for equirectangular_mode in equirectangular_modes:
        fractal = FractalTensors(conf, None, FT.generate_fractal_name(), (1, 1, 1), (0, 0), 2)
        fractal.load_fractal(fractal_name="",fullpath=file)

        output_filename = str(file)[0:-8]+"M_raycast"+".mp4"
        compiled_matrix = compile_fractal_step_real(fractal.matrix)
        voxel_to_vr_video(compiled_matrix, output_path=output_filename, eye_disparity=conf.eye_disparity, duration_sec=conf.video_length,fps =conf.fps, colormap="nipy_spectral",equirectangular_mode= None)
        after_time = time.time() -before_time
        print("Total time : ",after_time//60," minutes ",after_time % 60," seconds")
        continue
        file_size_bytes = os.path.getsize(output_filename)
        if file_size_bytes > 500000:
            print("Video selected : ",output_filename)
            voxel_to_vr_video(bailout, output_path=output_filename, eye_disparity=.08, duration_sec=30,
                              colormap="gist_earth", equirectangular_mode=None)
        else:
            print("Video rejected : ",output_filename)



def compute_fractal_from_frac_filename(fullpath):
    conf = Configuration()
    FT = FractalTensors
    fractal = FractalTensors(conf, None, FT.generate_fractal_name(), (512, 512, 512), (0, 0), 2)
    fractal.load_fractal(fractal_name="", fullpath=fullpath)
    fractal.compute_fractal()
    bailoutfilename = str(fullpath)[0:-5]+".bailout"
    fractal.save_Normalized_Bailout(bailoutfilename)
    return bailoutfilename


def generate_kindle():
    conf = Configuration()
    n = 0
    FT = FractalTensors
    mode = FT.get_dimensions_by_mode('low_def')
    while True:
        matrix = MatrixFormula(conf, (4, 4))
        matrix.create_random_matrix()
        fractal = FractalTensors(conf, matrix, FT.generate_fractal_name(), mode, (0, 0), 2)
        fractal.compute_fractal()
        relative_size = fractal.evaluate_video_size()
        if relative_size > .4:
            print("Adding " + fractal.fractal_name)
            fractal.save_fractal(filefolder="/kindle/",filename=f"note {relative_size:.3f}".replace(".","").replace("0","")+fractal.fractal_name+".frac")
def load_ga_using_kindle():
    conf = Configuration()
    FT = FractalTensors
    mode = FT.get_dimensions_by_mode('low_def')
    ga = GeneticAlgorithm(conf)
    directory_path = dirname(abspath(__file__))+"/kindle/"
    kindlespaths = os.listdir(directory_path)
    print(kindlespaths)
    kindlespaths = sorted(kindlespaths, reverse=True)
    print(kindlespaths)

    choosen_kindles = random.choices(kindlespaths,k=conf.poolsize)
    #choosen_kindles = kindlespaths[:conf.poolsize]
    print(choosen_kindles)

    for kindle in choosen_kindles:
        # matrix = MatrixFormula(conf, (4, 4))
        # matrix.create_random_matrix()
        print(kindle)
        fractal = FractalTensors(conf, None, FT.generate_fractal_name(), mode, (0, 0), 2)
        fractal.load_fractal(fractal_name="",filefolder="/kindle/",filename=kindle)
        fractal, matrix_note = ga.evaluate_matrix(fractal.matrix)
        ga.add_item(fractal.get_fractal(), matrix_note)
        print(fractal)
    print(ga)
    return ga

def simple_GA():
    conf = Configuration()
    n = 0
    FT = FractalTensors
    mode = FT.get_dimensions_by_mode('low_def')
    # highest_size = 0
    # best_matrix = None
    # ## GENERATE STARTING POOL
    # ga = GeneticAlgorithm(conf)
    # while n< conf.poolsize:
    #     matrix = MatrixFormula(conf,(4, 4))
    #     matrix.create_random_matrix()
    #     fractal = FractalTensors(conf,matrix, FT.generate_fractal_name(), mode, (0, 0), 2)
    #     fractal.compute_fractal()
    #     relative_size = fractal.evaluate_video_size()
    #     if relative_size > .4:
    #         print("Adding "+str(n)+" to pool. Name : ",fractal.fractal_name)
    #         ga.add_item(fractal.get_fractal(), relative_size)
    #         ga.add_item(fractal.get_fractal(), relative_size)
    #         ga.add_item(fractal.get_fractal(), relative_size)
    #
    #         n+=3
    # #ga.scale_notes(1)
    ## END GENERATE STARTING POOL
    ga = load_ga_using_kindle()
    n=0
    while True:
        similarity = ga.sample_pool_similarity()
        print("similarity",similarity)
        if similarity <= .9:
            conf.mutation_flip_probability = 0.0001
        elif similarity < .95:
            conf.mutation_flip_probability = 0.01

        elif similarity > .99:
            generate_hidef(ga, conf,5)
            os.execl(sys.executable, sys.executable, *sys.argv)
        ga.apply_random_flip_mutation_entire_pool()

        if random.random() < conf.crossing_probability:
            # Cross
            ga_item1 = ga.pick_item()
            ga_item2 = ga.pick_item()
            matrices = MatrixFormula(conf,(1,1)).cross_matrices(ga_item1['fractal']['matrix'], ga_item2['fractal']['matrix'])
            for matrix in matrices:
                fractal, matrix_note = ga.evaluate_matrix(matrix)
                ga.add_item(fractal.get_fractal(), matrix_note)

            # End Cross
        ga_item = ga.pick_item()
        matrix = ga_item['fractal']['matrix']
        if random.random() < conf.row_shift_probability:
            matrix.shift_rows()
        if random.random() < conf.mutation_probability:
            # Mutate
            ga_item = ga.pick_item()
            matrix =  ga_item['fractal']['matrix']
            fractal = FractalTensors(conf,ga_item['fractal']['matrix'], FractalTensors.generate_fractal_name(), mode,
                                     (0, 0), 2)
            fractal.compute_fractal()
            highest_note = fractal.evaluate_video_size()
            best_matrix = fractal.matrix
            fractal = fractal.reinit()
            #highest_note = 0
            #best_matrix = None
            for i in range(4):
                #fractal.matrix.mutate_matrix_item(i==0)
                continue_optimization = matrix.mutate_matrix_item(i == 0)
                fractal, matrix_note = ga.evaluate_matrix(matrix)
                ga.add_item(fractal.get_fractal(), matrix_note)

                if matrix_note > highest_note:
                    highest_note = matrix_note
                    best_matrix = fractal.matrix
                #fractal = fractal.reinit()
                if not continue_optimization:
                    break

            fractal.matrix = best_matrix
            ga.add_item(fractal.get_fractal(), highest_note)
            # End mutate
        print(ga)
        # Clean up
        ga.kill_overpooled_items()

        if (n+1) % 1000 == 0:
            generate_hidef(ga, conf,5)
        n += 1

def generate_hidef(ga,conf,count):
    fractals = ga.pick_top_items(count)
    for fractal in fractals:
        afractal = FractalTensors(conf, fractal['fractal']['matrix'], fractal['fractal']['fractal_name'],
                                 FractalTensors.get_dimensions_by_mode('render_desktop'), (0, 0), 2, False)
        if afractal.check_fractal_file_exist():
            continue
        afractal.save_fractal()
        afractal.compute_fractal()
        # fractal.videoFractal()
        afractal.encode_video_streaming()
def test_laptop():
    n = 0
    highest_size = 0
    best_matrix = None
    ga = GeneticAlgorithm()
    while True:

        current_timestamp = datetime.datetime.now().timestamp()
        fractal_name = str(current_timestamp).replace(".", "") + coolname.generate_slug()
        if best_matrix is None:
            matrix = MatrixFormula((4, 4))
            matrix.create_random_matrix()
            fractal = FractalTensors(matrix,fractal_name,(24, 24, 24), (0, 0), 2)
        else:
            fractal = FractalTensors(best_matrix, fractal_name, (24, 24, 24), (0, 0), 2)
        fractal.compute_fractal()
        relative_size = fractal.evaluate_video_size()
        print("relative_size,highest_size",relative_size,highest_size)
        if relative_size > highest_size:
            highest_size = relative_size
            best_matrix = fractal.matrix
        if n % 10 == 0:
            fractal.matrix = best_matrix
            ga.add_item(fractal.get_fractal(), highest_size)
            ga.kill_overpooled_items()
            print("Pool :",ga)
            highest_size = 0
            fractal = ga.pick_item()
            fractal = FractalTensors(fractal['fractal']['matrix'],fractal['fractal']['fractal_name'],(24, 24, 24), (0, 0), 2)
            fractal.matrix.mutate_matrix_item(True)


        else:
            fractal.matrix.mutate_matrix_item(False)
        #print(fractal.matrix)
        n += 1
        #exit(0)
    fractal.save_fractal()
    fractal2 = FractalTensors(None,fractal_name,(24, 24, 24), (0, 0), 2)
    fractal2.load_fractal(fractal_name)


   # fractal2.compute_fractal()

    # fractal.encode_video_streaming()

    # fractal.videoFractal()

    relative_size = fractal.evaluate_video_size()
    # normed_relative_size1.set(relative_size)
    print("relative_size,normed_relative_size", relative_size)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    device_name = socket.gethostname()
    print(f"Device Name: {device_name}")
    if device_name == 'arch-laptop':
        # matrix1 = MatrixFormula((3, 4))
        # matrix1.matrix = [['a','a','a','a'],['a','a','a','a'],['a','a','a','a']]
        # matrix2 = MatrixFormula((4, 4))
        # matrix2.matrix = [['b','b','b','b'],['b','b','b','b'],['b','b','b','b']]
        #
        # new_matrix1,new_matrix2 = MatrixFormula.cross_matrices(matrix1,matrix2)
        # print(new_matrix1)
        # print(new_matrix2)
        #load_ga_using_kindle()
        generate_VR_fractal()
        #simple_GA()
        #test_laptop()
    else:
        #compute_multiple_fractals_from_files("/home/carl/PycharmProjects/MatrixFractals/src/precomputed")
        generate_multiple_projection_from_matrices_ray_cast("/home/carl/PycharmProjects/MatrixFractals/src/precomputed")
        #generate_VR_fractal()

        #load_ga_using_kindle()
        #generate_kindle()
        #simple_GA()
        #find_fractals_cuda()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
