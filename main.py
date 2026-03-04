# This is a sample Python script.
import datetime
import random
import socket

from src.Configuration import Configuration
from src.FractalTensors import FractalTensors
from src.GeneticAlgorithm import GeneticAlgorithm
from src.MatrixFormula import MatrixFormula
from src.NormVar import NormVar
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

def simple_GA():
    conf = Configuration()
    n = 0
    FT = FractalTensors
    mode = FT.get_dimensions_by_mode('very_low_def')
    highest_size = 0
    best_matrix = None
    ga = GeneticAlgorithm()
    while n< conf.poolsize:
        matrix = MatrixFormula((4, 4))
        matrix.create_random_matrix()
        fractal = FractalTensors(matrix, FT.generate_fractal_name(), mode, (0, 0), 2)
        fractal.compute_fractal()
        relative_size = fractal.evaluate_video_size()
        if relative_size > .3:
            print("Adding "+str(n)+" to pool. Name : ",fractal.fractal_name)
            ga.add_item(fractal.get_fractal(), relative_size)
            n+=1
    ga.scale_notes(0.5)

    n=0
    while True:
        old_mode = mode
        mode = schedule_mode(n)
        if not old_mode == mode:
            ga.scale_notes(0.5)

        if random.random() < conf.crossing_probability:
            # Cross
            ga_item1 = ga.pick_item()
            ga_item2 = ga.pick_item()
            matrices = MatrixFormula.cross_matrices(ga_item1['fractal']['matrix'], ga_item2['fractal']['matrix'])
            for matrix in matrices:
                fractal = FractalTensors(matrix, FractalTensors.generate_fractal_name(), mode,
                                         (0, 0), 2)
                fractal.compute_fractal()
                video_size = fractal.evaluate_video_size()
                print(video_size)
                ga.add_item(fractal.get_fractal(), video_size)

            # End Cross
        if random.random() < conf.mutation_probability:
            # Mutate
            ga_item = ga.pick_item()
            fractal = FractalTensors(ga_item['fractal']['matrix'], FractalTensors.generate_fractal_name(), mode,
                                     (0, 0), 2)
            fractal.compute_fractal()
            highest_size = fractal.evaluate_video_size()
            best_matrix = fractal.matrix
            fractal = fractal.reinit()
            for i in range(4):
                fractal.matrix.mutate_matrix_item(i==0)
                fractal.compute_fractal()
                relative_size = fractal.evaluate_video_size()
                print("relative_size,highest_size", relative_size, highest_size)
                if relative_size > highest_size:
                    highest_size = relative_size
                    best_matrix = fractal.matrix
                fractal = fractal.reinit()

            fractal.matrix = best_matrix
            ga.add_item(fractal.get_fractal(), highest_size)
            # End mutate
        print(ga)
        # Clean up
        ga.kill_overpooled_items()

        if n > 1000 and (n+1) % 400 == 0:
            fractal = ga.pick_top_item()
            fractal = FractalTensors(fractal['fractal']['matrix'], fractal['fractal']['fractal_name'], FT.get_dimensions_by_mode('render_desktop'), (0, 0), 2, False)
            fractal.save_fractal()
            fractal.compute_fractal()
            # fractal.videoFractal()
            fractal.encode_video_streaming()
        n += 1

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
        simple_GA()
        #test_laptop()
    else:
        simple_GA()
        #find_fractals_cuda()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
