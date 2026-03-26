import random
import statistics
from operator import itemgetter

import numpy as np
import torch

from src.Configuration import Configuration
from src.FractalTensors import FractalTensors
from src.NormVar import NormVar


class GeneticAlgorithm:
    def __init__(self,conf):
        self.conf = conf
        self.pool = []
        self.note_sum = 0
        self.stage1 = NormVar()
        self.stage1.set(0)
        self.stage2 = NormVar()
        self.stage2.set(0)
        self.stage3 = NormVar()
        self.stage3.set(0)
    def add_item(self,item,note):
        self.pool.append({"fractal":item,"note":note})
        self.pool = sorted(self.pool, key=itemgetter('note'),reverse=True)
    def pick_item(self):
        picked = random.choices(self.pool, weights=[item['note'] ** 2 for item in self.pool], k=1)[0]
        print("Picked item note ",picked['note'])
        return picked
    def pick_top_items(self,count):
        self.pool = sorted(self.pool, key=itemgetter('note'), reverse=True)
        print("picked top item",self.pool[0:count])
        return self.pool[0:count]
    def kill_overpooled_items(self):
        count = len(self.pool)-self.conf.poolsize
        if count < 1 :
            return
        self.pool = sorted(self.pool, key=itemgetter('note'),reverse=True)
        self.pool = self.pool[0:self.conf.poolsize]

    def scale_notes(self,scalor):
        for item in self.pool:
            item['note'] = item['note']*scalor
    def apply_random_flip_mutation_entire_pool(self):
        for item in self.pool[1:]:
            if random.random() < self.conf.mutation_flip_probability:
                print("before",item['fractal']['matrix'],"item['note']",item['note'])
                item['fractal']['matrix'].mutate_flip_item()
                _, matrix_note = self.evaluate_matrix(item['fractal']['matrix'])
                item['note'] = matrix_note
                print("after",item['fractal']['matrix'],"item['note']",item['note'])

                print('flip mutation applied',item['fractal']['fractal_name'],matrix_note)
        self.pool = sorted(self.pool, key=itemgetter('note'),reverse=True)


    def evaluate_matrix(self,matrix):
        matrix_note = 0
        fractal_name = FractalTensors.generate_fractal_name()
        mode = FractalTensors.get_dimensions_by_mode('low_def')
        fractal = FractalTensors(self.conf,matrix, fractal_name, mode,
                                 (0, 0), 2,max_iter=100)
        fractal.compute_fractal()
        video_size = fractal.evaluate_video_size()
        self.stage1.set(video_size)
        matrix_note += video_size
        print("stage 1",video_size)
        print("self.stage1.get() ", self.stage1.get())
        if video_size <.4:
            print("Ended stage 1 ",matrix_note)
            return (fractal,matrix_note)

        mode = FractalTensors.get_dimensions_by_mode('medium_def')
        fractal = FractalTensors(self.conf,matrix, fractal_name, mode,
                                 (0, 0), 2,max_iter=100)
        fractal.compute_fractal()
        video_size = fractal.evaluate_video_size()
        self.stage2.set(video_size)
        print("stage 2", video_size)
        matrix_note += video_size * 2
        print("self.stage2.get() ",self.stage2.get() )
        if video_size <.2:
            print("Ended stage 2 ", matrix_note)
            return (fractal,matrix_note)
        mode = FractalTensors.get_dimensions_by_mode('higher_def')
        fractal = FractalTensors(self.conf,matrix,fractal_name, mode,
                                 (0, 0), 2,max_iter=100)
        fractal.compute_fractal()
        video_size = fractal.evaluate_video_size()
        print("stage 3", video_size)
        #self.stage3.set(video_size)
        matrix_note += video_size * 8
        print("Ended stage 3 ", matrix_note)
        return (fractal,matrix_note)

    import statistics

    def sample_pool_similarity(self) -> float:
        """
        Returns a score in [0, 1] indicating how similar the values are.
        - Near 1: values are very similar
        - Near 0: values are very different
        """
        population = [i['note'] for i in self.pool]
        if len(population) < 2:
            raise ValueError("Population must have at least 2 elements.")

        mean = statistics.mean(population)

        if mean == 0:
            # If mean is 0, fall back to std dev normalized by range
            std = statistics.stdev(population)
            value_range = max(population) - min(population)
            if value_range == 0:
                return 1.0  # All values are identical
            cv = std / value_range
        else:
            cv = statistics.stdev(population) / abs(mean)

        return 1 / (1 + cv)

    def __str__(self):
        return_string = ""
        for i in self.pool:
            return_string += "fractal : "+ str(i['fractal']['fractal_name']) + " note : "+str(i['note']) +"\n"
        return  return_string
# ga = GeneticAlgorithm()
# ga.add_item({'fractal_name':"blou1", 'matrix': "matrix1"},.1)
# ga.add_item({'fractal_name':"blou2", 'matrix': "matrix2"},.2)
# ga.add_item({'fractal_name':"blou3", 'matrix': "matrix3"},.4)
# ga.add_item({'fractal_name':"blou4", 'matrix': "matrix4"},.8)
# ga.add_item({'fractal_name':"blou5", 'matrix': "matrix5"},1.6)
# item = ga.pick_item()
# print(item)
# ga.kill_overpooled_items()
# print(ga)