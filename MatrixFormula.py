import random
from typing import List, Tuple
import torch

class MatrixFormula:
    def __init__(self,size: Tuple[int,int]):
        print("Formula Matrix size = ",size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.matrix = [[None for _ in range(size[0])] for _ in range(size[1])]
        self.types = {
            'complex' : [],
            'function' : ['sin','cos','square','sqrt','log','inverse'],
            'variable': ['C','D','Z']
        }
        #print(list(self.types.keys()))
        self.create_random_matrix()
    def process_matrix(self,fractal):
        result = fractal.get_empty_Z()
        for i in range(self.size[0]):
            # Start with the first item of the line
            line_Z = fractal.get_empty_Z() + self.compute_matrix_item(fractal,self.matrix[i][0])
            #print(line_Z.shape)
            for j in range(1,self.size[1]):
                current = self.compute_matrix_item(fractal,self.matrix[i][j])
                #print(line_Z.shape)
                line_Z *= current

            result += line_Z
        return result

    def create_random_matrix(self):

        for i in range(self.size[0]):
            for j in range(self.size[1]):
                choice = random.choice(list(self.types.keys()))
                #print(choice)
                match choice:
                    case 'complex':
                        real = 3 * torch.rand(1,device=self.device) - 1.5
                        imag = 3 * torch.rand(1,device=self.device) - 1.5
                        value = torch.complex(real, imag)
                    case 'function':
                        minus_sign = random.choice(['', '-'])
                        selected_function = random.choice(self.types['function'])
                        selected_variable = random.choice(self.types['variable'])
                        value = minus_sign+selected_function+"("+selected_variable+")"
                    case 'variable':
                        minus_sign = random.choice(['','-'])
                        selected_variable = random.choice(self.types['variable'])
                        value = minus_sign + selected_variable
                self.matrix[i][j] = value
                #self.compute_matrix_item(self.matrix[i][j])
        print(self.matrix)
    def compute_matrix_item(self,fractal,item):
        # Here we have a complex
        if torch.is_tensor(item):
            print('complex')
            return item
        # Here we have a function
        elif '(' in item:
            print('function')
            minus = 1
            if '-' in item:
                minus = -1
            fn = item.split('(')[0].replace('-','')
            var = item.split('(')[1][:-1]
            match fn:
                case 'sin':
                    match var:
                        case 'C':
                            return minus*torch.sin(fractal.get_C_var())
                        case 'D':
                            return minus*torch.sin(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sin(fractal.Z)
                case 'cos':
                    match var:
                        case 'C':
                            return minus*torch.cos(fractal.get_C_var())
                        case 'D':
                            return minus*torch.cos(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.cos(fractal.Z)
                case 'square':
                    match var:
                        case 'C':
                            return minus*torch.square(fractal.get_C_var())
                        case 'D':
                            return minus*torch.square(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.square(fractal.Z)
                case 'sqrt':
                    match var:
                        case 'C':
                            return minus*torch.sqrt(fractal.get_C_var())
                        case 'D':
                            return minus*torch.sqrt(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sqrt(fractal.Z)
                case 'log':
                    match var:
                        case 'C':
                            return minus * torch.log(fractal.get_C_var())
                        case 'D':
                            return minus * torch.log(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.log(fractal.Z)
                case 'inverse':
                    match var:
                        case 'C':
                            return minus * 1/fractal.get_C_var()
                        case 'D':
                            return minus * 1/fractal.get_D_var()
                        case 'Z':
                            return minus * 1/fractal.Z
        # Here we have a variable
        else:
            print('variable')
            minus = 1
            if '-' in item:
                minus = -1
            if "C" in item:
                print("C")
                return minus * fractal.get_C_var()

            if "D" in item:
                print("D")
                return minus * fractal.get_D_var()
            if "Z" in item:
                print("Z")
                return minus * fractal.Z

