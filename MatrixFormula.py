import random
from typing import List, Tuple
import torch

class MatrixFormula:
    def __init__(self,size: Tuple[int,int]):
        print("Formula Matrix size = ",size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.matrix = [[None for _ in range(size[0])] for _ in range(size[1])]
        self.currently_mutating = [0,0]
        self.types = {
            'complex' : [],
            'function' : ['sin','cos','tan','sinh','cosh','tanh','asin','acos','atan','asinh','acosh','atanh','sinc','exp','square','sqrt','rsqrt','log','angle','sigmoid','inverse','abs','relu'],
            'variable': ['C','D','Z']
        }
        #print(list(self.types.keys()))
        #self.create_random_matrix()

    def __str__(self):
        return_string = "["
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                return_string += str(self.matrix[i][j])
                if not j == self.size[1] -1:
                    return_string += ', '
            if not i == self.size[0] - 1:
                return_string += '\n'
        return return_string + ']'

    def __repr__(self):
        return self.__str__()

    def process_matrix(self,fractal):
        result = fractal.get_empty_window()
        for i in range(self.size[0]):
            # Start with the first item of the line
            line_Z = fractal.get_empty_window() + self.compute_matrix_item(fractal,self.matrix[i][0]) # Use windows gpu C,D and Z
            #print(line_Z.shape)
            for j in range(1,self.size[1]):
                current = self.compute_matrix_item(fractal,self.matrix[i][j])
                #print(line_Z.shape)
                line_Z *= current # * inverse

            result += line_Z # + inverse
        return result

    def create_random_matrix(self):

        for i in range(self.size[1]):
            for j in range(self.size[0]):
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
    @staticmethod
    def cross_matrices(matrix1,matrix2):
        if isinstance(matrix1, MatrixFormula):
            matrix1 = matrix1.matrix
        if isinstance(matrix2, MatrixFormula):
            matrix2 = matrix2.matrix
        matrix1_dimensions = (len(matrix1),len(matrix1[0]))
        matrix2_dimensions = (len(matrix2),len(matrix2[0]))
        if not matrix1_dimensions==matrix2_dimensions:
            print("Matrix dimensions do not match")
            exit(1)
        split_by = random.choice(['column','row'])
        if split_by == 'row':
            split_index = random.randint(1,matrix1_dimensions[0]-1)
            start1 = matrix1[0:split_index]
            start2 = matrix2[0:split_index]
            end1 = matrix1[split_index:]
            end2 = matrix2[split_index:]
            new_matrix1 = start1 + end2
            new_matrix2 = start2 + end1
        else:
            # Same thing but we transpose matrix at start and end
            matrix1 = [*zip(*matrix1)]
            matrix2 = [*zip(*matrix2)]
            split_index = random.randint(1, matrix1_dimensions[1] - 1)
            start1 = matrix1[0:split_index]
            start2 = matrix2[0:split_index]
            end1 = matrix1[split_index:]
            end2 = matrix2[split_index:]
            new_matrix1 = start1 + end2
            new_matrix2 = start2 + end1
            new_matrix1 = [list(i) for i in zip(*new_matrix1)]
            new_matrix2 = [list(i) for i in zip(*new_matrix2)]

        embedded_matrix1 = MatrixFormula(matrix1_dimensions)
        embedded_matrix1.matrix = new_matrix1
        embedded_matrix2 = MatrixFormula(matrix2_dimensions)
        embedded_matrix2.matrix = new_matrix2
        return (embedded_matrix1,embedded_matrix2)

    def mutate_matrix_item(self,change_current_mutation_index=True):
        if change_current_mutation_index:
            iindex, jindex = random.randint(0, self.size[0] - 1), random.randint(0, self.size[0] - 1)
            self.currently_mutating = [iindex, jindex]
        else:
            iindex, jindex = (self.currently_mutating[0],self.currently_mutating[1])

        #print(iindex,jindex)
        item = self.matrix[iindex][jindex]
        # Here we ha a tensor
        if torch.is_tensor(item):
            #print('complex')
            self.matrix[iindex][jindex] = item * torch.complex(3 * torch.rand(1, device=self.device) - 1.5, 3 * torch.rand(1, device=self.device) - 1.5)
        # Here we have a function
        elif '(' in item:
            options = ['flip_polarity','switch_function','switch_variable']
            option = random.choice(options)
            if option == 'flip_polarity':
                if '-' in item:
                    item.replace("-", "")
                else:
                    item = "-" + item
            elif option == 'switch_function':
                fn = item.split('(')[0].replace('-', '')
                new_fn = random.choice(self.types['function'])
                item.replace(fn,new_fn)
            elif option == 'switch_variable':
                options = ['flip_polarity', 'switch_variable']
                option = random.choice(options)
                if option == 'flip_polarity':
                    if '-' in item:
                        item.replace("-", "")
                    else:
                        item = "-" + item
                elif option == 'switch_variable':
                    var = item.split('(')[1][:-1]
                    new_var = random.choice(self.types['variable'])
                    item.replace(var, new_var)
            else:
                exit(1)

            self.matrix[iindex][jindex] = item
        # Here we have a variable
        else:
            options = ['flip_polarity', 'switch_variable']
            option = random.choice(options)
            if option == 'flip_polarity':
                if '-' in item:
                    item.replace("-", "")
                else:
                    item = "-" + item
            elif option == 'switch_variable':
                var = item.replace("-","")
                new_var = random.choice(self.types['variable'])
                item.replace(var, new_var)
            else:
                exit(1)
            self.matrix[iindex][jindex] = item
            #print('variable')

    def compute_matrix_item(self,fractal,item):
        # Here we have a complex
        if torch.is_tensor(item):
            #print('complex')
            return item
        # Here we have a function
        elif '(' in item:
            #print('function')
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
                            return minus * torch.sin(fractal.get_Z_var())
                case 'cos':
                    match var:
                        case 'C':
                            return minus*torch.cos(fractal.get_C_var())
                        case 'D':
                            return minus*torch.cos(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.cos(fractal.get_Z_var())
                case 'tan':
                    match var:
                        case 'C':
                            return minus * torch.tan(fractal.get_C_var())
                        case 'D':
                            return minus * torch.tan(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.tan(fractal.get_Z_var())
                case 'sinh':
                    match var:
                        case 'C':
                            return minus * torch.sinh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.sinh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sinh(fractal.get_Z_var())
                case 'cosh':
                    match var:
                        case 'C':
                            return minus * torch.cosh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.cosh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.cosh(fractal.get_Z_var())
                case 'tanh':
                    match var:
                        case 'C':
                            return minus * torch.tanh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.tanh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.tanh(fractal.get_Z_var())
                case 'asin':
                    match var:
                        case 'C':
                            return minus * torch.asin(fractal.get_C_var())
                        case 'D':
                            return minus * torch.asin(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.asin(fractal.get_Z_var())
                case 'acos':
                    match var:
                        case 'C':
                            return minus * torch.acos(fractal.get_C_var())
                        case 'D':
                            return minus * torch.acos(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.acos(fractal.get_Z_var())
                case 'atan':
                    match var:
                        case 'C':
                            return minus * torch.atan(fractal.get_C_var())
                        case 'D':
                            return minus * torch.atan(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.atan(fractal.get_Z_var())
                case 'asinh':
                    match var:
                        case 'C':
                            return minus * torch.asinh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.asinh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.asinh(fractal.get_Z_var())
                case 'acosh':
                    match var:
                        case 'C':
                            return minus * torch.acosh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.acosh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.acosh(fractal.get_Z_var())
                case 'atanh':
                    match var:
                        case 'C':
                            return minus * torch.atanh(fractal.get_C_var())
                        case 'D':
                            return minus * torch.atanh(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.atanh(fractal.get_Z_var())
                case 'sinc':
                    match var:
                        case 'C':
                            return minus * torch.sinc(fractal.get_C_var())
                        case 'D':
                            return minus * torch.sinc(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sinc(fractal.get_Z_var())
                case 'exp':
                    match var:
                        case 'C':
                            return minus * torch.exp(fractal.get_C_var())
                        case 'D':
                            return minus * torch.exp(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.exp(fractal.get_Z_var())
                case 'square':
                    match var:
                        case 'C':
                            return minus*torch.square(fractal.get_C_var())
                        case 'D':
                            return minus*torch.square(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.square(fractal.get_Z_var())
                case 'sqrt':
                    match var:
                        case 'C':
                            return minus*torch.sqrt(fractal.get_C_var())
                        case 'D':
                            return minus*torch.sqrt(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sqrt(fractal.get_Z_var())
                case 'rsqrt':
                    match var:
                        case 'C':
                            return minus * torch.rsqrt(fractal.get_C_var())
                        case 'D':
                            return minus * torch.rsqrt(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.rsqrt(fractal.get_Z_var())
                case 'log':
                    match var:
                        case 'C':
                            return minus * torch.log(fractal.get_C_var())
                        case 'D':
                            return minus * torch.log(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.log(fractal.get_Z_var())
                case 'angle':
                    match var:
                        case 'C':
                            return minus * torch.angle(fractal.get_C_var())
                        case 'D':
                            return minus * torch.angle(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.angle(fractal.get_Z_var())
                case 'sigmoid':
                    match var:
                        case 'C':
                            return minus * torch.sigmoid(fractal.get_C_var())
                        case 'D':
                            return minus * torch.sigmoid(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.sigmoid(fractal.get_Z_var())
                case 'inverse':
                    match var:
                        case 'C':
                            return minus * 1/fractal.get_C_var()
                        case 'D':
                            return minus * 1/fractal.get_D_var()
                        case 'Z':
                            return minus * 1/fractal.get_Z_var()
                case 'abs':
                    match var:
                        case 'C':
                            return minus * torch.abs(fractal.get_C_var())
                        case 'D':
                            return minus * torch.abs(fractal.get_D_var())
                        case 'Z':
                            return minus * torch.abs(fractal.get_Z_var())
                case 'relu':
                    match var:
                        case 'C':
                            return minus * torch.complex(torch.relu(fractal.get_C_var().real), torch.relu(fractal.get_C_var().imag))
                        case 'D':
                            return minus * torch.complex(torch.relu(fractal.get_D_var().real), torch.relu(fractal.get_D_var().imag))
                        case 'Z':
                            return minus * torch.complex(torch.relu(fractal.get_Z_var().real), torch.relu(fractal.get_Z_var().imag))


                case _:
                    print("fn does not exist ",fn)
                    exit()
        # Here we have a variable
        else:
            #print('variable')
            minus = 1
            if '-' in item:
                minus = -1
            if "C" in item:
                #print("C")
                return minus * fractal.get_C_var()

            if "D" in item:
                #print("D")
                return minus * fractal.get_D_var()
            if "Z" in item:
                #print("Z")
                return minus * fractal.get_Z_var()

