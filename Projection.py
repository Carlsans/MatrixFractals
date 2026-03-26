import torch


class Projection:
    def __init__(self,tensor,origin):
        self.tensor = tensor
        self.origin = origin

    def affine(self,origin,destination,y):
        slope = (destination[1] - origin[1])/(destination[0] - origin[0])
        b = origin[1]-slope*origin[0]
        print("slope",slope," b",b)
        # y = slope(x) + b
        # y - b = slope(x)
        # x = (y - b)/ slope
        x = (y - b)/slope
        print("(x,y)",x,y)
        return x

    def affine_pente(self,slope,y):
        b = self.origin[1] - slope * self.origin[0]
        #print("slope", slope, " b", b)
        x = (y - b) / slope
        return x
    def projection1d(self,n):
        linspace = torch.linspace(torch.pi,0,n)
        #print("linspace",linspace)
        tan_linspace = torch.tan(linspace)
        #print("tan_linspace",tan_linspace)
        for slope in tan_linspace:
            count,added_value = 0,0

            for y in range(0,self.tensor.shape[1]):
                result = self.somme_case(slope,y)
                if not result is None:
                    count += result[0]
                    added_value += result[1]
            if count != 0:
                print("added_value/count",(added_value/count))

    def somme_case(self,slope,y):
        x1 = self.affine_pente(slope, y + 0.0001).type(torch.int64)
        x2 = self.affine_pente(slope, y + 0.9999).type(torch.int64)

        if x1 < 0 or x1 > self.tensor.shape[0]-1:
            x1 = None
        else:
            ...#print("x1",x1.item())
        if x2 < 0 or x2 > self.tensor.shape[0]-1:
            x2 = None
        else:
            ...#print("x2",x2.item())
        if x1 is None and x2 is None:
            return None
        print("y", y)
        if not x1 == x2 and not (x1 is None or x2 is None):
            #print("self.tensor[x1,y]",self.tensor[x1][y].item(),"self.tensor[x2,y]",self.tensor[x2][y])
            return  torch.tensor([2]), self.tensor[x1,y] + self.tensor[x2,y]
        if x1 is None:
            #print("self.tensor[x2,y]",self.tensor[x2][y].item())
            return torch.tensor([1]),self.tensor[x2:x2+1, y:y+1]

        else:
            #print("self.tensor[x1,y]",self.tensor[x1][y].item())

            return  torch.tensor([1]),self.tensor[x1][y]
    def grid_sample(self):
        # Input shape (N,C,Hin,Win) ou N est batch, C le nombre de canaux ,H la hauteur et W la largeur
        test_tensor = torch.tensor([[1,2,3],[4,5,6],[7,8,9]]).reshape(1,1,3,3).to(torch.float32)
        print(test_tensor)

        # Grid Shape (N,Hout,Wout,2)
        test_grid = torch.zeros(1,test_tensor.shape[2],test_tensor.shape[3],2)

        #test_grid[0,0,0,1]=.1
        test_grid[0,:,:,0]=test_tensor.reshape(3,3)/9
        test_grid[0,:,:,1]=test_tensor.reshape(3,3)/9

        print(test_grid)

        outputgrid = torch.nn.functional.grid_sample(test_tensor,test_grid)
        print(outputgrid)

tensor_to_project = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])

proj = Projection(tensor_to_project,(1.5,.5))
proj.grid_sample()
#proj.projection1d(16)
#proj.affine((1.5,0.5),(3,3),2)