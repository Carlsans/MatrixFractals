class NormVar:
    def __init__(self,valuewhennone = 0):
        self.minvalue = float('inf')
        self.maxvalue = float('-inf')
        self.currentnormvalue = 0
        self.valuewhennone = valuewhennone

    def get(self):
        if self.currentnormvalue is None:
            return self.valuewhennone
        return self.currentnormvalue

    def set(self,value):
        if value is None:
            self.currentnormvalue = None
            return
        if value < self.minvalue:
            self.minvalue = value
        if value > self.maxvalue:
            self.maxvalue = value
        self.currentnormvalue = self.compute_normed_value(value)

    def compute_normed_value(self,value):
        if self.maxvalue-self.minvalue == 0:
            return 0
        normed_value = (value - self.minvalue)/(self.maxvalue-self.minvalue)
        return normed_value


# Test
# var1 = NormVar()
# var1.set(5)
# print(var1.get())
# var1.set(-5)
# print(var1.get())
# var1.set(2)
# print(var1.get())