import torch


class TimeFunctions:
    @staticmethod
    def time_from_sin(minrange,maxrange,divisions,force_cpu):
        if force_cpu:
            device = "cpu"
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        real_part = torch.linspace(minrange, maxrange, divisions, device=device)
        imag_part = torch.sin(real_part)*1j
        result = real_part+imag_part
        return result


