import numpy as np
x = [
    list(['K', 'P', 'E', 'Y', 'V', 'V', 'I', 'G']),
    list(['E', 'M', 'A', 'D', 'L', 'A', 'G', 'L']),
    list(['Q', 'D', 'C', 'C', 'Y', 'G', 'G', 'M']),
    list(['L', 'P', 'P', 'A', 'M', 'T', 'S', 'A']),
    list(['T', 'V', 'H', 'Y', 'G', 'S', 'L', 'A']),
    list(['G', 'V', 'G', 'R', 'E', 'S', 'Q'])
]

print(len(x))
xn = np.asarray(x)
#n, p = xn.shape
#print(n)
print(len(xn.shape))