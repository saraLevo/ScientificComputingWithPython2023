import math
import timeit

def semiFunction(x):
    f = math.sqrt(1 - x**2)
    return f
    
def Riemann(N):
    arg = []
    for x in range(1,N):
        i = -1 + x*(2/N)
        arg.append((2/N)*semiFunction(i))
    I = math.fsum(arg)
    return I

I = Riemann(100)
print('The integral value calculated with Riemann function is', I)
#The value obtained with N=100 is sufficiently close to the true value, still 
#one can check that for increasing values of N, the valu of I is more precise.

code = """ 
import math

def semiFunction(x):
    f = math.sqrt(1 - x**2)
    return f
    
def Riemann(N):
    arg = []
    for x in range(1,N):
        i = -1 + x*(2/N)
        arg.append((2/N)*semiFunction(i))
    I = math.fsum(arg)
    return I
 
I = Riemann(100)
"""

def timeRiemann(N):
    code = f""" 
import math

def semiFunction(x):
    f = math.sqrt(1 - x**2)
    return f
    
def Riemann(N):
    arg = []
    for x in range(1,N):
        i = -1 + x*(2/N)
        arg.append((2/N)*semiFunction(i))
    I = math.fsum(arg)
    return I
 
I = Riemann({N})
"""
    time = timeit.timeit(code, number = 1)
    return time #[s]

#t = 0
#N = 800000
#while t < 1.0:
#    t = timeRiemann(N)
#    N = N + 100
    #print(t)
#print(N)


#running for one minute
#t = []
#Nlist = []
#time = 0
#N = 10000000
#while time < 60.0:
#    time = timeRiemann(N)
#    t.append(time)
#    N = N + 1000000
#    Nlist.append(N)
#    print(N, time)
#print(N)

print(Riemann(45500000))
