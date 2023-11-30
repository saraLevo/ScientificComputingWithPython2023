##Reductions
import numpy as np

m = np.arange(12).reshape((3,4))
print(m)
mean_m = np.mean(m)
print('Mean of m:',mean_m)
mean_column = m.mean(0)
print('Mean of each column:', mean_column)
mean_row = m.mean(1)
print('Mean of each row:', mean_row)


##Output product
print("\n")
u = np.array([1, 3, 5, 7])
v = np.array([2, 4, 6, 8])

r1 = np.outer(u, v)
print('The outer product of u and v calculated with outer function:', r1)

r2 = np.zeros((4,4))
for i in range(len(u)):
	for j in range(len(v)):
		r2[i,j] = u[i]*v[j]
print('The outer product of u and v calculated with loops:', r2)

u_t = u.reshape(4,1)
r3 = u_t*v
print('The outer product calculated of u and v with broadcasting:', r3)


##Matrix masking
print("\n")
import numpy.random as npr
a = npr.rand(10, 6)*3
print('Matrix a:', a)

a[a < 0.3] = 0
print('Masked matrix:', a)


##Trigonometric functions
print("\n")
import math
import matplotlib.pyplot as plt

b = np.linspace(0, 2*math.pi, 100) #inclusive?
print('Array: ', b)


print('Extract every 10th element of b:', b[10:len(b):10])

print('Reverse b:', b[::-1])

print('The elements where the absolute difference between the sin and cos functions is below 0.1:', b[np.fabs(np.cos(b)-np.sin(b))<0.1])

fun_cos = np.cos(b)
fun_sin = np.sin(b)
markers = [i for i in range(len(b)) if abs(fun_cos[i]-fun_sin[i]) < 0.05]


#plt.plot(b, np.cos(b), label="cos(b)")
#plt.plot(b, np.sin(b), label="sin(b)")
#plt.scatter(b[markers], fun_cos[markers], color='red')
#plt.xlabel("b")
#plt.ylabel("f(b)")
#plt.grid(True)
#plt.show()


##Matrices
print("\n")
mul_table = np.fromfunction(lambda i, j: (i+1)*(j+1), (10,10))
print('10x10 multiplication table:', mul_table)

trace = sum([mul_table[i][i] for i in range(10)])
print('The trace of the matrix is:', trace)

#antid_mul_table = [mul_table[i][j] for i in range(9,0,-1) for j in range(10)]
antid_mul_table = np.diag(np.fliplr(mul_table))
print('Anti-diagonal matrix is:', antid_mul_table)

diagonaloffset_mul_table = [mul_table[i][i+1] for i in range(9)]
print('Diagonal matrix with offset 1 upwards:', diagonaloffset_mul_table)


##Broadcasting
print("\n")
cities = ['Chicago', 'Springfield', 'Saint-Louis', 'Tulsa', 'Oklahoma City', 'Amarillo', 'Santa Fe', 'Albuquerque', 'Flagstaff', 'Los Angeles']
pos = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544, 1913, 2448])*1.60934

distances = np.array([abs(pos[i] - pos) for i in range(len(pos))])
#arrange the name of cities on top and on the left side of the matrix 
print('Distances among the cities in km:', distances, '\n')


##Prime numbers sieve
print("\n")
import time

def isPrime(x):
    if x <= 1:
        return False
    for i in range(2,x):
        if x%i==0:
            return False
    return True
    
my_input = int(input('Compute the prime numbers up to (insert an integer number):'))

#x = np.arange(1,99)
#N = np.zeros_like(x, dtype=bool)

#for j in range(len(x)):
#    N[j] = isPrime(x[j])
#
#numpy_N = np.array(N)
#prime_num = x[N]

t0 = time.time()
x = np.arange(1,my_input)
N = np.zeros_like(x, dtype=bool)

for j in range(len(x)):
    N[j] = isPrime(x[j])

numpy_N = np.array(N)
prime_num = x[N]
t1 = time.time()

print(f'Time taken to find the prime numbers up to {my_input} is:', t1-t0)
print(f'Prime numbers up to {my_input} are:', prime_num)

t2 = time.time()
N1 = np.array([True for i in range(len(x))])

for i in range(2, int(math.sqrt(len(x)))+1):
    if N1[i] == True:
        j = np.arange(i**2, len(x), i)
        N1[j] = False

prime_num_erato = np.where(N1)[0]
t3 = time.time()
print(f'Time taken to find the prime numbers up to {my_input} with Eratosthenes sieve:',t3-t2)
print(f'Prime numbers up to {my_input} are:', prime_num_erato)


##Diffusion using random walk
print("\n")
import numpy.random as npr
import numpy as np

temp = npr.randint(2, size=(1000, 200))
print(temp)

#rd = np.zeros((1000,200))
#for i in range(1000):
#    for j in range(200):
#        if temp[i,j] == 0:
#            rd[i,j] = -1
#        else:
#            rd[i,j] = temp[i,j]


rd = np.array([temp[i, j] - 1 if temp[i,j] == 0 else temp[i,j] for i in range(1000) for j in range(200)])
rd = rd.reshape((1000,200))
print(rd)

walking_dist = np.array(rd.sum(axis=0))**2
print(walking_dist)
print(walking_dist.size)

def avg_distvstime(n):
    walking_dist_vec = np.ones((n, 1000))
    
    for l in range(n):
        temp = npr.randint(2, size=(1000, 200))
        rd = np.array([temp[i, j] - 1 if temp[i,j] == 0 else temp[i,j] for i in range(1000) for j in range(200)])
        rd = rd.reshape((1000,200))
        walking_dist = np.array(rd.sum(axis=1))
        walking_dist_vec[l, :] = walking_dist
        
    avg = np.average(walking_dist_vec, axis=0)
    return avg
    

avg_vec = avg_distvstime(10)
print('Dimensions',avg_vec.size)
plt.plot(range(1000), avg_vec)
plt.xlabel("steps")
plt.ylabel("avg of distances")
plt.grid(True)
plt.show()
