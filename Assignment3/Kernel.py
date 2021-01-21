import function as f

### 1.1a
m = 5                       # Mean examples: m = 0 for 1D and m = [0,0] for 2D
c = 1                       # Covariance matrix examples: c = 1 for 1D and c = [[1,0],[0,1]] for 2D 
k = 1000                    # Number of samples
bins = 10                   # Number of bins in histogram
hlist = [0.1, 1, 5, 10]     # Bandwidth in kernel density estimation
x = f.norm_data_generate(m,c,k,bins)
for h in hlist:
    f.mykde(x, h)

### 1b
m0 = 5
c0 = 1
m1 = 0
c1 = 0.04
k = 500
bins = 10
hlist = [0.1, 1, 5, 10]

x = f.Gauss_mixt_data_generate(m0, c0, k, m1, c1, k, bins, 0)
for h in hlist:
    f.mykde(x, h)

### 1.2
m0 = [1, 0]
c0 = [[0.9, 0.4], [0.4, 0.9]]
m1 = [0, 2.5]
c1 = [[0.9, 0.4], [0.4, 0.9]]
k = 500
hlist = [0.1, 1, 5, 10]

x = f.Gauss_mixt_data_generate(m0, c0, k, m1, c1, k, 0, 0)
for h in hlist:
    f.mykde(x, h)