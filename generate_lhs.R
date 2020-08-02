# Script ot generate Latin Hypercube Samples with lhs routine 
# Can further specify other methods - see lhs.pdf manual 
# n is the number of points we require 
# d is the dimensionality of the problem 

setwd('/home/harry/Dropbox/Semi-GP/dev/')
library(lhs)
n = 1000
d = 7
# X = maximinLHS(n, d)
X = optimumLHS(n, d)
write.csv(X, 'lhs/optimum_1000_7D')

