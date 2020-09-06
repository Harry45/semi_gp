# Script ot generate Latin Hypercube Samples with lhs routine 
# Can further specify other methods - see lhs.pdf manual 
# n is the number of points we require 
# d is the dimensionality of the problem 

setwd('/home/harry/Dropbox/Semi-GP/dev/')
library(lhs)
n = 1000
d = 6
X = maximinLHS(n, d)
# X = optimumLHS(n, d)
# X = randomLHS(n, d)
write.csv(X, 'lhs/maximin_1000_6D')

