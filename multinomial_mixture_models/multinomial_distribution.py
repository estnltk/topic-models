
# coding: utf-8

# In[22]:

from math import log, exp
from scipy.special import gammaln


# # Likelihood function for multinomial distribution
# 
# **Arguments**
# * Observation counts for each outcome type
# * Probabilities for each outcome type. They must sum to one

# In[28]:

def likelihood(counts, probs):
    ll = gammaln(sum(counts) + 1)
    for i, c_i in enumerate(counts):
        ll += c_i * log(probs[i]) - gammaln(c_i + 1)
    return exp(ll)


# returns a negative number
def loglikelihood(counts, probs):
    ll = gammaln(sum(counts) + 1)
    for i, c_i in enumerate(counts):
        ll += c_i * log(probs[i]) - gammaln(c_i + 1)
    return ll

# ## Test code  
# Multinomial sistribution with parameters $c =(7, 2, 3)$ and $p=(0.40, 0.35, 0.25)$ leads to the probability $0.02483$.
# 

# In[32]:

#likelihood(counts = [7, 2, 3], probs = [0.40, 0.35, 0.25])

