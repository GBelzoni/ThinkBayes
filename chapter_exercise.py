# -*- coding: utf-8 -*-
import numpy
import matplotlib
from matplotlib import pylab, mlab, pyplot
np = numpy
plt = pyplot
from IPython.display import display
from IPython.core.pylabtools import figsize, getfigs
from pylab import *
from numpy import *

from os import chdir
chdir('C:\Users\PHCostello\Documents\UbuntuHome\Projects\ThinkBayes')
from thinkbayes import *

#Test out plotting
x=linspace(0,2*pi,num=500)
y = sin(x)
plot(x,y)

###########################################################################
#COOKIES!!!
###########################################################################

#Suppose there are two bowls of cookies. Bowl 1 contains 30 vanilla cookies 
#and 10 chocolate cookies. Bowl 2 contains 20 of each.
#Now suppose you choose one of the bowls at random and, 
#without looking, select a cookie at random. 
#The cookie is vanilla. 
#
#What is the probability that it came from Bowl 1?

###########################################################################
#M&M
###########################################################################


#In 1995, they introduced blue M&M’s. 
#Before then, the color mix in a bag of plain M&M’s 
#was 30% Brown, 20% Yellow, 20% Red, 10% Green, 10% Orange, 10% Tan.
#
#Afterward it was 24% Blue , 20% Green, 16% Orange, 14% Yellow, 13% Red, 13% Brown.
#
#Suppose a friend of mine has two bags of M&M’s, 
#and he tells me that one is from 1994 and one from 1996. 
#He won’t tell me which is which, but he gives me one M&M from each bag. 
#One is yellow and one is green. 
#
#What is the probability that the yellow one came from the 1994 bag?

# Hyp A = Bag1 from 1994 and Bag2 from 1996
# Hyp B = Bag2 from 1994 and Bag1 from 1996
#Pr( A | given y and g from different bags) = Pr(y|first)P(g|second)P(A)/sum(...)
# = 0.2*0.2*0.5/( 0.2*0.2*0.5 + 0.14*0.1*0.5) = 2/(2+0.7) = 2/2.7

###########################################################################
###########################################################################
#Chapter 2
###########################################################################
###########################################################################


#Basic probability mass function
pmf = Pmf()
for x in [1,2,3,4,5,6]:
    pmf.Set(x,1/6.)
pmf.Prob(2)

#Setting arbitrary value then normalising.
pmf = Pmf()
for x in [1,2,3,4,5,6]:
    pmf.Set(x,1.)
print pmf.Prob(2)
pmf.Normalize()
print pmf.Prob(2)

pmf = Pmf()
for x in ['a','a','a','a','b','b','c']:
    pmf.Incr(x,1.)#Use Incr to add to freq in dist
pmf.Normalize()
pmf.Prob('a')

##Cookie problem
pmf = Pmf()
#Set priors
pmf.Set('Bowl 1' , 0.5)
pmf.Set('Bowl 2' , 0.5)
#Update with Likelyhood
pmf.Mult('Bowl 1', 0.75)
pmf.Mult('Bowl 2', 0.5)
#Normalise
pmf.Normalize()

print pmf.Prob('Bowl 1')

#Generalise
class Cookie(Pmf):
    
    def __init__(self, hypos):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo,1)
        self.Normalize()
    
    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()
        
    mixes = {
        'Bowl 1':dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2':dict(vanilla=0.5, chocolate=0.5),
        }
    
    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like
        
        

hypos = ['Bowl 1', 'Bowl 2']
pmf = Cookie(hypos)    

pmf.Update('vanilla')

for hypo, prob in pmf.Items():
    print hypo, prob
    
    
#####M&M problems

mix94 = dict(brown=30,
                 yellow=20,
                 red=20,
                 green=10,
                 orange=10,
                 tan=10)

mix96 = dict(blue=24,
                 green=20,
                 orange=16,
                 yellow=14,
                 red=13,
                 brown=13)
                 
hypoA = {'bag1' : mix94, 'bag2':mix96}
hypoB = {'bag1' : mix96, 'bag2':mix94}
hypothesis = {'A' : hypoA, 'B': hypoB}

class MandM(Suite):
    
    def Likelihood(self,data,hypo):
        
        bag, color = data
        like = hypothesis[hypo][bag][color]
        return like
        
        
suite = MandM('AB')    
suite.Print()     
            
suite.Update(('bag1', 'yellow'))
suite.Print()     
suite.Update(('bag2', 'green'))
suite.Print()    

         
##Exercises Chapter 2   

      #Generalise
class Cookie(Pmf):
    
    
    
    def __init__(self, hypos, 
                    Bowl1 = {vanilla:30, chocolate:10},
                    Bowl2 = {vanilla:20, chocolate:20}):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo,1)
        self.Normalize()
        self.Bowl1 = Bowl1
        self.Bowl2 = Bowl2        
    
    def Update(self, data):
        for hypo in self.Values():
            like = self.Likelihood(data, hypo)
            self.Mult(hypo, like)
        self.Normalize()
        
        
    mixes = {
        'Bowl 1':self.Bowl1,
        'Bowl 2':self.Bowl2,
        }
    
    def Likelihood(self, data, hypo):
        mix = self.mixes[hypo]
        like = mix[data]
        return like
        
        
#Exercise Chapter 2 kinda hard
#hypos = ['Bowl 1', 'Bowl 2']
#pmf = Cookie(hypos)    
#
#pmf.Update('vanilla')
#
#for hypo, prob in pmf.Items():
#    print hypo, prob      


###########################################################################
###########################################################################
#Chapter 3
###########################################################################
###########################################################################


class Dice(Suite):
    
    def Likelihood(self,data,hypo):
        
        if hypo<data:
            return 0
        else:
            return 1.0/hypo
        
        
suite = Dice([4,6,8,12,20])

suite.Update(6)
suite.Print()

for roll in [6,8,7,7,5,4]:
    suite.Update(roll)

suite.Print()


## A company has a has N trains and you see train numbered 60. What is the most
#probable number of trains?

class Train(Suite):
    
    def __init__(self, hypos, alpha=1.0):
        Pmf.__init__(self)
        for hypo in hypos:
            self.Set(hypo, hypo**(-alpha))
        self.Normalize()
        
        
    def Likelihood(self, data, hypo):
        if hypo < data:
            return 0
        else:
            return 1./hypo
            
hypos = range(1,1001)

suite = Train(hypos)
suite.Update(60)
suite.Mean()

xy = array([it for it in suite.Items()])
plot(xy[:,0],xy[:,1])
xy[0:5,:]
show()

#what about prior?

for data in [60,30,90]:
    suite.Update(data)
   
suite.Mean() 

interval = Percentile(suite, 5.), Percentile(suite,95)
print interval

cdf = suite.MakeCdf()
interval = cdf.Percentile(5.), cdf.Percentile(95)
interval


###############################################################################
#Chapter 3 exercises
###############################################################################

#Exercise 1   To write a likelihood function for the locomotive problem, we had to answer this question: “If the railroad has N locomotives, what is the probability that we see number 60?”
#The answer depends on what sampling process we use when we observe the locomotive. In this chapter, I resolved the ambiguity by specifying that there is only one train-operating company (or only one that we care about).
#But suppose instead that there are many companies with different numbers of trains. And suppose that you are equally likely to see any train operated by any company. In that case, the likelihood function is different because you are more likely to see a train operated by a large company.
#
#As an exercise, implement the likelihood function for this variation of the locomotive problem, and compare the results.

#Subset the total number of train into k groups
#You P(60|N, group_i size = sz_i, sum sz_i=N) 



###########################################################################
###########################################################################
#Chapter 4
###########################################################################
###########################################################################

#A statistical statement appeared in “The Guardian" on Friday January 4, 2002:
#When spun on edge 250 times, a Belgian one-euro coin came up heads 140 times and tails 110. 
#‘It looks very suspicious to me,’ said Barry Blight, a statistics lecturer at the London School of Economics.
#‘If the coin were unbiased, the chance of getting a result as extreme as that would be less than 7%.’
#But do these data give evidence that the coin is biased rather than fair?

hypos = range(0,101)

class Coin( Suite ):
    
    def Likelihood(self, data, hypo):
        
        if data=="H":
            return hypo/100.
        else:
            return 1 - hypo/100.
        
        
suite = Coin(hypos)
dataset = 'H' * 140 + 'T' * 110

for data in dataset:
    suite.Update(data)     

suite.Print()
x = suite.Values()
y =suite.d.values()
plot(x,y)
show()

suite.MaximumLikelihood()
print 'Mean', suite.Mean()
print 'Median', Percentile(suite, 50)
print 'CI', thinkbayes.CredibleInterval(suite, 90)

##### Beta distribution ###############

#Definition in thinkbayes.pdf
#TODO check conjugate prioness of beta dist
beta = Beta()
beta.Update( (140,110))
print beta.Mean()

x = linspace(0,1,num=100)
y = [ beta.EvalPdf(val) for val in x]
plot(x,y)
show()

#To make into pdf would need to scale as vals around 1e-75
#and making PmfDict can't handle little numbers due to floating point rounding
yscaled = [yv*1e80 for yv in y]
yscaled = array(yscaled)
yscaled = yscaled/sum(yscaled)

betaPmf = MakePmfFromDict(dict(zip(x, yscaled)))
x = linspace(0,1,num=100)
y = [ betaPmf.Prob(val) for val in x]
plot(x,y)
show()

###Exercises
#Exercise 1  
#Suppose that instead of observing coin tosses directly, 
#you measure the outcome using an instrument that is not always correct. 
#Specifically, suppose there is a probability y that an actual heads is reported as tails, 
#or actual tails reported as heads.

#Let a be the probability of misreporting the value

#Pr(p=x|y=H) = Pr(H|p=x)Pr(p=x)
#Pr(H|p=x) = x*a + (1-x)*(1-a) = prob observing getting head and reporting correctl + pr T and reporting incorrrectly
#Pr(T|p=x) = (1-x)*a + x*(1-a)


class CoinLatent( Suite ):
    
    def __init__(self, hypos, a=1.0):
        Suite.__init__(self, hypos)
        self.a = a 
    
    def set_a(a):
        self.a = a
    
    def Likelihood(self, data, hypo):
        
        prH = hypo/100.
        prT = 1 - hypo/100.
        a = self.a 
        
        if data=="H":
            return prH*a + prT*(1-a) 
        else:
            return prT*a + prH*(1-a) 
        
hypos = range(1,101)        
suite = CoinLatent(hypos,a=1-0.56)
dataset = 'H' * 140 + 'T' * 110

for data in dataset:
    suite.Update(data)     

#suite.Print()
x = suite.Values()
y =suite.d.values()
plot(x,y)
show()
aas = [1.0,0.9,0.7,0.6,0.58,0.56,0.53,0.5,0.3,0.1,0.0]

def genSuite(a):
    
    hypos = range(1,101)        
    suite = CoinLatent(hypos,a)
    dataset = 'H' * 140 + 'T' * 110
    
    for data in dataset:
        suite.Update(data)
        
    return suite
    


suite.MakeCdf().Percentile(50)

#Conclusion: we get 140H, 110T so approx 0.56 heads. If prob of reporting wrong is 0.56
# then we the we would think that most likely hyp near 100% heads, as then it would only be
# wrong reporting that is giving us the error. The distibution is increasing for a=0.56 with max
# max likelyhood of 0.3 at prior = 1.0 

# When reporting error is a=0.5, then 50/50 error means we have no info about the underlying dist
# and all priors are equally likely, so flat posterior

# When a<0.5 then error rate flips role of heas and tails and we get the same behaviour inverted, ie
# for a = 1- 0.56 then we have decreasing distribution with max at prior =0.0.

# So in end we have posterior P(p=x| data , a=alpha). I guess we could also put a prior on alpha and integrate it
# out as well

#CoinLatentIntAlpha

####Integrating out noise parameter in coin tossing problem - WRITEUP!!!
#hypo x's, alpha's
#CoinLatent, posteria given alpha, so Pr(x|data,alpha) = LikelyHood given alpha
#likely hood P(x|data, alpha)Pr(alpha)
###
#So generate pmf
errors = linspace(0.5,1,num=50)    
suites = [ genSuite(er) for er in errors]
x=56

sum([st.Prob(x) for st in suites])
pmf_coin = Pmf()
pmf_alpham = [sum([st.Prob(x) for st in suites]) for x in hypos]
hypos
for pr in zip(hypos,pmf_alpham):
    pmf_coin.Set(pr[0],pr[1])
pmf_coin.Normalize()

x = pmf_coin.Values()
y =pmf_coin.d.values()
plot(x,y)
show()

suite = CoinLatentAlphInt(hypos)
dataset = 'H' * 140 + 'T' * 110

for data in dataset:
    suite.Update(data)     

