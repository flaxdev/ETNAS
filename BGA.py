# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 12:45:24 2024

@author: cannavo
"""

# genetic algorithm search for continuous function optimization
from numpy.random import randint
from numpy.random import rand
import numpy as np
from tqdm import tqdm


# objective function to minimize
def objective(x):
	return 0


# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
    
    c1, c2 = p1.copy(), p2.copy()
    
    if len(p1)>2:
        if rand()<r_cross:
            if len(p1)==3:
                pt=1
            else:
                pt = randint(1, len(p1)-2)
                
            c1 = p1[:pt] + p2[pt:]
            c2 = p2[:pt] + p1[pt:]
        
    return [c1, c2]
    
# mutation operator
def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, chromosome=None):
    
   if (n_pop % 2):
       n_pop = n_pop + 1
	   
   # initial population of random bitstring
   pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
   # keep track of best solution    
   
   if chromosome is not None:
       pop[0] = chromosome.tolist()
   
   children = [[int(x) for x in y] for y in pop]     
   
   Table, Scores = list(), list()
   score = objective(children[0])
   Table.append(children[0])
   Scores.append(score)
   
   # enumerate generations
   for gen in tqdm(range(n_iter)):
       
		# replace population
        pop = children

		# evaluate all candidates in the population
        scores = []
        for d in pop:
                         
            try:
                isdone =  Table.index(d)
                
                #in rare case recompute anyway                
                if rand()<0.25:
                    score = objective(d)
                    
                    if score<Scores[isdone]:
                        Table.insert(0,d)
                        Scores.insert(0,score)
                    else:
                        Table.append(d)
                        Scores.append(score)                        
                else:                    
                    score = Scores[isdone]
                
            except ValueError:
                score = objective(d)
                Table.append(d)
                Scores.append(score)
				
            scores.append(score) 
			
		# check for new best (minimum) solution
        ibest = np.argmin(scores)
        best_eval = scores[ibest]
        best = pop[ibest]
                
		# select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        
		# create the next generation
        children = list()
        for i in range(0, n_pop, 2):
			# get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
            try:
                for c in crossover(p1, p2, r_cross):
    				# mutation
                    mutation(c, r_mut)
    				# store for next generation
                    children.append(c)
            except:
                    pass
                
        # add elite
        if len(children)<len(pop):
            children.append(best)
        else:
            pt = randint(len(pop))
            children[pt] = best
                
   np.savetxt("GA_Table.txt", np.array([x+[y] for x,y in zip(Table, Scores)] ), delimiter=",", fmt="%g") 

   Table = [x+[y] for x,y in zip(Table, Scores)] 
   return [best, best_eval, Table, pop]