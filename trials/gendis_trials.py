from GENDIS_python import gendis
from preprocessing_anomaly import preprocessing_anomaly
import numpy as np


genetic_extractor = gendis.genetic.GeneticExtractor(population_size=50, iterations=25, verbose=True, 
                                     mutation_prob=0.3, crossover_prob=0.3, 
                                     wait=10, max_len=len(X_train) // 2)

