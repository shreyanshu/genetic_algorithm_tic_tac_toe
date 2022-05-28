import math
import random

class GeneticAlgorithm:
    def __init__(self, number, goal):
        self.population = []
        self.goal = goal
        self.gene_pool = []
        self.number = number
        self.most_fit = {'fitness': 0}
        self.initialize_population()

    def initialize_population(self):
        for i in range(self.number):
            self.population.append({'chromosomes': self.make_individual(len(self.goal)),
                                    'fitness': 0})

    def make_individual(self, length):
        characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,'
        characters_length = len(characters)
        result = ''.join([characters[random.randint(0, characters_length-1)] for num in range(length)])
        return result

    def solve(self):
        for i in range(1000):
            self.calc_fitness_all()
            self.cross_parents()
            self.mutate()
        print(self.most_fit)

    def calc_fitness_all(self):
        most_fit = {'fitness':0}
        for members in self.population:
            match_counter = 0
            for index, gene in enumerate(members['chromosomes']):
                if gene == self.goal[index]:
                    match_counter += 1
                members['fitness'] = match_counter if match_counter > 0 else 1

            for _ in range(members['fitness']):
                self.gene_pool.append(members['chromosomes'])

            if members['fitness'] > most_fit['fitness']:
                most_fit = members

        if most_fit['fitness'] > self.most_fit['fitness']:
            self.most_fit = most_fit

        print('Most fitness so far is ', self.most_fit['fitness'])
        # if most_fit['fitness']:
        #     print('Chromosome ', self.most_fit['chromosomes'])

    def cross_parents(self):
        new_population = []
        pool = self.gene_pool
        for member in self.population:
            first_parent = pool[math.floor(random.random() * len(pool))]
            second_parent = pool[math.floor(random.random() * len(pool))]
            cross_section = math.floor(random.random() * len(self.goal) - 1)
            kid = first_parent[:cross_section] + second_parent[cross_section:]
            new_population.append({'chromosomes': kid, 'fitness': 0})
        self.population = new_population
        self.gene_pool = []

    def mutate(self):
        for members in self.population:
            mutate_chromosome = random.random() < 0.1
            # print(members, mutate_chromosome)
            if mutate_chromosome:
                for index, genes in enumerate(members['chromosomes']):
                    mutate_gene = random.random() < 0.05
                    if mutate_gene:
                        mutated_chromosome = members['chromosomes'][0:index] + self.make_individual(1) + \
                                             members['chromosomes'][index+1:]
                        members['chromosomes'] = mutated_chromosome


model = GeneticAlgorithm(600, 'You can run this GNN class')
print(model.population)
model.solve()
# print(model.population)
