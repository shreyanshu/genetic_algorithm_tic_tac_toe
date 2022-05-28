import math
import random
from neural_network import NeuralNetwork
import numpy as np

class Player:
    def __init__(self, id, env, play, model=None):
        self.id = id
        self.env = env
        self.play = play
        self.model = model
        self.position = ''
        self.free_indexes = 9
        self.won = False

    def get_free_indices(self):
        free_indices = []
        for i, row in enumerate(self.env.object_grid):
            for j, col in enumerate(row):
                if col == '-':
                    free_indices.append((i, j))
        return free_indices

    def vertical_win(self):
        for column in [0, 1, 2]:
            matches = 0
            for row in [0, 1, 2]:
                if self.env.object_grid[row][column] == self.play:
                    matches += 1
            if matches == 3:
                return True
        return False

    def horizontal_win(self):
        for row in [0, 1, 2]:
            matches = 0
            for column in [0, 1, 2]:
                if self.env.object_grid[row][column] == self.play:
                    matches += 1
            if matches == 3:
                return True
        return False

    def side_ways_win(self):
        matches = 0
        for row in [0,1,2]:
            if self.env.object_grid[row][row] == self.play:
                matches += 1
        if matches == 3:
            return True
        matches = 0
        for row in [0,1,2]:
            if self.env.object_grid[row][2-row] == self.play:
                matches += 1
        if matches == 3:
            return True
        return False

    def check_win(self):
        hor_win = self.horizontal_win()
        ver_win = self.vertical_win()
        side_win = self.side_ways_win()
        # print('Horizontal', hor_win)
        # print('Vertical', ver_win)
        # print('Diagonal', side_win)

        return hor_win or side_win or ver_win

    def move(self):
        free_places = self.get_free_indices()
        self.free_indexes = len(free_places)

        if self.free_indexes == 0:
            return self.env.object_grid

        move = self.think().argmax()
        dic = [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]][move]
        if self.env.object_grid[dic[0]][dic[1]] != '-':
            raise Exception('Error wrong placement')
        self.env.object_grid[dic[0]][dic[1]] = self.play
        if self.check_win():
            self.won = True
        return self.env.object_grid

    def think(self):
        state = np.array(self.env.object_grid).flatten()
        opponent = 'X' if self.play == 'O' else 'O'
        first_half = list(map(lambda x: 1 if x == self.play else 0, state))
        second_half = list(map(lambda x: 1 if x == opponent else 0, state))
        input = np.array(first_half+second_half).reshape((1, 18))
        prediction = self.model.predict(input)
        return prediction


class GeneticAlgorithm():
    def __init__(self, number):
        self.population = []
        self.gene_pool = []
        self.number = number
        self.most_fit = {'fitness': 0}
        self.initialize_population()
        self.elit_size = self.number * 0.3
        self.elites = []
        self.so_far = {}

    def initialize_population(self):
        for i in range(self.number):
            self.population.append({'chromosomes': self.make_individual(),
                                    'fitness': 0})

    def make_individual(self):
        return NeuralNetwork([18, 15, 15, 9])

    def solve(self):
        for i in range(200):
            print('SOLVE', i, '#'*100, len(self.population))
            self.calc_fitness_all()
            self.cross_parents()
            self.mutate()
        self.calc_fitness_all()

        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        best_model = sorted_population[0]['chromosomes']
        best_fitness = sorted_population[0]['fitness']
        print('best fitness', best_fitness)


    def mutate(self):
        for members in self.population:
            mutate_chromosome = random.random() < 0.1
            if mutate_chromosome:
                for index, weights in enumerate(members['chromosomes'].weights_flat):
                    mutate_gene = random.random() < 0.05
                    if mutate_gene:
                        members['chromosomes'].weights_flat[index] = (random.random()*2) - 1

                members['chromosomes'].reconstruct_weights()

    def cross_parents(self):
        new_population = []
        pool = self.gene_pool
        while (len(self.population)) > len(new_population) + self.elit_size:
            first_parent = pool[math.floor(random.random() * len(pool))]
            second_parent = pool[math.floor(random.random() * len(pool))]

            first_kid = self.make_individual()
            second_kid = self.make_individual()
            cross_section = math.floor(random.random() * len(first_parent.weights_flat))
            weights_first_kid = np.concatenate((first_parent.weights_flat[:cross_section],
                                                second_parent.weights_flat[cross_section:]))
            weights_second_kid = np.concatenate((second_parent.weights_flat[:cross_section],
                                                 first_parent.weights_flat[cross_section:]))
            first_kid.weights_flat = weights_first_kid
            first_kid.reconstruct_weights()

            second_kid.weights_flat = weights_second_kid
            second_kid.reconstruct_weights()

            cross_section = math.floor(random.random() * len(first_parent.bias_flat))
            bias_first_kid = np.concatenate((first_parent.bias_flat[:cross_section],
                                             second_parent.bias_flat[cross_section:]))
            bias_second_kid = np.concatenate((second_parent.bias_flat[:cross_section],
                                              first_parent.bias_flat[cross_section:]))
            first_kid.bias_flat = bias_first_kid
            first_kid.reconstruct_bias()
            second_kid.bias_flat = bias_second_kid
            second_kid.reconstruct_bias()

            new_population.append({'chromosomes': first_kid, 'fitness': 0})
            new_population.append({'chromosomes': second_kid, 'fitness': 0})

        sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
        new_population.extend(sorted_population[:int(len(self.population) * 0.3)])

        self.population = new_population
        self.gene_pool = []

    def calc_fitness_all(self):
        for member in self.population:
            total_games = 0
            total_games_lost = 0
            for index in range(len(self.population)):
                total_games += 2
                if not self.play_a_game(member['chromosomes'], self.population[index]['chromosomes'], True):
                    total_games_lost += 1
                if not self.play_a_game(member['chromosomes'], self.population[index]['chromosomes'], False):
                    total_games_lost += 1

            member['fitness'] = (total_games-total_games_lost)/total_games
            for i in range(math.floor(member['fitness']*1000)):
                self.gene_pool.append(member['chromosomes'])
            # TODO: Uncomment this
            print("Total Games {} total loss {}. Total agent in test run {}".format(total_games, total_games_lost,
                                                                                    len(self.gene_pool)))
        if len(self.gene_pool) < len(self.population):
            sorted_population = sorted(self.population, key=lambda x: x['fitness'], reverse=True)
            self.gene_pool.extend(list(map(lambda x: x['chromosomes'],
                                           sorted_population[:len(self.population)-len(self.gene_pool)])))


    def play_a_game(self, first_player, second_player, first_player_starts):
        class Environment:
            object_grid = [['-', '-', '-'], ['-', '-', '-'], ['-', '-', '-']]
            id = random.random()

            def print_grid(self):
                for row in self.object_grid:
                    print(' '.join(row))

        env = Environment()
        if first_player_starts:
            player_x = Player(0, env, 'X', first_player)
            player_o = Player(1, env, 'O', second_player)
        else:
            player_x = Player(0, env, 'X', second_player)
            player_o = Player(1, env, 'O', first_player)
        try:
            while True:
                player_x.move()
                if player_x.check_win():
                    return True
                if len(player_x.get_free_indices()) == 0:
                    return True
                player_o.move()
                if player_o.check_win():
                    return False
                if len(player_o.get_free_indices()) == 0:
                    return True
        except Exception as err:
            # print('i am here', err.args)
            return False


algo = GeneticAlgorithm(200)
algo.solve()