import pygame
import time
from collections import deque
import heapq
import tkinter as tk
from tkinter import messagebox
import random
import math
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from collections import defaultdict

pygame.init()

# Window
WIDTH, HEIGHT = 900, 700
CELL_SIZE, GRID_SIZE = 100, 3
GRID_OFFSET_X, GRID_OFFSET_Y = 300, 150

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("8-Puzzle Solver by Huynh Tan Vinh")

# Colors
BG_COLOR = (30, 30, 30)
TILE_COLOR = (0, 120, 255)
EMPTY_COLOR = (50, 50, 50)
BORDER_COLOR = (255, 255, 255)
BUTTON_COLOR = (50, 50, 50)
BUTTON_HOVER = (70, 70, 70)
BUTTON_CLICK = (30, 30, 30)
TEXT_COLOR = (255, 255, 255)
SOLVED_COLOR = (0, 255, 0)
SOLVING_COLOR = (255, 215, 0)
NOT_SOLVED_COLOR = (255, 0, 0)
SELECTED_ALGO_COLOR = (100, 100, 100)

# Fonts
TITLE_FONT = pygame.font.SysFont("Arial", 40, bold=True)
BUTTON_FONT = pygame.font.SysFont("Arial", 20)
TIMER_FONT = pygame.font.SysFont("Arial", 20, bold=True)

# Trạng thái đầu và cuốicuối
start_state = "265087431"
goal_state = "123456780"

start_input, goal_input = start_state, goal_state
input_active = None

algorithm_results = []
show_comparison = False
comparison_image = None
comparison_rect = pygame.Rect(150, 100, 600, 500)

def find_zero(state):
    idx = state.index('0')
    return idx // 3, idx % 3

def get_next_states(state):
    row, col = find_zero(state)
    next_states = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 3 and 0 <= new_col < 3:
            old_idx, new_idx = row * 3 + col, new_row * 3 + new_col
            next_state = list(state)
            next_state[old_idx], next_state[new_idx] = next_state[new_idx], next_state[old_idx]
            next_states.append(''.join(next_state))
    return next_states

def is_solvable(state, goal_state):
    def count_inversions(s):
        s = [int(x) for x in s if x != '0']
        return sum(1 for i in range(len(s)) for j in range(i + 1, len(s)) if s[i] > s[j])
    return (count_inversions(state) % 2) == (count_inversions(goal_state) % 2)

def heuristic(state, goal_state):
    goal_pos = {goal_state[i]: (i // 3, i % 3) for i in range(9)}
    return sum(abs(goal_pos[state[i]][0] - i // 3) + abs(goal_pos[state[i]][1] - i % 3)
               for i in range(9) if state[i] != '0')

# Algorithms
def bfs(start_state, goal_state):
    queue = deque([(start_state, [start_state], 0)])  # Initialize queue with (state, path, nodes_expanded)
    visited = {start_state}
    while queue:
        state, path, nodes = queue.popleft()
        if state == goal_state:
            return path, nodes, len(path) - 1, len(path) - 1
        for next_state in get_next_states(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state], nodes + 1))
    return None, 0, 0, 0

def ucs(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    queue = [(0, start_state, None, 0)]
    visited = {start_state: 0}
    parents = {}
    heapq.heapify(queue)
    while queue:
        cost, state, parent, nodes = heapq.heappop(queue)
        if state == goal_state:
            path = reconstruct_path(parents, start_state, goal_state)
            return path, nodes, len(path) - 1, cost
        for next_state in get_next_states(state):
            new_cost = cost + 1
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost
                parents[next_state] = state
                heapq.heappush(queue, (new_cost, next_state, state, nodes + 1))
    return None, 0, 0, 0

def reconstruct_path(parents, start, goal):
    path = [goal]
    while path[-1] != start:
        path.append(parents[path[-1]])
    return path[::-1]

def dfs(start_state, goal_state):
    stack = [(start_state, [start_state], 0)]
    visited = set()
    while stack:
        state, path, nodes = stack.pop()
        if state == goal_state:
            return path, nodes, len(path) - 1, len(path) - 1
        if state not in visited:
            visited.add(state)
            next_states = get_next_states(state)
            next_states.sort(key=lambda x: heuristic(x, goal_state))
            for next_state in reversed(next_states):
                if next_state not in visited:
                    stack.append((next_state, path + [next_state], nodes + 1))
    return None, 0, 0, 0

def ids(start_state, goal_state):
    depth, nodes_expanded = 0, 0
    while True:
        stack = [(start_state, [start_state], 0)]
        visited = set()
        while stack:
            state, path, d = stack.pop()
            if state == goal_state:
                return path, nodes_expanded, depth, len(path) - 1
            if d < depth and state not in visited:
                visited.add(state)
                for next_state in get_next_states(state):
                    if next_state not in visited:
                        stack.append((next_state, path + [next_state], d + 1))
                        nodes_expanded += 1
        depth += 1

def greedy_search(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    queue = [(heuristic(start_state, goal_state), start_state, 0)]
    parents = {start_state: None}
    visited = {start_state}
    nodes_expanded = 0
    heapq.heapify(queue)
    while queue:
        _, state, nodes = heapq.heappop(queue)
        if state == goal_state:
            path = reconstruct_path(parents, start_state, goal_state)
            return path, nodes, len(path) - 1, len(path) - 1
        next_states = sorted(get_next_states(state), key=lambda x: heuristic(x, goal_state))
        for next_state in next_states:
            if next_state not in visited:
                visited.add(next_state)
                parents[next_state] = state
                nodes_expanded += 1
                heapq.heappush(queue, (heuristic(next_state, goal_state), next_state, nodes_expanded))
    return None, nodes_expanded, 0, 0

def astar(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    queue = [(heuristic(start_state, goal_state), 0, start_state, 0)]
    parents = {start_state: None}
    g_scores = {start_state: 0}
    visited = set()
    nodes_expanded = 0
    heapq.heapify(queue)
    while queue:
        _, g, state, nodes = heapq.heappop(queue)
        if state == goal_state:
            path = reconstruct_path(parents, start_state, goal_state)
            return path, nodes, len(path) - 1, g
        if state not in visited:
            visited.add(state)
            for next_state in get_next_states(state):
                new_g = g + 1
                if next_state not in g_scores or new_g < g_scores[next_state]:
                    g_scores[next_state] = new_g
                    parents[next_state] = state
                    new_f = new_g + heuristic(next_state, goal_state)
                    nodes_expanded += 1
                    heapq.heappush(queue, (new_f, new_g, next_state, nodes_expanded))
    return None, nodes_expanded, 0, 0

def ida_star(start_state, goal_state):
    def search(state, g, threshold, path, nodes_expanded):
        f = g + heuristic(state, goal_state)
        if f > threshold:
            return f, None
        if state == goal_state:
            return f, path
        min_threshold = float('inf')
        next_states = sorted(get_next_states(state), key=lambda x: heuristic(x, goal_state))
        for next_state in next_states:
            if next_state not in path:
                nodes_expanded[0] += 1
                new_path = path + [next_state]
                new_f, result = search(next_state, g + 1, threshold, new_path, nodes_expanded)
                if result:
                    return new_f, result
                min_threshold = min(min_threshold, new_f)
        return min_threshold, None

    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    threshold = heuristic(start_state, goal_state)
    path = [start_state]
    nodes_expanded = [0]
    while True:
        new_threshold, result = search(start_state, 0, threshold, path, nodes_expanded)
        if result:
            return result, nodes_expanded[0], len(result) - 1, len(result) - 1
        if new_threshold == float('inf'):
            return None, nodes_expanded[0], 0, 0
        threshold = new_threshold

def simple_hill_climbing(start_state, goal_state, max_steps=500):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    state = start_state
    h = heuristic(state, goal_state)
    path = [state]
    visited = {state}
    nodes = 0
    for _ in range(max_steps):
        if state == goal_state:
            break
        next_states = sorted(get_next_states(state), key=lambda s: heuristic(s, goal_state))
        nodes += len(next_states)
        found_better = False
        for next_state in next_states:
            if next_state not in visited:
                next_h = heuristic(next_state, goal_state)
                if next_h < h:
                    state, h = next_state, next_h
                    path.append(state)
                    visited.add(state)
                    found_better = True
                    break
        if not found_better:
            unvisited = [s for s in next_states if s not in visited]
            if unvisited and random.random() < 0.1:
                state = random.choice(unvisited)
                h = heuristic(state, goal_state)
                path.append(state)
                visited.add(state)
            else:
                break
    return path, nodes, len(path) - 1, len(path) - 1

def steepest_ascent_hill_climbing(start_state, goal_state, max_steps=500):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    state, h = start_state, heuristic(start_state, goal_state)
    path, nodes = [state], 0
    visited = {state}
    for _ in range(max_steps):
        if state == goal_state:
            return path, nodes, len(path) - 1, len(path) - 1
        next_states = get_next_states(state)
        nodes += len(next_states)
        best_state, best_h = state, h
        for next_state in next_states:
            if next_state not in visited:
                next_h = heuristic(next_state, goal_state)
                if next_h < best_h:
                    best_state, best_h = next_state, next_h
        if best_state != state:
            state, h = best_state, best_h
            path.append(state)
            visited.add(state)
        else:
            unvisited = [s for s in next_states if s not in visited]
            if unvisited and random.random() < 0.1:
                state = random.choice(unvisited)
                h = heuristic(state, goal_state)
                path.append(state)
                visited.add(state)
            else:
                break
    return path, nodes, len(path) - 1, len(path) - 1

def stochastic_hill_climbing(start_state, goal_state, max_steps=500):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    state = start_state
    h = heuristic(state, goal_state)
    path = [state]
    visited = {state}
    nodes = 0
    for _ in range(max_steps):
        if state == goal_state:
            break
        next_states = get_next_states(state)
        nodes += len(next_states)
        better_states = [(s, heuristic(s, goal_state)) for s in next_states
                         if s not in visited and heuristic(s, goal_state) < h]
        if better_states:
            state, h = random.choice(better_states)
            path.append(state)
            visited.add(state)
        else:
            unvisited = [s for s in next_states if s not in visited]
            if unvisited and random.random() < 0.1:
                state = random.choice(unvisited)
                h = heuristic(state, goal_state)
                path.append(state)
                visited.add(state)
            else:
                break
    return path, nodes, len(path) - 1, len(path) - 1

def simulated_annealing(start_state, goal_state, initial_temp=100, cooling_rate=0.99, max_steps=500):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    state, h = start_state, heuristic(start_state, goal_state)
    path, nodes = [state], 0
    temp = initial_temp
    for _ in range(max_steps):
        if state == goal_state:
            return path, nodes, len(path) - 1, len(path) - 1
        next_states = get_next_states(state)
        nodes += len(next_states)
        next_state = random.choice(next_states)
        next_h = heuristic(next_state, goal_state)
        delta = next_h - h
        if delta < 0 or random.random() < math.exp(-delta / max(temp, 1e-6)):
            state, h = next_state, next_h
            path.append(state)
        temp *= cooling_rate
        if temp < 1e-6:
            break
    return path, nodes, len(path) - 1, len(path) - 1

def beam_search(start_state, goal_state, beam_width=4):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    
    queue = [(heuristic(start_state, goal_state), 0, start_state)]
    parents = {start_state: None}
    visited = set()
    nodes_expanded = 0
    
    heapq.heapify(queue)
    
    while queue:
        if len(queue) > beam_width:
            queue = heapq.nsmallest(beam_width, queue)
            heapq.heapify(queue)
        
        h, cost, state = heapq.heappop(queue)
        if state == goal_state:
            path = reconstruct_path(parents, start_state, goal_state)
            return path, nodes_expanded, len(path) - 1, cost
        
        if state not in visited:
            visited.add(state)
            for next_state in get_next_states(state):
                if next_state not in visited:
                    new_cost = cost + 1
                    new_h = new_cost + heuristic(next_state, goal_state)
                    parents[next_state] = state
                    nodes_expanded += 1
                    heapq.heappush(queue, (new_h, new_cost, next_state))
    
    return None, nodes_expanded, 0, 0

def and_or_search(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    stack = [(start_state, [start_state], 0)]
    visited = set()
    while stack:
        state, path, nodes_expanded = stack.pop()
        if state == goal_state:
            return path, nodes_expanded, len(path) - 1, len(path) - 1
        if state not in visited:
            visited.add(state)
            next_states = get_next_states(state)
            next_states.sort(key=lambda x: heuristic(x, goal_state))
            for next_state in reversed(next_states):
                if next_state not in visited:
                    stack.append((next_state, path + [next_state], nodes_expanded + 1))
    return None, 0, 0, 0

def belief_state_search(initial_belief_states, goal_state):
    if not any(is_solvable(state, goal_state) for state in initial_belief_states):
        return None, 0, 0, 0
    initial_state = next(state for state in initial_belief_states if is_solvable(state, goal_state))
    initial_belief = frozenset(initial_belief_states)
    queue = [(heuristic(initial_state, goal_state), 0, initial_belief, initial_state, [(initial_belief, initial_state)], [])]
    visited = {initial_belief: 0}
    nodes_expanded = 0
    actions = ["Up", "Down", "Left", "Right"]

    def apply_action_to_state(state, action):
        row, col = find_zero(state)
        if action == "Up" and row > 0:
            new_row, new_col = row - 1, col
        elif action == "Down" and row < 2:
            new_row, new_col = row + 1, col
        elif action == "Left" and col > 0:
            new_row, new_col = row, col - 1
        elif action == "Right" and col < 2:
            new_row, new_col = row, col + 1
        else:
            return state
        old_idx = row * 3 + col
        new_idx = new_row * 3 + new_col
        state_list = list(state)
        state_list[old_idx], state_list[new_idx] = state_list[new_idx], state_list[old_idx]
        return ''.join(state_list)

    def apply_action_to_belief(belief_state, action):
        new_states = set()
        for state in belief_state:
            new_state = apply_action_to_state(state, action)
            new_states.add(new_state)
        return frozenset(new_states)

    def belief_heuristic(belief):
        return min(heuristic(state, goal_state) for state in belief)

    heapq.heapify(queue)
    while queue:
        _, g, current_belief, current_state, path, action_path = heapq.heappop(queue)
        if goal_state in current_belief:
            state_path = [state for _, state in path]
            if state_path[-1] != goal_state:
                state_path.append(goal_state)
            return state_path, nodes_expanded, len(state_path) - 1, g
        for action in actions:
            next_belief = apply_action_to_belief(current_belief, action)
            next_state = apply_action_to_state(current_state, action)
            new_g = g + 1
            if next_belief not in visited or new_g < visited[next_belief]:
                visited[next_belief] = new_g
                nodes_expanded += 1
                h = belief_heuristic(next_belief)
                f = new_g + h
                heapq.heappush(queue, (f, new_g, next_belief, next_state, path + [(next_belief, next_state)], action_path + [action]))
    return None, nodes_expanded, 0, 0

def backtracking_search(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    visited = set()
    nodes_expanded = [0]

    def backtrack(state, path, depth):
        if state == goal_state:
            return path, True
        if state in visited:
            return None, False
        visited.add(state)
        next_states = get_next_states(state)
        nodes_expanded[0] += len(next_states)
        next_states.sort(key=lambda x: heuristic(x, goal_state))
        for next_state in next_states:
            if next_state not in visited:
                result, success = backtrack(next_state, path + [next_state], depth + 1)
                if success:
                    return result, True
        return None, False

    path, success = backtrack(start_state, [start_state], 0)
    if not success:
        return None, nodes_expanded[0], 0, 0
    return path, nodes_expanded[0], len(path) - 1, len(path) - 1

def genetic_algorithm(start_state, goal_state, population_size=50, generations=100, mutation_rate=0.1):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    actions = ["Up", "Down", "Left", "Right"]
    nodes_expanded = 0

    def apply_moves(state, moves):
        current = state
        for move in moves:
            row, col = find_zero(current)
            if move == "Up" and row > 0:
                new_row, new_col = row - 1, col
            elif move == "Down" and row < 2:
                new_row, new_col = row + 1, col
            elif move == "Left" and col > 0:
                new_row, new_col = row, col - 1
            elif move == "Right" and col < 2:
                new_row, new_col = row, col + 1
            else:
                continue
            old_idx = row * 3 + col
            new_idx = new_row * 3 + new_col
            state_list = list(current)
            state_list[old_idx], state_list[new_idx] = state_list[new_idx], state_list[old_idx]
            current = ''.join(state_list)
        return current

    def fitness(moves):
        state = apply_moves(start_state, moves)
        h = heuristic(state, goal_state)
        penalty = sum(1 for i in range(len(moves) - 1) if moves[i] == "Up" and moves[i + 1] == "Down" or 
                      moves[i] == "Down" and moves[i + 1] == "Up" or 
                      moves[i] == "Left" and moves[i + 1] == "Right" or 
                      moves[i] == "Right" and moves[i + 1] == "Left")
        return h + penalty

    def generate_individual(length):
        return [random.choice(actions) for _ in range(length)]

    def crossover(parent1, parent2):
        point = random.randint(1, len(parent1) - 1)
        return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

    def mutate(individual):
        if random.random() < mutation_rate:
            idx = random.randint(0, len(individual) - 1)
            individual[idx] = random.choice(actions)
        return individual

    population = [generate_individual(random.randint(5, 20)) for _ in range(population_size)]
    best_individual = None
    best_fitness = float('inf')

    for _ in range(generations):
        population = sorted(population, key=fitness)
        if fitness(population[0]) < best_fitness:
            best_individual = population[0]
            best_fitness = fitness(best_individual)
        final_state = apply_moves(start_state, best_individual)
        nodes_expanded += population_size
        if final_state == goal_state:
            path = [start_state]
            current = start_state
            for move in best_individual:
                next_state = apply_moves(current, [move])
                if next_state != current:
                    path.append(next_state)
                    current = next_state
            return path, nodes_expanded, len(path) - 1, len(path) - 1
        new_population = population[:population_size // 2]
        while len(new_population) < population_size:
            parent1, parent2 = random.choice(new_population), random.choice(new_population)
            child1, child2 = crossover(parent1[:], parent2[:])
            child1, child2 = mutate(child1), mutate(child2)
            new_population.extend([child1, child2])
        population = new_population[:population_size]

    path = [start_state]
    current = start_state
    for move in best_individual:
        next_state = apply_moves(current, [move])
        if next_state != current:
            path.append(next_state)
            current = next_state
    return path, nodes_expanded, len(path) - 1, len(path) - 1

def ac3_search(start_state, goal_state):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    
    # Bước 1: Kiểm tra tính hợp lệ của start_state bằng AC-3
    # Xây dựng domains: mỗi ô có domain là giá trị hiện tại (trừ ô trống)
    domains = {}
    for i in range(9):
        if start_state[i] == '0':
            domains[i] = list(range(9))  # Ô trống có thể là bất kỳ số nào
        else:
            domains[i] = [int(start_state[i])]  # Ô cố định chỉ có giá trị hiện tại
    
    # Xây dựng constraints: mỗi ô chỉ chứa một số duy nhất
    values = [int(start_state[i]) for i in range(9) if start_state[i] != '0']
    if len(values) != len(set(values)):
        return None, 0, 0, 0  # Nếu có số trùng lặp, không hợp lệ
    
    zero_pos = start_state.index('0')
    for val in values:
        if val in domains[zero_pos]:
            domains[zero_pos].remove(val)
    
    if not domains[zero_pos]:
        return None, 0, 0, 0
    
    queue = deque([(start_state, [start_state], 0)])  
    visited = {start_state}
    nodes_expanded = 0
    
    while queue:
        state, path, nodes = queue.popleft()
        if state == goal_state:
            return path, nodes_expanded, len(path) - 1, len(path) - 1
        for next_state in get_next_states(state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((next_state, path + [next_state], nodes + 1))
                nodes_expanded += 1
    
    return None, nodes_expanded, 0, 0

def kiem_thu(start_state, goal_state, max_steps=500):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    state = start_state
    h = heuristic(state, goal_state)
    path = [state]
    visited = {state}
    nodes_expanded = 0
    for _ in range(max_steps):
        if state == goal_state:
            return path, nodes_expanded, len(path) - 1, len(path) - 1
        next_states = get_next_states(state)
        nodes_expanded += len(next_states)
        unvisited = [s for s in next_states if s not in visited]
        if not unvisited:
            break
        heuristics = [(s, heuristic(s, goal_state)) for s in unvisited]
        if random.random() < 0.7 and heuristics:
            next_state = min(heuristics, key=lambda x: x[1])[0]
        else:
            next_state = random.choice(unvisited)
        state = next_state
        h = heuristic(state, goal_state)
        path.append(state)
        visited.add(state)
    return path, nodes_expanded, len(path) - 1, len(path) - 1

# Thêm các hàm hỗ trợ cho Q-Learning
class QLearningSolver:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(4))  
        self.actions = ["up", "down", "left", "right"]
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.3
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.01
        
    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        state_actions = self.q_table[state]
        return self.actions[np.argmax(state_actions)]
    
    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][self.actions.index(action)]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][self.actions.index(action)] = new_q
        
    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
# môi trường phức tạp, học cải thiện 
def q_learning(start_state, goal_state, max_episodes=1000, max_steps=100):
    if not is_solvable(start_state, goal_state):
        return None, 0, 0, 0
    
    solver = QLearningSolver()
    nodes_expanded = 0
    best_path = None
    best_steps = float('inf')
    
    for episode in range(max_episodes):
        state = start_state
        path = [state]
        total_reward = 0
        
        for step in range(max_steps):
            action = solver.get_action(state)
            next_state = apply_action(state, action)
            
            if not next_state:
                reward = -10  
                next_state = state
            else:
                nodes_expanded += 1
                path.append(next_state)
                h_current = heuristic(state, goal_state)
                h_next = heuristic(next_state, goal_state)
                reward = (h_current - h_next) * 10  # Phần thưởng dựa trên cải thiện heuristic
                
                if next_state == goal_state:
                    reward = 100  # Phần thưởng lớn khi đạt mục tiêu
                    if len(path) < best_steps:
                        best_path = path.copy()
                        best_steps = len(path) - 1
                    break
            
            solver.update_q_table(state, action, reward, next_state)
            state = next_state
        
        solver.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {solver.epsilon:.3f}, Best Steps: {best_steps}")
    
    if best_path:
        return best_path, nodes_expanded, len(best_path) - 1, len(best_path) - 1
    return None, nodes_expanded, 0, 0

def apply_action(state, action):
    """Áp dụng hành động lên trạng thái hiện tại"""
    row, col = find_zero(state)
    state_list = list(state)
    
    if action == "up" and row > 0:
        new_row, new_col = row - 1, col
    elif action == "down" and row < 2:
        new_row, new_col = row + 1, col
    elif action == "left" and col > 0:
        new_row, new_col = row, col - 1
    elif action == "right" and col < 2:
        new_row, new_col = row, col + 1
    else:
        return None  # Hành động không hợp lệ
    
    old_idx = row * 3 + col
    new_idx = new_row * 3 + new_col
    state_list[old_idx], state_list[new_idx] = state_list[new_idx], state_list[old_idx]
    return ''.join(state_list)

#
def draw_board(state, screen):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            idx = i * GRID_SIZE + j
            num = state[idx]
            rect = pygame.Rect(GRID_OFFSET_X + j * CELL_SIZE, GRID_OFFSET_Y + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, EMPTY_COLOR if num == '0' else TILE_COLOR, rect)
            if num != '0':
                text = BUTTON_FONT.render(num, True, TEXT_COLOR)
                screen.blit(text, text.get_rect(center=rect.center))
            pygame.draw.rect(screen, BORDER_COLOR, rect, 2)

def draw_button(screen, text, rect, is_hover=False, is_clicked=False, is_solving=False, is_selected=False):
    if is_selected:
        color = SELECTED_ALGO_COLOR
    else:
        color = BUTTON_CLICK if is_clicked else BUTTON_HOVER if is_hover else BUTTON_COLOR
    pygame.draw.rect(screen, color, rect, border_radius=5)
    display_text = f"{text} ({'On' if is_solving else 'Off'})" if "Solve" in text else text
    text_surf = BUTTON_FONT.render(display_text, True, TEXT_COLOR)
    screen.blit(text_surf, text_surf.get_rect(center=rect.center))

def draw_input_box(screen, text, rect, is_active=False):
    color = (100, 100, 100) if is_active else (50, 50, 50)
    pygame.draw.rect(screen, color, rect, border_radius=5)
    text_surf = BUTTON_FONT.render(text, True, TEXT_COLOR)
    screen.blit(text_surf, text_surf.get_rect(center=rect.center))

def draw_algorithm_grid(screen, algorithms, selected_algorithm, mouse_pos):
    algo_buttons = []
    cols, rows = 3, 6
    button_width, button_height = 70, 25
    padding_x, padding_y = 4, 4
    start_x, start_y = 30, 270
    grid_width = cols * button_width + (cols - 1) * padding_x + 10
    grid_height = rows * button_height + (rows - 1) * padding_y + 10
    grid_rect = pygame.Rect(start_x - 5, start_y - 5, grid_width, grid_height)
    pygame.draw.rect(screen, (45, 45, 45), grid_rect, border_radius=8)
    pygame.draw.rect(screen, BORDER_COLOR, grid_rect, 2, border_radius=8)
    algo_list = list(algorithms.keys())
    algo_font = pygame.font.SysFont("Arial", 14)
    for i in range(len(algo_list)):
        row = i // cols
        col = i % cols
        x = start_x + col * (button_width + padding_x)
        y = start_y + row * (button_height + padding_y)
        rect = pygame.Rect(x, y, button_width, button_height)
        is_hover = rect.collidepoint(mouse_pos)
        is_clicked = pygame.mouse.get_pressed()[0] and is_hover
        is_selected = algo_list[i] == selected_algorithm
        if is_selected:
            color = SELECTED_ALGO_COLOR
        elif is_hover:
            color = BUTTON_HOVER
        else:
            color = BUTTON_COLOR
        pygame.draw.rect(screen, color, rect, border_radius=5)
        pygame.draw.rect(screen, BORDER_COLOR, rect, 1, border_radius=5)
        text_surf = algo_font.render(algo_list[i], True, TEXT_COLOR)
        screen.blit(text_surf, text_surf.get_rect(center=rect.center))
        algo_buttons.append((rect, algo_list[i]))
    return algo_buttons

def draw_results(screen, elapsed_time, nodes, depth, cost, steps, solved, solving):
    screen.blit(TIMER_FONT.render("Results", True, TEXT_COLOR), (650, 150))
    status_text = "SOLVING..." if solving else "SOLVED!" if solved else "NOT SOLVED YET!"
    status_color = SOLVING_COLOR if solving else SOLVED_COLOR if solved else NOT_SOLVED_COLOR
    screen.blit(TIMER_FONT.render(status_text, True, status_color), (650, 190))
    screen.blit(TIMER_FONT.render(f"Runtime: {elapsed_time:.2f} sec", True, TEXT_COLOR), (650, 225))
    screen.blit(TIMER_FONT.render(f"Search Depth: {depth}", True, TEXT_COLOR), (650, 260))
    screen.blit(TIMER_FONT.render(f"Path to Goal: {steps}", True, TEXT_COLOR), (650, 300))
    screen.blit(TIMER_FONT.render(f"Nodes Expanded: {nodes}", True, TEXT_COLOR), (650, 335))

def draw_comparison_chart():
    global comparison_image, show_comparison
    if not algorithm_results:
        show_message_box("Vui lòng chạy ít nhất một thuật toán để so sánh!")
        return
    
    names = [result[0] for result in algorithm_results]
    runtimes = [round(result[1], 3) for result in algorithm_results]  # Làm tròn đến 3 chữ số thập phân
    path_costs = [result[2] if result[2] != float('inf') else 0 for result in algorithm_results]
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots()
    runtime_bars = ax.bar(x - width/2, runtimes, width, label='Thời gian (s)', color='skyblue')
    cost_bars = ax.bar(x + width/2, path_costs, width, label='Chi phí đường đi', color='salmon')
    
    # Thêm giá trị số trên mỗi cột
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}' if isinstance(height, float) else f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # Điều chỉnh vị trí text
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    add_labels(runtime_bars)
    add_labels(cost_bars)
    
    ax.set_xlabel('Thuật toán')
    ax.set_title('So sánh thời gian chạy và chi phí đường đi')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_chart.png')
    plt.close()
    
    comparison_image = pygame.image.load('comparison_chart.png')
    comparison_image = pygame.transform.scale(comparison_image, (600, 500))
    show_comparison = True

def swap_tiles(state, pos):
    row, col = pos
    zero_row, zero_col = find_zero(state)
    if abs(row - zero_row) + abs(col - zero_col) == 1:
        state_list = list(state)
        zero_idx, click_idx = zero_row * 3 + zero_col, row * 3 + col
        state_list[zero_idx], state_list[click_idx] = state_list[click_idx], state_list[zero_idx]
        return ''.join(state_list)
    return state

def show_message_box(message):
    root = tk.Tk()
    root.withdraw()
    result = messagebox.askokcancel("Thông báo", message)
    root.destroy()
    return result

def shuffle_state(state):
    state_list = list(state)
    for _ in range(10):
        next_states = get_next_states(''.join(state_list))
        state_list = list(random.choice(next_states))
    return ''.join(state_list)

# Algorithm dictionary
algorithms = {
    "BFS": bfs, "DFS": dfs, "UCS": ucs, "IDS": ids, 
    "Greedy": greedy_search, "A*": astar, "IDA*": ida_star,
    "Beam": beam_search, 
    "Simple HC": simple_hill_climbing, 
    "Steepest HC": steepest_ascent_hill_climbing,
    "SHC": stochastic_hill_climbing, 
    "SA": simulated_annealing,
    "AND-OR": and_or_search,
    "Belief": belief_state_search,
    "Backtrack": backtracking_search,
    "Genetic": genetic_algorithm,  
    "AC-3": ac3_search,
    "Kiem Thu": kiem_thu,
     "Q-Learning": q_learning,
}

# Game variables
selected_algorithm = "DFS"
path = None
current_state = start_state
current_step = 0
auto_play = False
auto_speed = 0.5
last_step_time = 0
running = True
solved = False
solving = False
is_solving = False
timer_start = None
elapsed_time = 0
nodes_expanded = 0
search_depth = 0
path_cost = 0
total_steps = 0
is_shuffled = False

# UI elements
start_rect = pygame.Rect(50, 150, 180, 40)
goal_rect = pygame.Rect(50, 220, 180, 40)
solve_rect = pygame.Rect(650, 400, 180, 40)
reset_rect = pygame.Rect(650, 450, 180, 40)
shuffle_rect = pygame.Rect(650, 500, 180, 40)
compare_rect = pygame.Rect(650, 550, 180, 40)

clock = pygame.time.Clock()

def reset_game():
    global start_input, goal_input, start_state, goal_state, path, current_step, current_state, auto_play, solved, solving, is_solving, timer_start, elapsed_time, nodes_expanded, search_depth, path_cost, total_steps, is_shuffled, algorithm_results, show_comparison, comparison_image
    start_input, goal_input = "265087431", "123456780"
    start_state, goal_state = start_input, goal_input
    path, current_step, current_state = None, 0, start_state
    auto_play, solved, solving, is_solving = False, False, False, False
    timer_start, elapsed_time, nodes_expanded, search_depth, path_cost, total_steps, is_shuffled = None, 0, 0, 0, 0, 0, False
    algorithm_results = []
    show_comparison = False
    comparison_image = None

def solve_puzzle():
    global is_solving, start_state, goal_state, solving, path, nodes_expanded, search_depth, path_cost, total_steps, current_step, current_state, timer_start, last_step_time, auto_play, solved, elapsed_time, algorithm_results
    if is_shuffled and not show_message_box("Vui lòng reset trước khi giải!"):
        return
    is_solving = True
    start_state, goal_state = start_input, goal_input
    solving = True
    timer_start = time.time()
    if selected_algorithm == "Belief":
        path, nodes_expanded, search_depth, path_cost = belief_state_search({start_state}, goal_state)
    else:
        path, nodes_expanded, search_depth, path_cost = algorithms[selected_algorithm](start_state, goal_state)
    elapsed_time = time.time() - timer_start
    for i, result in enumerate(algorithm_results):
        if result[0] == selected_algorithm:
            algorithm_results[i] = (selected_algorithm, elapsed_time, path_cost)
            break
    else:
        algorithm_results.append((selected_algorithm, elapsed_time, path_cost if path else float('inf')))
    if not path or (path[-1] != goal_state and selected_algorithm in ["Stochastic HC", "Simulated Annealing"]):
        print(f"Không tìm thấy lời giải với {selected_algorithm}!")
        path = [start_state]
        total_steps = 0
        solved = solving = is_solving = False
    else:
        total_steps = len(path) - 1
        print(f"Giải thành công với {selected_algorithm} trong {total_steps} bước.")
        auto_play = True
        solved = False
    current_step, current_state = 0, start_state
    last_step_time = time.time()

# Main loop
while running:
    clock.tick(60)
    screen.fill(BG_COLOR)  

    mouse_pos = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    rect = pygame.Rect(GRID_OFFSET_X + j * CELL_SIZE, GRID_OFFSET_Y + i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    if rect.collidepoint(mouse_x, mouse_y):
                        current_state = swap_tiles(current_state, (i, j))
                        if current_state != start_state:
                            is_shuffled = True
                        break
            algo_buttons = draw_algorithm_grid(screen, algorithms, selected_algorithm, mouse_pos)
            for rect, algo in algo_buttons:
                if rect.collidepoint(event.pos):
                    selected_algorithm = algo
                    break
            if start_rect.collidepoint(event.pos):
                input_active = "start"
            elif goal_rect.collidepoint(event.pos):
                input_active = "goal"
            else:
                input_active = None
            if solve_rect.collidepoint(event.pos):
                if is_solving:
                    is_solving = auto_play = solving = solved = False
                    timer_start = elapsed_time = 0
                else:
                    solve_puzzle()
            elif reset_rect.collidepoint(event.pos):
                reset_game()
            elif shuffle_rect.collidepoint(event.pos):
                current_state = shuffle_state(current_state)
                is_shuffled = True
            elif compare_rect.collidepoint(event.pos):
                draw_comparison_chart()
                show_comparison = True
            elif show_comparison and not comparison_rect.collidepoint(event.pos):
                show_comparison = False
        elif event.type == pygame.KEYDOWN and input_active:
            if event.key == pygame.K_BACKSPACE:
                if input_active == "start" and start_input:
                    start_input = start_input[:-1]
                elif input_active == "goal" and goal_input:
                    goal_input = goal_input[:-1]
            elif event.unicode.isdigit() and event.unicode in "012345678" and len(locals()[f"{input_active}_input"]) < 9:
                locals()[f"{input_active}_input"] += event.unicode

    # Update timer if solving
    if timer_start and not solved:
        elapsed_time = time.time() - timer_start

    # Auto-play logic
    if auto_play and path and current_step < len(path) - 1:
        if time.time() - last_step_time >= auto_speed:
            current_step += 1
            current_state = path[current_step]
            last_step_time = time.time()
            if current_step == len(path) - 1:
                auto_play = solving = is_solving = False
                solved = True

    draw_board(current_state, screen)  
    screen.blit(TIMER_FONT.render("Initial state", True, TEXT_COLOR), (50, 120))
    draw_input_box(screen, start_input, start_rect, input_active == "start")
    screen.blit(TIMER_FONT.render("Goal state", True, TEXT_COLOR), (50, 190))
    draw_input_box(screen, goal_input, goal_rect, input_active == "goal")
    algo_buttons = draw_algorithm_grid(screen, algorithms, selected_algorithm, mouse_pos)  # Draw algorithm grid
    draw_results(screen, elapsed_time, nodes_expanded, search_depth, path_cost, total_steps, solved, solving)

    # Draw buttons
    is_solve_hover = solve_rect.collidepoint(mouse_pos)
    is_reset_hover = reset_rect.collidepoint(mouse_pos)
    is_shuffle_hover = shuffle_rect.collidepoint(mouse_pos)
    is_compare_hover = compare_rect.collidepoint(mouse_pos)
    draw_button(screen, "Solve Puzzle", solve_rect, is_solve_hover, pygame.mouse.get_pressed()[0] and is_solve_hover, is_solving)
    draw_button(screen, "Reset", reset_rect, is_reset_hover, pygame.mouse.get_pressed()[0] and is_reset_hover)
    draw_button(screen, "Shuffle", shuffle_rect, is_shuffle_hover, pygame.mouse.get_pressed()[0] and is_shuffle_hover)
    draw_button(screen, "So Sánh", compare_rect, is_compare_hover, pygame.mouse.get_pressed()[0] and is_compare_hover)

    if show_comparison and comparison_image:
        try:
            screen.blit(comparison_image, comparison_rect.topleft)
        except Exception as e:
            print(f"Error displaying comparison image: {e}")
            show_comparison = False

    # Check if the current state matches the goal state
    if current_state == goal_state and not auto_play:
        solved, solving, timer_start = True, False, None

    pygame.display.flip()

pygame.quit()