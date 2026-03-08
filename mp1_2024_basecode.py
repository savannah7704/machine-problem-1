#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: szczurpi

This program implements the uniform cost search algorithm for solving a maze
(1 point cost for each move)
THIS IS CHANGED TO A* SEARCH WITH A HEURISTIC FUNCTION.

Artificial Intelligence
MP1: Robot Navigation
SEMESTER: Spring 2026
NAME: Savannah Stumpf, Alexander Tardecilla, and Anthony Viglielmo
"""


import numpy as np
import queue # Needed for frontier queue
from heapq import heapify

class MazeState():
    """ Stores information about each visited state within the search """
    # Define constants
    SPACE = 0
    WALL = 1
    EXIT = 2
    VISITED = 3
    PATH = 4
    START_MARK = 5
    END_MARK = 6

    MAZE_FILE = 'maze2024.txt'
    maze = np.loadtxt(MAZE_FILE, dtype=np.int32)  
    start = tuple(np.array(np.where(maze==5)).flatten())
    ends = list(zip(np.where(maze==2)[0], np.where(maze==2)[1])) # List of exit positions (in case there are multiple exits in the maze)
    move_num = 0 # Used by show_path() to count moves in the solution path
    
    def reset_state():
        """ Resets the static variables to prepare for a new search """
        MazeState.maze = np.loadtxt(MazeState.MAZE_FILE, dtype=np.int32)  
        MazeState.start = tuple(np.array(np.where(MazeState.maze==5)).flatten())
        MazeState.ends = list(zip(np.where(MazeState.maze==2)[0], np.where(MazeState.maze==2)[1]))
        MazeState.move_num = 0
    
    def __init__(self, conf=start, g=0, pred_state=None, pred_action=None):
        """ Initializes the state with information passed from the arguments """
        self.pos = conf         # Configuration of the state - current coordinates
        self.gcost = g          # Path cost
        self.pred = pred_state  # Predecesor state
        self.action_from_pred = pred_action  # Action from predecesor state to current state
    
    def __hash__(self):
        """ Returns a hash code so that it can be stored in a set data structure """
        return self.pos.__hash__()
    
    def is_goal(self):
        """ Returns true if current position is same as the exit position """
        return self.maze[self.pos] == MazeState.EXIT
    
    def __eq__(self, other):
        """ Checks for equality of states by positions only """
        return self.pos == other.pos
    
    def __lt__(self, other):
        """ Allows for ordering the states by the path (g) cost """
        return (self.gcost + self.heuristic()) < (other.gcost + other.heuristic())  
    
    def __str__(self):
        """ Returns the maze representation of the state """
        a = np.array(self.maze)
        a[self.start] = MazeState.START_MARK
        a[self.ends] = MazeState.EXIT
        return str(a)

    def show_path(self):
        """ Recursively outputs the list of moves and states along path """
        if self.pred is not None:
            self.pred.show_path()
        
        if MazeState.move_num==0:
            print('START')
        else:
            print('Move',MazeState.move_num, 'ACTION:', self.action_from_pred)
        MazeState.move_num = MazeState.move_num + 1
        self.maze[self.pos] = MazeState.PATH

    def heuristic(self):
        """Heuristic function for A* search"""

        rows, cols = self.maze.shape # Get the total number of rows and columns in the maze
        x1, y1 = self.pos #get the current position of the agent
        best = float('inf') # initialize best to infinity

        for (x2, y2) in MazeState.ends: # Loop through all exit positions
            dx = abs(x1 - x2) # calculate the horizontal distance to the exit
            dy = abs(y1 - y2) #calculate the vertical distance to the exit

            dx = min(dx, rows - dx) #account for wrap-around in horizontal direction
            dy = min(dy, cols - dy) #account for wrap-around in vertical direction

            distance = dx + dy #calculate the distance to the exit
            best = min(best, distance) #update best if this exit is closer than the previously found closest exit

        return best


    def get_new_pos(self, move):
        """ Returns a new position from the current position and the specified move """
        rows = self.maze.shape[0]
        cols = self.maze.shape[1]
        match move:
            case 'up':
                if self.pos[0] - 1 >= 0:
                    new_pos = (self.pos[0]-1, self.pos[1])
                else: # Wrap around to bottom
                    new_pos = (rows - 1,self.pos[1])
            case 'down':
                if self.pos[0] + 1 < rows:
                    new_pos = (self.pos[0] + 1, self.pos[1])
                else: # Wrap around to top
                    new_pos = (0, self.pos[1])
            case 'left':
                if self.pos[1] - 1 >= 0:
                    new_pos = (self.pos[0], self.pos[1] - 1)
                else: # Wrap around to right
                    new_pos = (self.pos[0], cols - 1)
            case 'right':
                if self.pos[1] + 1 < cols:
                    new_pos = (self.pos[0], self.pos[1] + 1)
                else: # Wrap around to left
                    new_pos = (self.pos[0], 0)
            case _:
                raise('wrong direction for checking move')
        return new_pos
        
    def can_move(self, move):
        """ Returns true if agent can move in the given direction """
        new_pos = self.get_new_pos(move)
        return self.maze[new_pos]!=MazeState.WALL
                    
    def gen_next_state(self, move):
        """ Generates a new MazeState object by taking move from current state """
        new_pos = self.get_new_pos(move)
        if self.maze[new_pos] != MazeState.EXIT:
            self.maze[new_pos] = MazeState.VISITED
        return MazeState(new_pos, self.gcost+1, self, move)
            
# Display the heading info
print('Artificial Intelligence')
print('MP1: Robot navigation')
print('SEMESTER: Spring 2026')
print('NAME: Savannah Stumpf, Alexander Tardecilla, and Anthony Viglielmo')
print()

print('INITIAL MAZE')
"""
# load start state onto frontier priority queue
frontier = queue.PriorityQueue() # This does best-first search
#frontier = queue.LifoQueue() # This would do depth-first search
#frontier = queue.Queue() # This would do breadth-first search

start_state = MazeState()
frontier.put(start_state)
print(start_state)
# Keep a closed set of states to which optimal path was already found
closed_set = set()
"""

# Expand state (up to 4 moves possible)
possible_moves = [
    ['right','down','up'], # disable left
    ['left', 'down', 'up'], # disable right
    ['left', 'right', 'up'], # disable down
    ['left', 'right', 'down'] # disable up
]

best_move = None
best_length = float('inf')
best_states = 0

print(MazeState.maze)

for i, moves in enumerate(possible_moves):
    disabled_move = (set(['up', 'down', 'left', 'right']) - set(moves)).pop()
    MazeState.reset_state()
    MazeState.move_num = 0
    frontier = queue.PriorityQueue()
    start_state = MazeState()
    frontier.put(start_state)
    closed_set = set()
    found = False
    num_states = 0

    print(f"\nSOLUTION AFTER DISABLED MOVE: {disabled_move}")
    
    while not frontier.empty():
        # Choose state at front of priority queue
        next_state = frontier.get()
        num_states = num_states + 1
    
        # If goal then quit and return path
        if next_state.is_goal():
            found = True
            next_state.show_path()
            print(next_state)
            break
    
        # Add state chosen for expansion to closed_set
        closed_set.add(next_state)
  
        # Expanding the node
        for move in moves:
            if next_state.can_move(move):
                neighbor = next_state.gen_next_state(move)
                if neighbor in closed_set:
                    continue
                if neighbor not in frontier.queue:                           
                    frontier.put(neighbor)
                else:
                    if neighbor.gcost < frontier.queue[frontier.queue.index(neighbor)].gcost:
                        frontier.queue[frontier.queue.index(neighbor)] = neighbor
                        heapify(frontier.queue)
    
    move_path_length = MazeState.move_num-1
    
    if found == False:
        MazeState.reset_state()
        print(MazeState.maze)
        print('\nNo solution')
    else:
        print('\nNumber of states visited =', num_states)
        print('\nLength of shortest path = ', move_path_length)
        if move_path_length < best_length:
            best_length = move_path_length
            best_move = disabled_move


print('BEST MOVE: disable ', best_move)
print('SHORTEST PATH LENGTH FOR BEST MOVE: ', best_length)