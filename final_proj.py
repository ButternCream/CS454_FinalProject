# make a DFA that accepts strings over {a,b,c} such that eveyr substring
# of length 4 has at least one occurence of each letter a,b,c.

# how to make DFA
# hardest part is -  how do i do the transition function
# create a dictionary. key = state_name + inputsymbol  the element is state_name to switch to
# once i have the dictionary, i create an array of state_name 

import numpy as np
import itertools
import threading
import sys
import time
import progressbar

DONE = False

# Scott's state class
class state:
	def __init__(self, name, parent=None, final=False):
		"""
		Constructor
		:param name: States name
		:param parent: The parent of the current state
		:param final: if it's a final state
		"""
		self.name = name
		self.transition_list = dict() # Empty map to append to later. Pretty much a map of state transitions
		self.parent = parent # Parent state
		self.parent_transition = None # Keep track of the number we transitioned on
		self.final = final
	def add_transition(self, char, to_state):
		"""
		Add transition to current states transition map. Basically delta function
		:param char: The character you transition on
		:param to_state: The state that character goes to
		:return:
		"""
		self.transition_list[char] = to_state

def build_dfa(K, S):
    """
    :param K: The positive integer we're finding multiples of, using numbers containing only the digits in the set S (see below)
    :param S: Subset of digits {0-9}. Alphabet for the DFA
    :return: DFA Transition Table/Matrix
    """
    S = list(set(S)) # Remove duplicates and put in order
    states = K+1 # K remainders + start state = rows
    #matrix = [state('S')] # Special 'start' state
    matrix = [state(s) for s in range(states-1)] # 0 to K-1 states

    # Build DFA Matrix
    for row in matrix:
        if row.name == 'S' or row.name == 0: # S and 0 are always the same
            if row.name == 0: row.final = True # Remainder of 0 which means its the final state
            [row.add_transition(i, i % K) for i in S]
        else:
            next_state = int(str(row.name) + str(S[0])) % K # Concat row and col into int and % K. example: K = 7, row = 1, col = 1, 11 % 7 = 4, so 4 is next state
            row.add_transition(S[0],next_state) # Add a transition to row dictionary. Kind of like the delta function d(character, state to go to)
            col = 0 # Keep track of previous column
            for c in S[1:]:
                next_state = (next_state + (c - S[col])) # Example K = 7, S = [1,5], say next state is 4 (from prev col S[0] = 1) so, 4 + (current_col = S[1] = 5 - prev_col = S[col] = S[0] = 1) = 4+(5-1) = 8
                if next_state >= K: next_state -= K # If next_state >= K, (above) we got 8 so 8-7 = 1
                elif next_state < 0: next_state += K # If next_state < K - Probably could do mod above but mod is cubic while add/sub are linear
                row.add_transition(c, next_state) # Add the transition of current col to next_state
                col += 1

    return matrix

def find_string(dfa, start=0, perm="", return_values=None, b=None):
    """
    :param dfa: The DFA to perform BFS on
    :return: The smallest integer of the given multiple K
    """
    global DONE

    if b:
        b.wait()

    S = set()
    Q = list()
    dfa[start].parent = None
    S.add(dfa[start].name)
    Q.append(dfa[start])
    curr = None
    while Q:
        curr = Q.pop(0)
        if curr.final:
            break
        for key,value in curr.transition_list.items():
            if value not in S:
                S.add(value)
                Q.append(dfa[value])
                dfa[value].parent = curr
                dfa[value].parent_transition = str(key)

    if not curr.final or curr is None:
        if return_values is not None:
            DONE = True
            return_values.append(None)
        return None # No possible integer, example: K = 4, S = [1,3,5]
    else:
        number = ""
        '''while curr.parent != None:
            number += curr.parent_transition
            curr = curr.parent
        '''
        if return_values is not None:
            DONE = True
            return_values.append(perm+number[::-1])
        return number[::-1] # Reverse


def getState(dfa, transitions):
    current_state = dfa[0]
    for t in transitions:
        next_state = current_state.transition_list[t]
        current_state = dfa[next_state]
        
    return current_state.name

def getStartingStates(dfa, alphabet, N):
    perms = itertools.product(alphabet,repeat=N)
    states = [getState(dfa,p) for p in perms]
    return states


def main():
    global DONE
    K = int(input("K = ")) # 13
    S = input("S = ")# Enter input like S = 2 5 (separated by spaces)
    N = int(input("Max Permutations = ")) # Perms length
    S = S.split()
    S = [int(x) for x in S]
    speedup_list = []
    MAX = int(input("Number of Tests = "))
    dfa = build_dfa(K,S)
    for n in range(1,N+1):
        print("Running with %s permutations." % str(n))
        states = getStartingStates(dfa,S,n)
        perms = itertools.product(S,repeat=n)
        perm_strings = []
        for p in perms:
            perm_strings.append((''.join([str(x) for x in p])))
        bar = progressbar.ProgressBar(max_value=MAX)
        for _ in range(MAX):
            return_values = list() # return values from threads
            #print(states)
            threads = []
            b = threading.Barrier(len(states))
            for i,s in enumerate(states):
                t = threading.Thread(target = find_string, args = (dfa, s, perm_strings[i], return_values, b))
                threads.append(t)       

            for t in threads:
                t.start()

            t0 = time.time()

            while not DONE:
                continue

            t1 = time.time()
            p_total = t1-t0
            #print("Parallel Time: ",p_total)
                
            t0 = time.time()
            s = find_string(dfa)
            t1 = time.time()
            s_total = t1-t0
            #print("Serial Time: ", s_total)

            if p_total == 0:
                p_total = 1.0e-10

            speedup = s_total / p_total
            speedup_list.append(speedup)

            #print("Speedup: ", speedup)
            #print(return_values)
            #print(min(return_values, key=len))
            bar.update(_)

        print("Speedup AVG: ", sum(speedup_list)/float(len(speedup_list)))
        print("Number of threads: ", len(threads))
        print()
    
main()
