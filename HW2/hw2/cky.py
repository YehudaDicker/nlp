"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self,tokens):
        """
        Membership checking. Parse the input tokens and return True if 
        the sentence is in the language described by the grammar. Otherwise
        return False
        """
        # TODO, part 2

        n = len(tokens)
        table = [[[] for j in range (n+1)] for i in range(n)]

        # Populate diagonal of table
        for i, token in enumerate(tokens):
            table[i][i+1] = [lhs for lhs, rhs, prob in self.grammar.rhs_to_rules[(token,)] if prob > 0]

        for span in range(2, n+1):
            for i in range(n-span+1):
                j = i + span
                for k in range(i+1, j):
                    for rule in self.grammar.lhs_to_rules.keys():
                        for b in table[i][k]:
                            for c in table[k][j]:
                                if (b,c) in self.grammar.rhs_to_rules:
                                    for a, rhs, prob in self.grammar.rhs_to_rules[(b,c)]:
                                        if a not in table[i][j] and prob > 0:
                                            table[i][j].append(a)
        return self.grammar.startsymbol in table[0][n]
       
    def parse_with_backpointers(self, tokens):
        n = len(tokens)

        # Initialize tables with None
        backpointer_table = {(i, j): {} for i in range(n) for j in range(i, n+1)}
        probability_table = {(i, j): {} for i in range(n) for j in range(i, n+1)}

        #print("Initial Backpointer Table:", backpointer_table)
        #print("Initial Probability Table:", probability_table)

        # Base case tables
        for i, token in enumerate(tokens):
            rules = self.grammar.rhs_to_rules.get((token,), [])
            if rules:
                backpointer_table[(i, i+1)] = {}
                probability_table[(i, i+1)] = {}
                for lhs, rhs, prob in rules:
                    backpointer_table[(i, i+1)][lhs] = token
                    probability_table[(i, i+1)][lhs] = math.log(prob)

        #print(f"After populating base case for token {token}:")
        #print("Backpointer Table:", backpointer_table)
       # print("Probability Table:", probability_table)

        #print(self.grammar.rhs_to_rules)
        # Filling the rest of the tables
        for span in range(2, n+1):
            for i in range(n-span+1):
                j = i + span
                for k in range(i+1, j):
                    for rules in self.grammar.rhs_to_rules.keys():
                        for rule in self.grammar.rhs_to_rules[rules]:
                            #print(rule)
                            lhs, rhs, prob = rule
                            #print(lhs, rhs, prob )
                            if len(rhs) == 2:
                                b, c = rhs
                                #print(b, c)
                                if b in backpointer_table[(i, k)] and c in backpointer_table[(k, j)]:
                                    log_prob = math.log(prob) + probability_table[(i, k)][b] + probability_table[(k, j)][c]
                
                                    if lhs not in probability_table[(i, j)] or log_prob > probability_table[(i, j)][lhs]:
                                        backpointer_table[(i, j)][lhs] = ((b, i, k), (c, k, j))
                                        probability_table[(i, j)][lhs] = log_prob
                            
        #print("Final Backpointer Table:", backpointer_table)
        #print("Final Probability Table:", probability_table)

        return backpointer_table, probability_table

def get_tree(chart,i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    # TODO: Part 4

    #if (i, j) in chart and chart[(i, j)] is not None and nt in chart[(i, j)]:
    backpointer = chart[(i, j)][nt]
    if j == i + 1:
        return (nt, backpointer)
    else:
        left_child = get_tree(chart, backpointer[0][1], backpointer[0][2], backpointer[0][0])
        right_child = get_tree(chart, backpointer[1][1], backpointer[1][2], backpointer[1][0])
        return (nt, left_child, right_child)
    return None 
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        #print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        assert check_table_format(chart)
        assert check_probs_format(probs)
        
