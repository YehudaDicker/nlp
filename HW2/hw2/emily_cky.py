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

        n = len(tokens)
        table = {}

        for i in range(n):
            for j in range(i, n+1): 
                table[(i, j)] = set()
        
        for i in range(0, n): 
            for key, rules in grammar.rhs_to_rules.items():
                if tokens[i] == key[0] and len(key) == 1:
                    for lhs in rules:
                        table[(i, i+1)].add(lhs[0])

        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                temp = set()
                for k in range(i+1, j):
                    for key in grammar.rhs_to_rules.keys():
                        for rules in grammar.rhs_to_rules[key]:                       
                            if len(rules[1]) == 2:
                                if rules[1][0] in table[(i, k)] and rules[1][1] in table[(k, j)]: 
                                    temp.add(rules[0])
                    table[(i, j)] = temp
    
        if grammar.startsymbol in table[(0, n)]: 
            return True
        return False 
       
    def parse_with_backpointers(self, tokens):
        """
        Parse the input tokens and return a parse table and a probability table.
        """
        n = len(tokens)
        table = {}
        probs = {}

        for i in range(n):
            for j in range(i, n+1): 
                table[(i, j)] = {}
                probs[(i, j)] = {}
        
        for i in range(n):  
            rules = self.grammar.rhs_to_rules[(tokens[i],)]
            for rule in rules:
                    lhs = rule[0]  
                    prob = rule[2]  
                    table[(i, i+1)][lhs] = tokens[i]
                    probs[(i, i+1)][lhs] = math.log2(prob)

        for length in range(2, n+1):
            for i in range(0, n-length+1):
                j = i + length
                for k in range(i+1, j):
                    for key in self.grammar.rhs_to_rules.keys():
                        for rules in self.grammar.rhs_to_rules[key]:    
                            if len(rules[1]) == 2:
                                if rules[1][0] in table[(i, k)] and rules[1][1] in table[(k, j)]:
                                    prob = math.log2(rules[2]) + probs[(i, k)][rules[1][0]] + probs[(k, j)][rules[1][1]]
                                    if rules[0] not in probs[(i,j)] or prob > probs[(i, j)][rules[0]]:
                                        backpointer = ((rules[1][0], i, k), (rules[1][1], k, j))
                                        probs[(i, j)][rules[0]] = prob
                                        table[(i, j)][rules[0]] = backpointer
        return table, probs


def get_tree(chart, i,j,nt): 
    """
    Return the parse-tree rooted in non-terminal nt and covering span i,j.
    """
    entry = chart[(i, j)][nt]
    if j== i + 1:
        return nt, entry
    else: 
        return (nt, get_tree(chart, entry[0][1], entry[0][2], entry[0][0]), get_tree(chart, entry[1][1], entry[1][2], entry[1][0]))
    return None 
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        #toks = ['miami', 'flights','cleveland', 'from', 'to','.']
        #print(parser.is_in_language(toks))
        table, probs = parser.parse_with_backpointers(toks)
        #print(table[0,len(toks)][grammar.startsymbol])
        #print(probs[0,len(toks)][grammar.startsymbol])
        print(get_tree(table, 0, len(toks), grammar.startsymbol))
        assert check_table_format(table) 
        assert check_probs_format(probs)
        
