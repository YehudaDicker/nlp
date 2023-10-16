"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum, isclose

class Pcfg(object): 
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        # TODO, Part 1
        
        # Check if each rule is in CNF
        for lhs, rules in self.lhs_to_rules.items():
            prob_sum = fsum(rule[2] for rule in rules)
            if not isclose(prob_sum, 1.0, rel_tol=1e-9):
                print(f"Probabilites for {lhs} do not sum to 1.0")
                return False

            for rule in rules:
                rhs = rule[1]
                if len(rhs) != 2 and not (len(rhs) == 1 and rhs[0].isupper() == False):
                    print(f"Invalid CNF for rule: {lhs} -> {rhs}")
                    return False

        return True 

def test_grammars(filenames, expected_results):
    for filename, expected in zip(filenames, expected_results):
        with open(filename, 'r') as grammar_file:
            grammar = Pcfg(grammar_file)
            result = grammar.verify_grammar()
            print(f"Failed on {filename}: expected {expected}, got {result}") if result != expected else print(f"Passed for {filename}")

# List of test files and their expected results (True for valid, False for invalid)
test_files = ['valid_grammar.cfg', 'invalid_grammar.cfg', 'invalid_prob.cfg']
expected_results = [True, False, False]

# Run the tests
#test_grammars(test_files, expected_results)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python <script_name> <grammar_file>")
    else:
        try:
            with open(sys.argv[1],'r') as grammar_file:
                grammar = Pcfg(grammar_file)
                is_valid = grammar.verify_grammar()
                if is_valid:
                    print(f"Error: The grammar {sys.argv[1]} is a valid PCFG in CNF.")
                else:
                    print(f"Error: The grammar {sys.argv[1]} is not a valid PCFG in CNF.")
        except FileNotFoundError:
            print(f"Error: The fi;le {sys.argv[1]} was not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}.")
        





