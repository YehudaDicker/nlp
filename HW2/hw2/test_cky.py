from cky import CkyParser  # Make sure to use the actual path where your classes are defined
from grammar import Pcfg
# Load the grammar
with open('atis3.pcfg', 'r') as grammar_file:
    grammar = Pcfg(grammar_file)
    parser = CkyParser(grammar)

# Test case 1
tokens1 = ['flights', 'from', 'miami', 'to', 'cleveland', '.']
result1 = parser.is_in_language(tokens1)
print(f"Test case 1 result: {result1}")  # Expected: True

# Test case 2
tokens2 = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
result2 = parser.is_in_language(tokens2)
print(f"Test case 2 result: {result2}")  # Expected: False

# Add more test cases as needed
