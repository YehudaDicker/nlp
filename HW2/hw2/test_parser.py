from cky import CkyParser, check_table_format, check_probs_format
from grammar import Pcfg  

def test_parser(parser, sentence):
    table, probs = parser.parse_with_backpointers(sentence)
    
    print("Testing sentence:", " ".join(sentence))
    print("\nTable:")
    for key, value in table.items():
        print(key, value)
    
    print("\nProbs:")
    for key, value in probs.items():
        print(key, value)

    # Check the format of the tables
    print("\nTable Format Valid:", check_table_format(table))
    print("Prob Format Valid:", check_probs_format(probs))

    # Add more specific checks or assertions as needed, for example:
    # assert some specific condition, e.g., a known parse for a test sentence

if __name__ == "__main__":
    with open('atis3.pcfg', 'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        parser = CkyParser(grammar)
        
        test_sentences = [
            ['flights', 'from', 'miami', 'to', 'cleveland', '.'],
            # Add more test sentences as needed
        ]

        for sentence in test_sentences:
            test_parser(parser, sentence)
            print("\n" + "="*50 + "\n")  # Print separator between test cases
