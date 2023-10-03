import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    sequence = ["START"] * (n-1) + sequence + ["END"] if n != 1 else ["START"] + sequence + ["END"]
    
    res = []
    for i in range(len(sequence) - n + 1):  
        ngram = tuple(sequence[i:i+n])  
        res.append(ngram)
    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
        
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        
        self.total_words = 0
        self.count_ngrams(generator)                
        self.total_words = sum(self.unigramcounts.values())



    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = defaultdict(int) # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here

        for sentence in corpus:

            unigrams = get_ngrams(sentence, 1)
            bigrams = get_ngrams(sentence, 2)
            trigrams = get_ngrams(sentence, 3)
           
            for unigram in unigrams:
                self.unigramcounts[unigram] = 1 + self.unigramcounts.get(unigram, 0)
                        
            for bigram in bigrams:
                self.bigramcounts[bigram] = 1 + self.bigramcounts.get(bigram, 0)

            for trigram in trigrams:
                 self.trigramcounts[trigram] = 1 + self.trigramcounts.get(trigram, 0)
            
        return

    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram[0] == "START":
            return 0
        bigram = tuple(trigram[0:2])
        return 1.0 / len(self.lexicon) if (bigram not in self.bigramcounts or trigram not in self.trigramcounts) else self.trigramcounts[trigram] / self.bigramcounts[bigram]  

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """    
        if bigram[0] == "START":
            return 0
        unigram = tuple([bigram[0]])
        return 1.0 / len(self.lexicon) if (unigram not in self.unigramcounts or bigram not in self.bigramcounts) else self.bigramcounts[bigram] / self.unigramcounts[unigram]
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        if unigram == "START" or unigram not in self.unigramcounts:
            return 0
        return self.unigramcounts[unigram] / self.total_words

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        return result            

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0

        rawTrigramProb = self.raw_trigram_probability(trigram)
        rawBigramProb = self.raw_bigram_probability(trigram[:2])
        rawUnigramProb = self.raw_unigram_probability(trigram[:1])

        #print(f"Raw trigram probability: {rawTrigramProb}")
        #print(f"Raw bigram probability: {rawBigramProb}")
        #print(f"Raw unigram probability: {rawUnigramProb}")
        
        smoothed = lambda1*rawTrigramProb + lambda2*rawBigramProb + lambda3*rawUnigramProb

        return smoothed
       
        
    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        trigrams = get_ngrams(sentence, 3)

        logProb = 0
        for trigram in trigrams:
            logProb += math.log2(self.smoothed_trigram_probability(trigram))
        
        return logProb

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_prob = 0
        total_words_corpus = 0
        for sentence in corpus:
            log_prob += self.sentence_logprob(sentence)
            total_words_corpus += len(sentence) - 1

        avg_logProb = log_prob / total_words_corpus

        p = 2**(-avg_logProb)

        return p


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))

            total += 1

            if pp < pp2:
                correct += 1
    
        for f in os.listdir(testdir2):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))

            total += 1
            if pp2 < pp:
                correct += 1
            
        return correct/total 

  
        
if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ 
    # python -i trigram_model.py /Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/hw1_data/brown_train.txt
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    ''' 
    #Testing get_ngram
    gram = ("i","am")
    res = get_ngrams(gram, 2)
    print(res)
    

    #Testing count_ngrams
    corpus = [["i", "love", "nlp"], ["i", "love", "programming"]]
    model.count_ngrams(corpus)
    print(model.trigramcounts)
    

    #Testing Raw Probability Calculations:

    # 1. Directly printing raw probabilities:

    # For unigram
    unigram = ("the",)
    print(f"Raw unigram probability for {unigram}: {model.raw_unigram_probability(unigram)}")

    # For bigram
    bigram = ("the", "cat")
    print(f"Raw bigram probability for {bigram}: {model.raw_bigram_probability(bigram)}")

    # For trigram
    trigram = ("the", "black", "cat")
    print(f"Raw trigram probability for {trigram}: {model.raw_trigram_probability(trigram)}")

    # 2. Testing for Specific n-grams:

    test_unigrams = [("the",), ("a",), ("an",)]
    test_bigrams = [("the", "cat"), ("a", "dog"), ("he", "said")]
    test_trigrams = [("the", "black", "cat"), ("a", "big", "dog"), ("he", "has", "said")]

    print("\nRaw Unigram Probabilities:")
    for uni in test_unigrams:
        print(f"{uni}: {model.raw_unigram_probability(uni)}")

    print("\nRaw Bigram Probabilities:")
    for bi in test_bigrams:
        print(f"{bi}: {model.raw_bigram_probability(bi)}")

    print("\nRaw Trigram Probabilities:")
    for tri in test_trigrams:
        print(f"{tri}: {model.raw_trigram_probability(tri)}")
    

    trigram = ("i", "love", "nlp")
    expected_smoothed_prob = (1/3.0) * (0.5) + (1/3.0) * (1.0) + (1/3.0) * (2/3)  # Using the earlier calculated probabilities
    print("\nExpected smoothed prob:", expected_smoothed_prob)
    print("Model:", model.smoothed_trigram_probability(trigram))


    sentence = "i love nlp"
    trigrams_in_sentence = get_ngrams(sentence.split(), 3)
    manual_log_prob = sum([math.log2(model.smoothed_trigram_probability(trigram)) for trigram in trigrams_in_sentence])

    print("\nExpected sentence log prob:", manual_log_prob)
    print("Model:", model.sentence_logprob(sentence.split()))
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)
    
    print(get_ngrams(["THE"], 1))
    print(model.raw_trigram_probability(("START", "START", "THE")))
    print(model.raw_bigram_probability(("START", "THE")))
    print(model.raw_unigram_probability(("THE")))

    
    # Testing perplexity: 
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(f"Perplexity on test set: {pp}")
    

    #Essay scoring experiment: 
    acc = essay_scoring_experiment('/Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/hw1_data/ets_toefl_data/train_high.txt', '/Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/hw1_data/ets_toefl_data/train_low.txt', "/Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/hw1_data/ets_toefl_data/test_high", "/Users/yehudadicker/Downloads/NLP/GitHub/nlp/HW1/hw1_data/ets_toefl_data/test_low")
    print(acc)
''' 