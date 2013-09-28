# -*- coding: utf-8 -*-

import re

#import numpy


class SearchClient():
  """
  The idea is to use a composite MinHash to compare the similarity of the query word and the
  words in the indexed text.

  For example:
      client = SearchClient('data/big.txt', n_gram_upto=4)
      client.lookup('Phœnician or Greek navigators')  # not optimised _at all_


  A composite MinHash makes 2 sets of characters from a word:
      The first is just a set of the characters
      The second is the set of the characters in position (See make_positions() function below
        for description of weighting)

  In the simple implementation of this text fuzzy search, we calculate the set values in advance,
  then scan over the list of words to find the best match.
  This implementation is unlikely to be performan but is hopefully O(Nm) on searching
  (where N is number of words in index and `m` is average length of each of N words with the
    query word)


  A more adanced idea (which I'd be surprised if someone hasn't already implementated) is to try
  to build an index by taking N words and performing a Minhash between the words and all the
  other words.  (Obviously this would be N^2 on building index so maybe not so good after all.)
  We produce a NxN matrix of the composite Minhash between each pair of words.
  We choose the J most highly constrained connections of the Matrix and relax them to produce J
  ordered lists of word & composite MinHash values.

  Then we perform a binary search through the lists by composite Minhashing the query string Q
  against the top and bottom values of each J list, discarding "some" of worst matching J lists
  on each binary interation cycle.
  For some data this may lead to O(lgN m) but this won't work for uniformly distributed permutations,
  where you'd end up needing N lists and hence O(NlgN m) up from O(Nm) for the simple implementation
  """
  def __init__(self, file_name='data/min_hash.txt', n_gram_upto=1):
    self.words = self.make_list(file_name, n_gram_upto)

  def make_positions(self, word):
    """
    For a word, return a set of its letters and their positions.

    e.g.
    input:  `Hah, 5!`
    output: set(['H-1', 'a-2', 'h-3', ',-4', ' -5', '5-6', '!-7'])
    """
    word_positions = set()
    for i, letter in enumerate(word):
      word_positions.add(letter + '-' + unicode(i))
    return word_positions

  def prepare_word(self, word, downcase=True):
    if downcase:
      word = word.lower()
    return {'word': word, 'set': set(word), 'pset': self.make_positions(word)}

  def com_min_hash(self, word1, word2, weighting=0.7):
    """
    Composite MinHash
    Weighting is towards the "in position" MinHash, i.e. for the words:
    hannah and nah, a weighting of 0 would give a completely positive
    composite sim hash value (i.e. 1), and a weighting of 1 would give
    a completely different Minhash value (i.e. 0)
    """
    ws1 = word1['set']
    ws2 = word2['set']
    wps1 = word1['pset']
    wps2 = word2['pset']

    return weighting * (float(len(wps1 & wps2))/len(wps1 | wps2)) + (1 - weighting) * (float(len(ws1 & ws2))/len(ws1 | ws2))

  def read_words(self, file_name, n_gram_upto):
    """
    phrase can be a single word or multiple, single space seperated words
    """
    phrases = {}
    line_splitter = re.compile(r' |\[|\]|\n')
    lineNumber = 0
    window = []
    with open(file_name) as f:
      line = f.readline()
      while line:
        lineNumber += 1
        line = re.sub(r',|\.|:|;|\?|\n|\(|\)|!|\{|\}|\||/|\\|\^', '', line).decode('utf8')
        for word in line_splitter.split(line):
          word = word.strip()
          if word:
            # add a new term and discard an old one
            window = [word] + window[:(n_gram_upto-1)]
            for i in range(1, n_gram_upto+1):
              phrase = u' '.join(reversed(window[:i]))
              if phrase in phrases:
                phrases[phrase].add(lineNumber)
              else:
                phrases[phrase] = {lineNumber}
        line = f.readline()
    return phrases

  def prepare_words(self, words):
    prepared_words = []
    for word, lines in words.items():
      prepared_word = self.prepare_word(word)
      # add the line numbers (will be as a set at this point)
      prepared_word['lines'] = lines
      prepared_words.append(prepared_word)
    return prepared_words

  def make_list(self, file_name, n_gram_upto):
    words = self.read_words(file_name, n_gram_upto)
    words = self.prepare_words(words)
    return words

  def sort_results(self, results):
    def order(result1, result2):
      return -1 if result1['score'] > result2['score'] else 1
    return sorted(results, order)

  def convert_lines(self, results):
    # convert the sets of line numbers to sorted lists
    for result in results:
      result['lines'] = sorted(list(result['lines']))

  def basic_lookup(self, query):
    prepared_query = self.prepare_word(query)
    results = []
    for prepared_word in self.words:
      result = {'score': self.com_min_hash(prepared_word, prepared_query), 'word': prepared_word['word'], 'lines': prepared_word['lines']}
      results.append(result)
    return self.sort_results(results)

  def lookup(self, query, limit=8, threshold=0.6):
    """
    Return a maximum of 8, and only if they are over 0.6 in score
    """
    query_words = query.split(' ')
    results = self.basic_lookup(query)[:limit]
    for query_word in query_words:
      results += self.basic_lookup(query_word)[:limit]
    # remove duplicate hits:
    accepted_results = {}
    for proposed_result in results:
      word = proposed_result['word']
      if word not in accepted_results or accepted_results[word]['score'] < proposed_result['score']:
        accepted_results[word] = proposed_result
    accepted_results = accepted_results.values()

    results = self.sort_results(accepted_results)
    self.convert_lines(results)
    return [result for result in results if result['score'] >= threshold]
