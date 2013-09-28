import re

#import numpy


class SearchClient():
  """
  The idea is to use a composite MinHash to compare the similarity of the query word and the
  words in the indexed text.

  A composite MinHash makes 2 sets of characters from a word:
      The first is just a set of the characters
      The second is the set of the characters in position (See makePositions() function below
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
  def __init__(self, file_name='data/min_hash.txt'):
    self.words = makeList(file_name)

  def makePositions(word):
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

  def prepareWord(word, downcase=True):
    if downcase:
      word = word.lower()
    return {'word': word, 'set': set(word), 'pset': makePositions(word)}

  def comMinHash(word1, word2, weighting=0.7):
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

  def readWords(file_name):
    words = {}
    line_splitter = re.compile(r' |\[|\]|\n')
    lineNumber = 0
    with open(file_name) as f:
      line = f.readline()
      while line:
        lineNumber += 1
        line = re.sub(r',|\.|:|;|\?|\n|\(|\)|!|\{|\}|\||/|\\|\^', '', line).decode('utf8')
        for word in line_splitter.split(line):
          if word in words:
            words[word].add(lineNumber)
          else:
            words[word] = {lineNumber}
        line = f.readline()
    return words

  def prepareWords(words):
    prepared_words = []
    for word, lines in words.items():
      prepared_word = prepareWord(word)
      prepared_word['lines'] = lines
      prepared_words.append(prepared_word)
    return prepared_words

  def makeList(file_name):
    words = readWords(file_name)
    words = prepareWords(words)
    return words

  def sort_scores(self, scores):
    def order(result1, result2):
      return -1 if result1['score'] > result2['score'] else 1
    return sorted(scores, order)

  def basic_lookup(self, query):
    prepared_query = prepareWord(query)
    scores = []
    for prepared_word in self.words:
      result = {'score': comMinHash(prepared_word, prepared_query), 'word': prepared_word['word']}
      scores.append(result)
    return self.sort_scores(scores)

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

    results = self.sort_scores(accepted_results)
    return [result for result in results if result['score'] >= threshold]
