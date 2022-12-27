from itertools import chain, combinations as combinations_

def combinations(string):
    return map(lambda params: ''.join(params), chain.from_iterable(combinations_(string, i) for i in range(1, len(string))))