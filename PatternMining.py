# File containing implementations for common pattern mining algorithms
import copy
import itertools
# -------------------------------------------------------------------------
# APRIORI
# -------------------------------------------------------------------------
def apriori(database, minSupp):
    """
    Implementation of the apriori algorithm.
    Expects a database in the form of a list of lists [[],[],[]].
    Support for (sub)sequences is defined as the relative amount of exact matches in the database.
    A subsequence is an uninterrupted part of a sequence.
    """
    # preprocessing: sort all database entries
    for transaction in database:
        transaction.sort()
    # gather elements as first candidates
    candidates = []
    for transaction in database:
        for element in transaction:
            if [element] not in candidates:
                candidates.append([element])

    frequent_patterns = []
    while(len(candidates) > 0):
        current_frequent_patters = []
        for candidate in candidates:
            support = sum([subset(candidate, transaction) for transaction in database]) / len(database)
            if support >= minSupp:
                frequent_patterns.append((candidate, support))
                current_frequent_patters.append(candidate)
        candidates = apriori_gen(current_frequent_patters)

    return frequent_patterns



def apriori_gen(candidates):
    """
    Candidate generation for the apriori algorithm, given the previous set of candidates.
    Utilizes the apriori-principle to generate the next candidate set.
    Requires ordered elements to avoid duplicates.
    """
    if len(candidates) < 2:
        return []
    new_candidates = []
    old_length = len(candidates[0])
    # create new elements
    for elem1, elem2 in itertools.combinations_with_replacement(candidates, 2):
        if elem1[:old_length-1] == elem2[1:old_length]:
            new_element = elem2 + [elem1[old_length-1]]
            new_candidates.append(new_element)
        if elem2[:old_length-1] == elem1[1:old_length]:
            new_element = elem1 + [elem2[old_length-1]]
            new_candidates.append(new_element)
    for elem in new_candidates:
        elem.sort()
    # ensure uniqueness of elements
    pointer = 0
    while pointer < len(new_candidates) - 1:
        if new_candidates[pointer] in new_candidates[pointer + 1:]:
            new_candidates.pop(pointer)
        else:
            pointer += 1
    return new_candidates

def subset(sub, l):
    """
    Checks if a given list "sub" is a subset of another list "l"
    """
    l_copy = copy.deepcopy(l)
    for el in sub:
        if el in l_copy:
            l_copy.remove(el)
        else:
            return False
    return True

# ---------------------------------------------------------------------------
# GSP
# ---------------------------------------------------------------------------

def subsequence(sub, seq):
    """
    Checks if a given sequence "sub" is a subsequence of another sequence "seq"
    """
    l = len(sub)
    L = len(seq)
    if l > L:
        return False
    for i in range(L-l+1):
        if(seq[i:i+l] == sub):
            return True
    return False
def gsp_gen(candidates):
    """
    Candidate generation for the GSP algorithm, given the previous set of candidates.
    Utilizes the apriori-principle to generate the next candidate set.
    """
    if len(candidates) < 2:
        return []
    new_candidates = []
    old_length = len(candidates[0])
    # create new elements
    for elem1, elem2 in itertools.combinations(candidates, 2):
        if elem1[:old_length-1] == elem2[1:old_length]:
            new_element = elem2 + [elem1[old_length-1]]
            new_candidates.append(new_element)
        if elem2[:old_length-1] == elem1[1:old_length]:
            new_element = elem1 + [elem2[old_length-1]]
            new_candidates.append(new_element)

    # ensure uniqueness of elements
    pointer = 0
    while pointer<len(new_candidates)-1:
        if new_candidates[pointer] in new_candidates[pointer+1:]:
            new_candidates.pop(pointer)
        else:
            pointer += 1
    return new_candidates

def GSP(database, minSupp):
    """
    Implementation of the Generalized Sequential Pattern (GSP) algorithm.
    This particular version expects a database consisting of sequences of items (not itemsets).
    Support for (sub)sequences is defined as the relative amount of exact matches in the database.
    A subsequence is an uninterrupted part of a sequence.
    """
    frequent_patterns = []
    # gather elements as first candidates
    candidates = []
    for sequence in database:
        for element in sequence:
            if [element] not in candidates:
                candidates.append([element])

    while(len(candidates) > 0):
        current_frequent_patters = []
        for candidate in candidates:
            support = sum([subsequence(candidate, sequence) for sequence in database]) / len(database)
            if support >= minSupp:
                frequent_patterns.append(candidate)
                current_frequent_patters.append(candidate)
        candidates = gsp_gen(current_frequent_patters)

    return frequent_patterns