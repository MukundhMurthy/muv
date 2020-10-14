"""
Miscellaneous utilities.
"""
import numpy as np

from muv.descriptors import MUVDescriptors

import numpy as np
import muv as mv
from muv import EmbeddingFilter
import rdkit
import sys
from typing import List
from scipy.spatial import distance_matrix, cdist

class MUV_generator(object):
    def __init__(self, actives, decoys, thresholds):
        self.actives = actives
        self.decoys = decoys
        self.thresholds = thresholds

        #calculate MUV descriptors for actives and decoys
        ad = mv.calculate_descriptors(self.actives)
        dd = mv.calculate_descriptors(self.decoys)

        #Calculate the euclidean distance matrix between actives and decoys, turn into masked array type and turn off
        dm = distance_matrix(ad, dd)
        dm = np.ma.array(dm, mask=np.ones_like(dm, dtype=bool))
        dm.mask = False
        self.dm = dm

    def select_diverse_actives(self):
        indices = mv.kennard_stone(self.dm, self.thresholds['ks'])
        self.diverse_actives = self.actives[indices]

    def select_similar_decoys(self, da_ratio):
        list_decoys = []
        for active in self.diverse_actives:
            decoys = np.argsort(self.dm[active])[:da_ratio]
            self.dm.mask[:, decoys] = True
            list_decoys.append(decoys)
        self.dm.mask = False

    def embed_filter(self, n_neighbors):
        ef = EmbeddingFilter(self.thresholds['ef'], n_neighbors=n_neighbors)
        activeind = ef.filter(self.dm)
        if self.diverse_actives is not None:
            final_actives = [active for active in activeind if active in self.diverse_actives]

def calculate_descriptors(self, mols):
    """
    Calculate MUV descriptors for molecules.

    Parameters
    ----------
    mols : iterable
        Molecules.
    """
    describer = MUVDescriptors()
    x = []
    for mol in mols:
        x.append(describer(mol))
    x = np.asarray(x)
    return x


def kennard_stone(d, k):
    """
    Use the Kennard-Stone algorithm to select k maximally separated
    examples from a dataset.

    See Kennard and Stone, Technometrics 1969, 11, 137-148.

    Algorithm
    ---------
    1. Choose the two examples separated by the largest distance. In the
        case of a tie, use the first examples returned by np.where.
    2. For the remaining k - 2 selections, choose the example with the
        greatest distance to the closest example among all previously
        chosen points.

    Parameters
    ----------
    d : ndarray
        Pairwise distance matrix between dataset examples.
    k : int
        Number of examples to select.
    """
    assert 1 < k < d.shape[0]
    chosen = []

    # choose initial points
    first = np.where(d == np.amax(d))
    chosen.append(first[0][0])
    chosen.append(first[1][0])
    d = np.ma.array(d, mask=np.ones_like(d, dtype=bool))

    # choose remaining points
    while len(chosen) < k:
        d.mask[:, chosen] = False
        d.mask[chosen] = True
        print d
        p = np.ma.argmax(np.ma.amin(d, axis=1))
        chosen.append(p)

    return chosen
