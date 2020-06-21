import scipy.io as scio
import numpy as np
from astropy import wcs


class GXBox(object):
    """This class wraps IDL Box structure used in GX Simulator"""

    # noinspection PyTypeChecker
    def __init__(self, filename):
        self._box = scio.readsav(filename).box
        self.refids = [ptr.ID[0].decode('utf-8') for ptr in self._box['REFMAPS'][0].OMAP[0].POINTER[0].PTRS[0]
                       if type(ptr) == np.recarray]

    @property
    def bx(self, order='C'):
        if order == 'C':
            return self._box.bx[0]
        else:
            return np.transpose(self._box.bx[0], (2, 1, 0))

    @property
    def by(self, order='C'):
        if order == 'C':
            return self._box.by[0]
        else:
            return np.transpose(self._box.by[0], (2, 1, 0))

    @property
    def bz(self, order='C'):
        if order == 'C':
            return self._box.bz[0]
        else:
            return np.transpose(self._box.bz[0], (2, 1, 0))

    @property
    def index(self):
        """INDEX OF BASE MAPS"""

        # REMOVE KEYS THAT DON'T SATISFY FITS HEADER REQUIREMENTS
        keystorem = []

        index = [dict(zip(self._box.index[0].dtype.names, val)) for val in self._box.index[0]][0]

        for key in index.keys():
            # CONVERT BYTES VALUES TO STRING
            if type(index[key]) == bytes:
                index[key] = index[key].decode('utf-8')

            if len(key) > 8:
                keystorem.append(key)

        for key in keystorem:
            del index[key]

        index['COMMENT'] = 'Walk through sunpy'
        index['HISTORY'] = 'GX Box -> SunPy'
        index['DATE_OBS'] = index['DATE_OBS'].replace("T", " ")

        return index

    def refmap(self, id):
        """RETURNS REFERENCE MAP WITH ID"""
        return self._rmap(self.refids.index(id)) if id in self.refids else None

    # def _rmap(self, position):
    #     """CONVERTS REFERENCE MAP AT POSITION TO SUNPY MAP"""
    #
    #     draft = self._box['REFMAPS'][0].OMAP[0].POINTER[0].PTRS[0][position]

