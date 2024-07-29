# data
from enum import Enum
from functools import cache

import numpy as np
from mendeleev import element as md_element

# Conversion factors
TO_CM = 219474.63067
TO_EV = 27.2113839
TO_BOHR = 1.88973
TO_ANGSTROM = 0.5291761
FORCE_MASS = 1822.88853


@cache
def atomic_number(element: str) -> int:
    """Returns the atomic number for the element"""
    el = md_element(element)
    return el.atomic_number


AM_DICT = {
    "s": 0,
    "p": 1,
    "d": 2,
    "f": 3,
    "g": 4,
    "h": 5,
    "i": 6,
    "j": 7,
    "k": 8,
    "l": 9,
}
"""Dictionary converting letter-value angular momenta to l quantum number"""

INV_AM_DICT = dict((v, k) for k, v in AM_DICT.items())
"""Dictionary converting back from l quantum number to letter value"""

_EVEN_TEMPERED_DATA = {}
"""Dictionary with pre-optimised even-tempered expansions for atoms"""

ETParams = list[tuple[float, float, int]]
"""Parameters for an even-tempered expansion. The tuple contains:
    a float of the starting exponent;
    a float of the spacing between exponents;
    an int of the total number of primitive exponents"""

_WELL_TEMPERED_DATA = {}
"""Dictionary with pre-optimised well-tempered expansions for atoms"""

WTParams = list[tuple[float, float, float, float, int]]
"""Parameters for a well-tempered expansion. The tuple contains:
    a float of the starting exponent;
    a float of the primary spacing between exponents;
    a float of the gamma parameter;
    a float of the delta parameter;
    an int of the total number of primitive exponents"""

"""Dictionary with pre-optimised Legendre polynomial expansions for atoms"""
_LEGENDRE_DATA = {
    'H': [
        [
            -0.5852780356752019,
            -4.208143931324614,
            0.29506043797320314,
            -0.22703973086131857,
            -0.03937673349771134,
            -0.036565239743579954,
        ]
    ],
    'He': [
        [
            0.3365604169411115,
            -4.349324475254306,
            0.2569845242860431,
            -0.16330096396962618,
            0.022824746541758315,
            -0.01992495279028017,
        ]
    ],
    'Li': [
        [
            0.7795721765030852,
            -6.451814714010956,
            0.5002748413877769,
            -0.31162177278973724,
            0.20044566564042415,
            0.01932138358560287,
        ]
    ],
    'Be': [
        [
            1.3345949521378189,
            -6.461604056305345,
            0.4589078358885121,
            -0.3638324525545,
            0.1816435073341283,
            0.021604857734403925,
        ]
    ],
    'B': [
        [
            1.8309258419868915,
            -6.463214688111899,
            0.4344040947825959,
            -0.4034024005144653,
            0.14660637599540707,
            0.009030495837624681,
        ],
        [
            -1.192195925057745,
            -3.7454842638864276,
            0.06322855923500376,
            -0.1629559822272255,
            -0.00910941042915516,
            -0.019390289097594847,
        ],
    ],
    'C': [
        [
            2.217349812524922,
            -6.468298783198044,
            0.4176041743993609,
            -0.4263571622433264,
            0.12911822297360726,
            0.003342086288550511,
        ],
        [
            -0.758298083413343,
            -3.7762660604080995,
            0.018467306448720355,
            -0.17257619155776865,
            -0.004634515781188141,
            -0.01832883926046141,
        ],
    ],
    'N': [
        [
            2.5365557047481566,
            -6.473456508112158,
            0.4040189211386716,
            -0.4420155953464171,
            0.11761386353448015,
            -0.00045680156164003147,
        ],
        [
            -0.41188109977592435,
            -3.794468635854243,
            -0.0016444639463253719,
            -0.17098491720280024,
            0.004268702357133291,
            -0.015606541535690195,
        ],
    ],
    'O': [
        [
            2.8083425172796037,
            -6.481177909780908,
            0.38792698406360626,
            -0.46037134170397803,
            0.09984989060810086,
            -0.008315306658921051,
        ],
        [
            -0.2381752492670329,
            -3.8675875713382735,
            -0.04808151475718213,
            -0.16998172208409168,
            0.012619505334633851,
            -0.014363149522390203,
        ],
    ],
    'F': [
        [
            3.0488361301595397,
            -6.487822409073642,
            0.3763837401569423,
            -0.47247724724509177,
            0.09036775701936836,
            -0.011561889493476083,
        ],
        [
            0.007942562034886736,
            -3.8948762580944996,
            -0.07200321297895391,
            -0.17139603994005048,
            0.017320864778432303,
            -0.013220576668452493,
        ],
    ],
    'Ne': [
        [
            3.261881913756225,
            -6.492936944897167,
            0.36819878713540843,
            -0.4810997634168346,
            0.08410415253724086,
            -0.013807462333186528,
        ],
        [
            0.22604238431330953,
            -3.916059303542598,
            -0.0922847795406194,
            -0.17622285934768325,
            0.018084377863380522,
            -0.013053929518511113,
        ],
    ],
    'Na': [
        [
            2.8329286491246415,
            -8.419114954461612,
            0.46676195001172255,
            -0.6504448621897473,
            0.2329399810169266,
            0.007173464457045574,
        ],
        [
            1.2864400480550016,
            -5.202978192775552,
            0.16295662990443105,
            -0.3568453661436226,
            0.009963282406270235,
            -0.0607906492630238,
        ],
    ],
    'Mg': [
        [
            4.8682630636718205,
            -7.824738998289657,
            0.8373938006147699,
            -0.6257899063135539,
            0.019397334958551123,
            -0.14095973728075872,
        ],
        [
            0.13745040956175467,
            -5.69312909183676,
            -0.0007813113759242204,
            -0.27714098791649,
            0.11833556536887292,
            -0.017753966390246755,
        ],
    ],
    'Al': [
        [
            3.402230446793065,
            -8.266035641003544,
            0.5546670001248568,
            -0.6739566974638637,
            0.15403855218207146,
            -0.03252710348928738,
        ],
        [
            0.2452198638425205,
            -5.292618068564417,
            -0.14173853654556168,
            -0.2760046329012772,
            0.08542197439217612,
            -0.014077995683691322,
        ],
    ],
    'Si': [
        [
            3.6417265420822504,
            -8.207628608964864,
            0.5901248012179916,
            -0.6759189525188765,
            0.1318704450392033,
            -0.04454995401386687,
        ],
        [
            0.5086336968183198,
            -5.192396020971051,
            -0.08690600540246016,
            -0.2652999020682108,
            0.0838370564934744,
            -0.015001749096980673,
        ],
    ],
    'P': [
        [
            3.8400710552029382,
            -8.163108099995815,
            0.6205834858596087,
            -0.6744065342899902,
            0.11492387284934538,
            -0.05436093287413064,
        ],
        [
            0.7316044222144655,
            -5.128252271337691,
            -0.06509989409126535,
            -0.2756502690813274,
            0.07140767832888092,
            -0.018849886879164973,
        ],
    ],
    'S': [
        [
            4.008971931424867,
            -8.134390399888098,
            0.6334055860399179,
            -0.6829861327490292,
            0.09412037150406671,
            -0.06605038101033628,
        ],
        [
            0.8414581055585477,
            -5.135114599548462,
            -0.08561979988969565,
            -0.2937967114119469,
            0.07132319999698347,
            -0.01541215175144012,
        ],
    ],
    'Cl': [
        [
            4.163397734105068,
            -8.110152771707861,
            0.6459479127155887,
            -0.6859858178169782,
            0.07947041308438291,
            -0.07580833963190303,
        ],
        [
            1.0055280073473738,
            -5.108005763580298,
            -0.0854976539165474,
            -0.30962175665098846,
            0.06283461597108986,
            -0.01621356288045616,
        ],
    ],
    'Ar': [
        [
            4.304384453328594,
            -8.087380622556559,
            0.6581304333133784,
            -0.6875060359346178,
            0.06925989159260118,
            -0.08224110653793498,
        ],
        [
            1.1569045981393935,
            -5.080491816647825,
            -0.08155065675170743,
            -0.32099259480744224,
            0.055210485501791536,
            -0.0177574595257872,
        ],
    ],
}

LegParams = list[tuple[tuple, int]]


class GROUNDSTATE_MULTIPLICITIES(Enum):
    H = 2
    He = 1
    Li = 2
    Be = 1
    B = 2
    C = 3
    N = 4
    O = 3
    F = 2
    Ne = 1
    Na = 2
    Mg = 1
    Al = 2
    Si = 3
    P = 4
    S = 3
    Cl = 2
    Ar = 1
    K = 2
    Ca = 1
    Sc = 2
    Ti = 3
    V = 4
    Cr = 7
    Mn = 6
    Fe = 5
    Co = 4
    Ni = 3
    Cu = 2
    Zn = 1
    Ga = 2
    Ge = 3
    As = 4
    Se = 3
    Br = 2
    Kr = 1


def get_even_temper_params(atom: str = "H", accuracy: float = 1e-5) -> ETParams:
    """Searches for the relevant even tempered expansion
    from _EVEN_TEMPERED_DATA
    """
    if atom in _EVEN_TEMPERED_DATA:
        log_acc = -np.log10(accuracy)
        index = max(4, log_acc) - 4
        index = int(min(index, 3))
        return _EVEN_TEMPERED_DATA[atom][index]
    else:
        return []


def get_legendre_params(atom: str = "H") -> LegParams:
    """Searches for the relevant Legendre polynomial-based expansion
    from _LEGENDRE_DATA
    """
    if atom in _LEGENDRE_DATA:
        return _LEGENDRE_DATA[atom]
    else:
        return []


def get_well_temper_params(atom: str = "H", accuracy: float = 1e-5) -> WTParams:
    """Searches for the relevant well tempered expansion
    from _WELL_TEMPERED_DATA
    """
    if atom in _WELL_TEMPERED_DATA:
        log_acc = -np.log10(accuracy)
        index = max(4, log_acc) - 4
        index = int(min(index, 3))
        return _WELL_TEMPERED_DATA[atom][index]
    else:
        return []


"""Essentially exact numerical Hartree-Fock energies for all atoms
   in Hartree. Ref: Saito 2009, doi.org/10.1016/j.adt.2009.06.001"""
_ATOMIC_HF_ENERGIES = {
    1: -0.5,
    2: -2.86167999561,
    3: -7.43272693073,
    4: -14.5730231683,
    5: -24.5290607285,
    6: -37.688618963,
    7: -54.4009342085,
    8: -74.80939847,
    9: -99.4093493867,
    10: -128.547098109,
    11: -161.858911617,
    12: -199.614636425,
    13: -241.876707251,
    14: -288.854362517,
    15: -340.718780975,
    16: -397.504895917,
    17: -459.482072393,
    18: -526.817512803,
    19: -599.164786767,
    20: -676.758185925,
    21: -759.735718041,
    22: -848.405996991,
    23: -942.884337738,
    24: -1043.35637629,
    25: -1149.86625171,
    26: -1262.4436654,
    27: -1381.41455298,
    28: -1506.87090819,
    29: -1638.96374218,
    30: -1777.84811619,
    31: -1923.26100961,
    32: -2075.35973391,
    33: -2234.23865428,
    34: -2399.8676117,
    35: -2572.44133316,
    36: -2752.05497735,
    37: -2938.35745426,
    38: -3131.54568644,
    39: -3331.68416985,
    40: -3538.99506487,
    41: -3753.59772775,
    42: -3975.54949953,
    43: -4204.78873702,
    44: -4441.53948783,
    45: -4685.88170428,
    46: -4937.92102407,
    47: -5197.6984731,
    48: -5465.13314253,
    49: -5740.16915577,
    50: -6022.93169531,
    51: -6313.48532075,
    52: -6611.78405928,
    53: -6917.98089626,
    54: -7232.13836387,
    55: -7553.93365766,
    56: -7883.54382733,
    57: -8221.0667026,
    58: -8566.87268128,
    59: -8921.18102813,
    60: -9283.88294453,
    61: -9655.09896927,
    62: -10034.9525472,
    63: -10423.5430217,
    64: -10820.6612101,
    65: -11226.5683738,
    66: -11641.4525953,
    67: -12065.2898028,
    68: -12498.1527833,
    69: -12940.1744048,
    70: -13391.4561931,
    71: -13851.8080034,
    72: -14321.2498119,
    73: -14799.812598,
    74: -15287.5463682,
    75: -15784.5331876,
    76: -16290.6485954,
    77: -16806.1131497,
    78: -17331.0699646,
    79: -17865.4000842,
    80: -18408.9914949,
    81: -18961.8248243,
    82: -19524.0080381,
    83: -20095.5864271,
    84: -20676.500915,
    85: -21266.8817131,
    86: -21866.7722409,
    87: -22475.8587125,
    88: -23094.3036664,
    89: -23722.1920622,
    90: -24359.622444,
    91: -25007.1098723,
    92: -25664.3382676,
    93: -26331.4549589,
    94: -27008.7194421,
    95: -27695.8872166,
    96: -28392.7711729,
    97: -29099.8316144,
    98: -29817.418916,
    99: -30544.9721855,
    100: -31282.777599,
    101: -32030.9329688,
    102: -32789.5121404,
    103: -33557.9504126,
    104: -34336.6215955,
    105: -35125.5446447,
    106: -35924.7569387,
    107: -36734.3244057,
    108: -37554.1214298,
    109: -38384.3424294,
    110: -39225.1624771,
    111: -40076.3544159,
    112: -40937.7978561,
    113: -41809.5353119,
    114: -42691.6571511,
    115: -43584.1991337,
    116: -44487.1002441,
    117: -45400.4748133,
    118: -46324.3558151,
}

_ATOMIC_LEGENDRE_COEFFS = {
    'O': [
        [
            1.614709e00,
            -5.148965e00,
            5.235046e-02,
            -1.726141e-01,
            1.476052e-01,
            1.898161e-02,
        ],  # cc-pVDZ
        [
            -0.97793704,
            1.77658338,
            -0.27137494,
            0.02623075,
            0.0351218,
            -0.06996131,
        ],  # 3 functions initial guess
    ]
}
