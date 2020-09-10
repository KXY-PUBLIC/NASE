from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal')

PRIMITIVES = [
    'none',
    'skip_connect',
    'sep_conv_3x3_2d',
    'sep_conv_5x5_2d',
    'sep_conv_3x3_1d',
    'TransE',
    'TransR',
    #'TransH'
]
CLASSIFIER = [
    'Trans_cal',
    'Conv_cal',
    'Fc_cal',
    'DistMult_cal',
    'SimplE_cal'
]

arc_1 = Genotype(normal=[('sep_conv_3x3_1d', 0), ('sep_conv_3x3_1d', 1), ('sep_conv_3x3_1d', 2), ('Conv_cal', 0)])
arc_2 = Genotype(normal=[('TransE', 0), ('skip_connect', 1), ('sep_conv_3x3_2d', 2), ('Conv_cal', 0)])
arc_3 = Genotype(normal=[('TransE', 0), ('sep_conv_3x3_2d', 1), ('sep_conv_3x3_2d', 2), ('TransE', 0), ('skip_connect', 1), ('TransR', 2), ('Conv_cal', 0)])



