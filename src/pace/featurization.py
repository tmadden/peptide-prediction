import numpy as np

def do_FMLN_encoding(peplist, m=8, n=3):
    """
    First m last n. e.g. for FELT encoding, the default, m=8, n=3

    :param peplist: the list of peptides to encode

    :param m: use the first m residues

    :param n: concatenated with the last n residues

    :returns: encoded peptide list
    """

    return [p[0:m] + p[-n:] for p in peplist]


amino_acids = "ACDEFGHIKLMNPQRSTVWY"

aa5d = {"A": [0.354311, 3.76204, -11.0357, -0.648649, 2.82792],
        "R": [7.5734, -10.135, 2.48594, -4.29106, -5.6871],
        "N": [11.2937, 1.06685, 2.71827, 1.96258, -0.859314],
        "D": [13.4195, -1.60027, -0.325263, 3.7422, 2.43733],
        "C": [-5.84613, 4.88503, 1.62632, 9.39709, -5.84334],
        "Q": [6.59904, -5.16578, -0.696992, 0.582121, -1.74988],
        "E": [9.78784, -7.86097, -7.31842, 2.61123, 4.73404],
        "G": [9.65497, 15.7781, -0.557594, 0.299376, 1.65613],
        "H": [1.01864, -4.96926, 0.952556, 4.65696, -0.328102],
        "I": [-15.634, 1.99332, -2.04451, -3.24324, -1.67176],
        "L": [-11.8251, 0.505348, -6.15677, -4.55717, 3.21852],
        "K": [10.7622, -9.51739, -1.02226, -5.40541, -0.421845],
        "M": [-10.585, -3.95856, -3.60113, 5.33888, 1.20304],
        "F": [-14.571, -0.645723, 1.67278, -0.033264, 3.24977],
        "P": [7.66197, 8.02942, 9.45586, -3.57588, 5.99957],
        "S": [8.81349, 6.68183, -0.348496, -1.13098, -3.06228],
        "T": [3.01164, 4.12701, -0.348496, -2.19543, -4.28095],
        "W": [-13.1095, -5.22193, 9.03767, 1.38046, 4.6403],
        "Y": [-6.24473, -1.60027, 9.87406, -1.59667, -1.42177],
        "V": [-12.1352, 3.81819, -4.34459, -3.25988, -4.67154],
        "X": [0.0, 0.0, 0.0, 0.0, 0.0]}
# note the final amino acid, X. this is in some of the test data (rarely though)


def do_5d_encoding(peplist):
    """
    5D encoding using amino acid multi dimensional scaling properties

    :param peplist: the list of peptides to encode

    :returns: encoded peptide list
    """
    e = []
    for p in peplist:
        d = []
        for c in p:
            d = d + aa5d[c]
        e.append(d)

    return e

# a dictionary containing the 7d encoding of the HLA alleles.
hla7d = {"A0101": [324.766, -42.3269, -9.42882, -35.6718, 169.164, -83.6432, 116.185],
         "A0201": [333.475, -85.0009, -65.4655, 70.537, -140.327, 82.7846, 4.63228],
         "A0202": [344.584, -62.1799, -58.103, 73.5427, -128.878, 102.699, -1.45338],
         "A0203": [361.85, -26.4017, -126.734, 4.86373, -107.062, 116.761, -12.5105],
         "A0204": [358.149, -90.8417, -28.4419, 76.2574, -121.966, 54.3282, -22.6128],
         "A0205": [332.949, -62.0543, -60.4789, 63.3079, -134.188, 96.2142, -19.5234],
         "A0206": [322.256, -84.6104, -67.8361, 60.5319, -145.458, 76.8616, -13.0925],
         "A0207": [346.455, -81.7856, -55.6056, 76.3887, -146.425, 95.1263, 5.07423],
         "A0211": [359.37, -54.7699, -61.9689, 65.3301, -124.59, 87.8622, -15.5576],
         "A0301": [285.522, -8.30883, -49.5867, -35.1263, 25.2693, -58.4191, 26.4301],
         "A1101": [278.017, 4.32194, -34.657, -31.813, 81.4289, -58.3968, 61.2415],
         "A1102": [285.501, 4.94663, -39.8893, -33.831, 86.9057, -63.3131, 62.8656],
         "A2301": [192.451, -231.225, 201.831, -66.6266, 59.3638, 137.106, -31.3582],
         "A2402": [237.685, -224.811, 240.209, -49.2404, 90.6779, 153.934, -21.6082],
         "A2407": [229.738, -216.824, 234.222, -55.2818, 96.4689, 156.96, -26.4362],
         "A2501": [141.232, -167.11, -64.8661, -209.027, 168.433, 21.7572, -37.5103],
         "A2601": [225.674, -13.8584, -147.672, -48.73, 94.9768, -55.898, 6.13798],
         "A2902": [242.649, -71.756, 0.66427, 31.4797, 37.961, -91.0264, 6.98198],
         "A3001": [279.857, -14.7696, 4.80853, 21.1388, 22.5339, -74.5164, 25.8432],
         "A3002": [246.518, -63.8862, 10.8413, 29.9867, 60.0964, -99.409, 6],
         "A3101": [280.054, -38.4118, 0.894747, 17.0068, 16.1997, -67.6481, 58.4783],
         "A3201": [129.9, -201.877, 74.7643, -108.84, 76.3698, -10.8929, 3.53094],
         "A3301": [256.137, -41.1085, -30.3413, 30.9549, 45.954, -73.1464, -14.4398],
         "A3303": [250.84, -37.9172, -23.4074, 31.0795, 38.0265, -67.9381, -3.90866],
         "A3401": [236.39, 34.3235, -155.355, -61.707, 70.4852, 10.7366, -1.80241],
         "A3402": [267.471, 3.76176, -106.123, -51.947, 83.6979, -78.0087, -32.8851],
         "A3601": [296.601, -16.0174, -13.4141, -21.6886, 112.476, -93.645, 56.6737],
         "A6601": [236.844, 18.4454, -163.663, -61.3822, 80.2009, -35.8095, 10.7005],
         "A6801": [315.35, -50.7132, -68.6625, 32.7446, 7.1676, -45.0664, -46.2176],
         "A6802": [256.781, -59.2995, -105.007, 54.7035, -46.7105, 12.145, -65.7847],
         "A7401": [220.182, -38.6808, -14.3565, 6.00363, 0.246707, -54.2285, 45.2022],
         "B0702": [-120.711, 86.7746, -24.7856, 105.302, 143.364, -15.8973, -128.968],
         "B0704": [-124.872, 81.6682, -34.0575, 107.76, 146.229, -7.83603, -124.556],
         "B0801": [-153.426, 12.0477, 16.5871, 171.572, 91.7852, -6.30467, -77.5888],
         "B1301": [-267.383, -220.387, -1.67179, 9.4758, -84.1126, 30.3925, 31.3891],
         "B1302": [-262.215, -222.599, 30.2632, 45.8475, -26.426, 15.9047, 2.08898],
         "B1402": [-148.316, 28.9807, -37.4203, 73.3018, 92.2331, -25.3369, -53.8978],
         "B1501": [-187.877, -30.2155, -91.7606, 0.492643, -38.4483, -67.3954, 37.4444],
         "B1502": [-192.472, -47.6736, -127.934, -16.7632, -41.8775, -110.559, 40.9629],
         "B1503": [-221.983, -18.2716, -93.3081, 31.4279, -9.58137, -40.2643, 38.9182],
         "B1510": [-215.021, -26.319, -105.145, 55.9558, -12.3584, -13.8524, -48.1012],
         "B1517": [-116.151, -162.46, 37.2685, -210.873, -20.6844, -39.4808, 14.675],
         "B1801": [-176.431, -44.6879, -69.9558, 122.489, 27.4772, -17.0521, 12.6456],
         "B2705": [-114.901, -124.754, 49.6316, 38.1179, 119.595, 34.6247, 37.5389],
         "B3501": [-206.243, -48.9683, -90.5699, 15.0337, -70.0101, -91.7229, -4.17562],
         "B3503": [-211.611, -48.9634, -78.1823, 42.7428, -90.9641, -47.3979, -36.9566],
         "B3507": [-223.207, -49.705, -104.495, 14.4976, -74.9959, -102.902, -11.4399],
         "B3701": [-163.667, -124.428, 60.7883, 73.7676, 47.7489, 126.191, 67.166],
         "B3801": [-203.486, -190.947, 16.4836, 10.3351, 62.6206, 106.855, -99.4469],
         "B3802": [-191.313, -171.067, 0.0800488, 38.4261, 52.4389, 89.494, -78.4657],
         "B4001": [-246.582, -36.5647, -22.9408, 228.283, -25.5104, 22.8211, 76.089],
         "B4002": [-170.638, -53.6915, -25.2698, 195.333, 33.4546, 47.2693, 69.1927],
         "B4006": [-176.216, -61.9119, -35.3419, 198.867, 38.8483, 33.6022, 69.0447],
         "B4201": [-95.1638, 40.691, 13.7983, 144.529, 97.2544, 11.2255, -121.821],
         "B4402": [-282.635, -214.431, 33.5992, -22.2319, 52.5092, 23.4498, 165.85],
         "B4403": [-274.886, -228.035, -1.91267, -32.5896, 8.8742, 12.1218, 158.141],
         "B4501": [-273.033, -53.4082, 2.07051, 160.449, 10.0731, 33.0808, 147.48],
         "B4601": [-96.98, 110.364, -68.3334, -90.4927, -77.2145, -48.4396, -11.7466],
         "B4901": [-314.865, -164.287, 22.8002, -49.417, 29.8044, 96.3947, 88.7368],
         "B5001": [-264.912, -15.9007, -86.9117, 89.7673, -14.8012, 2.52236, 113.391],
         "B5101": [-253.124, -177.153, -20.2465, -93.7343, 41.0953, 18.4837, -120.525],
         "B5201": [-260.05, -172.115, 19.1753, -76.5082, 11.154, 37.1966, -43.6924],
         "B5301": [-249.69, -184.2, -3.59567, -113.015, -27.5088, -16.2248, -41.1715],
         "B5401": [-162.196, 42.885, 2.32797, 47.273, -6.62129, -56.5085, -101.107],
         "B5501": [-163.593, 40.8951, -45.8945, 26.5072, 42.3958, -71.1925, -110.493],
         "B5502": [-153.487, 7.54264, -6.84504, 72.5769, 6.94411, -67.8642, -101.993],
         "B5601": [-174.507, -1.17092, -18.1727, 56.4199, -4.22751, -87.1624, -94.2318],
         "B5701": [-125.868, -238.13, 83.1821, -162.054, -106.102, -122.31, -11.836],
         "B5703": [-122.64, -247.918, 100.764, -115.853, -129.493, -67.7032, -59.67],
         "B5801": [-162.986, -182.92, 66.8455, -162.546, -141.857, -33.3733, -2.19717],
         "B5802": [-144.128, -167.694, 139.375, -136.127, -89.3486, -60.0733, -22.5441],
         "C0102": [-27.5086, 253.261, 60.0832, 11.0726, 33.403, 85.608, -12.9594],
         "C0202": [-66.8217, 230.536, -35.7476, -106.042, -32.1818, 36.9678, 16.0958],
         "C0302": [-87.7161, 177.227, -56.4427, -104.639, -70.5721, -12.0704, 27.4498],
         "C0303": [-87.913, 189.107, -46.111, -80.9729, -100.866, 9.58792, -1.88126],
         "C0304": [-80.6525, 168.845, -49.6165, -76.0399, -89.6026, 19.0713, 3.15259],
         "C0401": [-20.5546, 268.514, 59.8682, -45.0003, 61.6652, 121.617, -4.60906],
         "C0403": [-49.4761, 249.561, 30.898, -72.8017, 7.18241, 83.2845, -11.2701],
         "C0501": [-52.0165, 242.156, 20.3453, -54.835, -16.6971, 95.5259, -27.2367],
         "C0602": [-46.6338, 268.249, 33.4088, -89.0955, 55.0966, 15.831, 45.4416],
         "C0701": [-72.5735, 247.666, 7.70436, -36.9451, 23.2499, -15.8764, 77.5906],
         "C0702": [-76.6601, 269.536, 11.7907, -32.2402, 8.03214, 43.0429, 74.5395],
         "C0704": [-83.2365, 292.148, 32.7894, 12.4237, 40.9147, 93.7188, 53.2889],
         "C0801": [-66.8049, 201.389, 1.70581, -10.3739, -65.3183, 77.3472, -28.5976],
         "C0802": [-59.7972, 247.731, 3.85076, -29.6895, -12.0939, 82.7751, -28.1321],
         "C1202": [-55.8652, 223.027, -35.5587, -72.4501, -31.8726, 28.8749, 4.86058],
         "C1203": [-44.978, 233.445, 7.35961, -77.3969, -10.6832, -13.7036, -5.57605],
         "C1402": [-16.6864, 247.1, 93.1332, -38.481, 53.3043, 13.2742, 11.4426],
         "C1403": [-19.3572, 261.843, 92.2812, -46.6671, 52.6951, 15.6073, 9.88445],
         "C1502": [-93.1043, 197.009, -28.973, -100.831, -73.5043, -17.2725, -22.5644],
         "C1601": [-41.7196, 212.433, 59.5754, -42.8664, -2.35308, -17.7098, 20.321],
         "C1701": [-76.1826, 218.541, -19.9087, -54.7388, -56.7312, 57.7515, -4.95788],
         "G0101": [139.215, 131.822, 343.099, 124.984, -87.553, -138.057, 3.97286],
         "G0103": [140.516, 134.158, 348.06, 126.989, -88.7754, -141.79, 3.23507],
         "G0104": [140.747, 134.572, 348.944, 127.347, -88.9943, -142.455, 3.10482]}


# HLA SNPS in 6 regionally organized sets
snpSets95 = [[32, 45, 46, 52, 62, 63, 103, 107, 163, 166, 167, 171, 177, 178, 180],
        [65, 66],
        [9, 24, 67, 69, 99, 113, 156],
        [70, 71, 97, 114],
        [12, 73, 95, 116, 152],
        [76, 77, 79, 80, 81, 82, 83, 143]]

snpSets16 = [[32, 45, 46, 52, 62, 63, 103, 107, 163, 166, 167, 171],
        [65, 66],
        [9, 24, 67, 69, 99, 113, 156],
        [70, 71, 97, 114],
        [12, 73, 95, 116, 152],
        [76, 77, 79, 80, 81, 82, 83]]

# for explanation of the following stuff for MHC featurization and bringing into the neural net, see prepareMHCinput.py
# ved is vector of encoding dimension
vedAll95 = [2, 1, 1, 1, 1, 2, 1, 1, 6, 2, 2, 2, 1, 2, 1, 3, 2, 1, 2, 1, 2, 1, 1, 3, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 2, 1, 1,
        1, 1, 2, 2, 2, 3, 2, 5, 2, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 5, 3, 1, 3, 3, 7, 1, 3, 5, 3, 1, 3, 3,
        1, 4, 3, 1, 3, 4, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 3, 1, 1, 2, 5, 1, 8, 1, 5, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 1,
        1, 2, 6, 1, 6, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 2, 2, 2, 3, 1, 3, 1,
        2, 2, 3, 6, 1, 1, 1, 5, 1, 3, 1, 1, 2, 1, 4, 1, 1, 2, 3, 1, 2, 2, 2, 1, 2, 1, 1, 1, 3, 3, 1, 2, 1, 2]

vedAll16 = [1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 
        1, 1, 1, 2, 1, 2, 2, 4, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 5, 3, 1, 3, 3, 5, 1, 2, 4, 3, 1, 2, 3, 1, 3, 
        3, 1, 2, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 4, 1, 5, 1, 3, 1, 1, 1, 2, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 5, 1, 
        4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 3, 1, 1, 
        1, 5, 1, 2, 1, 1, 2, 1, 3, 1, 1, 2, 3, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2]


def get_vedAll(allele_set_number):
    rval = None
    if allele_set_number == 16:
        rval = vedAll16
    elif allele_set_number == 95:
        rval = vedAll95
    else:
        raise Exception(
            'in get_vedAll, expecting 16 or 95, got {}'.format(allele_set_number))
    return rval

def get_snpSets(allele_set_number):
    rval = None
    if allele_set_number == 16:
        rval = snpSets16
    elif allele_set_number == 95:
        rval = snpSets95
    else:
        raise Exception(
            'in get_snpSets, expecting 16 or 95, got {}'.format(allele_set_number))
    return rval   


# allele sets are used for building machine learning algorithms that aggregate data across "close" alleles
# should ultimately put in the code which goes from the hla7d and forms (possibly overlapping) allele clusters
# but for now I've done this in matlab: computeHLAdistanceMatrix.m
allele_sets_16 = [[0, 1, 2, 3, 4, 9],
                  [0, 5, 7, 8, 9],
                  [0, 6, 9],
                  [11, 12, 13, 14, 15],
                  [10, 13, 14, 15]]
               
a16_names = ['A0101', 'A0201', 'A0203', 'A0204', 'A0207', 'A0301', 'A2402', 'A2902', 'A3101', 'A6802',
             'B3501', 'B4402', 'B4403', 'B5101', 'B5401', 'B5701']


allele_sets_95 = [[34, 35, 41, 43, 47, 48, 49, 54, 55, 56, 58, 60, 61, 62, 67, 68, 69, 70],
                  [0, 12, 13, 14, 15, 21, 92, 93, 94],
                  [0, 1, 2, 3, 4, 5, 6, 7, 8, 28, 29],
                  [0, 9, 10, 11, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30],
                  [31, 32, 33, 36, 37, 38, 39, 40, 42, 44, 45, 46, 51, 52, 53, 56, 59, 63, 64, 65, 66],
                  [31, 32, 33, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 56, 59, 63, 64, 65],
                  [57, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91],
                  [71, 72, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 90, 91]]

a95_names = ['A0101', 'A0201', 'A0202', 'A0203', 'A0204', 'A0205', 'A0206', 'A0207', 'A0211', 'A0301', 
             'A1101', 'A1102', 'A2301', 'A2402', 'A2407', 'A2501', 'A2601', 'A2902', 'A3001', 'A3002', 
             'A3101', 'A3201', 'A3301', 'A3303', 'A3401', 'A3402', 'A3601', 'A6601', 'A6801', 'A6802', 
             'A7401', 'B0702', 'B0704', 'B0801', 'B1301', 'B1302', 'B1402', 'B1501', 'B1502', 'B1503', 
             'B1510', 'B1517', 'B1801', 'B2705', 'B3501', 'B3503', 'B3507', 'B3701', 'B3801', 'B3802', 
             'B4001', 'B4002', 'B4006', 'B4201', 'B4402', 'B4403', 'B4501', 'B4601', 'B4901', 'B5001', 
             'B5101', 'B5201', 'B5301', 'B5401', 'B5501', 'B5502', 'B5601', 'B5701', 'B5703', 'B5801', 
             'B5802', 'C0102', 'C0202', 'C0302', 'C0303', 'C0304', 'C0401', 'C0403', 'C0501', 'C0602', 
             'C0701', 'C0702', 'C0704', 'C0801', 'C0802', 'C1202', 'C1203', 'C1402', 'C1403', 'C1502', 
             'C1601', 'C1701', 'G0101', 'G0103', 'G0104']

def get_allele_sets(allele_set_number):
    rsets = None
    rnames = None
    if allele_set_number == 16:
        rsets = allele_sets_16
        rnames = a16_names
    elif allele_set_number == 95:
        rsets = allele_sets_95
        rnames = a95_names
    else:
        raise Exception(
            'in get_allele_sets, expecting 16 or 95, got {}'.format(allele_set_number))
    return rsets, rnames


def split_into_sets(xydata, xa, classmembers, canames):
    """
    Partitions a full dataset into subsets based on given allele sets (called classmembers).

    :param xydata: this is the machine learning ready data, as a numpy array where last column is y data to predict
    
    :param xa: allele names that go along with the xydata

    :param classmembers: a list of lists. classmembers[0] is the list of allele integer identifiers that constitute the first (0th) allele set
    
    :param canames: these are the allele names, the indexing of this list corresponds to the allele integer identifier (could have also used a dictionary)

    :return: the xydata split into subsets (possibly overlapping, i.e. rows can go into more than one set)
             and, for each set, the allele list (classmembers) corresponding to it.
    """

    # replace allele names with unique integers [where they are in a16_names, or a95_names : called canames in general]:
    xai = []
    for i in range(len(xa)):
        xai.append(canames.index(xa[i]))

    # return will be a list of numpy arrays, which are the xydata with only the correct rows.
    xysets = []
    forestmembers = []
    # now step through and form 5 (8 if 95 allele dataset) subsets
    for i in range(len(classmembers)):
        # select all the rows for these classmembers
        myi = np.where(np.in1d(xai, classmembers[i]))
        # xysets.append(xydata[tuple(myi),:])
        # save if we found some samples that are in this set:
        if len(myi[0]) > 0:
            xysets.append(xydata[tuple(myi)])
            forestmembers.append(classmembers[i])
    return xysets, forestmembers

def get_all_sets_for_allele(aname, forestmembers, canames):
    """
    Find the sets that a partiocular allele belongs to.

    :param aname: the allele name    
 
    :param forestmembers: list of lists. forestmembers[0] is the first allele set, etc.
  
    :param canames: these are the allele names, the indexing of this list corresponds to forestmembers (integers)

    :return: list of forests to use for the particular allele
    """

    anum = canames.index(aname)
    useforest = []
    for i in range(len(forestmembers)):
        if anum in forestmembers[i]:
            useforest.append(i)
    return useforest
