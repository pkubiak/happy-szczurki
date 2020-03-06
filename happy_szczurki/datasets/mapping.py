"""Module contain mapping of record string labels into ints"""

LABELS_MAPPING = {
    None: 0,
    '22-kHz': 1,
    '22kHz': 1,
    'SH': 2,
    'FM': 3,
    'RP': 4,
    'FL': 5,
    'ST': 6,
    'CMP': 7,
    'IU': 8,
    'TR': 9,
    'RM': 10
}

REV_LABELS_MAPPING = {
    value: key for key, value in LABELS_MAPPING.items()
}
