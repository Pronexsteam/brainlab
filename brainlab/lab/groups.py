"""Именованные группы клеток. У червя — классический контур касания (Chalfie 1985)."""
WORM = {
    "anterior_touch": ["ALML", "ALMR", "AVM"],
    "posterior_touch": ["PLML", "PLMR"],
    "backward_cmd": ["AVAL", "AVAR", "AVDL", "AVDR", "AVEL", "AVER"],
    "forward_cmd": ["AVBL", "AVBR", "PVCL", "PVCR"],
}
