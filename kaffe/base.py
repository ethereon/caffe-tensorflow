class KaffeError(Exception):
    pass

# Ordering of the blobs
IDX_WEIGHTS  = 0
IDX_BIAS     = 1

# The tensors are ordered (c_o, c_i, h, w) or (n, c, h, w)
IDX_N        = 0
IDX_C        = 1
IDX_C_OUT    = 0
IDX_C_IN     = 1
IDX_H        = 2
IDX_W        = 3

