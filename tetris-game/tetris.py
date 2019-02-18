import numpy as np

tiles = [
    # ##
    # ##
    np.asmatrix([[1, 1],
                 [1, 1]],
                dtype=np.int32),
    #  # 
    # ###
    np.asmatrix([[0, 1, 0],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    #  ##
    # ##
    np.asmatrix([[0, 1, 1],
                 [1, 1, 0],
                 [0, 0, 0]],
                dtype=np.int32),
    # ##
    #  ##
    np.asmatrix([[1, 1, 0],
                 [0, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    # #
    # ###
    np.asmatrix([[1, 0, 0],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    #   #
    # ###
    np.asmatrix([[0, 0, 1],
                 [1, 1, 1],
                 [0, 0, 0]],
                dtype=np.int32),
    # ####
    np.asmatrix([[1, 1, 1, 1],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0],
                 [0, 0, 0, 0]],
                dtype=np.int32)
    ]

