import torch
import torch.nn as nn
import torch.nn.functional as func
import ntetris as tetris

COLUMN_OFFSET = tetris.PADDING
DROPPABLE_COLUMNS = tetris.COLUMNS + COLUMN_OFFSET
INPUT_LENGTH = 2 * tetris.COLUMNS + tetris.NUM_TILES

class DepthsNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.dense1 = nn.Linear(INPUT_LENGTH, 128, bias=True)
        self.dense2 = nn.Linear(128, 128, bias=True)
        self.dense3 = nn.Linear(128, 4 * DROPPABLE_COLUMNS, bias=True)



    def forward(self, depths, tile_id):
        one_hot_tile_id = func.one_hot(tile_id, num_classes=tetris.NUM_TILES)
        normalized_depths = depths / tetris.ROWS - 0.5
        relative_depths = depths - depths.max(dim=1,keepdim=True).values
        transformed_input = torch.concat([normalized_depths, relative_depths, one_hot_tile_id], dim=-1)

        data = self.dense1(transformed_input)
        data = func.relu(data)
        data = self.dense2(data)
        data = func.relu(data)
        output = self.dense3(data)
        return output


