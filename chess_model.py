import torch
import torch.nn as nn


class ChessModel(nn.Module):
    def __init__(self, board_obs_pretrained=None):
        '''
        Main chess model class holding board observation block and move block
        :param board_obs_pretrained: load pretrained weights if exists
        '''
        super().__init__()
        self.board_nn = BoardObsBlock()  # Encoder transfer bitmap to general representation
        if board_obs_pretrained is not None:
            self.board_nn.load_state_dict(board_obs_pretrained)
            print("Loaded from pretrained")

        self.next_moves_nn = MoveBlock()  # Move block NN

    def forward(self, board, generated_boards):
        board = torch.tensor(board)
        next_move_embds = self.apply_next_move_embds(generated_boards)
        next_move_embds = torch.stack(next_move_embds, dim=0)
        output = self.board_nn(board)
        next_move_output, board_with_moves = self.next_moves_nn(next_move_embds, output)

        return next_move_output

    def apply_next_move_embds(self, generated_boards):
        new_l = []
        for gen in generated_boards:
            new_l.append(self.board_nn(torch.Tensor(gen)))

        return new_l


class BoardObsBlock(nn.Module):
    def __init__(self, embedding=False):
        '''
        Board observation encoder added extra layer (100,1) to compare with real board evaluation score and able to fit
        properly the model, scores normalized by 100
        :param embedding: add extra layer for fine tuning or not
        '''
        super().__init__()
        self.block = [nn.Linear(773, 500), nn.Tanh(), nn.Linear(500, 256), nn.Tanh(), nn.Linear(256, 400), nn.Tanh(),
                      nn.Linear(400, 400),
                      nn.Tanh(), nn.Linear(400, 256), nn.Tanh(), nn.Linear(256, 100)]
        self.last_layer = None
        if embedding:
            self.block.append(nn.Tanh())
            self.block.append(nn.Linear(100, 1))

        self.block = nn.Sequential(*self.block)

    def forward(self, board):
        obs_output = self.block(board)
        return obs_output


class MoveBlock(nn.Module):
    def __init__(self, num_attention_layers=8, attention_size=512, output_size=30):
        '''
        Move block gets boards representation of N legal moves and calculates their probabilities
        :param num_attention_layers: number of blocks
        :param attention_size: number of linear layer for each layer
        :param output_size: N moves

        Output: vector of ranked moves
        '''
        super().__init__()
        self.moves_dim = output_size
        self.first_layer = [nn.Linear(self.moves_dim * 200, attention_size), nn.ReLU()]
        self.first_layer = nn.Sequential(*self.first_layer)
        self.attention_block = torch.nn.MultiheadAttention(attention_size, num_attention_layers)
        self.last_layer = [nn.ReLU(), nn.Linear(attention_size, output_size)]
        self.last_layer = nn.Sequential(*self.last_layer)

    def forward(self, input, board):
        board = board.expand(self.moves_dim, 100)
        input = torch.cat([board, input], dim=1)
        input = torch.flatten(input)
        output = self.first_layer(input)
        output = output.unsqueeze(0)
        attention_output, attention_weights = self.attention_block(output, output, output)
        output = self.last_layer(attention_output)
        return output, board


# for future ideas
class MoveGeneratorBlock(nn.Module):
    def __init__(self, depth=4, num_attention_layers=3, attention_size=512, output_size=64):
        super().__init__()
        self.board_state = self.BoardObsBlock()
        self.blocks = []
        for i in range(depth):
            move_block = MoveBlock(num_attention_layers, attention_size, output_size)
            self.blocks.append(move_block)
