import torch
from chess_model import ChessModel, BoardObsBlock
import os
import chess
import chess.pgn
import chess.svg
from chess_data import ChessData
from chess_evaluator import ChessEvaluator
from torch.nn.functional import normalize
from torch.utils.data import DataLoader
import time
from tqdm import tqdm


class ChessTrainer:
    def __init__(self, model_type,pretrained=None):
        '''
        Two options in model type:
            - base
            - move
        :param model_type: base or move
        '''
        self.board = chess.Board()
        self.chess_data = ChessData()
        self.chess_eval = ChessEvaluator()
        if model_type == 'base':
            self.chess_model = BoardObsBlock(embedding=True)
        elif model_type == 'move':
            self.chess_model = ChessModel(pretrained)
        else:
            raise Exception("Unrecognized model")

    def train_score_model(self, train, val, bs=32, epochs=1, lr=0.001):
        '''
        Train function for board representation evaluated against observation score
        position score normalized by 100
        :param train: train dataloader
        :param val: validation dataloader
        :param bs: batch size
        :param epochs: number of epochs
        :param lr: learning rate
        :return: None
        '''
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print("Device: ", device)
        self.chess_model.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        train_loader = DataLoader(train, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val, batch_size=bs, shuffle=True)

        self.chess_model.train()
        for i in tqdm(range(epochs)):
            cost = None
            start_time = time.time()
            for batch_ind, batch in enumerate(train_loader):
                score = torch.tensor(batch[1]).type(torch.DoubleTensor)

                bitmap_boards = []
                # convert boards to bitmap
                for obs in batch[0]:
                    self.board.set_fen(obs)
                    board_bitmap = self.chess_data.get_bitboard(self.board)
                    bitmap_boards.append(board_bitmap)

                board_obs = torch.Tensor(bitmap_boards).to(
                    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

                model_output = self.chess_model(board_obs)
                model_output = torch.reshape(model_output, (-1,))
                cost = self.loss(score, model_output)

                self.optimizer.zero_grad()

                cost.backward()

                self.optimizer.step()


            with torch.set_grad_enabled(False):
                cost = None
                for batch_ind, batch in tqdm(enumerate(val_loader)):
                    score = torch.tensor(batch[1]).type(torch.DoubleTensor)

                    bitmap_boards = []
                    # convert boards to bitmap
                    for obs in batch[0]:
                        self.board.set_fen(obs)
                        board_bitmap = self.chess_data.get_bitboard(self.board)
                        bitmap_boards.append(board_bitmap)

                    board_obs = torch.Tensor(bitmap_boards)

                    model_output = self.chess_model(board_obs)
                    cost = self.loss(score, model_output)

                print("Cost on validation:", cost.item())

            # decrease LR each epoch
            lr = lr * 0.999
            self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)

            print("Cost: ", cost.item(), "Time per epoch:", time.time() - start_time)

    def train_move_model(self, data=None):
        '''
        Training of move block model which gets representation of board (output from base model) and need to predict
        vector of probabilities on legal moves vector
        :param data:
        :return:
        '''
        lr = 0.001
        self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)
        self.loss = torch.nn.MSELoss()
        self.chess_model.train()
        # in case we want play itself
        if data is None:
            while True:
                try:
                    next_move_candidates, all_moves, scores = self.chess_data.generate_boards(self.board)
                    current_board = self.chess_data.get_bitboard(self.board)

                    logits = self.chess_model(current_board, next_move_candidates)
                    scores = torch.Tensor(scores)
                    scores = normalize(scores, p=2.0, dim=0)  # normalize scores to match loss function
                    best_move_vector = scores
                    cost = self.loss(logits, best_move_vector)
                    self.optimizer.zero_grad()

                    cost.backward()

                    self.optimizer.step()

                    # taking best move
                    if self.board.turn:
                        self.board.push(all_moves[scores.index(max(scores))])
                    else:
                        if isinstance(all_moves[scores.index(min(scores))], int):
                            self.board.push(all_moves[0])
                        else:
                            self.board.push(all_moves[scores.index(min(scores))])

                    print(self.board.fen())
                    if self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_repetition() or \
                            self.board.is_insufficient_material() or self.board.is_game_over():
                        self.board.reset()
                        print("New Game!", cost.item(), "Result: ", self.board.outcome().result())
                        lr = lr * 0.99
                        self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)
                except:
                    self.board.reset()
                    lr = lr * 0.99
                    self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)
                    print("New Game!")

        # in case we have training file
        else:
            all_files = os.listdir(data)
            for file in all_files:
                if 'pgn' in file:
                    print(file)
                    file = open(data + file, 'r')
                    f_pgn = chess.pgn.read_game(file)
                    while f_pgn is not None:
                        print("New game")
                        for move in list(f_pgn.mainline_moves()):

                            try:
                                self.board.push(move)
                            except AssertionError:
                                move_parsed = chess.Move.from_uci(move)
                                self.board.push(move_parsed)
                            except Exception as p:
                                print("Picking best move due Exception ", str(p))
                                next_move_candidates, all_moves, scores = self.chess_data.generate_boards(self.board)
                                if self.board.turn:
                                    self.board.push(all_moves[scores.index(max(scores))])
                                else:
                                    if isinstance(all_moves[scores.index(min(scores))], int):
                                        self.board.push(all_moves[0])
                                    else:
                                        self.board.push(all_moves[scores.index(min(scores))])

                                print("Exception take best move")

                            next_move_candidates, all_moves, scores = self.chess_data.generate_boards(self.board)
                            current_board = self.chess_data.get_bitboard(self.board)
                            logits = self.chess_model(current_board, next_move_candidates)
                            scores = torch.Tensor(scores)
                            scores = normalize(scores, p=2.0, dim=0)  # normalize scores to match loss function
                            best_move_vector = scores
                            logits = torch.reshape(logits, (-1,))
                            self.cost = self.loss(logits, best_move_vector)
                            self.optimizer.zero_grad()

                            self.cost.backward()

                            self.optimizer.step()

                            if self.board.is_checkmate() or self.board.is_stalemate() or self.board.is_fivefold_repetition() or self.board.is_repetition() or \
                                    self.board.is_insufficient_material() or self.board.is_game_over():
                                self.board.reset()
                                print("New Game! Optimizer activated", self.cost.item())
                                lr = lr * 0.99
                                self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)
                        result = f_pgn.headers['Result']
                        print("Game status end", f_pgn.is_end(), result)
                        f_pgn = chess.pgn.read_game(file)
                        self.board.reset()
                        # decrease LR each game, might change the policy
                        lr = lr * 0.99
                        self.optimizer = torch.optim.Adam(self.chess_model.parameters(), lr=lr)
                        print("Loss:", self.cost)

        print("Training end ")


    def save(self, path):
        cfg = self.chess_model.state_dict()
        torch.save(cfg, path)
        return True
