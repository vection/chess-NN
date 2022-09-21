import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import chess.pgn
from chess_evaluator import ChessEvaluator
import pickle
from torch.utils.data import Dataset
from multiprocessing import Pool

class ChessData:
    def __init__(self, path='E:/Aviv/chess_project/Lichess Elite Database/Lichess Elite Database/'):
        '''
        class to handle all data manipulations
        :param path: folder of pgn games
        '''
        self.path = path
        self.chess_eval = ChessEvaluator()

    def generate_boards(self, board, padding=30):
        '''
        Generate N moves of size padding by given board
        :param board: board to generate on
        :param padding: padding number which max length of moves
        :return: list of new boards, list of moves taken, list of scores
        '''
        # legal_moves = list(self.board.legal_moves) ### for testing
        legal_moves = self.chess_eval.get_best_moves(board.fen())
        all_moves = []
        scores = []
        new_boards = []
        for move in legal_moves:
            copy_board = board.copy()
            move_to_make = chess.Move.from_uci(move['Move'])
            copy_board.push(move_to_make)
            if move['Centipawn'] is None:
                scores.append(0)
            else:
                scores.append(move['Centipawn'])
            board_unicode = self.get_bitboard(copy_board)  # self.chess_emb.convert_board(copy_board.unicode())
            new_boards.append(board_unicode)
            all_moves.append(move_to_make)

        if len(all_moves) < padding:
            for i in range(int(padding - len(all_moves))):
                new_boards.append([0 for i in range(773)])
                scores.append(0)
                all_moves.append(0)
        return new_boards, all_moves, scores

    @staticmethod
    def load_score_dataset(pkl_path, train_rate=0.8):
        if isinstance(pkl_path, str):
            with open(pkl_path, 'rb') as f:
                dataset = pickle.load(f)
        else:
            dataset = pkl_path
        df = pd.DataFrame(dataset)
        train_df = df.sample(frac=train_rate, random_state=200).reset_index(drop=True)
        valid_df = df.drop(train_df.index).reset_index(drop=True)

        return BoardScoreDataset(train_df), BoardScoreDataset(valid_df)

    def create_score_dataset(self, path=None):
        '''
        Dataset creation from folder of files or list of files
        :param path: path to folder / list of files
        :return: dict with data
        '''
        all_files = os.listdir(self.path)
        # if path is not None:
        #     all_files = path
        current_boards = []
        all_scores = []
        board = chess.Board()
        for file in tqdm(all_files):

            if 'pgn' in file:
                print(file)
                file = open(self.path + file, 'r')
                f_pgn = chess.pgn.read_game(file)
                while f_pgn is not None:
                    # print("New game")
                    for move in list(f_pgn.mainline_moves()):
                        try:
                            board.push(move)
                        except AssertionError:
                            move_parsed = chess.Move.from_uci(move)
                            board.push(move_parsed)
                        current_board = board.fen()
                        position_score = self.chess_eval.calculate_score(board)
                        all_scores.append(position_score)
                        current_boards.append(current_board)

                    result = f_pgn.headers['Result']
                    # print("Game status end", f_pgn.is_end(), result)
                    f_pgn = chess.pgn.read_game(file)
                    board.reset()

        return {'current_board': current_boards, 'scores': all_scores}

    def create_move_dataset(self, path=None):
        '''
        Dataset creation from folder of files or list of files
        :param path: path to folder / list of files
        :return: dict with data
        '''
        all_files = os.listdir(self.path)
        if path is not None:
            all_files = path
        next_moves = []
        current_boards = []
        all_scores = []
        current_board_score = []
        board = chess.Board()
        for file in tqdm(all_files):

            if 'pgn' in file:
                print(file)
                file = open(self.path + file, 'r')
                f_pgn = chess.pgn.read_game(file)
                while f_pgn is not None:
                    # print("New game")
                    for move in list(f_pgn.mainline_moves()):
                        try:
                            board.push(move)
                        except AssertionError:
                            move_parsed = chess.Move.from_uci(move)
                            board.push(move_parsed)
                        next_move_candidates, all_moves, scores = self.generate_boards(board)
                        current_board = board.fen()
                        position_score = self.chess_eval.calculate_score(board)
                        all_scores.append(scores)
                        current_boards.append(current_board)
                        next_moves.append(next_move_candidates)
                        current_board_score.append(position_score)

                    result = f_pgn.headers['Result']
                    # print("Game status end", f_pgn.is_end(), result)
                    f_pgn = chess.pgn.read_game(file)
                    board.reset()

        return {'current_board': current_boards, 'scores': all_scores, 'next_move': next_moves,
                'current_board_score': current_board_score}

    def save(self, path, data):
        '''
        Save function
        :param path: pkl path
        :param data: dict data
        :return: True if success
        '''
        with open(path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True

    def get_bitboard(self, board):
        '''
        Transform board to general representation
        :param board: type chess.Board
        :return:  bitboard representation of the state of the game
        64 * 6 + 5 dim binary numpy vector
        64 squares, 6 pieces, '1' indicates the piece is at a square
        5 extra dimensions for castling rights queenside/kingside and whose turn
        '''

        bitboard = np.zeros(64 * 6 * 2 + 5)

        piece_idx = {'p': 0, 'n': 1, 'b': 2, 'r': 3, 'q': 4, 'k': 5}

        for i in range(64):
            if board.piece_at(i):
                color = int(board.piece_at(i).color) + 1
                bitboard[(piece_idx[board.piece_at(i).symbol().lower()] + i * 6) * color] = 1

        bitboard[-1] = int(board.turn)
        bitboard[-2] = int(board.has_kingside_castling_rights(True))
        bitboard[-3] = int(board.has_kingside_castling_rights(False))
        bitboard[-4] = int(board.has_queenside_castling_rights(True))
        bitboard[-5] = int(board.has_queenside_castling_rights(False))

        return bitboard


class BoardScoreDataset(Dataset):
    def __init__(self, data):
        '''
        Dataset object for base model
        :param data:
        '''
        self.scores = [score / 100 for score in data['scores']]  # normalize all scores by 100
        self.board_obs = data['current_board']

    def __len__(self):
        return len(self.scores)

    def __getitem__(self, idx):
        return self.board_obs[idx], self.scores[idx]




def merge_dataset_files(files):
    positions = []
    scores = []
    for pkl_path in files:
        with open(pkl_path, 'rb') as f:
            dataset = pickle.load(f)

            [scores.append(i) for i in dataset['scores']]
            [positions.append(i) for i in dataset['current_board']]
    data = {'current_board':positions, 'scores':scores}
    return ChessData.load_score_dataset(data, 0.8)

def parse_score_multi_results(res):
    positions = []
    scores = []
    for proc_result in res:
        [positions.append(i) for i in proc_result[0]]
        [scores.append(i) for i in proc_result[1]]

    return positions,scores

def create_score_dataset_multi(all_files, max_process=os.cpu_count(),save_path=None):
    def save(save_p,pos,sc):
        data = {'current_board': pos, 'scores': sc}
        '''
        Save function
        :param path: pkl path
        :param data: dict data
        :return: True if success
        '''
        with open(save_p, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return True
    proc_pool = Pool(processes=max_process)
    print("Multiprocessing activated with {} workers".format(max_process))
    res = proc_pool.map(create_score_dataset_static, all_files)
    position,scores = parse_score_multi_results(res)
    if save is not None:
        save(save_path, position,scores)
    return position,scores

def create_score_dataset_static(file):
    '''
    Dataset creation from folder of files or list of files
    :param path: path to folder / list of files
    :return: dict with data
    '''
    current_boards = []
    all_scores = []
    board = chess.Board()
    if 'pgn' in file:
        print(file)
        file = open(file, 'r')
        f_pgn = chess.pgn.read_game(file)
        while f_pgn is not None:
            # print("New game")
            for move in list(f_pgn.mainline_moves()):
                try:
                    board.push(move)
                except AssertionError:
                    move_parsed = chess.Move.from_uci(move)
                    board.push(move_parsed)
                current_board = board.fen()
                position_score = ChessEvaluator.calculate_score(board)
                all_scores.append(position_score)
                current_boards.append(current_board)

            result = f_pgn.headers['Result']
            # print("Game status end", f_pgn.is_end(), result)
            f_pgn = chess.pgn.read_game(file)
            board.reset()

    return [current_boards, all_scores]
