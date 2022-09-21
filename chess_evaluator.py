from stockfish import Stockfish
import chess
import numpy as np

piece_values = {'P': 10, 'N': 30, 'B': 30, 'R': 50, 'Q': 90, 'K': 100, 'p': -10, 'n': -30, 'b': -30,
                    'r': -50,
                    'q': -90, 'k': -100}

position_values = {
    'P': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                   [1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0, 1.0],
                   [0.5, 0.5, 1.0, 2.5, 2.5, 1.0, 0.5, 0.5],
                   [0.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0],
                   [0.5, -0.5, -1.0, 0.0, 0.0, -1.0, -0.5, 0.5],
                   [0.5, 1.0, 1.0, -2.0, -2.0, 1.0, 1.0, 0.5],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),

    'N': np.array([[-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0],
                   [-4.0, -2.0, 0.0, 0.0, 0.0, 0.0, -2.0, -4.0],
                   [-3.0, 0.0, 1.0, 1.5, 1.5, 1.0, 0.0, -3.0],
                   [-3.0, 0.5, 1.5, 2.0, 2.0, 1.5, 0.5, -3.0],
                   [-3.0, 0.0, 1.5, 2.0, 2.0, 1.5, 0.0, -3.0],
                   [-3.0, 0.5, 1.0, 1.5, 1.5, 1.0, 0.5, -3.0],
                   [-4.0, -2.0, 0.0, 0.5, 0.5, 0.0, -2.0, -4.0],
                   [-5.0, -4.0, -3.0, -3.0, -3.0, -3.0, -4.0, -5.0]]),

    'B': np.array([[-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0],
                   [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                   [-1.0, 0.0, 0.5, 1.0, 1.0, 0.5, 0.0, -1.0],
                   [-1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, -1.0],
                   [-1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, -1.0],
                   [-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0],
                   [-1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, -1.0],
                   [-2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0]]),

    'R': np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                   [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                   [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                   [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                   [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                   [-0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5],
                   [0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0]]),

    'Q': np.array([[-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0],
                   [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
                   [-1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                   [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                   [-0.5, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, -0.5],
                   [-1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, -1.0],
                   [-1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, -1.0],
                   [-2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0]]),

    'K': np.array([[-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                   [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                   [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                   [-3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0],
                   [-2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0],
                   [-1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0],
                   [2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0],
                   [2.0, 3.0, 1.0, 0.0, 0.0, 1.0, 3.0, 2.0]])}

class ChessEvaluator:

    def __init__(self, stockfish_path='D://Downloads//stockfish_15_win_x64_avx2//stockfish_15_x64_avx2'):
        '''
        Evaluation class to generate all evaluation functions in the game
        to generate moves with scores I use stockfish
        :param stockfish_path: path to stockfish engine
        '''
        self.stock = Stockfish(path='D://Downloads//stockfish_15_win_x64_avx2//stockfish_15_x64_avx2')

    def get_best_moves(self, fen_position):
        self.stock.set_fen_position(fen_position)
        all_moves = self.stock.get_top_moves(30)
        return all_moves

    @staticmethod
    def calculate_score(board, position_ranking=True):
        # Position of pieces is not taken into account for their strength
        if position_ranking == 'None':
            total_eval = 0
            pieces = list(board.piece_map().values())

            for piece in pieces:
                total_eval += piece_values[str(piece)]

            return total_eval

        else:
            positionTotalEval = 0
            pieces = board.piece_map()

            for j in pieces:
                file = chess.square_file(j)
                rank = chess.square_rank(j)

                piece_type = str(pieces[j])
                positionArray = position_values[piece_type.upper()]

                if piece_type.isupper():
                    flippedPositionArray = np.flip(positionArray, axis=0)
                    positionTotalEval += piece_values[piece_type] + flippedPositionArray[rank, file]

                else:
                    positionTotalEval += piece_values[piece_type] - positionArray[rank, file]

            return positionTotalEval

    def minimax(self, board, depth, alpha, beta, maximizingPlayer, bestMove='h1h3'):
        if depth == 0 or board.is_game_over():
            if (board.turn is False):
                return self.position_evaluation(board, self.piece_values, self.position_values), bestMove
            else:
                return -1 * self.position_evaluation(board, self.piece_values, self.position_values), bestMove

        if maximizingPlayer:
            maxEval = -np.inf
            for child in [str(i).replace("Move.from_uci(\'", '').replace('\')', '') for i in
                          list(board.legal_moves)]:
                board.push(chess.Move.from_uci(child))
                eval_position = self.minimax(board, depth - 1, alpha, beta, False)[0]
                board.pop()
                maxEval = np.maximum(maxEval, eval_position)
                alpha = np.maximum(alpha, eval_position)
                if beta <= alpha:
                    break
            return maxEval

        else:
            minEval = np.inf
            minMove = np.inf
            for child in [str(i).replace("Move.from_uci(\'", '').replace('\')', '') for i in
                          list(board.legal_moves)]:
                board.push(chess.Move.from_uci(child))
                eval_position = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                minEval = np.minimum(minEval, eval_position)
                if minEval < minMove:
                    minMove = minEval
                    bestMin = child

                beta = np.minimum(beta, eval_position)
                if beta <= alpha:
                    break

            return minEval, bestMin

    # def check_rank(self, board, depth=3):
    #     final_eval = 0
    #
    #     for i in range(depth):
    #         legal_moves = list(board.legal_moves)
    #         scores = []
    #         for move in legal_moves:
    #             copy_board = board.copy()
    #             copy_board.push(move)
    #             scores.append((move, self.calculate_score(copy_board), deepcopy(copy_board)))
    #
    #         scores_sorted = sorted(scores, key=lambda item: item[1],reverse=not board.turn)
    #         final_eval = scores_sorted[0]
    #         board = scores_sorted[0][2]
    #
    #     return final_eval[1],final_eval[0]
    #
    #
    #
    # def calculate_best_move(self, board, depth=3):
    #     legal_moves = list(board.legal_moves)
    #     rank_moves = []
    #     for move in legal_moves:
    #         copy_board = board.copy()
    #         copy_board.push(move)
    #
    #         rank_moves.append((move, self.check_rank(copy_board,depth)[0]))
    #
    #     scores_sorted = sorted(rank_moves, key=lambda item: item[1],reverse=not board.turn)
    #     return scores_sorted[0][0],scores_sorted[0][1]
