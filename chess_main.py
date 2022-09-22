import torch
import os
import chess
import chess.pgn
import chess.svg
from chess_data import ChessData,create_score_dataset_multi
from chess_evaluator import ChessEvaluator
from chess_trainer import ChessTrainer


'''
Chess solution based on NN with affinity to DeepChess paper 
The solution architecture:
Base block model - input is bitmap representation of board, output is score for the position (basic current position evaluation)
Move block model - input is bitmap representation of board, bitmap representation of N next move possibles 
                   output is vector of N moves with probabilities 
                   
Project planning:
- Fine tune base model as much as possible
- Train move model with pretrained base model
- Evaluate against human / bots
'''
class Board:
    def __init__(self):
        self.board = chess.Board()
        self.chess_eval = ChessEvaluator()
        self.chess_data = ChessData()
        self.chess_model = None

    def train_base_model(self, dataset_file,config={'bs' : 128, 'epochs': 5, 'train_split': 0.8,'save_path':'E:/Aviv/chess_project/score_model_1.pth'}):
        train, val = self.chess_data.load_score_dataset(dataset_file,train_rate=config['train_split'])
        base_model = ChessTrainer(model_type='base')
        base_model.train_score_model(train, val, bs=config['bs'], epochs=config['epochs'],save_checkpoint=config['save_path'])
        base_model.save(config['save_path'])

    def train_move_model(self, folder_path,base_pretrained=None, config={'bs' : 128, 'epochs': 5, 'train_split': 0.8,'save_path':'E:/Aviv/chess_project/move_model_1.pth'}):
        move_model = ChessTrainer(model_type='move',pretrained=base_pretrained)
        move_model.train_move_model(folder_path)
        move_model.save(config['save_path'])

    def play(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError


#
# if __name__ == "__main__":
#     torch.set_default_dtype(torch.float64)
#     print(torch.cuda.is_available())
#     print(torch.version.cuda)
#     path = 'E:/Aviv/chess_project/Lichess Elite Database/Lichess Elite Database/'
#     print(os.listdir(path))
#     board = Board()
    # all_files = os.listdir(path)
    # all_files = [path + f for f in all_files]
    # #res = board.chess_data.create_score_dataset(path)
    # res = create_score_dataset_multi(all_files[60:70],max_process=5,save_path=('E:/Aviv/chess_project/data_batch_7.pkl'))
    # print("Finished one batch")
    # res = create_score_dataset_multi(all_files[70:80], max_process=5, save_path=('E:/Aviv/chess_project/data_batch_8.pkl'))
    # #print(res)
    # board.chess_data.save('E:/Aviv/chess_project/data_all.pkl',res)

    #train board observation model
    # train, val = board.chess_data.load_score_dataset('E:/Aviv/chess_project/data.pkl')
    # chess_trainer = ChessTrainer(model_type='base')
    # chess_trainer.train_score_model(train,val,bs=256,epochs=25)
    # chess_trainer.save('E:/Aviv/chess_project/score_model_2.pth')


    # train move model
    # chess_trainer = ChessTrainer(model_type='move')
    # chess_trainer.train_move_model(path)

    # csv = pd.read_csv('E:/Aviv/chess_project/data.csv')
    # print(csv)
    # board.train(path)
    # torch.save(board.chess_model, 'E:/Aviv/chess_project/chess_weights.pth')
