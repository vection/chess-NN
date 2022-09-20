# Chess Neural Network Model 

## Introduction:
The game of chess known as the oldest game of history, presented in 7th century in the old texts. What makes this game so attractive is the complexity of analyzing moves and able to see the few moves upfront your opponent. 
The history with computer chess started in 18th century only after a thousand years after invented. There are many approaches when we talk about computer and chess, naïve implementation would be alpha beta algorithm with declaring depth parameter which suffers from computation limit. Newer approach presented as DeepChess aiming to solve the game by using Deep Learning approach.
The approach presented in DeepChess is dedicated to choose between position 1 to position 2 by given 2 positions. They trained Pos2Vec model which aims to learn representation of the board and siamise model to evaluate two positions and score who is better.

## Suggested Approach
Seems like Pos2Vec did quite good job in representing the board, Instead of evaluating between two positions, evaluate with N number of moves.
For each position generate N top moves, for my experiment I chose N=30 which seems to be reasonably decent number as most of the time number of legal moves is lower. For generating the N top moves I used Stockfish engine, for testing I will use legal_moves method from chess python library.
The suggested network Encode the board presentation with BoardObs outputs vector of 100 neurons, then pass it to MoveBlock which takes all generated possible moves, pass it through BoardObs. Then input from current board and next board concatenated and inserted to the network. The motivation is to link between current position to next position to understand both positions in one layer.
For base model training I add extra layer 100->1 and y label it with position score evaluation.
The evaluation of MoveBlock would be N scores per move, the scores must be normalized. I chose MSE loss function to minimize the error to match the score.
The database I used is "Lichess rated games  2400+ elo".

## Model architecture
### Base observation model 
![base_chess_model_diagram](https://user-images.githubusercontent.com/28596354/191191765-fc2e1bb3-79ae-4e45-af19-fe5020a7e0cc.png)


### Move model
![move_chess_model](https://user-images.githubusercontent.com/28596354/191204492-bf275dab-4011-47d0-beaf-d66303770952.png)

## Preparing the data
To feed the data there are 2 inputs - current board representation and next moves representation 

## Training phase
The model train on the database running on the games, for each position generate top N moves with scores, then feed current board and generated boards to network with loss function on logits and scores. The move eventually made is taken by game and not top move score, this is to see more cases and create sense of random picking. For further experiments, can create more random behavior to see more situations.



