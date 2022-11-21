import random
import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

print("Centaurea 0.1")

coef = 2000

mulpv = 1

def get_material(fen):
    
    board = chess.Board(fen)
    
    w, b = chess.WHITE, chess.BLACK
    
    if board.turn == chess.BLACK:
        w, b = b, w
    
    material = 0.0
    material += len(board.pieces(chess.PAWN,   w)) * 1 *  1
    material += len(board.pieces(chess.PAWN,   b)) * 1 * -1
    material += len(board.pieces(chess.KNIGHT, w)) * 3 *  1
    material += len(board.pieces(chess.KNIGHT, b)) * 3 * -1
    material += len(board.pieces(chess.BISHOP, w)) * 3 *  1
    material += len(board.pieces(chess.BISHOP, b)) * 3 * -1
    material += len(board.pieces(chess.ROOK,   w)) * 5 *  1
    material += len(board.pieces(chess.ROOK,   b)) * 5 * -1
    material += len(board.pieces(chess.QUEEN,  w)) * 9 *  1
    material += len(board.pieces(chess.QUEEN,  b)) * 9 * -1
    
    return material * 100


class Simple_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(768, 2048)
        self.lin2 = nn.Linear(2048, 2048)
        self.lin3 = nn.Linear(2048, 2048)
        self.lin4 = nn.Linear(2048, 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        x = torch.relu(x)
        x = self.lin4(x)
        x = torch.clamp(x, max = 1.0, min = 0.0)
        return x


def conv2cp(x):
    return int(coef * (x - 0.5) * 2)


def get_scp(fen):
    
    board = chess.Board(fen)
    
    w, b = chess.WHITE, chess.BLACK
    
    if board.turn == chess.BLACK:
        board = board.transform(chess.flip_vertical)
        # board = board.transform(chess.flip_horizontal)
        w, b = b, w

    data = np.zeros((12, 8, 8))
    
    for i in board.pieces(chess.PAWN, w):
        data[0, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.PAWN, b):
        data[1, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.KNIGHT, w):
        data[2, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.KNIGHT, b):
        data[3, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.BISHOP, w):
        data[4, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.BISHOP, b):
        data[5, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.ROOK, w):
        data[6, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.ROOK, b):
        data[7, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.QUEEN, w):
        data[8, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.QUEEN, b):
        data[9, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.KING, w):
        data[10, 7 - i // 8, i % 8] = 1
    
    for i in board.pieces(chess.KING, b):
        data[11, 7 - i // 8, i % 8] = 1
    
    return data


try:
    net = torch.load("model.pt", map_location=torch.device('cpu'))
    print("NN loaded")
except:
    print("NN not found")
    
    
board = chess.Board()
while True:
    newStr = input()
    file = open("log.log", 'a')
    file.write(newStr + '\n')
    file.close()
    if 'stop' in newStr:
        exit()
    if 'position' in newStr:
        if 'startpos' in newStr:
            board = chess.Board()
        elif 'fen' in newStr:
            board = chess.Board(" ".join(newStr.split()[2:8]))
        if 'moves' in newStr:
            if 'fen' in newStr:
                for move in newStr.split()[9:]:
                    board.push(chess.Move.from_uci(move))
            else:
                for move in newStr.split()[3:]:
                    board.push(chess.Move.from_uci(move))
    elif 'go' in newStr:
        ar = []
        z = board.turn * 2 - 1

        for move1 in board.legal_moves:
            board.push(move1)
            if board.is_checkmate():
                evl = 9000
            elif board.is_stalemate() or board.is_repetition(2):
                evl = 0
            else:
                evl = -conv2cp(net(torch.Tensor([get_scp(board.fen())])))
            board.pop()
            ar.append([evl, str(move1), board.is_capture(move1)])
        ar = sorted(ar)

        pos_moves = []
        for move in ar[::-1]:
            if ar[-1][0] - move[0] < 20:
                pos_moves.append(move)
            else:
                break
        
        best_eval, best_move, flag = random.choice(pos_moves)
        for pv in range(min(mulpv, len(ar))):
            print('info depth 1 seldepth 1 multipv {} score cp {} nodes 1 nps 1 tbhits 0 time 1 pv '.format(pv + 1, ar[-pv - 1][0]) + str(ar[-pv - 1][1]))
        print(f'bestmove {best_move}')
        
    elif newStr == 'uci':
        print('id name Centaurea')
        print('id author Thamgr')
        print("option name model type string default model.pt")
        print("option name MultiPV type spin default 1 min 1 max 500")
        print('uciok')
    elif newStr == 'isready':
        print('readyok')
    elif newStr.startswith("setoption name"):
        options = {}
        parts = newStr.split(" ", 2)
        parts = parts[-1].split(" value ")
        name, value = parts[0], parts[1]
        if name == "model":
            try:
                net = torch.load(value)
                net.eval()
                print("NN loaded!")
            except:
                print("Bad net :(")
        if name == "MultiPV":
            mulpv = int(value)
        
    if board.is_checkmate():
        exit()
    if board.is_stalemate():
        exit()
