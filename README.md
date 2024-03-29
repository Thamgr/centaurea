# Centaurea

## Neural network for playing chess!

The input is flattened matrices for all 12 possible piece types. For a given position centaurea returns evaluation in range [0, 1]. 0 means win for the side, whose move it is.

The net includes 3 hidden layers, 2048 neurons each. It uses ReLU activation after each layer.

<img src="https://i.imgur.com/0plKaGJ.png" width="300">

Training dataset was build on random generated positions; evaluations were provided by top chess engine Stockfish 11.

Playing at depth 1, Centaurea is capable of beating weak chess engines. The following game was played against random-move bot:

<img src="https://i.imgur.com/TOZB3zI.gif" width="200">

We can see that it takes opponent's hanging pieces and retrieve own ones. Probably due to 1-depth searhc, it often misses simple tactics. Yet it has already reached quite competitive level of play!

[Trained weights for uci engine](https://drive.google.com/file/d/1u1cizMC0JXSVcM7QkQ9HcdqMwnffgxC4/view?usp=sharing)
