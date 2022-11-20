# Centaurea

Neural network for playing chess!

The input is flattened matrices for all 12 possible piece types. For a given position centaurea returns evaluation in range [0, 1]. 0 means win for the side, whose move it is.

The net includes 3 hidden layers, 2048 neurons each. It uses ReLU activation after each layer.

![arch](https://i.imgur.com/0plKaGJ.png)
