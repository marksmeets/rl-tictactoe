# rl-tictactoe
Experiments in reinforcement learning using java and dl4j

Run TDZero for a TD(0) based learner with tables.
QLambda is an implementation that makes use of eligibility traces
TDZeroNN and QLambdaNN are the same, but instead of a lookup table, they use a neural network as a function approximator. Unfortunately, this doesn't work yet. Anybody willing to help?

For the neural nets we use the deeplearning4j framework.
TDZero, SarsaLambda and TDLambda are table based. They work. TDZeroNN and TDLambdaNN are neural net based, and don't seem to work.

If you run TDZeroNN, you can see things go wrong. Instead of discovering that a move to an already taken spot is bad (reward -1) it seems to more and more take illegal moves. Looking at the output of the neural net, it looks like all outputs are pulled down (ie, become more negative) when in fact only one should. The rewards are either -1 (lost, or illegal) or +1 (won).

I've already experimented a lot with the learning rate, weight init, different activation functions and more (or less) neurons in the network, but they all seem to have the same problem. Also added (and removed again, to keep the example concise) double Q networks with experience replay.

One interesting point to note: if in the function QNNOneHot:argMax you uncomment the line that basically filters out any illegal moves, then TDZeroNN actually learns to play the game. After a short while it loses less than 1% of the games. Probably less with more neurons in the hidden layer, but good enough for me. Obviously, I want it to learn not to take illegal moves.