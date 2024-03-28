
/*
g++ -O2 -std=c++11 common.cpp gem/environment.cpp main.cpp PGmult.cpp -I "./LSTM" LSTM/node.cpp LSTM/model.cpp LSTM/PVUnit.cpp LSTM/layer.cpp LSTM/layers/lstmlayer.cpp LSTM/layers/policy.cpp LSTM/layers/conv.cpp LSTM/layers/pool.cpp LSTM/params.cpp && ./a.out

rsync -r PG_test kevindu@login.rc.fas.harvard.edu:./MultiagentSnake --exclude .git/
*/

#include "gem/environment.h"
#include "lstm.h"

#ifndef PG_h
#define PG_h

class PGInstance{
public:
    Environment env;
    double reward;
    double value;
    double policy[actionCount][numActions];
};

class PGMult{
public:
    const static double constexpr epsilon = 0.01;
    const static double constexpr alpha = 0.001;
    const static double constexpr regRate = 0.00001;

    LSTM::Model net;
    LSTM::Data* netInput;
    LSTM::Data* netOutput;
    LSTM::Model structure;

    string gameOutFile;

    PGMult(LSTM::Model structure_, string gameOutFile_);
    double rollout(bool print=false);
    void accGrad(PGInstance instance);
    void train(int batchSize, int numIter);
};

#endif
