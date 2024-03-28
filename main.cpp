
#include "PGmult.h"

int main(){
    LSTM::Model structure = LSTM::Model(LSTM::Shape(numFeatures));
    structure.addDense(60);
    structure.addOutput(actionCount*numActions);
    PGMult trainer(structure, "game.out");
    trainer.train(32, 100000);

    for(int i=0; i<5; i++){
        {
            ofstream gameOut (trainer.gameOutFile, ios::app);
            gameOut << "Game " << i << '\n';
            gameOut.close();
        }
        trainer.rollout(true);
    }
}