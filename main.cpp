
#include "PGmult.h"

int main(){
    LSTM::PVUnit structure;
    structure.commonBranch = new LSTM::Model(LSTM::Shape(numFeatures));
    structure.commonBranch->addDense(100);
    structure.initPV();
    structure.policyBranch->addDense(50);
    structure.policyBranch->addOutput(actionCount*numActions);
    structure.valueBranch->addDense(30);
    structure.valueBranch->addOutput(1);
    PGMult trainer(&structure, "game.out");
    trainer.train(32, 100000);

    for(int i=0; i<10; i++){
        {
            ofstream gameOut (trainer.gameOutFile, ios::app);
            gameOut << "Game " << i << '\n';
            gameOut.close();
        }
        trainer.rollout(true);
    }
}