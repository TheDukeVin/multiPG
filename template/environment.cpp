
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;
}

string Environment::toString(){
    return "";
}

void Environment::validActions(){
    for(int i=0; i<actionCount; i++){
        validActs[i] = vector<int>();
    }
}

double Environment::makeAction(){
    double reward = 0;
    for(int i=0; i<actionCount; i++){
        executedAction[i] = false;
    }
    
    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
    }
    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }
}