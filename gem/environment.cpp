
#include "environment.h"

Environment::Environment(){
    timeIndex = 0;
    endState = false;

    for(int i=0; i<numSlots; i++){
        gems[i] = randomN(numGems);
        locked[i] = false;
    }
}

string Environment::toString(){
    string s = "Gems:   ";
    for(int i=0; i<numSlots; i++){
        s += to_string(gems[i]) + ' ';
    }
    s += "\nLocked: ";
    for(int i=0; i<numSlots; i++){
        s += to_string(locked[i]) + ' ';
    }
    s += '\n';
    return s;
}

void Environment::validActions(){
    for(int i=0; i<actionCount; i++){
        validActs[i] = vector<int>();
    }
    for(int i=0; i<numSlots; i++){
        if(locked[i]) validActs[i] = vector<int>{0};
        else validActs[i] = vector<int>{0, 1};
    }
}

double Environment::makeAction(){
    double reward = 0;
    for(int i=0; i<actionCount; i++){
        executedAction[i] = false;
    }

    for(int i=0; i<numSlots; i++){
        if(actions[i] == 1){
            locked[i] = true;
        }
        if(!locked[i]){
            gems[i] = randomN(numGems);
        }
        executedAction[i] = true;
    }
    
    timeIndex ++;
    if(timeIndex == timeHorizon){
        endState = true;
        int numType[numGems];
        for(int i=0; i<numGems; i++){
            numType[i] = 0;
        }
        for(int i=0; i<numSlots; i++){
            numType[gems[i]] ++;
        }
        for(int i=0; i<numGems; i++){
            reward += pow(numType[i], 2);
        }
        // reward -= (pow(numSlots, 2) + numSlots*numGems - numSlots) / numGems;
    }
    return reward;
}

void Environment::getFeatures(double* features){
    for(int i=0; i<numFeatures; i++){
        features[i] = 0;
    }

    // Generate equivalent permutation to minimize biases
    // Eventually we want to not have to do this.
    int perm[numGems];
    for(int i=0; i<numGems; i++){
        perm[i] = i;
    }
    // shuffle(perm, perm + numGems, default_random_engine{dev()});

    for(int i=0; i<numSlots; i++){
        features[i*numGems + perm[gems[i]]] = 1;
        features[numSlots*numGems + i] = locked[i];
    }
}