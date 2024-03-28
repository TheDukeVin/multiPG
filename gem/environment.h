
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_multiact/common.h"

#ifndef environment_h
#define environment_h

const int numSlots = 10;
const int numGems = 5;

const int timeHorizon = 3;
const double discountFactor = 1;
const int numActions = 2;
const int actionCount = numSlots;
const int numFeatures = numSlots*(numGems+1);

class Environment{
public:
    int timeIndex;
    bool endState;

    int gems[numSlots];
    bool locked[numSlots];

    vector<int> validActs[actionCount];
    int actions[actionCount];

    // In environments where we don't necessarily execute every action
    bool executedAction[actionCount];

    Environment();
    string toString();
    void validActions();
    double makeAction(); // returns reward
    void getFeatures(double* features);
};

#endif