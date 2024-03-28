
#include "/Users/kevindu/Desktop/Coding/Tests:experiments/PG_multiact/common.h"

#ifndef environment_h
#define environment_h

const int timeHorizon = 0;
const double discountFactor = 0;
const int numActions = 0;
const int actionCount = 0;
const int numFeatures = 0;

class Environment{
public:
    int timeIndex;
    bool endState;

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