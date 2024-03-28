
#include "PGmult.h"

PGMult::PGMult(LSTM::Model structure_, string gameOutFile_){
    gameOutFile = gameOutFile_;
    ofstream gameOut (gameOutFile_);
    gameOut.close();

    structure = structure_;
    structure.randomize(0.1);
    structure.resetGradient();
    netInput = new LSTM::Data(structure.inputSize);
    netOutput = new LSTM::Data(structure.outputSize);
    net = LSTM::Model(&structure, NULL, netInput, netOutput);
    net.copyParams(&structure);
}

double PGMult::rollout(bool print){
    Environment env;
    vector<PGInstance> trajectory;
    for(int t=0; t<timeHorizon; t++){
        PGInstance instance;
        env.validActions();
        env.getFeatures(netInput->data);
        net.forwardPass();
        for(int i=0; i<actionCount; i++){
            double policy[numActions];
            computeSoftmaxPolicy(netOutput->data + i*numActions, numActions, env.validActs[i], policy);
            for(int j=0; j<numActions; j++){
                instance.policy[i][j] = policy[j];
            }

            int action;
            if(randUniform() < epsilon){
                action = env.validActs[i][randomN(env.validActs[i].size())];
            }
            else{
                action = sampleDist(policy, numActions);
            }
            env.actions[i] = action;
        }
        instance.env = env;
        trajectory.push_back(instance);

        trajectory[t].reward = env.makeAction();

        if(print){
            ofstream gameOut (gameOutFile, ios::app);
            gameOut << trajectory[t].env.toString();
            gameOut << "Policy: ";
            for(int i=0; i<actionCount; i++){
                if(trajectory[t].env.validActs[i].size() == 1){
                    gameOut << ". ";
                }
                else{
                    gameOut << trajectory[t].policy[i][1] << ' ';
                }
            }
            gameOut << "\nActions: ";
            for(int i=0; i<actionCount; i++){
                if(trajectory[t].env.validActs[i].size() == 1){
                    gameOut << ". ";
                }
                else{
                    gameOut << trajectory[t].env.actions[i] << ' ';
                }
            }
            gameOut << "\nReward: " << trajectory[t].reward << "\n\n";
        }
        if(env.endState) break;
    }
    double value = 0;
    double total_reward = 0;
    for(int t=trajectory.size()-1; t>=0; t--){
        if(t < trajectory.size()-1){
            value *= pow(discountFactor, trajectory[t+1].env.timeIndex - trajectory[t].env.timeIndex);
        }
        value += trajectory[t].reward;
        total_reward += trajectory[t].reward;
        trajectory[t].value = value;

        accGrad(trajectory[t]);
    }
    return total_reward;
}

void PGMult::accGrad(PGInstance instance){
    instance.env.getFeatures(netInput->data);
    net.forwardPass();
    net.resetGradient();
    for(int i=0; i<netOutput->size; i++){
        netOutput->gradient[i] = 0;
    }
    for(int i=0; i<actionCount; i++){
        for(auto a : instance.env.validActs[i]){
            netOutput->gradient[i*numActions + a] = 0.0001 * netOutput->data[a] + (instance.policy[i][a] - (a == instance.env.actions[i])) * instance.value;
        }
    }
    net.backwardPass();
    structure.accumulateGradient(&net);
}

void PGMult::train(int batchSize, int numIter){
    ofstream fout("score.out");
    double sum = 0;
    int evalPeriod = 1000;
    double evalSum = 0;
    unsigned start_time = time(0);
    string controlLog = "control.out";
    {
        ofstream controlOut ("control.out");
        controlOut.close();
    }
    for(int it=0; it<numIter; it++){
        for(int i=0; i<batchSize; i++){
            double value = rollout();
            sum += value;
            if(it >= numIter/2){
                evalSum += value;
            }
        }
        structure.updateParams(alpha, -1, regRate);
        net.copyParams(&structure);
        if(it % evalPeriod == 0){
            if(it > 0){
                fout << ',';
            }
            double avgScore = sum / batchSize / evalPeriod;
            fout << avgScore;
            {
                ofstream controlOut(controlLog, ios::app);
                controlOut << "Iteration " << it << " Time: " << (time(0) - start_time) << ' ' << avgScore << '\n';
                controlOut.close();
            }
            sum = 0;
        }
    }
    fout << '\n';
    {
        ofstream controlOut(controlLog, ios::app);
        controlOut << "Evaluation score: " << (evalSum / batchSize / (numIter/2)) << '\n';
        controlOut.close();
    }
}