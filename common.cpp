
#include "PGmult.h"

int randomN(int N){
    uniform_int_distribution<std::mt19937::result_type> dist(0,N-1);
    return dist(rng);
}

double randUniform(){
    uniform_real_distribution<double> distribution(0.0,1.0);
    return distribution(rng);
}

void computeSoftmaxPolicy(double* logits, int size, vector<int> validActions, double* policy){
    double maxLogit = -1e+10;
    for(auto a : validActions){
        if(logits[a] > maxLogit){
            maxLogit = logits[a];
        }
    }
    double sum = 0;
    for(auto a : validActions){
        sum += exp(logits[a] - maxLogit);
    }
    for(int i=0; i<size; i++){
        policy[i] = -1;
    }
    for(auto a : validActions){
        policy[a] = exp(logits[a] - maxLogit) / sum;
    }
}

int sampleDist(double* dist, int N){
    double sum = 0;
    for(int i=0; i<N; i++){
        if(dist[i] >= 0) sum += dist[i];
    }
    if(abs(sum - 1) > 1e-07){
        string s = "Invalid distribution\n";
        for(int i=0; i<N; i++){
            s += to_string(dist[i]) + ' ';
        }
        s += '\n';
        ofstream errOut ("err.out");
        errOut<<s;
        errOut.close();
    }
    assert(abs(sum - 1) < 1e-07);

    double parsum = 0;
    double randReal = randUniform();
    
    int index = -1;
    for(int i=0; i<N; i++){
        if(dist[i] < 0){
            continue;
        }
        parsum += dist[i];
        if(randReal < parsum + 1e-06){
            index = i;
            break;
        }
    }
    assert(index != -1);
    return index;
}