#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Slide.h"

using namespace std;

class Individual
{
    vector<int> chromosome;
    size_t chromosomeLength;

public:
    int fitness;
    // Constructor
    Individual(vector<int> chromosome);
    // Getters
    vector<int> getChromosome();
    int getFitness();
    size_t getChromosomeLength();
    // Setters
    void setFitness(int fitness);
    //
    Individual mate(Individual parent2);
    int calculateFitness();
    // Overloading < operator
    bool operator<(const Individual &ind2) const
    {
        return this->fitness > ind2.fitness;
    }
};

#endif //INDIVIDUAL_H