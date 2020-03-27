#include "Individual.h"
#include "PhotoSelector.h"

Individual::Individual(vector<int> chromosome)
{
    this->chromosome = chromosome;
    chromosomeLength = chromosome.size();
};

// Perform mating and produce new offspring
Individual Individual::mate(Individual par2)
{
    unordered_set<int> used;
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0, chromosomeLength);
    size_t slideIndex;
    vector<int> childChromosome;
    size_t added = 0;

    while (added < chromosomeLength)
    {
        // random probability
        double p = dis(generator) / chromosomeLength;

        // if prob is less than 0.45, insert gene
        // from parent 1
        if (p < 0.45)
        {
            childChromosome.push_back(chromosome.at(added));
            added++;
        }

        // if prob is between 0.45 and 0.90, insert
        // gene from parent 2
        else if (p < 0.90)
        {
            childChromosome.push_back(par2.getChromosome().at(added));
            added++;
        }

        // otherwise insert random gene(mutate),
        // for maintaining diversity
        else
        {
            slideIndex = dis(generator);
            auto search = used.find(slideIndex);
            if (search == used.end())
            {
                childChromosome.push_back(slideIndex);
                used.insert(slideIndex);
                added++;
            }
        }
    }

    return Individual(childChromosome);
};

vector<int> Individual::getChromosome()
{
    return chromosome;
}

int Individual::getFitness()
{
    return fitness;
}

size_t Individual::getChromosomeLength()
{
    return chromosomeLength;
}

void Individual::setFitness(int fitness)
{
    this->fitness = fitness;
}
