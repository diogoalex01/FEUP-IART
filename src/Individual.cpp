#include "Individual.h"
#include "PhotoSelector.h"

Individual::Individual(vector<int> chromosome)
{
    this->chromosome = chromosome;
    chromosomeLength = chromosome.size();
};

// Perform mating and produce new offspring
Individual Individual::mate(Individual secondParent)
{
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0, chromosomeLength - 1);

    unordered_set<int> usedSlides;
    vector<int> childChromosome;

    size_t added = 0;
    size_t lastFromFirstParent = 0;
    size_t lastFromSecondParent = 0;
    size_t blockIndex = 0;

    size_t blockSize = 10;

    while (childChromosome.size() < chromosomeLength)
    {
        while (added < blockSize && childChromosome.size() < chromosomeLength && blockIndex + lastFromFirstParent < chromosomeLength)
        {
            auto search = usedSlides.find(chromosome.at(blockIndex + lastFromFirstParent));

            if (search == usedSlides.end())
            {
                childChromosome.push_back(chromosome.at(blockIndex + lastFromFirstParent));
                added++;
                usedSlides.insert(chromosome.at(blockIndex + lastFromFirstParent));
            }

            blockIndex++;
        }

        lastFromFirstParent = blockIndex;
        added = 0;
        blockIndex = 0;

        while (added < blockSize && childChromosome.size() < chromosomeLength && blockIndex + lastFromSecondParent < chromosomeLength)
        {
            auto search = usedSlides.find(secondParent.getChromosome().at(blockIndex + lastFromSecondParent));

            if (search == usedSlides.end())
            {
                childChromosome.push_back(secondParent.getChromosome().at(blockIndex + lastFromSecondParent));
                added++;
                usedSlides.insert(secondParent.getChromosome().at(blockIndex + lastFromSecondParent));
            }

            blockIndex++;
        }

        lastFromSecondParent = blockIndex;
        added = 0;
        blockIndex = 0;
    }

    return Individual(childChromosome);
};

void Individual::mutate()
{
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0, chromosomeLength - 1);
    size_t nMutations = chromosomeLength / 10;
    size_t firstIndex, secondIndex;

    while (nMutations > 0)
    {
        firstIndex = dis(generator);
        secondIndex = dis(generator);

        swap(chromosome.at(firstIndex), chromosome.at(secondIndex));

        nMutations--;
    }
}

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
