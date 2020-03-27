#include "PhotoSelector.h"

// Constructor

PhotoSelector::PhotoSelector(vector<Photo> vPhotos, vector<Photo> hPhotos, int heuristic)
{
    this->vPhotos = vPhotos;
    this->hPhotos = hPhotos;
    this->heuristic = heuristic;
}

// Getters

int PhotoSelector::getCurrentScore()
{
    return currentScore;
}

vector<Photo> PhotoSelector::getVertical()
{
    return vPhotos;
}

vector<Photo> PhotoSelector::getHorizontal()
{
    return hPhotos;
}

vector<Slide> PhotoSelector::getCurrentSlides()
{
    return currentSlides;
}

// Setters

void PhotoSelector::setVertical(vector<Photo> vPhotos)
{
    this->vPhotos = vPhotos;
}

void PhotoSelector::setHorizontal(vector<Photo> hPhotos)
{
    this->hPhotos = hPhotos;
}

// Main Functions

void PhotoSelector::makeSlides()
{
    unordered_set<string> hTagsSet;
    vector<string> tags;

    for (auto it = hPhotos.begin(); it != hPhotos.end(); it++)
    {
        Slide slide({(*it)});
        tags = (*it).getTags();
        slide.setTags(tags);
        slide.setOrientation('H');
        currentSlides.push_back(slide);
    }

    printf("Horizontal slides finished!\n");

    for (auto it = vPhotos.begin(); it != vPhotos.end(); it++)
    {
        Slide slide({});
        slide.setOrientation('V');

        if (!(*it).getUsed())
        {
            findVerticalPair((*it), slide);
        }
    }

    printf("Vertical slides finished!\n");
    lastSlideIndex = getCurrentSlides().size() - 1;
    printf("Finished making %zu slides!\n", lastSlideIndex + 1);
    evaluateScore();
    printf("Initial score: %d\n", getCurrentScore());
    populationSize = 2500;
    maxGenerations = 25;
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    switch (heuristic)
    {
    case 1:
        hillClimbing();
        break;

    case 2:
        simulatedAnnealing();
        break;

    case 3:
        tabuSearch();
        break;

    case 4:
        geneticAlgorithm();
        break;
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    cout << "Elapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
    evaluateScore();
    // printf("Final Score: %d\n", getCurrentScore());
    // printf("State Changes: %ld\n", stateChanges);
}

void PhotoSelector::findVerticalPair(Photo &photo, Slide &slide)
{
    // if the number of vertical photos is odd, the last one is not used
    if (photo.getID() == vPhotos.at(vPhotos.size() - 1).getID() && vPhotos.size() % 2 != 0)
        return;

    for (auto it = vPhotos.begin(); it != vPhotos.end(); ++it)
    {
        if (it->getID() != photo.getID() && !it->getUsed())
        {
            slide.setPhotos({photo, *it});
            slide.setTags(photo.getTags(), it->getTags());
            photo.setUsed(true);
            it->setUsed(true);
            currentSlides.push_back(slide);
            return;
        }
    }
}

void PhotoSelector::getRandomIndexes(size_t &firstSlideIndex, size_t &secondSlideIndex, mt19937 generator)
{
    uniform_int_distribution<int> dis(0, lastSlideIndex);

    firstSlideIndex = dis(generator);
    secondSlideIndex = dis(generator);

    // if the same slide is chosen, find two new slides
    while (firstSlideIndex == secondSlideIndex)
    {
        firstSlideIndex = dis(generator);
        secondSlideIndex = dis(generator);
    }

    // secondSlideIndex always after firstSlideIndex
    if (firstSlideIndex > secondSlideIndex)
    {
        swap(firstSlideIndex, secondSlideIndex);
    }
}

void PhotoSelector::genPairVerticalSlides(Slide &s1, Slide &s2, Slide firstSlide, Slide secondSlide, mt19937 generator)
{
    uniform_int_distribution<int> dis2(0, 1);

    size_t firstPhoto = dis2(generator);
    size_t secondPhoto = dis2(generator);

    s1.setPhotos({firstSlide.getPhotos().at(firstPhoto),
                  secondSlide.getPhotos().at(secondPhoto)});
    s2.setPhotos({firstSlide.getPhotos().at(firstPhoto ^ 1),     // flip used photo index
                  secondSlide.getPhotos().at(secondPhoto ^ 1)}); // flip used photo index

    s1.setID(firstSlide.getID());
    s2.setID(secondSlide.getID());

    s1.setOrientation('V');
    s2.setOrientation('V');

    s1.setTags(firstSlide.getPhotos().at(firstPhoto).getTags(),
               secondSlide.getPhotos().at(secondPhoto).getTags());

    s2.setTags(firstSlide.getPhotos().at(firstPhoto ^ 1).getTags(),    // flip used photo index
               secondSlide.getPhotos().at(secondPhoto ^ 1).getTags()); // flip used photo index
}

int PhotoSelector::getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide)
{
    int commonTags = 0;   // Number of common tags
    int currentSlideTags; // Number of tags only in the current slide
    int nextSlideTags;    // Number of tags only in the next slide

    // Find the number of common tags in a given transition
    for (auto slideItr = currentSlide.begin(); slideItr != currentSlide.end(); slideItr++)
    {
        if (nextSlide.find((*slideItr)) != nextSlide.end())
            commonTags++;
    }

    currentSlideTags = currentSlide.size() - commonTags;
    nextSlideTags = nextSlide.size() - commonTags;

    int subResult = min(nextSlideTags, currentSlideTags);
    int finalResult = min(subResult, commonTags);

    return finalResult;
}

void PhotoSelector::compareScores(Slide &firstSlide, Slide &secondSlide, int &scoreBefore, int &scoreAfter, size_t firstSlideIndex, size_t secondSlideIndex)
{
    // A - B - C .... G (H) I
    // A -> B + B -> C + G -> H + H -> I
    // Compare with
    // A -> H + H -> C + G -> B + B -> I

    scoreBefore = 0;
    scoreAfter = 0;

    // firstSlideIndex is not the first slide
    if (firstSlideIndex != 0)
    {
        scoreBefore += getTransitionScore(currentSlides.at(firstSlideIndex - 1).getTags(), currentSlides.at(firstSlideIndex).getTags());
        scoreAfter += getTransitionScore(currentSlides.at(firstSlideIndex - 1).getTags(), secondSlide.getTags());
    }

    // secondSlideIndex is not the last slide
    if (secondSlideIndex != lastSlideIndex)
    {
        scoreBefore += getTransitionScore(currentSlides.at(secondSlideIndex).getTags(), currentSlides.at(secondSlideIndex + 1).getTags());
        scoreAfter += getTransitionScore(firstSlide.getTags(), currentSlides.at(secondSlideIndex + 1).getTags());
    }

    // firstSlideIndex and secondSlideIndex are not consecutive
    if (secondSlideIndex - firstSlideIndex > 1)
    {
        scoreBefore += getTransitionScore(currentSlides.at(firstSlideIndex).getTags(), currentSlides.at(firstSlideIndex + 1).getTags());
        scoreAfter += getTransitionScore(secondSlide.getTags(), currentSlides.at(firstSlideIndex + 1).getTags());
    }

    scoreBefore += getTransitionScore(currentSlides.at(secondSlideIndex - 1).getTags(), currentSlides.at(secondSlideIndex).getTags());
    scoreAfter += getTransitionScore(currentSlides.at(secondSlideIndex - 1).getTags(), firstSlide.getTags());
}

string PhotoSelector::tabuEntry(size_t firstIndex, size_t secondIndex, int kind)
{
    string entry;

    // fisrtIndex is not the first slide
    if (firstIndex != 0)
        entry = to_string(currentSlides.at(firstIndex - 1).getID()) + ".";

    entry += to_string(currentSlides.at(firstIndex).getID()) + "." + to_string(currentSlides.at(firstIndex + 1).getID());

    switch (kind)
    {
    case 0: // Swap Photos
        entry += "Photo";
        break;

    case 1: // Swap Position
        entry += "Position";
        break;
    }

    entry += "." + to_string(currentSlides.at(secondIndex - 1).getID()) + "." + to_string(currentSlides.at(secondIndex).getID());

    // secondIndex is not the last slide
    if (secondIndex != lastSlideIndex)
        entry += "." + to_string(currentSlides.at(secondIndex + 1).getID());

    return entry;
}

bool PhotoSelector::isTabu(string entry)
{
    string front;
    tabuAux = tabuList;

    while (!tabuAux.empty())
    {
        front = tabuAux.front();

        if (front == entry)
            return true;

        tabuAux.pop();
    }

    return false;
}

vector<int> PhotoSelector::createChromosome(size_t rangeOfIndexes)
{
    unordered_set<int> used;
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0, rangeOfIndexes);
    size_t slideIndex;
    vector<int> genome;

    while (genome.size() < rangeOfIndexes)
    {
        slideIndex = dis(generator);
        auto search = used.find(slideIndex);
        if (search == used.end())
        {
            genome.push_back(slideIndex);
            used.insert(slideIndex);
        }
    }

    return genome;
}

// Heuristics

void PhotoSelector::hillClimbing()
{
    printf("[Hill Climbing]\n");

    maxAttempts = lastSlideIndex + 1;
    printf("Max Number of Attempts: %ld\n", maxAttempts);

    while (nAttempts < maxAttempts)
    {
        nAttempts++;
        stateChanges++;
        neighbouringHC();
    }
}

void PhotoSelector::simulatedAnnealing()
{
    printf("[Simulated Annealing]\n");
    numIterations = lastSlideIndex + 1;
    printf("Number of Iterations: %ld Per Temperature Drop\n", numIterations);

    // keeps annealing till reaching the minimum temperature
    while (T > Tmin)
    {
        printf("T is %f\n", T);
        for (int i = 0; i < numIterations; i++)
        {
            stateChanges++;
            neighbouringSA();
        }

        T *= alpha; // decreases T: cooling phase
    }
}

void PhotoSelector::tabuSearch()
{
    printf("[Tabu Search]\n");
    maxAttempts = lastSlideIndex + 1;
    printf("Max Attempts: %ld\n", maxAttempts);

    while (nAttempts < maxAttempts)
    {
        nAttempts++;
        stateChanges++;
        neighbouringTS();
    }
}

// Neighbouring Functions

void PhotoSelector::neighbouringHC()
{
    random_device device;
    mt19937 generator(device());

    int scoreBefore = 0, scoreAfter = 0;
    size_t firstSlideIndex;
    size_t secondSlideIndex;

    getRandomIndexes(firstSlideIndex, secondSlideIndex, generator);

    Slide firstSlide = currentSlides.at(firstSlideIndex);
    Slide secondSlide = currentSlides.at(secondSlideIndex);

    if (firstSlide.getOrientation() == 'V' && secondSlide.getOrientation() == 'V')
    {
        uniform_int_distribution<int> dis2(0, 1);
        int changeOption = dis2(generator);

        Slide s1({}, -1), s2({}, -1);

        // Swap photos between vertical slides
        if (changeOption)
        {
            genPairVerticalSlides(s1, s2, firstSlide, secondSlide, generator);
            compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

            if (scoreAfter > scoreBefore)
            {
                currentSlides.at(firstSlideIndex) = s1;
                currentSlides.at(secondSlideIndex) = s2;
                nAttempts = 0;
            }
        }
        // Swap position between vertical slides
        else
        {
            compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

            if (scoreAfter > scoreBefore)
            {
                vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
                vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
                iter_swap(firstSlideIter, secondSlideIter);
                nAttempts = 0;
            }
        }
    }
    // Swap position between horizontal slides
    else
    {
        compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

        if (scoreAfter > scoreBefore)
        {
            vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
            vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
            iter_swap(firstSlideIter, secondSlideIter);
            nAttempts = 0;
        }
    }
}

void PhotoSelector::neighbouringSA()
{
    random_device device;
    mt19937 generator(device());
    uniform_real_distribution<> saDis(0.0, 1.0);

    double chance = saDis(generator);
    int scoreBefore = 0, scoreAfter = 0;
    size_t firstSlideIndex, secondSlideIndex;

    getRandomIndexes(firstSlideIndex, secondSlideIndex, generator);

    Slide firstSlide = currentSlides.at(firstSlideIndex);
    Slide secondSlide = currentSlides.at(secondSlideIndex);

    if (firstSlide.getOrientation() == 'V' && secondSlide.getOrientation() == 'V')
    {
        uniform_int_distribution<int> dis2(0, 1);
        int changeOption = dis2(generator);

        Slide s1({}, -1), s2({}, -1);

        // Swap photos between verical slides
        if (changeOption)
        {
            Slide s1({}, -1), s2({}, -1);
            genPairVerticalSlides(s1, s2, firstSlide, secondSlide, generator);

            compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);
            if (exp((scoreAfter - scoreBefore) / T) > chance)
            {
                currentSlides.at(firstSlideIndex) = s1;
                currentSlides.at(secondSlideIndex) = s2;
            }
        }
        // Swap position between verical slides
        else
        {
            compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

            if (exp((scoreAfter - scoreBefore) / T) > chance)
            {
                vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
                vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
                iter_swap(firstSlideIter, secondSlideIter);
            }
        }
    }
    // Swap position between horizontal slides
    else
    {
        compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

        if (exp((scoreAfter - scoreBefore) / T) > chance)
        {
            vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
            vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
            iter_swap(firstSlideIter, secondSlideIter);
        }
    }
}

void PhotoSelector::neighbouringTS()
{
    random_device device;
    mt19937 generator(device());

    int scoreBefore = 0, scoreAfter = 0;
    size_t firstSlideIndex;
    size_t secondSlideIndex;
    string entry;

    getRandomIndexes(firstSlideIndex, secondSlideIndex, generator);

    Slide firstSlide = currentSlides.at(firstSlideIndex);
    Slide secondSlide = currentSlides.at(secondSlideIndex);

    if (firstSlide.getOrientation() == 'V' && secondSlide.getOrientation() == 'V')
    {
        uniform_int_distribution<int> dis2(0, 1);
        int changeOption = dis2(generator);

        Slide s1({}, -1), s2({}, -1);

        // Swap photos between vertical slides
        if (changeOption)
        {
            genPairVerticalSlides(s1, s2, firstSlide, secondSlide, generator);

            entry = tabuEntry(firstSlideIndex, secondSlideIndex, 0);
            compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

            if (!isTabu(entry) && scoreAfter > scoreBefore)
            {
                currentSlides.at(firstSlideIndex) = s1;
                currentSlides.at(secondSlideIndex) = s2;
                nAttempts = 0;
                tabuList.push(entry);

                if (tabuList.size() > lastSlideIndex)
                {
                    tabuList.pop();
                }
            }
        }
        // Swap position between vertical slides
        else
        {
            compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);
            entry = tabuEntry(firstSlideIndex, secondSlideIndex, 1);

            if (!isTabu(entry) && scoreAfter > scoreBefore)
            {
                vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
                vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
                iter_swap(firstSlideIter, secondSlideIter);
                nAttempts = 0;

                tabuList.push(entry);

                if (tabuList.size() > lastSlideIndex)
                {
                    tabuList.pop();
                }
            }
        }
    }
    // Swap position between horizontal slides
    else
    {
        compareScores(firstSlide, secondSlide, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);
        entry = tabuEntry(firstSlideIndex, secondSlideIndex, 1);

        if (!isTabu(entry) && scoreAfter > scoreBefore)
        {
            vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
            vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
            iter_swap(firstSlideIter, secondSlideIter);
            nAttempts = 0;

            tabuList.push(entry);

            if (tabuList.size() > lastSlideIndex)
            {
                tabuList.pop();
            }
        }
    }
}

void PhotoSelector::geneticAlgorithm()
{
    unordered_set<int> used;
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0, populationSize / 2);
    int generation = 0;
    vector<int> bestIndividual;
    int maxFitness = 0;
    int firstParentIndex, secondParentIndex;
    Individual firstParent({}), secondParent({}), offspring({});

    vector<Individual> population;
    Individual newIndividual({});
    int individualFitness;

    // create initial population
    for (int i = 0; i < populationSize; i++)
    {
        vector<int> chromosome = createChromosome(lastSlideIndex);
        newIndividual = Individual(chromosome);
        individualFitness = calculateFitness(newIndividual);
        newIndividual.setFitness(individualFitness);
        population.push_back(newIndividual);
    }

    while (generation < maxGenerations)
    {
        printf("Generation %d ", generation);
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        sort(population.begin(), population.end());

        if (population[0].getFitness() > maxFitness)
        {
            maxFitness = population[0].getFitness();

            bestIndividual = population[0].getChromosome();
        }

        vector<Individual> newGeneration;

        int s = 0.1 * populationSize;

        for (int i = 0; i < s; i++)
            newGeneration.push_back(population[i]);

        s = 0.9 * populationSize;
        for (int i = 0; i < s; i++)
        {
            firstParentIndex = dis(generator);
            secondParentIndex = dis(generator);
            firstParent = population[firstParentIndex];
            secondParent = population[secondParentIndex];
            offspring = firstParent.mate(secondParent);
            individualFitness = calculateFitness(offspring);
            offspring.setFitness(individualFitness);
            newGeneration.push_back(offspring);
        }

        population = newGeneration;
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        cout << "Elapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;

        generation++;
    }

    printf("Final Score: %d\n", maxFitness);
}

// Evaluation Functions

int PhotoSelector::calculateFitness(Individual individual)
{
    int fitness = 0, transitionScore;
    vector<int> allIndexes = individual.getChromosome();

    if (individual.getChromosomeLength() <= 1)
        return 0;

    for (size_t i = 0; i < individual.getChromosomeLength() - 1; i++)
    {
        transitionScore = getTransitionScore(currentSlides.at(allIndexes.at(i)).getTags(), currentSlides.at(allIndexes.at(i + 1)).getTags());
        fitness += transitionScore;
    }

    return fitness;
};

void PhotoSelector::evaluateScore()
{
    int transitionScore;
    currentScore = 0;

    if (currentSlides.size() <= 1)
        return;

    for (auto slideItr = currentSlides.begin(); slideItr != currentSlides.end() - 1; slideItr++)
    {
        auto nextSlideItr = slideItr + 1;
        transitionScore = getTransitionScore(slideItr->getTags(), nextSlideItr->getTags());
        currentScore += transitionScore;
    }
}