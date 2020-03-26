#include "PhotoSelector.h"

// Constructor

PhotoSelector::PhotoSelector(vector<Photo> vPhotos, vector<Photo> hPhotos)
{
    this->vPhotos = vPhotos;
    this->hPhotos = hPhotos;
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

    int heuristic = 1; // ------------------- HARD CODED --------------------

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
    }

    chrono::steady_clock::time_point end = chrono::steady_clock::now();

    cout << "Elapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
    evaluateScore();
    printf("Final Score: %d\n", getCurrentScore());
    printf("State Changes: %ld\n", stateChanges);
}

void PhotoSelector::findVerticalPair(Photo &photo, Slide &slide)
{
    // if the number of vertical photos is odd, the last one is not used
    if (photo.getID() == vPhotos.at(vPhotos.size() - 1).getID() && vPhotos.size() % 2 != 0)
    {
        return;
    }

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

int PhotoSelector::getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide)
{
    int commonTags = 0;   // Number of common tags
    int currentSlideTags; // Number of tags only in the current slide
    int nextSlideTags;    // Number of tags only in the next slide

    // Find the number of common tags in a given transition
    for (auto slideItr = currentSlide.begin(); slideItr != currentSlide.end(); slideItr++)
    {
        if (nextSlide.find((*slideItr)) != nextSlide.end())
        {
            commonTags++;
        }
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

// Heuristics

void PhotoSelector::hillClimbing()
{
    maxAttempts = lastSlideIndex + 1;
    printf("Max Attempts: %ld\n", maxAttempts);

    while (nAttempts < maxAttempts)
    {
        nAttempts++;
        stateChanges++;
        neighbouringHC();
    }
}

void PhotoSelector::simulatedAnnealing()
{
    numIterations = lastSlideIndex + 1;
    printf("Num Iterations: %ld\n", numIterations);

    // keeps annealing till reaching the minimum temperature
    while (T > Tmin)
    {
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
    maxAttempts = lastSlideIndex + 1;
    printf("Max Attempts: %ld\n", maxAttempts);

    while (nAttempts < maxAttempts)
    {
        nAttempts++;
        stateChanges++;
        neighbouringHC();
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

        size_t firstPhoto = dis2(generator);
        size_t secondPhoto = dis2(generator);

        Slide
            s1({firstSlide.getPhotos().at(firstPhoto),
                secondSlide.getPhotos().at(secondPhoto)},
               firstSlide.getID()),
            s2({firstSlide.getPhotos().at(firstPhoto ^ 1),    // flip used photo index
                secondSlide.getPhotos().at(secondPhoto ^ 1)}, // flip used photo index
               secondSlide.getID());

        s1.setTags(firstSlide.getPhotos().at(firstPhoto).getTags(),
                   secondSlide.getPhotos().at(secondPhoto).getTags());

        s2.setTags(firstSlide.getPhotos().at(firstPhoto ^ 1).getTags(),    // flip used photo index
                   secondSlide.getPhotos().at(secondPhoto ^ 1).getTags()); // flip used photo index

        s1.setOrientation('V');
        s2.setOrientation('V');

        compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

        if (scoreAfter > scoreBefore)
        {
            currentSlides.at(firstSlideIndex) = s1;
            currentSlides.at(secondSlideIndex) = s2;
            nAttempts = 0;
        }
    }
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

        size_t firstPhoto = dis2(generator);
        size_t secondPhoto = dis2(generator);

        Slide
            s1({firstSlide.getPhotos().at(firstPhoto),
                secondSlide.getPhotos().at(secondPhoto)},
               firstSlide.getID()),
            s2({firstSlide.getPhotos().at(firstPhoto ^ 1),    // flip used photo index
                secondSlide.getPhotos().at(secondPhoto ^ 1)}, // flip used photo index
               secondSlide.getID());

        s1.setOrientation('V');
        s2.setOrientation('V');

        s1.setTags(firstSlide.getPhotos().at(firstPhoto).getTags(),
                   secondSlide.getPhotos().at(secondPhoto).getTags());

        s2.setTags(firstSlide.getPhotos().at(firstPhoto ^ 1).getTags(),    // flip used photo index
                   secondSlide.getPhotos().at(secondPhoto ^ 1).getTags()); // flip used photo index

        compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

        if (exp((scoreAfter - scoreBefore) / T) > chance)
        {
            currentSlides.at(firstSlideIndex) = s1;
            currentSlides.at(secondSlideIndex) = s2;
        }
    }
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

    getRandomIndexes(firstSlideIndex, secondSlideIndex, generator);

    Slide firstSlide = currentSlides.at(firstSlideIndex);
    Slide secondSlide = currentSlides.at(secondSlideIndex);

    if (firstSlide.getOrientation() == 'V' && secondSlide.getOrientation() == 'V')
    {
        uniform_int_distribution<int> dis2(0, 1);

        size_t firstPhoto = dis2(generator);
        size_t secondPhoto = dis2(generator);

        Slide
            s1({firstSlide.getPhotos().at(firstPhoto),
                secondSlide.getPhotos().at(secondPhoto)},
               firstSlide.getID()),
            s2({firstSlide.getPhotos().at(firstPhoto ^ 1),    // flip used photo index
                secondSlide.getPhotos().at(secondPhoto ^ 1)}, // flip used photo index
               secondSlide.getID());

        s1.setTags(firstSlide.getPhotos().at(firstPhoto).getTags(),
                   secondSlide.getPhotos().at(secondPhoto).getTags());

        s2.setTags(firstSlide.getPhotos().at(firstPhoto ^ 1).getTags(),    // flip used photo index
                   secondSlide.getPhotos().at(secondPhoto ^ 1).getTags()); // flip used photo index

        s1.setOrientation('V');
        s2.setOrientation('V');

        compareScores(s2, s1, scoreBefore, scoreAfter, firstSlideIndex, secondSlideIndex);

        if (scoreAfter > scoreBefore)
        {
            currentSlides.at(firstSlideIndex) = s1;
            currentSlides.at(secondSlideIndex) = s2;
            nAttempts = 0;
        }
    }
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

// Evaluation Function

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