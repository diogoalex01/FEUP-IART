#include "PhotoSelector.h"
#include <random>
#include <chrono>
#include <math.h>
long attempts = 0;

PhotoSelector::PhotoSelector(vector<Photo> vPhotos, vector<Photo> hPhotos)
{
    this->nAttempts = 0;
    this->currentScore = 0;
    this->vPhotos = vPhotos;
    this->hPhotos = hPhotos;

    cout << "made photo selector" << this->vPhotos.size() << " " << hPhotos.size() << endl;
}

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

vector<Slide> PhotoSelector::getFinalSlides()
{
    return finalSlides;
}

void PhotoSelector::setVertical(vector<Photo> vPhotos)
{
    this->vPhotos = vPhotos;
}

void PhotoSelector::setHorizontal(vector<Photo> hPhotos)
{
    this->hPhotos = hPhotos;
}

void PhotoSelector::makeSlides()
{
    unordered_set<string> hTagsSet;
    vector<string> tags;

    for (auto it = hPhotos.begin(); it != hPhotos.end(); it++)
    {
        Slide slide({(*it)});
        tags = (*it).getTags();
        slide.setTags(tags);
        currentSlides.push_back(slide);
    }

    for (auto it = vPhotos.begin(); it != vPhotos.end(); it++)
    {
        Slide slide({});

        if (!(*it).getUsed())
        {
            findVerticalPair((*it), slide);
        }
    }

    // for (Slide slide : currentSlides)
    // {
    //     // cout << "\n-----------------------------\n"
    //     //      << slide.getID() << " has " << slide.getPhotos().size() << " photos and "
    //     //      << "has the following tags:" << endl;
    //     for (Photo i : slide.getPhotos())
    //     {
    //         printf("Slide %d has photo %d\n", slide.getID(), i.getID());
    //     }
    //     for (string tags : slide.getTags())
    //     {
    //         cout << tags << endl;
    //     }
    // }

    printf("Finished making %zu slides!\n", getCurrentSlides().size());
    updateCurrentScore();
    printf("Current score: %d\n", getCurrentScore());

    maxAttempts = 10000;//pow(currentSlides.size(), 2);
    printf("Max Attempts: %ld\n", maxAttempts);
    while (nAttempts < maxAttempts)
    {
        nAttempts++;
        attempts++;
        makeChange();
    }

    printf("Current score: %d\n", getCurrentScore());
    printf("Attempts: %ld\n", attempts);
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
            // printf("encontrou commun: %s\n", (*slideItr).c_str());
            commonTags++;
        }
    }

    currentSlideTags = currentSlide.size() - commonTags;
    nextSlideTags = nextSlide.size() - commonTags;

    int subResult = min(nextSlideTags, currentSlideTags);
    int finalResult = min(subResult, commonTags);
    // printf("Common tags: %d\n  S1 tags: %d\n  S2 tags: %d\n  Result: %d\n", commonTags, currentSlideTags, nextSlideTags, finalResult);
    return finalResult;
}

void PhotoSelector::updateCurrentScore()
{
    int transitionScore;
    if (currentSlides.size() <= 1)
        currentScore = 0;

    for (auto slideItr = currentSlides.begin(); slideItr != currentSlides.end() - 1; slideItr++)
    {
        auto nextSlideItr = slideItr + 1;
        transitionScore = getTransitionScore(slideItr->getTags(), nextSlideItr->getTags());
        //printf("Transition Score: %d\n", transitionScore);
        currentScore += transitionScore;
    }
    //printf("\nCurrent Score: %d\n", currentScore);
}

void PhotoSelector::updateCurrentScore(int change)
{
    this->currentScore += change;

    //printf("\nCurrent Score: %d\n", currentScore);
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

void PhotoSelector::makeChange()
{
    int randomIndex = 1; //(rand() % 1);
    // cout << "Permutation option: " << randomIndex << endl;

    if (randomIndex)
    {
        // Permutation between slides
        permuteSlides();
    }
    else
    {
        // Permutation between photos
        permutePhotos();
    }
}

void PhotoSelector::permuteSlides()
{
    random_device device;
    mt19937 generator(device());
    uniform_int_distribution<int> dis(0,749);
     size_t firstSlideIndex  = dis(generator);
     size_t secondSlideIndex  = dis(generator);

    while (firstSlideIndex == secondSlideIndex)
    {
        firstSlideIndex = dis(generator);
        secondSlideIndex =  dis(generator);
    }
    //printf("Selected slides %zu and %zu\n", firstSlideIndex, secondSlideIndex);

    int scoreBefore = 0, scoreAfter = 0;
    // secondSlideIndex always after firstSlideIndex
    if (firstSlideIndex > secondSlideIndex)
    {
        int temp = firstSlideIndex;
        firstSlideIndex = secondSlideIndex;
        secondSlideIndex = temp;
    }

    // A (B) C .... G (H) I
    // A -> B + B -> C + G -> H + H -> I
    // Compare with
    // A -> H + H -> C + G -> B + B -> I

    if (firstSlideIndex != 0)
    {
        scoreBefore += getTransitionScore(currentSlides.at(firstSlideIndex - 1).getTags(), currentSlides.at(firstSlideIndex).getTags());
        scoreAfter += getTransitionScore(currentSlides.at(firstSlideIndex - 1).getTags(), currentSlides.at(secondSlideIndex).getTags());
    }
    else if (secondSlideIndex != getCurrentSlides().size() - 1)
    {
        scoreBefore += getTransitionScore(currentSlides.at(secondSlideIndex).getTags(), currentSlides.at(secondSlideIndex + 1).getTags());
        scoreAfter += getTransitionScore(currentSlides.at(firstSlideIndex).getTags(), currentSlides.at(secondSlideIndex + 1).getTags());
    }

    scoreBefore += getTransitionScore(currentSlides.at(firstSlideIndex).getTags(), currentSlides.at(firstSlideIndex + 1).getTags());
    scoreBefore += getTransitionScore(currentSlides.at(secondSlideIndex - 1).getTags(), currentSlides.at(secondSlideIndex).getTags());
    scoreAfter += getTransitionScore(currentSlides.at(secondSlideIndex - 1).getTags(), currentSlides.at(firstSlideIndex).getTags());
    scoreAfter += getTransitionScore(currentSlides.at(secondSlideIndex).getTags(), currentSlides.at(firstSlideIndex + 1).getTags());
    printf("Attempts number %ld\n", attempts);
    //printf("total score before swap: %d\n", getCurrentScore());
    //printf("transitions score before: %d\ntransitions score after: %d\n", scoreBefore, scoreAfter);
    if (scoreAfter > scoreBefore)
    {
        //printf("[swaping %zu and %zu]\n", firstSlideIndex, secondSlideIndex);
        //printf("total score before swap: %d\n", getCurrentScore());
        vector<Slide>::iterator firstSlideIter = currentSlides.begin() + firstSlideIndex;
        vector<Slide>::iterator secondSlideIter = currentSlides.begin() + secondSlideIndex;
        iter_swap(firstSlideIter, secondSlideIter);
        updateCurrentScore(scoreAfter - scoreBefore);
        nAttempts = 0;
    }
}
