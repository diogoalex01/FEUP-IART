#ifndef PhotoSelector_H
#define PhotoSelector_H

#include <queue>
#include <algorithm>
#include <random>
#include <math.h>
#include <chrono>

#include "Photo.h"
#include "Slide.h"

class PhotoSelector
{
    int currentScore = 0;
    long stateChanges = 0;
    long nAttempts = 0;
    long maxAttempts;

    double T = 1;         // initial and final temperature
    double Tmin = 0.0001; // temperature at which iteration terminates
    double alpha = 0.9;   // decrease in temperature
    long numIterations;   // number of iterations of annealing before decreasing temperature;

    vector<Photo> vPhotos;
    vector<Photo> hPhotos;
    vector<Slide> currentSlides;
    size_t lastSlideIndex;

public:
    // Constructor
    PhotoSelector(vector<Photo> vp, vector<Photo> hp);
    // Getters
    int getCurrentScore();
    vector<Photo> getVertical();
    vector<Photo> getHorizontal();
    vector<Slide> getCurrentSlides();
    // Setters
    void setVertical(vector<Photo> vp);
    void setHorizontal(vector<Photo> hp);
    // Main Functions
    void makeSlides();
    void findVerticalPair(Photo &photo, Slide &slide);
    void getRandomIndexes(size_t &firstSlideIndex, size_t &secondSlideIndex, mt19937 generator);
    int getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide);
    void compareScores(Slide &firstSlide, Slide &secondSlide, int &scoreBefore, int &scoreAfter, size_t firstSlideIndex, size_t secondSlideIndex);
    // Heuristics
    void hillClimbing();
    void simulatedAnnealing();
    void tabuSearch();
    // Neighbouring Functions
    void neighbouringHC(); //neighbour function for hill climbing
    void neighbouringSA(); //neighbour function for simulated annealing
    void neighbouringTS();//neighbour function for tabu search
    // Evaluation Function
    void evaluateScore();
};

#endif //PhotoSelector_H
