#ifndef PhotoSelector_H
#define PhotoSelector_H

#include <queue>
#include <algorithm>
#include <time.h>
#include <tgmath.h>

#include "Photo.h"
#include "Slide.h"

class PhotoSelector
{
    int currentScore;
    long nAttempts;
    long maxAttempts;
    vector<Photo> vPhotos;
    vector<Photo> hPhotos;
    vector<Slide> currentSlides;
    vector<Slide> finalSlides;

public:
    // Constructor
    PhotoSelector(vector<Photo> vp, vector<Photo> hp);
    // Getters
    int getCurrentScore();
    vector<Photo> getVertical();
    vector<Photo> getHorizontal();
    vector<Slide> getCurrentSlides();
    vector<Slide> getFinalSlides();
    // Setters
    void setMaxAttemps();
    void setVertical(vector<Photo> vp);
    void setHorizontal(vector<Photo> hp);
    // Main Functions
    void updateCurrentScore();
    void updateCurrentScore(int change);
    void makeSlides();
    int getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide);
    // Heuristics
    void findVerticalPair(Photo &photo, Slide &slide);
    void makeChange();
    //
    void permuteSlides();
    void permutePhotos();
    // void SAfindVerticalPair(Photo &photo, Slide &slide);
    // void SAmakeSlideshow();
};

#endif //PhotoSelector_H
