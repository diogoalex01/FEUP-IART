#ifndef PhotoSelector_H
#define PhotoSelector_H

#include <queue>
#include <algorithm>
#include <time.h>

#include "Photo.h"
#include "Slide.h"

class PhotoSelector
{
    vector<Photo> vPhotos;
    vector<Photo> hPhotos;
    vector<Slide> initialSlides;
    vector<Slide> finalSlides;

public:
    // Constructor
    PhotoSelector(vector<Photo> vp, vector<Photo> hp);
    // Getters
    vector<Photo> getVertical();
    vector<Photo> getHorizontal();
    vector<Slide> getInitialSlides();
    vector<Slide> getFinalSlides();
    // Setters
    void setVertical(vector<Photo> vp);
    void setHorizontal(vector<Photo> hp);
    // Main Functions
    void makeSlides();
    int getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide);
    int getFinalScore();
    // Heuristics
    void HCfindVerticalPair(Photo &photo, Slide &slide);
    void HCmakeSlideshow();
    void SAfindVerticalPair(Photo &photo, Slide &slide);
    void SAmakeSlideshow();
};

#endif //PhotoSelector_H
