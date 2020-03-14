#ifndef PhotoSelector_H
#define PhotoSelector_H

#include <queue>
#include <algorithm>

#include "Photo.h"
#include "Slide.h"

using namespace std;

class PhotoSelector
{
    vector<Photo> vPhotos;
    vector<Photo> hPhotos;
    vector<Slide> initialSlides;
    vector<Slide> result;

public:
    // Constructor
    PhotoSelector();
    PhotoSelector(vector<Photo> vp, vector<Photo> hp);
    // Getters
    vector<Photo> getVertical();
    vector<Photo> getHorizontal();
    // Setters
    void setVertical(vector<Photo> vp);
    void setHorizontal(vector<Photo> hp);

    void makeSlides();
    void HCfindVerticalPair(Photo &photo, Slide &slide);
};

#endif //PhotoSelector_H
