#ifndef Slide_H
#define Slide_H

#include "Photo.h"
#include <unordered_set>

class Slide
{
    int id;
    vector<Photo> photos;
    unordered_set<string> tags; // unique tags
    bool used;                  // is some slide's match
    static int slideID;
    char orientation;

public:
    // Constructor
    Slide(vector<Photo> photos);
    Slide(vector<Photo> photos, int slideID);
    // Getters
    int getID();
    vector<Photo> getPhotos();
    unordered_set<string> getTags();
    bool getUsed();
    bool getHasMatch();
    char getOrientation();
    // Setters
    void setID(int newId);
    void setPhotos(vector<Photo> photos);
    void setTags(vector<string> newTags);
    void setTags(vector<string> tags1, vector<string> tags2);
    void setUsed(bool newValue);
    void setHasMatch(bool newValue);
    void setOrientation(char orientation);
};

#endif //Slide_H
