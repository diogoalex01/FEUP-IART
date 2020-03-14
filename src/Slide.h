#ifndef Slide_H
#define Slide_H

#include "Photo.h"
#include <unordered_set>

class Slide
{
    int id;
    std::vector<Photo> photos;
    std::unordered_set<std::string> tags; // unique tags
    bool used;

public:
    // Constructor
    Slide(int id, std::vector<Photo> photos);
    // Getters
    int getID();
    std::vector<Photo> getPhotos();
    std::unordered_set<std::string> getTags();
    // Setters
    void setID(int newId);
    void setPhotos(std::vector<Photo> photos);
    void setTags(std::vector<std::string> tags);
};

#endif //Slide_H
