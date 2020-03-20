#include "Slide.h"

int Slide::slideID = 0;

Slide::Slide(vector<Photo> photos)
{
    id = slideID;
    slideID++;
    this->photos = photos;
    used = false;
}

int Slide::getID()
{
    return id;
}

vector<Photo> Slide::getPhotos()
{
    return photos;
}

unordered_set<string> Slide::getTags()
{
    return tags;
}

bool Slide::getUsed()
{
    return used;
}

void Slide::setID(int newId)
{
    this->id = newId;
}

void Slide::setPhotos(vector<Photo> photos)
{
    this->photos = photos;
}

void Slide::setTags(unordered_set<string> tags)
{
    this->tags = tags;
}

void Slide::setUsed(bool newValue)
{
    used = newValue;
}
