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

void Slide::setTags(vector<string> newTags)
{
    this->tags.clear();
    this->tags.insert(newTags.begin(), newTags.end());
}

void Slide::setTags(vector<string> tags1, vector<string> tags2)
{
    this->tags.clear();
    this->tags.insert(tags1.begin(), tags1.end());
    this->tags.insert(tags2.begin(), tags2.end());
}

void Slide::setUsed(bool newValue)
{
    used = newValue;
}
