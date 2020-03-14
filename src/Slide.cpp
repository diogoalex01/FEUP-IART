#include "Slide.h"

Slide::Slide(int id, std::vector<Photo> photos)
{
    this->id = id;
    this->photos = photos;
    used = false;
}

int Slide::getID()
{
    return id;
}

std::vector<Photo> Slide::getPhotos()
{
    return photos;
}

std::unordered_set<std::string> Slide::getTags()
{
    return tags;
}

void Slide::setID(int newId)
{
    this->id = newId;
}

void Slide::setPhotos(std::vector<Photo> photos)
{
    this->photos = photos;
}

void Slide::setTags(std::vector<std::string> tags)
{
    this->tags.clear();
    this->tags.insert(tags.begin(), tags.end());
}