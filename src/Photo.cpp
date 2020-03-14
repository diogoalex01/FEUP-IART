#include "Photo.h"

Photo::Photo(int id, char orientation, std::vector<std::string> tags)
{
    this->id = id;
    this->orientation = orientation;
    this->tags = tags;
    used = false;
}

int Photo::getID()
{
    return id;
}

char Photo::getOrientation()
{
    return orientation;
}

std::vector<std::string> Photo::getTags()
{
    return tags;
}

bool Photo::getUsed()
{
    return used;
}

void Photo::setUsed(bool newValue)
{
    used = newValue;
}
