#include "Photo.h"

int Photo::photoID = 0;

Photo::Photo(char orientation, vector<string> tags)
{
    id = photoID;
    photoID++;
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

vector<string> Photo::getTags()
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
