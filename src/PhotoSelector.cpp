#include "PhotoSelector.h"

PhotoSelector::PhotoSelector()
{
    cout << "made photo selector" << endl;
}

PhotoSelector::PhotoSelector(vector<Photo> vPhotos, vector<Photo> hPhotos)
{
    this->vPhotos = vPhotos;
    this->hPhotos = hPhotos;

    cout << "made photo selector" << this->vPhotos.size() << " " << hPhotos.size() << endl;
}

vector<Photo> PhotoSelector::getVertical()
{
    return vPhotos;
}

vector<Photo> PhotoSelector::getHorizontal()
{
    return hPhotos;
}

void PhotoSelector::setVertical(vector<Photo> vPhotos)
{
    this->vPhotos = vPhotos;
}

void PhotoSelector::setHorizontal(vector<Photo> hPhotos)
{
    this->hPhotos = hPhotos;
}

void PhotoSelector::makeSlides()
{
    Slide slide(-1, {});
    int slideID = 0;

    for (auto it = hPhotos.begin(); it != hPhotos.end(); it++)
    {
        slide.setID(slideID);
        slide.setPhotos({(*it)});
        slide.setTags(it->getTags());

        initialSlides.push_back(slide);
        slideID++;
        cout << "Added hor to slide " << slideID - 1 << endl;
    }

    for (auto it = vPhotos.begin(); it != vPhotos.end(); it++)
    {
        slide.setID(slideID);
        cout << "Vert number " << it->getID() << endl;
        HCfindVerticalPair((*it), slide);
        slideID++;
    }

    for (Slide slide : initialSlides)
    {
        cout << "\n-----------------------------\n"
             << slide.getID() << " has " << slide.getPhotos().size() << " photos and "
             << "has the following tags:" << endl;
        for (string tags : slide.getTags())
        {
            cout << tags << endl;
        }
    }
}

void PhotoSelector::HCfindVerticalPair(Photo &photo, Slide &slide)
{
    // printf("no hc há %zu photos verticais\n", vPhotos.size());
    vector<string> allTags;

    for (auto it = vPhotos.begin(); it != vPhotos.end(); ++it)
    {
        // printf("ID da photo %d \t ID do it %d\n", photo.getID(), it->getID());
        if (it->getID() != photo.getID() && !it->getUsed())
        {
            set_union(photo.getTags().begin(), photo.getTags().end(), it->getTags().begin(), it->getTags().end(), back_inserter(allTags));

            cout << "\nPhoto Tags ------\n";
            for (string tag : photo.getTags())
            {
                cout << tag << endl;
            }

            cout << "\nItr Tags ------\n";
            for (string tag : it->getTags())
            {
                cout << tag << endl;
            }

            cout << "\nAll Tags ------\n";
            for (string tag : allTags)
            {
                cout << tag << endl;
            }
            cout << "\n------\n";

            if (allTags.size() > photo.getTags().size())
            {
                slide.setPhotos({photo, *it});
                slide.setTags(allTags);
                photo.setUsed(true);
                it->setUsed(true);
                initialSlides.push_back(slide);
                break;
            }
        }
    }
}