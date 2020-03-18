#include "PhotoSelector.h"

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

vector<Slide> PhotoSelector::getInitialSlides()
{
    return initialSlides;
}

vector<Slide> PhotoSelector::getFinalSlides()
{
    return finalSlides;
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
    unordered_set<string> hTagsSet;
    vector<string> tags;

    for (auto it = hPhotos.begin(); it != hPhotos.end(); it++)
    {
        Slide slide({(*it)});
        hTagsSet.clear();
        tags = (*it).getTags();
        hTagsSet.insert(tags.begin(), tags.end());
        slide.setTags(hTagsSet);
        initialSlides.push_back(slide);
    }

    for (auto it = vPhotos.begin(); it != vPhotos.end(); it++)
    {
        Slide slide({});
        // HCfindVerticalPair((*it), slide);
        if (!(*it).getUsed())
        {
            SAfindVerticalPair((*it), slide);
        }
    }

    for (Slide slide : initialSlides)
    {
        cout << "\n-----------------------------\n"
             << slide.getID() << " has " << slide.getPhotos().size() << " photos and "
             << "has the following tags:" << endl;
        for (Photo i : slide.getPhotos())
        {
            printf("Slide %d has photo %d\n", slide.getID(), i.getID());
        }
        for (string tags : slide.getTags())
        {
            cout << tags << endl;
        }
    }

    // HCmakeSlideshow();
    // SAmakeSlideshow();
}

int PhotoSelector::getTransitionScore(unordered_set<string> currentSlide, unordered_set<string> nextSlide)
{
    int commonTags = 0;   // Number of common tags
    int currentSlideTags; // Number of tags only in the current slide
    int nextSlideTags;    // Number of tags only in the next slide

    // Find the number of common tags in a given transition
    for (auto slideItr = currentSlide.begin(); slideItr != currentSlide.end(); slideItr++)
    {
        if (nextSlide.find((*slideItr)) != nextSlide.end())
        {
            printf("encontrou commun: %s\n", (*slideItr).c_str());
            commonTags++;
        }
    }

    currentSlideTags = currentSlide.size() - commonTags;
    nextSlideTags = nextSlide.size() - commonTags;

    int subResult = min(nextSlideTags, currentSlideTags);
    int finalResult = min(subResult, commonTags);
    printf("Common tags: %d\nS1 tags: %d\n S2 tags: %d\nResult: %d\n", commonTags, currentSlideTags, nextSlideTags, finalResult);

    return finalResult;
}

int PhotoSelector::getFinalScore()
{
    int finalScore = 0;

    if (finalSlides.size() <= 1)
        return 0;

    for (auto slideItr = finalSlides.begin(); slideItr != finalSlides.end() - 1; slideItr++)
    {
        auto nextSlideItr = slideItr + 1;
        finalScore += getTransitionScore(slideItr->getTags(), nextSlideItr->getTags());
    }

    printf("\nFinal Score: %d\n", finalScore);
    return finalScore;
}

void PhotoSelector::HCfindVerticalPair(Photo &photo, Slide &slide)
{
    unordered_set<string> allTags;
    vector<string> tags;

    for (auto it = vPhotos.begin(); it != vPhotos.end(); ++it)
    {
        if (it->getID() != photo.getID() && !it->getUsed())
        {
            tags = (*it).getTags();
            allTags.insert(tags.begin(), tags.end());
            tags = photo.getTags();
            allTags.insert(tags.begin(), tags.end());

            // If number of distinct tags is larger than the number of
            // tags in the current photo, then there must be at least one new tag.
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

void PhotoSelector::HCmakeSlideshow()
{
    // Inital state is the first slide
    finalSlides.push_back(initialSlides.at(0));
    initialSlides.at(0).setUsed(true);
    unordered_set<string> currentTags = initialSlides.at(0).getTags();

    for (auto slideItr = initialSlides.begin() + 1; slideItr != initialSlides.end(); slideItr++)
    {
        if (!slideItr->getUsed() && getTransitionScore(currentTags, slideItr->getTags()) > 0)
        {
            slideItr->setUsed(true);
            finalSlides.push_back((*slideItr));
            currentTags = slideItr->getTags();
        }
    }
}

void PhotoSelector::SAfindVerticalPair(Photo &photo, Slide &slide)
{
    unordered_set<string> allTags;
    vector<string> tags;
    size_t bestMatch;
    size_t maxDistinct = 0;
    bool foundMatch = false;

    for (auto it = vPhotos.begin(); it != vPhotos.end(); ++it)
    {
        if (it->getID() != photo.getID() && !it->getUsed())
        {
            tags = (*it).getTags();
            allTags.insert(tags.begin(), tags.end());
            tags = photo.getTags();
            allTags.insert(tags.begin(), tags.end());

            // If number of distinct tags is larger than the number of
            // tags in the current photo, then there must be at least one new tag.

            if (allTags.size() - photo.getTags().size() > maxDistinct)
            {
                foundMatch = true;
                maxDistinct = allTags.size() - photo.getTags().size();
                bestMatch = it - vPhotos.begin();
            }
        }
    }
    
    if (foundMatch)
    {
        slide.setPhotos({photo, vPhotos.at(bestMatch)});
        slide.setTags(allTags);
        photo.setUsed(true);
        vPhotos.at(bestMatch).setUsed(true);
        initialSlides.push_back(slide);
    }
}
