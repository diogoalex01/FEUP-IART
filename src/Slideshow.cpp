#include "Photo.h"
#include "Slide.h"
#include "Slideshow.h"
#include "PhotoSelector.h"

std::vector<Photo> vPhotos;
std::vector<Photo> hPhotos;

int photoID = 0;

void parseInputFile(std::string &inputFileName, PhotoSelector &photoSelector)
{
    std::ifstream inputFile("../input/" + inputFileName);

    if (inputFile.is_open())
    {
        std::string numberOfPhotos, photo;
        getline(inputFile, numberOfPhotos);

        while (!inputFile.eof())
        {
            getline(inputFile, photo);

            if (photo.length() == 0)
                continue;

            parsePhoto(photo);
        }

        photoSelector.setVertical(vPhotos);
        photoSelector.setHorizontal(hPhotos);
        printf("\n----\nthere are %zu vertical photos\nand there are %zu horizontal photos\n", photoSelector.getVertical().size(), photoSelector.getHorizontal().size());
        inputFile.close();
    }
    else
    {
        std::cerr << "Input file not found, try again!" << std::endl;
    }
}

void parsePhoto(std::string photoLine)
{
    int index;

    // Photo orientation
    index = photoLine.find(' ');
    char orientation = photoLine.substr(0, index)[0];
    photoLine.erase(0, index + 1);

    // Number of tags
    index = photoLine.find(' ');
    int numberTags = stoi(photoLine.substr(0, index));
    photoLine.erase(0, index + 1);

    std::vector<std::string> tags;

    // Photo tags
    for (int j = 0; j < numberTags; j++)
    {
        index = photoLine.find(' ');
        std::string tag = photoLine.substr(0, index);
        tags.push_back(tag);
        photoLine.erase(0, index + 1);
    }

    Photo photo(photoID, orientation, tags);
    photoID++;

    std::cout << photoID << std::endl;
    std::cout << orientation << std::endl;
    for (unsigned int k = 0; k < tags.size(); k++)
    {
        std::cout << tags.at(k) << std::endl;
    }

    if (orientation == 'V')
    {
        vPhotos.push_back(photo);
    }
    else
    {
        hPhotos.push_back(photo);
    }
}

void writeOutputFile(std::string &ouputFileName)
{
    std::ofstream outputFile("../output/" + ouputFileName);

    if (outputFile.is_open())
    {
        std::vector<Slide> slideshow;

        // To test
        Photo p0(0, 'H', {});
        Photo p1(1, 'H', {});
        Photo p2(2, 'V', {});
        Photo p3(3, 'V', {});

        Slide s0(0, {p0});
        Slide s1(0, {p3});
        Slide s2(0, {p1, p2});

        slideshow.push_back(s0);
        slideshow.push_back(s1);
        slideshow.push_back(s2);
        //

        int slideshowSize = slideshow.size();
        outputFile << slideshowSize << std::endl;

        for (auto slideItr = slideshow.begin(); slideItr != slideshow.end(); slideItr++)
        {
            for (auto photoItr = (*slideItr).getPhotos().begin(); photoItr != (*slideItr).getPhotos().end(); photoItr++)
            {
                outputFile << (*photoItr).getID();

                if (photoItr != (*slideItr).getPhotos().end() - 1) // Spaces only if there's more than 1 photo in the slide.
                    outputFile << " ";
            }

            outputFile << std::endl;
        }

        outputFile.close();
    }
    else
    {
        std::cout << "Unable to open file!";
    }
}

int main()
{
    PhotoSelector photoSelector;
    std::string inputFileName = "a_example.txt";
    // std::cout << "Enter input file name: ";
    // std::cin >> inputFileName;

    std::cout << "Opening " << inputFileName << "!" << std::endl;
    parseInputFile(inputFileName, photoSelector);
    photoSelector.makeSlides();
    std::string ouputFileName = "example.txt";
    writeOutputFile(ouputFileName);

    return 0;
}