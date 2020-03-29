#include "Photo.h"
#include "Slide.h"
#include "Slideshow.h"
#include "PhotoSelector.h"

vector<Photo> vPhotos;
vector<Photo> hPhotos;

void parseInputFile(string &inputFileName, PhotoSelector &photoSelector)
{
    ifstream inputFile("../input/" + inputFileName);

    if (inputFile.is_open())
    {
        string numberOfPhotos, photo;
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
        printf("[Found %zu vertical photos]\n", vPhotos.size());
        printf("[Found %zu horizontal photos]\n", hPhotos.size());
        inputFile.close();
    }
    else
    {
        cerr << "Input file not found, try again!" << endl;
    }
}

void parsePhoto(string photoLine)
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

    vector<string> tags;

    // Photo tags
    for (int j = 0; j < numberTags; j++)
    {
        index = photoLine.find(' ');
        string tag = photoLine.substr(0, index);
        tags.push_back(tag);
        photoLine.erase(0, index + 1);
    }

    Photo photo(orientation, tags);
    if (orientation == 'V')
        vPhotos.push_back(photo);
    else
        hPhotos.push_back(photo);
}

void writeOutputFile(string &ouputFileName, PhotoSelector &photoSelector)
{
    ofstream outputFile("../output/" + ouputFileName);

    if (outputFile.is_open())
    {
        vector<Slide> slideshow = photoSelector.getCurrentSlides();

        int slideshowSize = slideshow.size();
        outputFile << slideshowSize << endl;

        for (auto slideItr = slideshow.begin(); slideItr != slideshow.end(); slideItr++)
        {
            vector<Photo> photos = (*slideItr).getPhotos();
            for (size_t i = 0; i < photos.size(); i++)
            {
                outputFile << photos.at(i).getID();

                if (i == 0 && photos.size() == 2) // Spaces only if there's more than 1 photo in the slide.
                    outputFile << " ";
            }

            outputFile << endl;
        }

        outputFile.close();
    }
    else
    {
        cout << "Unable to open file!";
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        printf("Usage: ./Slideshow input_file heuristic");
        printf("Missing arguments!\n");
        exit(-2);
    }

    string input_file = argv[1];
    int heuristic = stoi(argv[2]);

    PhotoSelector photoSelector({}, {}, heuristic);
    string inputFileName = input_file;
    // cout << "Enter input file name: ";
    // cin >> inputFileName;

    cout << "Opening " << inputFileName << "!" << endl;
    parseInputFile(inputFileName, photoSelector);
    photoSelector.makeSlides();
    photoSelector.getCurrentScore();
    string ouputFileName = "example.txt";
    writeOutputFile(ouputFileName, photoSelector);

    return 0;
}