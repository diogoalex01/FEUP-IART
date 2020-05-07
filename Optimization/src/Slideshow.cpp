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
        printf("   [Found %zu vertical photos  ]\n", vPhotos.size());
        printf("   [Found %zu horizontal photos]\n", hPhotos.size());
        inputFile.close();
    }
    else
    {
        cerr << "Input file not found, try again!" << endl
             << endl;
        exit(-1);
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
        exit(-1);
    }
}

void hillClimbingParameters(PhotoSelector &photoSelector)
{
    int maxAttempts;
    cout << "Maximum number of unsuccessful attempts? ";
    cin >> maxAttempts;
    cout << endl;

    // start counting elapsed time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    cout << "---------------------------------------------------" << endl
         << endl;

    photoSelector.setMaxAttempts(maxAttempts);
    photoSelector.hillClimbing();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "\n\nElapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
}

void simulatedAnnealingParameters(PhotoSelector &photoSelector)
{
    double T;
    double Tmin;
    double alpha;
    long numIterations;

    cout << "Initial temperature: ";
    cin >> T;
    cout << "Minimum temperature: ";
    cin >> Tmin;
    cout << "Temperature drop: ";
    cin >> alpha;
    cout << "Number of iterations per temperature drop: ";
    cin >> numIterations;
    cout << endl;

    // start counting elapsed time
    photoSelector.setTemperature(T);
    photoSelector.setTmin(Tmin);
    photoSelector.setAlpha(alpha);
    photoSelector.setNumIterations(numIterations);

    // start counting elapsed time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    cout << "---------------------------------------------------" << endl
         << endl;

    photoSelector.simulatedAnnealing();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "\n\nElapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
}

void tabuSearchParameters(PhotoSelector &photoSelector)
{
    int maxAttempts, fifoSize;
    cout << "Maximum number of unsuccessful attempts? ";
    cin >> maxAttempts;
    cout << "Tabu list size: ";
    cin >> fifoSize;
    cout << endl;

    // start counting elapsed time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    cout << "---------------------------------------------------" << endl
         << endl;

    photoSelector.setMaxAttempts(maxAttempts);
    photoSelector.setTabuListSize(fifoSize);
    photoSelector.tabuSearch();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "\n\nElapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
}

void geneticAlgorithmParameters(PhotoSelector &photoSelector)
{
    int populationSize;
    int maxGenerations;
    int maxAttempts;

    cout << "Population size: ";
    cin >> populationSize;
    cout << "Number of generations: ";
    cin >> maxGenerations;
    cout << "Maximum number of consecutive generations without improvement: ";
    cin >> maxAttempts;
    cout << endl;

    photoSelector.setMaxAttempts(maxAttempts);

    // start counting elapsed time
    chrono::steady_clock::time_point begin = chrono::steady_clock::now();

    cout << "---------------------------------------------------" << endl
         << endl;

    photoSelector.setPopulationSize(populationSize);
    photoSelector.setMaxGenerations(maxGenerations);
    photoSelector.geneticAlgorithm();

    chrono::steady_clock::time_point end = chrono::steady_clock::now();
    cout << "\n\nElapsed Time: " << chrono::duration_cast<chrono::milliseconds>(end - begin).count() << " [ms]" << endl;
}

void chooseParameters(PhotoSelector &photoSelector, int heuristic)
{
    photoSelector.setHeuristic(heuristic);

    switch (heuristic)
    {
    case 1:
        hillClimbingParameters(photoSelector);
        break;

    case 2:
        simulatedAnnealingParameters(photoSelector);
        break;

    case 3:
        tabuSearchParameters(photoSelector);
        break;

    case 4:
        geneticAlgorithmParameters(photoSelector);
        break;
    }

    photoSelector.evaluateScore();
    printf("\nInitial score: %d\nFinal score: %d\n", photoSelector.getInitialScore(), photoSelector.getCurrentScore());
    photoSelector.evaluateScore();
    cout << "---------------------------------------------------" << endl
         << endl;
}

int main()
{
    int heuristic;
    string inputFileName;

    cout << endl
         << "----------------------------" << endl
         << "------- Optimization -------" << endl
         << "----------------------------" << endl;
    cout << endl;
    cout << "File name: ";
    cin >> inputFileName;

    cout << "\n- Opening " << inputFileName << "!" << endl;
    PhotoSelector photoSelector({}, {});
    parseInputFile(inputFileName, photoSelector);
    photoSelector.makeSlides();
    photoSelector.getCurrentScore();

    cout << endl;
    cout << "Which heuristic do you want to try?" << endl
         << endl;
    cout << "1: Hill Climbing" << endl;
    cout << "2: Simulated Annealing " << endl;
    cout << "3: Tabu Search" << endl;
    cout << "4: Genetic Algorithm" << endl
         << endl;
    cout << "Heuristic: ";
    cin >> heuristic;
    cout << endl;

    chooseParameters(photoSelector, heuristic);

    string ouputFileName = "example.txt";
    writeOutputFile(ouputFileName, photoSelector);

    return 0;
}