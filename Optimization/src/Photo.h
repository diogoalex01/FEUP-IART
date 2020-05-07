#ifndef Photo_H
#define Photo_H

#include <iostream>
#include <string>
#include <vector>

using namespace std;

class Photo
{
    int id;
    char orientation; // either horizontal (H) or vertical (V)
    vector<string> tags;
    bool used;
    static int photoID;

public:
    // Constructor
    Photo(char orientation, vector<string> tags);
    // Getters
    int getID();
    char getOrientation();
    vector<string> getTags();
    bool getUsed();
    // Setters
    void setUsed(bool newValue);
};

#endif //Photo_H
