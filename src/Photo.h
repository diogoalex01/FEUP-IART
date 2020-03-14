#ifndef Photo_H
#define Photo_H

#include <iostream>
#include <string>
#include <vector>

class Photo
{
    int id;
    char orientation; // either horizontal (H) or vertical (V)
    std::vector<std::string> tags;
    bool used;

public:
    // Constructor
    Photo(int id, char orientation, std::vector<std::string> tags);
    // Getters
    int getID();
    char getOrientation();
    std::vector<std::string> getTags();
    bool getUsed();
    // Setters
    void setUsed(bool newValue);
};

#endif //Photo_H
