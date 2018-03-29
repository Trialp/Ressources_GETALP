#ifndef __TOOLS_H__
#define __TOOLS_H__
#include <string.h>
struct param
{
    bool debugMode;
    string srcFile;     
    string trgFile;     
    string bimodelFile;
    string outputFileName;
};

vector<string> stringToVector ( string s, string tok );

#endif
