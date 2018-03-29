#include <fstream>
#include <iostream>
#include <string.h>
#include <vector>
#include "bitexts.h"
#include "tools.h"





void usage();

void readCommandLineArguments ( unsigned int argc, char *argv[] , param & p)
{
    p.bimodelFile="";
    p.debugMode=0;
    p.outputFileName="";
    p.srcFile="";
    p.trgFile="";

    string s ( "" );
    string infos ("");
    for ( unsigned int i = 1; i < argc; i++ )
    {
	s= argv[i];
        // Command line string
        if (i < argc-1)
        {
            infos = argv[i+1];
        }
        // Reading
        if ( s.compare ( "-s" ) == 0 )
        {
            p.srcFile = infos;
        }
        else if ( s.compare ( "-t" ) == 0 )
        {
            p.trgFile = infos;
        }
        else if ( s.compare ( "-m" ) == 0 )
        {
            p.bimodelFile = infos;
        }
        else if ( s.compare ( "-o" ) == 0 )
        {
            p.outputFileName = infos;
        }
        else if ( s.compare ( "--debugMode" ) == 0 )
        {
            p.debugMode = true;
        }
        else if ( s.compare ( "--help" ) == 0 )
        {
            usage();
        }
        else if ( s.find ( "-" ) == 0 )
        {
	    cerr << "ERROR : tercpp: unknown option :" << s <<endl;
            usage();
        }
    }
}

void usage()
{
    cerr<<"Usage : "<<endl<<"\tscorebitexts -s source_file[.gz] -t target_file[.gz] -m model_file -o output_file"<<endl;
    exit(0);
}


int main ( int argc, char *argv[] )
{
    param myParams;
    readCommandLineArguments ( argc,argv, myParams);
//     cerr << Tools::printParams(myParams) << endl; 
//     exit(0);
    if ((int)myParams.srcFile.size()==0) 
    {
        cerr << "ERROR : main : source file is not set !" << endl;
        usage();
//         return 0;
    }
    if ((int)myParams.trgFile.size()==0)
    {
        cerr << "ERROR : main : target file is not set !" << endl;
        usage();
//         return 0;
    }
    if ((int)myParams.bimodelFile.size()==0)
    {
        cerr << "ERROR : main : bimodel file is not set !" << endl;
        usage();
//         return 0;
    }
    bitexts l_bitext(myParams.srcFile,myParams.trgFile);
    l_bitext.loadModel(myParams.bimodelFile);
    l_bitext.score();
    l_bitext.save(myParams.outputFileName);
    return EXIT_SUCCESS;
}
