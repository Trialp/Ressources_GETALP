#ifndef __BITEXT_H__
#define __BITEXT_H__
#include <string>
//#include <ext/hash_map>
#include <iostream>
#include <monolingual.hpp>
#include <bilingual.hpp>
#include <string.h>
// #include <gzstream.h>


using namespace std;

    class bitexts
    {
        private:
            string m_filename_src;
            string m_filename_trg;
            BilingualModel * m_bimodel;
            vector<string> m_src_data;
            vector<string> m_trg_data;
            vector<float> m_scores;

        public:
            bitexts(string filename_src, string filename_trg);
            bitexts();
            void load(string filename_src, string filename_trg);
            void load();
            void score();
            void save(string filename);
            void loadModel(string filename);
            void setSrcFilename(string filename);
            void setTrgFilename(string filename);
            string getSrcFilename();
            string getTrgFilename();
    };


#endif