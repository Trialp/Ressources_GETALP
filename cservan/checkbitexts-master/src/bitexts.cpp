#include "bitexts.h"


using namespace std;


bitexts::bitexts()
{
    m_filename_src="";
    m_filename_trg="";
}
bitexts::bitexts(string filename_src, string filename_trg)
{
    m_filename_src=filename_src;
    m_filename_trg=filename_trg;
    load();
}
string bitexts::getSrcFilename()
{
    return m_filename_src;
}
string bitexts::getTrgFilename()
{
    return m_filename_trg;
}
void bitexts::load()
{
    string l_line;
//     if (m_filename_src.find(".gz") > -1)
//     {
//         igzstream l_srcFile;
//         igzstream l_trgFile;
//         l_srcFile.open(m_filename_src);
//         l_trgFile.open(m_filename_trg);
//         if(l_srcFile.is_open())
//         {
//             while (getline(l_srcFile,l_line))
//             {
//                 src_data.push_back(l_line);
//             }
//         }
//         if(l_trgFile.is_open())
//         {
//             while (getline(l_trgFile,l_line))
//             {
//                 trg_data.push_back(l_line);
//             }
//         }
//     }
//     else
//     {
        ifstream l_srcFile;
        ifstream l_trgFile;
        l_srcFile.open(m_filename_src);
        l_trgFile.open(m_filename_trg);
        if(l_srcFile.is_open())
        {
            while (getline(l_srcFile,l_line))
            {
                m_src_data.push_back(l_line);
            }
        }
        l_srcFile.close();
        if(l_trgFile.is_open())
        {
            while (getline(l_trgFile,l_line))
            {
                m_trg_data.push_back(l_line);
            }
        }
        l_trgFile.close();
        if (m_src_data.size() != m_trg_data.size())
        {
            cerr << "ERROR bitexts::load: source and target data sizes are different! "<< (int)m_src_data.size()<< "\t"<< (int)m_trg_data.size()<<endl;
            exit(1);
        }
//     }
}
void bitexts::score()
{
    int l_cpt=0;
    while (l_cpt < (int)m_src_data.size())
    {
        m_scores.push_back(m_bimodel->similaritySentence(m_src_data.at(l_cpt),m_trg_data.at(l_cpt)));
        l_cpt++;
        // if (l_cpt % 10 == 0) { cerr << '.';  }
        // if (l_cpt % 100 == 0) { cerr << " [" << l_cpt << "]\n" << flush; }
    }
}
void bitexts::save(string filename)
{
    int l_cpt=0;
    ofstream ofile;
    ofile.open(filename);
    while (l_cpt < (int)m_src_data.size())
    {
        ofile << m_src_data.at(l_cpt)<< "\t" << m_trg_data.at(l_cpt) << "\t" << m_scores.at(l_cpt) <<endl;
        l_cpt++;
        // if (l_cpt % 10 == 0) { cerr << '.';  }
        // if (l_cpt % 100 == 0) { cerr << " [" << l_cpt << "]\n" << flush; }
    }
    ofile.close();

}

void bitexts::loadModel(string filename)
{
    m_bimodel=new BilingualModel(new BilingualConfig());
    m_bimodel->load(filename);
}


