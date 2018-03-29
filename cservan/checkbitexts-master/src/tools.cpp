

vector<string> stringToVector ( string s, string tok )
{
    vector<string> to_return;
    string to_push ( "" );
    bool pushed = false;
    string::iterator sIt;
    for ( sIt = s.begin(); sIt < s.end(); sIt++ )
    {
        pushed = false;
        for ( string::iterator sTok = tok.begin(); sTok < tok.end(); sTok++ )
        {
            if ( ( *sIt ) == ( *sTok ) )
            {
                to_return.push_back ( to_push );
                to_push = "";
                pushed = true;
            }
        }
        if ( !pushed )
        {
            to_push.push_back ( ( *sIt ) );
        }
    }
    to_return.push_back ( to_push );
    return to_return;
}
