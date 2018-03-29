using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace UI
{
    public class CaseClickedEventArgs
    {
        private readonly Case _case;

        public CaseClickedEventArgs(Case c)
        {
            _case = c;
        }

        public Case ClickedCase {
          get { return _case; }
        }
    }
}
