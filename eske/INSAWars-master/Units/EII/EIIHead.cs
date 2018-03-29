using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.EII
{
    [Serializable]
    class EIIHead : Head
    {
        public EIIHead(Case location, Player player)
            : base(location, player)
        {
        }
    }
}
