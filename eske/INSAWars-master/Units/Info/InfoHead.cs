using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.Info
{
    [Serializable]
    class InfoHead : Head
    {
        public InfoHead(Case location, Player player)
            : base(location, player)
        {
        }
    }
}
