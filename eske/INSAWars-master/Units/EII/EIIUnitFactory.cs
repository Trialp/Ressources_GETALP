using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.EII
{
    [Serializable]
    class EIIUnitFactory : AbstractUnitFactory
    {
        public override Head CreateHead(Case position, Player player)
        {
            return new EIIHead(position, player);
        }

        public override Student CreateStudent(Case position, Player player)
        {
            return new EIIStudent(position, player);
        }

        public override Teacher CreateTeacher(Case position, Player player)
        {
            return new EIITeacher(position, player);
        }
    }
}
