using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.Info
{
    [Serializable]
    class InfoUnitFactory : AbstractUnitFactory
    {
        public override Head CreateHead(Case position, Player player)
        {
            return new InfoHead(position, player);
        }

        public override Student CreateStudent(Case position, Player player)
        {
            return new InfoStudent(position, player);
        }

        public override Teacher CreateTeacher(Case position, Player player)
        {
            return new InfoTeacher(position, player);
        }
    }
}
