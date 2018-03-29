using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;
using INSAWars.Units;

namespace INSAWars.Units.Info
{
    [Serializable]
    class InfoCivilization : ICivilization
    {
        public AbstractUnitFactory UnitFactory
        {
            get { return new InfoUnitFactory(); }
        }

        public override string ToString()
        {
            return "Info";
        }
    }
}
