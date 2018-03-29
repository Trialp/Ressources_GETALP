using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace INSAWars.Units
{
    public interface ICivilization
    {
        AbstractUnitFactory UnitFactory
        {
            get;
        }
    }
}
