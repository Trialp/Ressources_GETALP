using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace INSAWars.Units
{
    public class CivilizationFactory
    {
        public static ICivilization GetCivilizationByName(string name)
        {
            switch (name)
            {
                case "INFO":
                    return new Info.InfoCivilization();
                case "EII":
                    return new EII.EIICivilization();
                default:
                    return new Info.InfoCivilization();
            }
        }
    }
}
