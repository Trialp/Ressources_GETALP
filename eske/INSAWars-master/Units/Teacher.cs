using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units
{
    [Serializable]
    public abstract class Teacher : Unit
    {
        public Teacher(Case location, Player player)
            : base(location, player)
        {
        }

        public override bool CanBuildCity()
        {
            return !Location.HasCity;
        }

        public override string Texture
        {
            get { return "Teacher"; }
        }

        public override string ToString()
        {
            return "Teacher";
        }
    }
}
