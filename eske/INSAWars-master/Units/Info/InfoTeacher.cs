using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.Info
{
    [Serializable]
    class InfoTeacher : Teacher
    {
        public override int AttackBase { get { return 0; } }
        public override int DefenseBase { get { return 1; } }
        public override int HitPoints { get { return 1; } }
        public override int MovementPoints { get { return 3; } }

        public InfoTeacher(Case location, Player player)
            : base(location, player)
        {
        }
    }
}
