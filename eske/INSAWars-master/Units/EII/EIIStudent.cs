using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.EII
{
    [Serializable]
    class EIIStudent : Student
    {
        public override int AttackBase { get { return 3; } }
        public override int DefenseBase { get { return 3; } }
        public override int HitPoints { get { return 10; } }
        public override int MovementPoints { get { return 2; } }

        public EIIStudent(Case location, Player player)
            : base(location, player)
        {
        }
    }
}
