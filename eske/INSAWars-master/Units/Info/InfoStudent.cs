using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units.Info
{
    [Serializable]
    class InfoStudent : Student
    {
        public override int AttackBase { get { return 4; } }
        public override int DefenseBase { get { return 2; } }
        public override int HitPoints { get { return 10; } }
        public override int MovementPoints { get { return 2; } }

        public InfoStudent(Case location, Player player)
            : base(location, player)
        {
        }
    }
}
