using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Game;

namespace INSAWars.Units
{
    [Serializable]
    public abstract class Head : Unit
    {
        public override int AttackBase { get { return 0; } }
        public override int DefenseBase { get { return 2; } }
        public override int HitPoints { get { return 5; } }
        public override int MovementPoints { get { return 3; } }

        public override int AttackBonus { get { return 0; } }
        public override int DefenseBonus { get { return 0; } }

        public Head(Case position, Player player)
            : base(position, player)
        {
        }

        public override void Kill()
        {
            base.Kill();
            _player.Head = null;
        }

        public override bool CanBuildCity()
        {
            return false;
        }

        public override string Texture
        {
            get { return "Head"; }
        }

        public override string ToString()
        {
            return "Head";
        }
    }
}
