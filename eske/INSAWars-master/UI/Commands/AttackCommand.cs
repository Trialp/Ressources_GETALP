using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;
using INSAWars.Units;

namespace UI.Commands
{
    /// <summary>
    /// The attack command is used when the player wants to attack a unit on a given case.
    /// </summary>
    public class AttackCommand : CaseCommand
    {
        private Unit _unit;

        /// <summary>
        /// The player's unit that must attack.
        /// </summary>
        /// <param name="u"></param>
        public AttackCommand(Unit u)
        {
            _unit = u;
        }

        public override void Execute(Case selectedCase)
        {
            _unit.Attack(selectedCase);
        }

        public override bool CanExectute(Case selectedCase)
        {
            return _unit.CanAttack(selectedCase);
        }
    }
}