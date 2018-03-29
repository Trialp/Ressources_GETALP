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
    /// Provides a command to move a unit.
    /// </summary>
    public class MoveUnitCommand : CaseCommand
    {
        private Unit _unit;
        private Game _game;

        /// <summary>
        /// Note: ideally, this class should not be coupled to Game. Refactoring needed.
        /// </summary>
        /// <param name="game"></param>
        /// <param name="unit"></param>
        public MoveUnitCommand(Game game, Unit unit)
        {
            _unit = unit;
            _game = game;
        }

        public override void Execute(Case selectedCase)
        {
            _unit.MoveTo(selectedCase);
        }

        public override bool CanExectute(Case selectedCase)
        {
            return _unit.CanMoveTo(selectedCase) && _game.IsVisible(selectedCase);
        }
    }
}
