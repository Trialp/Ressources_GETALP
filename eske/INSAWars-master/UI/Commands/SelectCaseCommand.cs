using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace UI.Commands
{
    /// <summary>
    /// Provides a command to select a case.
    /// </summary>
    public class SelectCaseCommand : CaseCommand
    {
        private Game _game;
        private GameControl _control;
        
        /// <summary>
        /// Note: ideally, this class should not be coupled to Game. Refactoring needed.
        /// </summary>
        /// <param name="g"></param>
        /// <param name="control"></param>
        public SelectCaseCommand(Game g, GameControl control)
        {
            _game = g;
            _control = control;
        }

        public override void Execute(Case selectedCase)
        {
            _control.SelectCase(selectedCase);
        }

        public override bool CanExectute(Case selectedCase)
        {
            return _game.IsVisible(selectedCase);
        }
    }
}
