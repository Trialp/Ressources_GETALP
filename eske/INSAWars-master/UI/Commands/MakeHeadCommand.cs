using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace UI.Commands
{
    /// <summary>
    /// Provides a command to create a new Head unit.
    /// </summary>
    public class MakeHeadCommand : BasicCommand
    {
        private Case _case;

        public MakeHeadCommand(Case c)
        {
            _case = c;
        }

        public override void Execute()
        {
            _case.City.MakeHead();
        }

        public override bool CanExecute()
        {
            return _case.HasCity && _case.City.CanMakeHead();
        }
    }
}
