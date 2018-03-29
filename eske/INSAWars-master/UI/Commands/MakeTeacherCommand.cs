using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace UI.Commands
{
    /// <summary>
    /// Provides a command to make a Teacher unit.
    /// </summary>
    public class MakeTeacherCommand : BasicCommand
    {
        private Case _case;

        public MakeTeacherCommand(Case c)
        {
            _case = c;
        }

        public override void Execute()
        {
            _case.City.MakeTeacher();
        }

        public override bool CanExecute()
        {
            return _case.HasCity && _case.City.CanMakeTeacher();
        }
    }
}
