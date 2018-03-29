using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using INSAWars.Game;

namespace UI.Commands
{
    /// <summary>
    /// Provides the basis for a Command Pattern related to Case commands.
    /// </summary>
    public abstract class CaseCommand
    {
        public abstract void Execute(Case selectedCase);
        public abstract bool CanExectute(Case selectedCase);
    }
}
