using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace INSAWars.Game
{
    /// <summary>
    /// Adds iron to a given case.
    /// </summary>
    [Serializable]
    public class IronCaseDecorator : CaseDecorator
    {

        public override int Iron
        {
            get { return decoratedCase.Iron + 2; }
        }

        public IronCaseDecorator(Case decoratedCase)
            : base(decoratedCase) {}
    }
}
