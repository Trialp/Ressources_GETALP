using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace INSAWars.Game
{
    /// <summary>
    /// Adds food to a given case.
    /// </summary>
    [Serializable]
    public class FoodCaseDecorator : CaseDecorator
    {
        public override int Food
        {
            get { return decoratedCase.Food + 2; }
        }

        public FoodCaseDecorator(Case decoratedCase)
            : base(decoratedCase) {}

    }
}
