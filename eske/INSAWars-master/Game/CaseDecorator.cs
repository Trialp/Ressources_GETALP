using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Units;

namespace INSAWars.Game
{
    /// <summary>
    /// Defines a Decorator Pattern for cases.
    /// </summary>
    [Serializable]
    public abstract class CaseDecorator : Case
    {
        protected Case decoratedCase;

        public CaseDecorator(Case decoratedCase)
            : base(decoratedCase.X, decoratedCase.Y)
        {
            this.decoratedCase = decoratedCase;
        }

        public override string Texture
        {
            get { return decoratedCase.Texture; }
        }

        public override int Food
        {
            get { return decoratedCase.Food; }
        }

        public override int Iron
        {
            get { return decoratedCase.Iron; }
        }
        
        public override string ToString()
        {
            return decoratedCase.ToString();
        }
    }
}
