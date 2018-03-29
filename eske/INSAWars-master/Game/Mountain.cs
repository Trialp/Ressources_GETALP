using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace INSAWars.Game
{
    [Serializable]
    public class Mountain : Case
    {
        public override int Food
        {
            get { return 0; }
        }

        public override int Iron
        {
            get { return 3; }
        }

        public override string Texture
        {
            get { return "Mountain"; }
        }

        public Mountain(int x, int y) : base(x, y) { }

        public override string ToString()
        {
            return "Mountain";
        }
    }
}
