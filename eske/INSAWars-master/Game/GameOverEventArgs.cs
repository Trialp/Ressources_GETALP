using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace INSAWars.Game
{
    public class GameOverEventArgs
    {
        private Player _winner;

        public GameOverEventArgs(Player winner)
        {
            _winner = winner;
        }

        public Player Winner
        {
            get { return _winner; }
        }
    }
}
