#region usings
using System;
using System.IO;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using INSAWars.Units;
using INSAWars.MVVM;
using System.ComponentModel;
#endregion

namespace INSAWars.Game
{
    /// <summary>
    /// Defines a game and some utilities methods.
    /// </summary>
    [Serializable]
    public class Game : ObservableObject
    {
        #region members
        public static Random random = new Random();

        private List<Player> players;
        private Queue<Player> alivePlayers;
        private Map map;
        private int nbTurns;
        private Boolean _over;
        #endregion

        #region events
        public delegate void GameOver(object sender, GameOverEventArgs e);
        [field: NonSerialized]
        public event GameOver GameIsOver;
        #endregion

        #region properties
        public bool Over
        { 
            get { return _over; }
        }

        public List<Player> Players
        {
            get { return players; }
        }

        public int NbTurns { 
            get { return nbTurns; }
            set
            {
                SetProperty(ref nbTurns, value);
            }
        }

        public Map Map { get { return map; } }

        public Player CurrentPlayer { get { return alivePlayers.Peek(); } }
        #endregion

        #region constructors
        /// <summary>
        /// Creates a new game with the given map and the given players.
        /// Players play turn by turn until only one player remains.
        /// </summary>
        /// <param name="map">Map of the game.</param>
        /// <param name="players">Players of the game.</param>
        public Game(Map map, List<Player> players)
        {
            this.players = players;
            this.alivePlayers = new Queue<Player>(players);

            this.map = map;

            this._over = false;
            this.nbTurns = 0;
        }
        #endregion

        #region methods

        /// <summary>
        /// Builds a city on the case on which the given teacher stands.
        /// Note: this should not be in this class. Refactoring needed.
        /// </summary>
        /// <param name="name">Name of the city.</param>
        /// <param name="teacher">Teacher building the city.</param>
        public void BuildCity(Unit teacher)
        {
            Player player = teacher.Player;
            Case position = teacher.Location;

            City city = new City(position, player, map.TerritoryAround(position, City.Radius));

            position.BuildCity(city);
            player.AddCity(city);

            // The teacher sacrifices himself to build the city (may he rest in peace).
            teacher.Kill();
        }

        /// <summary>
        /// Makes the game go to the next turn.
        /// Changes the current player, and updates the state of the game.
        /// </summary>
        public void NextTurn()
        {
            Player player;

            int count = players.Count;
            // Updates the queue of alive players (remove the dead ones).
            for (int i = 0; i < count; i++)
            {
                player = alivePlayers.Dequeue();
                
                if (!player.IsDead)
                {
                    alivePlayers.Enqueue(player);
                }
            }

            // If there are less than 1 alive player left, then the game is over.
            if (alivePlayers.Count <= 1)
            {
                _over = true;

                var handler = GameIsOver;
                if (handler != null)
                {
                    handler(this, new GameOverEventArgs(alivePlayers.First()));
                }

                return;
            }
            else
            {
                NbTurns++;
                // Puts the player to the end of the queue.
                player = alivePlayers.Dequeue();
                alivePlayers.Enqueue(player);
                player.NextTurn();
            }
        }

        /// <summary>
        /// Tells wether or not the given case is visible by the current player.
        /// Note: this should not be in this class. Refactoring needed.
        /// </summary>
        /// <param name="c">Given case</param>
        /// <returns>True if the case is in the field of view of the current player, false otherwise.</returns>
        public bool IsVisible(Case c)
        {
            foreach (Case _c in map.TerritoryAround(c, 3))
            {
                if ((_c.HasCity || _c.IsUsed) && _c.Occupant == CurrentPlayer)
                {
                    return true;
                }
            }

            foreach (Case _c in map.TerritoryAround(c, 2))
            {
                if (_c.HasUnits && _c.Occupant == CurrentPlayer)
                {
                    return true;
                }
            }

            return false;
        }
        #endregion
    }
}
